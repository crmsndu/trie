"""Unit tests for pool_configurator.py -- CPU only, no GPU required.

Tests the end-to-end computation: available_bytes -> MemoryPoolConfig,
verifying tokens are correct, constraints are respected, and memory
invariants hold (tokens * per_token_cost <= available_bytes).
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextlib.contextmanager
def mock_cpu_env(kv_size=2, tp_size=1, swa_eviction_interval=4):
    """Mock GPU-dependent functions for CPU-only testing.

    swa_eviction_interval pins SGLANG_SWA_EVICTION_INTERVAL (decode batches between
    SWA evictions) to a small value so the chunk-cap formula stays hand-computable;
    only SWAChunkCapPoolConfigurator reads it.
    """
    from sglang.srt.environ import envs

    with (
        patch("torch._utils._element_size", return_value=kv_size),
        patch(
            "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
            return_value=tp_size,
        ),
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(swa_eviction_interval),
    ):
        yield


def _make_model_runner(
    *,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
    num_layers=32,
    use_mla_backend=False,
    is_hybrid_swa=False,
    full_attention_layer_ids=None,
    swa_attention_layer_ids=None,
    swa_num_kv_heads=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    swa_full_tokens_ratio=0.5,
    page_size=1,
    mambaish_config=None,
    disable_radix_cache=False,
    chunked_prefill_size=None,
    disable_overlap_schedule=False,
    sliding_window_size=None,
    speculative_num_draft_tokens=None,
    max_speculative_num_draft_tokens=None,
    speculative_algorithm=None,
    speculative_num_steps=None,
    speculative_eagle_topk=None,
    disaggregation_mode="null",
    max_running_requests=None,
    disaggregation_decode_extra_slots=0,
    kv_cache_dtype="fake_bf16",
    dsa_model=False,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    dsa_prefill_backend="flashmla_sparse",
    dsa_decode_backend="flashmla_sparse",
    enable_hisparse=False,
    enable_dsa_cp_shared_kv_cache=False,
):
    """Create a mock ModelRunner with the fields configurators need."""
    mr = MagicMock()

    mr.use_mla_backend = use_mla_backend
    mr.is_draft_worker = False
    mr.num_effective_layers = num_layers
    mr.start_layer = 0
    mr.end_layer = num_layers
    mr.dp_size = 1
    mr.page_size = page_size
    mr.mambaish_config = mambaish_config
    mr.is_hybrid_swa = is_hybrid_swa
    mr.sliding_window_size = sliding_window_size
    mr.enable_hisparse = enable_hisparse

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.is_hybrid_swa = is_hybrid_swa
    mc.full_attention_layer_ids = (
        full_attention_layer_ids
        if full_attention_layer_ids is not None
        else list(range(num_layers))
    )
    mc.swa_attention_layer_ids = (
        swa_attention_layer_ids if swa_attention_layer_ids is not None else []
    )
    mc.swa_head_dim = swa_head_dim or head_dim
    mc.swa_v_head_dim = swa_v_head_dim or v_head_dim
    mc.get_num_kv_heads = lambda tp_size: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size: swa_num_kv_heads or num_kv_heads
    mc.kv_lora_rank = kv_lora_rank
    mc.qk_rope_head_dim = qk_rope_head_dim
    mc.hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM" if dsa_model else "LlamaForCausalLM"],
        index_topk=2048 if dsa_model else None,
        index_head_dim=128 if dsa_model else None,
    )
    mr.model_config = mc

    mr.kv_cache_dtype = kv_cache_dtype

    sa = SimpleNamespace()
    sa.swa_full_tokens_ratio = swa_full_tokens_ratio
    sa.page_size = page_size
    sa.disable_radix_cache = disable_radix_cache
    sa.chunked_prefill_size = chunked_prefill_size
    sa.disable_overlap_schedule = disable_overlap_schedule
    sa.speculative_num_draft_tokens = speculative_num_draft_tokens
    sa.max_speculative_num_draft_tokens = (
        max_speculative_num_draft_tokens or speculative_num_draft_tokens
    )
    sa.speculative_algorithm = speculative_algorithm
    sa.speculative_num_steps = speculative_num_steps
    sa.speculative_eagle_topk = speculative_eagle_topk
    sa.disaggregation_mode = disaggregation_mode
    sa.max_running_requests = max_running_requests
    sa.disaggregation_decode_extra_slots = disaggregation_decode_extra_slots
    sa.dsa_prefill_backend = dsa_prefill_backend
    sa.dsa_decode_backend = dsa_decode_backend
    sa.enable_dsa_cp_shared_kv_cache = enable_dsa_cp_shared_kv_cache
    mr.server_args = sa

    spec = MagicMock()
    spec.is_eagle.return_value = False
    spec.is_standalone.return_value = False
    spec.is_dflash.return_value = False
    spec.is_none.return_value = True
    mr.spec_algorithm = spec

    return mr


KV_SIZE = 2  # bf16


def _full_per_token(mr):
    mc = mr.model_config
    return mc.get_num_kv_heads(1) * (mc.head_dim + mc.v_head_dim) * KV_SIZE


def _swa_per_token(mr):
    mc = mr.model_config
    return mc.get_swa_num_kv_heads(1) * (mc.swa_head_dim + mc.swa_v_head_dim) * KV_SIZE


def _actual_memory_used(mr, config):
    """Compute actual memory consumed by the pool sizes in config."""
    mc = mr.model_config
    full_pt = _full_per_token(mr)
    swa_pt = _swa_per_token(mr)
    nf = len(mc.full_attention_layer_ids)
    ns = len(mc.swa_attention_layer_ids)

    if mr.is_hybrid_swa:
        full = config.full_max_total_num_tokens or 0
        swa = config.swa_max_total_num_tokens or 0
        return full * full_pt * nf + swa * swa_pt * ns
    else:
        return config.max_total_num_tokens * full_pt * (nf + ns)


class TestDefaultConfigurator(unittest.TestCase):
    """Default (MHA): available_bytes -> tokens, memory invariant holds."""

    def _run(self, available_bytes, page_size=1, **kwargs):
        mr = _make_model_runner(page_size=page_size, **kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, cfg, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_page_alignment(self):
        available = 10_000_000
        _, _, config = self._run(available, page_size=128)
        self.assertEqual(config.max_total_num_tokens % 128, 0)

    def test_constraint_respected(self):
        """calculate_pool_sizes_from_max_tokens respects the limit."""
        mr, cfg, config = self._run(10_000_000)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(100, page_size=1)
        self.assertEqual(constrained.max_total_num_tokens, 100)

    def test_constraint_page_aligned(self):
        mr, cfg, _ = self._run(10_000_000, page_size=128)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(1000, page_size=128)
        self.assertEqual(constrained.max_total_num_tokens, 896)  # 1000 // 128 * 128

    def test_no_swa_fields(self):
        _, _, config = self._run(10_000_000)
        self.assertIsNone(config.full_max_total_num_tokens)
        self.assertIsNone(config.swa_max_total_num_tokens)


class TestDSAFP8Configurator(unittest.TestCase):
    NUM_LAYERS = 78
    RAW_MAIN_DIM = 576
    SCALED_MAIN_DIM = 656
    INDEXER_DIM = 132

    def _make_configurator(
        self,
        *,
        kv_cache_dtype,
        cp_size=1,
        cuda=True,
        npu=False,
        dsa_prefill_backend="flashmla_sparse",
        dsa_decode_backend="flashmla_sparse",
        enable_hisparse=False,
        draft_num_layers=0,
        vmm_granularities=(2 * 1024 * 1024,),
        index_topk_freq=None,
        index_skip_topk_offset=None,
    ):
        mr = _make_model_runner(
            num_layers=self.NUM_LAYERS,
            use_mla_backend=True,
            dsa_model=True,
            kv_cache_dtype=kv_cache_dtype,
            dsa_prefill_backend=dsa_prefill_backend,
            dsa_decode_backend=dsa_decode_backend,
            enable_hisparse=enable_hisparse,
            enable_dsa_cp_shared_kv_cache=cp_size > 1,
        )
        if draft_num_layers:
            mr.spec_algorithm.is_eagle.return_value = True
            mr.spec_algorithm.is_none.return_value = False
            mr.eagle_draft_num_layers = draft_num_layers
        if index_topk_freq is not None:
            mr.model_config.hf_config.index_topk_freq = index_topk_freq
        if index_skip_topk_offset is not None:
            mr.model_config.hf_config.index_skip_topk_offset = index_skip_topk_offset

        with (
            patch(
                "sglang.srt.model_executor.pool_configurator.current_platform.is_cuda",
                return_value=cuda,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.current_platform.is_npu",
                return_value=npu,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_attention_cp_size",
                return_value=cp_size,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_rank_major_shared_tensor_granularities",
                return_value=vmm_granularities,
            ),
        ):
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
            )

            return DefaultPoolConfigurator(mr)

    def _cell_size(self, **kwargs):
        return self._make_configurator(**kwargs)._cell_size

    def test_cuda_fp8_nonshared_uses_scaled_main_layout(self):
        cell_size = self._cell_size(kv_cache_dtype=torch.float8_e4m3fn)
        self.assertEqual(
            cell_size,
            (self.SCALED_MAIN_DIM + self.INDEXER_DIM) * self.NUM_LAYERS,
        )
        self.assertEqual(cell_size, 61_464)

    def test_cuda_fp8_cp8_shares_scaled_main_pool_only(self):
        cell_size = self._cell_size(kv_cache_dtype=torch.float8_e4m3fn, cp_size=8)
        expected = (
            self.SCALED_MAIN_DIM * self.NUM_LAYERS // 8
            + self.INDEXER_DIM * self.NUM_LAYERS
        )
        self.assertEqual(cell_size, expected)
        self.assertEqual(cell_size, 16_692)

    def test_cuda_fp8_cp8_with_one_mtp_layer(self):
        cell_size = self._cell_size(
            kv_cache_dtype=torch.float8_e4m3fn,
            cp_size=8,
            draft_num_layers=1,
        )
        self.assertEqual(cell_size, 16_906)

    def test_cuda_fp8_cp8_uses_only_index_producer_layers(self):
        cell_size = self._cell_size(
            kv_cache_dtype=torch.float8_e4m3fn,
            cp_size=8,
            draft_num_layers=1,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        # Target producers: 0, 1, 2, then every fourth layer 6..74 (21 total).
        expected = (
            self.SCALED_MAIN_DIM * self.NUM_LAYERS // 8
            + self.INDEXER_DIM * 21
            + self.SCALED_MAIN_DIM // 8
            + self.INDEXER_DIM
        )
        self.assertEqual(cell_size, expected)
        self.assertEqual(cell_size, 9_382)

    def test_cuda_fp8_cp8_active_index_layers_with_three_mtp_layers(self):
        configurator = self._make_configurator(
            kv_cache_dtype=torch.float8_e4m3fn,
            cp_size=8,
            draft_num_layers=3,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        expected = (
            self.SCALED_MAIN_DIM * self.NUM_LAYERS // 8
            + self.INDEXER_DIM * 21
            + 3 * (self.SCALED_MAIN_DIM // 8 + self.INDEXER_DIM)
        )
        self.assertEqual(configurator._cell_size, expected)
        self.assertEqual(configurator._cell_size, 9_810)
        vmm_config = configurator._dsa_cp_shared_vmm_config
        self.assertIsNotNone(vmm_config)
        self.assertEqual(vmm_config.num_main_layers, 81)
        self.assertEqual(vmm_config.num_index_layers, 24)

    def test_npu_keeps_all_index_layers(self):
        cell_size = self._cell_size(
            kv_cache_dtype=torch.float8_e4m3fn,
            cuda=False,
            npu=True,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        expected = (self.RAW_MAIN_DIM + self.INDEXER_DIM) * self.NUM_LAYERS
        self.assertEqual(cell_size, expected)

    def test_cuda_fp8_cp8_vmm_padding_limits_capacity(self):
        from sglang.srt.distributed.device_communicators.vmm_utils import (
            get_padded_first_dim_for_vmm,
        )

        configurator = self._make_configurator(
            kv_cache_dtype=torch.float8_e4m3fn,
            cp_size=8,
            draft_num_layers=1,
        )

        # The pre-fix run selected this capacity with its old 16,116 B/token
        # coefficient. Use the lower edge of the corresponding profiled budget.
        available_bytes = 3_078_912 * 16_116
        config = configurator.calculate_pool_sizes(available_bytes, page_size=64)
        self.assertEqual(config.max_total_num_tokens, 2_804_032)

        vmm_config = configurator._dsa_cp_shared_vmm_config
        self.assertIsNotNone(vmm_config)
        self.assertEqual(
            vmm_config.required_bytes(config.max_total_num_tokens, 64),
            49_619_139_072,
        )
        self.assertLessEqual(
            vmm_config.required_bytes(config.max_total_num_tokens, 64),
            available_bytes,
        )
        self.assertGreater(
            vmm_config.required_bytes(config.max_total_num_tokens + 64, 64),
            available_bytes,
        )

        # 3,078,912 tokens request 6,014 local pages, but 656-byte rows and a
        # 2 MiB VMM granularity require a 131,072-row (2,048-page) boundary.
        padded_rows, aligned_bytes = get_padded_first_dim_for_vmm(
            (6_014 * 64, self.SCALED_MAIN_DIM),
            dtype=torch.uint8,
            granularity=2 * 1024 * 1024,
        )
        self.assertEqual(padded_rows // 64, 6_144)
        self.assertEqual(aligned_bytes, 257_949_696)

    def test_cuda_fp8_cp8_vmm_sizing_covers_fallback_granularity(self):
        available_bytes = 3_078_912 * 16_116
        small_granularity = self._make_configurator(
            kv_cache_dtype=torch.float8_e4m3fn,
            cp_size=8,
            draft_num_layers=1,
            vmm_granularities=(256 * 1024,),
        )
        both_paths = self._make_configurator(
            kv_cache_dtype=torch.float8_e4m3fn,
            cp_size=8,
            draft_num_layers=1,
            vmm_granularities=(256 * 1024, 2 * 1024 * 1024),
        )

        self.assertEqual(
            small_granularity.calculate_pool_sizes(
                available_bytes, 64
            ).max_total_num_tokens,
            2_885_504,
        )
        self.assertEqual(
            both_paths.calculate_pool_sizes(available_bytes, 64).max_total_num_tokens,
            2_804_032,
        )

    def test_raw_dsa_layouts_remain_unchanged(self):
        cases = (
            (
                "bf16",
                torch.bfloat16,
                True,
                "flashmla_sparse",
                "flashmla_sparse",
                False,
            ),
            ("trtllm", torch.float8_e4m3fn, True, "trtllm", "trtllm", False),
            ("hip", torch.float8_e4m3fn, False, "tilelang", "aiter", False),
            (
                "npu",
                torch.float8_e4m3fn,
                False,
                "flashmla_sparse",
                "flashmla_sparse",
                False,
            ),
            (
                "hisparse",
                torch.float8_e4m3fn,
                True,
                "flashmla_sparse",
                "flashmla_sparse",
                True,
            ),
        )
        for (
            name,
            dtype,
            cuda,
            prefill_backend,
            decode_backend,
            enable_hisparse,
        ) in cases:
            with self.subTest(name=name):
                cell_size = self._cell_size(
                    kv_cache_dtype=dtype,
                    cuda=cuda,
                    dsa_prefill_backend=prefill_backend,
                    dsa_decode_backend=decode_backend,
                    enable_hisparse=enable_hisparse,
                )
                kv_size = torch._utils._element_size(dtype)
                expected = (
                    self.RAW_MAIN_DIM * kv_size + self.INDEXER_DIM
                ) * self.NUM_LAYERS
                self.assertEqual(cell_size, expected)

    def test_model_runner_and_pool_share_scaled_layout_calculation(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLM"],
                    index_topk=2048,
                ),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
            kv_cache_dtype=torch.float8_e4m3fn,
            server_args=SimpleNamespace(
                dsa_prefill_backend="flashmla_sparse",
                dsa_decode_backend="flashmla_sparse",
            ),
        )
        pool_dim = DSATokenToKVPool.calculate_kv_cache_dim(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            kv_cache_dtype=torch.float8_e4m3fn,
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_sparse",
            is_hip_platform=False,
        )
        runner_dim = ModelRunnerKVCacheMixin.calculate_mla_kv_cache_dim(runner)
        self.assertEqual(pool_dim, 656)
        self.assertEqual(runner_dim, pool_dim)

        runner.server_args.dsa_decode_backend = "trtllm"
        self.assertEqual(
            ModelRunnerKVCacheMixin.calculate_mla_kv_cache_dim(runner), 576
        )

    def test_non_dsa_fp8_mla_remains_unchanged(self):
        mr = _make_model_runner(
            num_layers=self.NUM_LAYERS,
            use_mla_backend=True,
            dsa_model=False,
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        with (
            patch(
                "sglang.srt.model_executor.pool_configurator.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
                return_value=1,
            ),
        ):
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
            )

            cell_size = DefaultPoolConfigurator(mr)._cell_size
        self.assertEqual(cell_size, self.RAW_MAIN_DIM * self.NUM_LAYERS)


class TestHybridSWAConfigurator(unittest.TestCase):
    """Hybrid SWA: full/swa split, ratio, memory invariant."""

    def _make_swa_runner(self, full_layers=16, swa_layers=16, ratio=0.5, page_size=1):
        return _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens ~= full_tokens * ratio (within page alignment)"""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            mr, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(swa, int(full * ratio), f"ratio={ratio}")

    def test_ratio_with_page_alignment(self):
        """With page alignment, swa_tokens = align(full_tokens * ratio)"""
        available = 10_000_000
        mr, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For hybrid, max_total_num_tokens = full_max_total_num_tokens"""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run"""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget (but less than profiled due to constraint)."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, mr.server_args.page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        # constrained should use roughly half the memory
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts"""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )

    def test_chunk_cache_cap_accounts_for_spec_topk_page_rounding(self):
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=True,  # spec-v1: no double allocation
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # spec-v1 (overlap off): decode_alloc = max(ceil_align(3+4,4)*2,
        # ceil_align(5,4)) = 16. trailing = 8 + 20 + page(4) = 32; per req =
        # 32 + 16 = 48. Global prefill = 1*chunk(4) + page(4) = 8.
        # cap = 48 * 2 + 8 = 104 -> ceil_align(104, 4) = 104.
        self.assertEqual(config.swa_max_total_num_tokens, 104)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_doubles_decode_alloc_for_spec_v2_overlap(self):
        # Overlap on -> spec-v2: decode_alloc = 2 * get_alloc_len_per_decode =
        # 2 * max(steps*topk, max_draft) = 2 * max(6, 5) = 12 (page=1, since the
        # v2 allocator does not support page>1 & topk>1). trailing = 8 + 20 +
        # page(1) = 29; per req = 29 + 12 = 41. Global prefill =
        # 2*chunk(4) + page(1) = 9; cap = 41 * 2 + 9 = 91.
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=1,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=False,  # spec-v2: 2 * get_alloc_len_per_decode
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        self.assertEqual(config.swa_max_total_num_tokens, 91)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_drops_prefill_for_disagg_decode(self):
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # disagg decode drops the prefill term: per req = 4 + 1 + 4 + 1 = 10 (as above).
        self.assertEqual(config.swa_max_total_num_tokens, 100)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_prefill_holds_window_plus_chunk(self):
        # Non-decode (prefill) engine: each request keeps its decode footprint, while
        # in-flight chunked-prefill tokens are a global batch budget -- two chunks
        # under overlap.
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=16,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            disaggregation_mode="prefill",
            disable_overlap_schedule=False,  # overlap -> 2 chunks in flight
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # per req = trailing(window(8) + eviction(4) + page(4)) + decode_alloc(4)
        # = 20. Global prefill = 2*chunk(16) + page(4) = 36.
        # cap = 20 * max_running_requests(2) + 36 = 76.
        self.assertEqual(config.swa_max_total_num_tokens, 76)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_disagg_decode_pre_alloc(self):
        # decode adds disaggregation_decode_extra_slots in-transfer slots to the
        # request count (num_reserved_decode_tokens is a full-pool concern, not SWA).
        available = 2_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
            disaggregation_decode_extra_slots=2,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # active per req = 4 + 1 + 4 + 1 = 10 for the 10 running requests; the 2
        # in-transfer extra slots hold only window + page = 4 + 1 = 5 each.
        # cap = 10 * 10 + 5 * 2 = 110.
        self.assertEqual(config.swa_max_total_num_tokens, 110)
        self.assertLessEqual(_actual_memory_used(mr, config), available)


class TestAllSWAConfigurator(unittest.TestCase):
    """All-SWA (full_layers=0): special case."""

    def _run(self, available_bytes, ratio=0.5, page_size=1, **kwargs):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[],
            swa_attention_layer_ids=list(range(32)),
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=ratio,
            page_size=page_size,
            **kwargs,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_full_max_is_zero(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.full_max_total_num_tokens, 0)

    def test_max_total_equals_swa(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.swa_max_total_num_tokens)

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        swa_pt = _swa_per_token(mr)
        ns = len(mr.model_config.swa_attention_layer_ids)
        used = config.swa_max_total_num_tokens * swa_pt * ns
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_constraint_respected(self):
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(500, page_size=1)
        self.assertEqual(config.max_total_num_tokens, 500)
        self.assertEqual(config.swa_max_total_num_tokens, 500)


class TestEagleConfigurator(unittest.TestCase):
    """EAGLE: draft KV cache must be accounted for so total allocation fits in budget."""

    def test_eagle_does_not_exceed_budget(self):
        """Total memory (target + draft KV cache) must not exceed available."""
        available = 10_000_000
        num_layers = 32
        eagle_draft_num_layers = 4

        mr = _make_model_runner(num_layers=num_layers)
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_standalone.return_value = False
        mr.spec_algorithm.is_none.return_value = False
        mr.eagle_draft_num_layers = eagle_draft_num_layers

        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, 1)

        full_pt = _full_per_token(mr)
        total_layers = num_layers + eagle_draft_num_layers
        used = config.max_total_num_tokens * full_pt * total_layers
        self.assertLessEqual(used, available)


class TestFactory(unittest.TestCase):
    def test_default_for_non_swa(self):
        mr = _make_model_runner(is_hybrid_swa=False)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, DefaultPoolConfigurator)

    def test_swa_for_hybrid(self):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(16)),
            swa_attention_layer_ids=list(range(16, 32)),
            swa_num_kv_heads=4,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                HybridSWAPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, HybridSWAPoolConfigurator)

    def test_chunk_cap_configurator_selection(self):
        # SWAChunkCapPoolConfigurator is selected only when max_running_requests is set.
        def _cfg(max_running_requests):
            mr = _make_model_runner(
                is_hybrid_swa=True,
                full_attention_layer_ids=[0],
                swa_attention_layer_ids=[1],
                swa_num_kv_heads=4,
                disable_radix_cache=True,
                chunked_prefill_size=4,
                sliding_window_size=8,
                max_running_requests=max_running_requests,
            )
            with mock_cpu_env():
                from sglang.srt.model_executor.pool_configurator import (
                    create_memory_pool_configurator,
                )

                return create_memory_pool_configurator(mr)

        from sglang.srt.model_executor.pool_configurator import (
            SWAChunkCapPoolConfigurator,
        )

        self.assertIsInstance(_cfg(2), SWAChunkCapPoolConfigurator)
        self.assertNotIsInstance(_cfg(None), SWAChunkCapPoolConfigurator)


if __name__ == "__main__":
    unittest.main()
