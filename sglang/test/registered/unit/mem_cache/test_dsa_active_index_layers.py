import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.model_config import get_dsa_index_producer_layer_ids
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import (
    compute_state_layout_hash,
    resolve_dsa_state_tensor_mapping,
    setup_state_kv_args,
    validate_state_layout_compatibility,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    compute_dsa_indexer_storage_layout_namespace,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    get_pool_storage_namespace,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _DsaStrategy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dsa_config(**kwargs):
    values = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        num_hidden_layers=12,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


class TestDSAIndexProducerLayers(unittest.TestCase):
    def test_frequency_and_offset(self):
        config = _dsa_config(
            num_hidden_layers=78,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        layer_ids = get_dsa_index_producer_layer_ids(config)
        self.assertEqual(layer_ids, [0, 1, 2, *range(6, 78, 4)])
        self.assertEqual(len(layer_ids), 21)

    def test_pattern_and_pipeline_range(self):
        config = _dsa_config(
            num_hidden_layers=8,
            index_topk_pattern="FSSFFSFS",
        )
        self.assertEqual(
            get_dsa_index_producer_layer_ids(config, 2, 7),
            [3, 4, 6],
        )

    def test_persistent_indexer_namespace_tracks_layout(self):
        compact = compute_dsa_indexer_storage_layout_namespace(
            [0, 2, 6], 8448, torch.uint8
        )
        full = compute_dsa_indexer_storage_layout_namespace(
            list(range(79)), 8448, torch.uint8
        )
        changed_stride = compute_dsa_indexer_storage_layout_namespace(
            [0, 2, 6], 16896, torch.uint8
        )

        self.assertEqual(
            compact,
            compute_dsa_indexer_storage_layout_namespace([0, 2, 6], 8448, torch.uint8),
        )
        self.assertNotEqual(compact, full)
        self.assertNotEqual(compact, changed_stride)
        self.assertEqual(
            get_pool_storage_namespace(
                PoolName.INDEXER,
                SimpleNamespace(storage_layout_namespace=compact),
            ),
            f"indexer.{compact}",
        )


class TestDSAActiveIndexPool(unittest.TestCase):
    def _make_pool(self, index_layer_ids):
        return DSATokenToKVPool(
            size=128,
            page_size=64,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=4,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=160,
            start_layer=2,
            end_layer=6,
            index_layer_ids=index_layer_ids,
        )

    def test_compact_mapping_and_state_order(self):
        pool = self._make_pool([2, 5])
        self.assertEqual(pool.index_layer_ids, (2, 5))
        self.assertEqual(pool.index_layer_num, 2)
        self.assertEqual(len(pool.index_k_with_scale_buffer), 2)
        self.assertIs(
            pool.get_index_k_with_scale_buffer(2),
            pool.index_k_with_scale_buffer[0],
        )
        self.assertIs(
            pool.get_index_k_with_scale_buffer(5),
            pool.index_k_with_scale_buffer[1],
        )
        self.assertEqual(pool.get_index_state_layer_ids(), [2, 5])
        ptrs, data_lens, item_lens = pool.get_state_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(len(data_lens), 2)
        self.assertEqual(len(item_lens), 2)

    def test_inactive_layer_fails_fast(self):
        pool = self._make_pool([2, 5])
        with self.assertRaisesRegex(
            RuntimeError, "layer 3 has no materialized index KV cache"
        ):
            pool.get_index_k_with_scale_buffer(3)

    def test_move_kv_cache_copies_packed_page_rows(self):
        pool = self._make_pool([2])
        buf = pool.get_index_k_with_scale_buffer(2)
        src_loc = 65
        dst_loc = 130
        src_page, src_offset = divmod(src_loc, pool.page_size)
        dst_page, dst_offset = divmod(dst_loc, pool.page_size)
        k_bytes_per_page = pool.page_size * pool.index_head_dim
        keys = buf[:, :k_bytes_per_page].view(
            buf.shape[0], pool.page_size, pool.index_head_dim
        )
        scales = buf[:, k_bytes_per_page:].view(buf.shape[0], pool.page_size, 4)
        keys[src_page, src_offset].fill_(17)
        scales[src_page, src_offset].fill_(29)

        pool.move_kv_cache(
            torch.tensor([dst_loc], dtype=torch.int64),
            torch.tensor([src_loc], dtype=torch.int64),
        )

        self.assertTrue(
            torch.equal(keys[dst_page, dst_offset], keys[src_page, src_offset])
        )
        self.assertTrue(
            torch.equal(scales[dst_page, dst_offset], scales[src_page, src_offset])
        )

    def test_draft_style_pool_keeps_every_layer(self):
        pool = self._make_pool([2, 3, 4, 5])
        self.assertEqual(pool.index_layer_num, pool.layer_num)
        for layer_id in range(2, 6):
            self.assertIsNotNone(pool.get_index_k_with_scale_buffer(layer_id))

    def test_rejects_invalid_mapping(self):
        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            self._make_pool([5, 2])
        with self.assertRaisesRegex(ValueError, "within the pool layer range"):
            self._make_pool([1, 2])

    def test_pd_state_ids_include_ordered_draft_namespace(self):
        target = self._make_pool([2, 5])
        draft = DSATokenToKVPool(
            size=128,
            page_size=64,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=160,
            start_layer=0,
            end_layer=1,
            index_layer_ids=[0],
        )
        kv_args = KVArgs()
        setup_state_kv_args(
            kv_args,
            target,
            draft_token_to_kv_pool=draft,
            total_kv_layers=6,
        )
        self.assertEqual(kv_args.state_types, [StateType.DSA])
        self.assertEqual(kv_args.state_layer_ids, [[2, 5, 6]])
        self.assertEqual(len(kv_args.state_data_ptrs[0]), 3)
        self.assertEqual(len(kv_args.state_layout_hashes[0]), 64)

        validate_state_layout_compatibility(
            kv_args,
            kv_args.state_layer_ids,
            kv_args.state_layout_hashes,
            remote_state_item_lens=kv_args.state_item_lens,
            peer="matching-decode",
        )
        remote_ids = [[0, 2, 6, 5, 9]]
        remote_lens = [
            [1, *kv_args.state_item_lens[0][::2], kv_args.state_item_lens[0][1], 1]
        ]
        remote_hashes = [
            compute_state_layout_hash(StateType.DSA, remote_ids[0], remote_lens[0])
        ]
        self.assertEqual(
            resolve_dsa_state_tensor_mapping(
                kv_args,
                0,
                remote_ids,
                remote_hashes,
                remote_state_tensors=[[1] * len(remote_ids[0])],
                remote_state_item_lens=remote_lens,
                peer="pp1-decode",
            ),
            [(0, 1), (1, 3), (2, 2)],
        )
        missing_ids = [[2, 6]]
        missing_lens = [[kv_args.state_item_lens[0][0], kv_args.state_item_lens[0][2]]]
        missing_hashes = [
            compute_state_layout_hash(StateType.DSA, missing_ids[0], missing_lens[0])
        ]
        with self.assertRaisesRegex(RuntimeError, "state layout mismatch"):
            validate_state_layout_compatibility(
                kv_args,
                missing_ids,
                missing_hashes,
                remote_state_item_lens=missing_lens,
                peer="wrong-decode",
            )
        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            validate_state_layout_compatibility(
                kv_args,
                [],
                [],
                remote_state_item_lens=[],
                peer="old-decode",
            )
        with self.assertRaisesRegex(RuntimeError, "tensor count mismatch"):
            validate_state_layout_compatibility(
                kv_args,
                kv_args.state_layer_ids,
                kv_args.state_layout_hashes,
                [[1, 2]],
                kv_args.state_item_lens,
                peer="truncated-decode",
            )

    def test_zero_producer_stage_has_empty_state_mapping(self):
        pool = self._make_pool([])
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, pool, total_kv_layers=6)

        self.assertEqual(kv_args.state_layer_ids, [[]])
        self.assertEqual(
            resolve_dsa_state_tensor_mapping(
                kv_args,
                0,
                [],
                [],
                peer="pp-stage-with-no-producers",
            ),
            [],
        )

    def test_zero_producer_stage_omits_hicache_sidecar_allocation(self):
        kvcache = SimpleNamespace(
            layer_num=4,
            index_layer_num=0,
            kv_cache_dim=160,
        )
        cache = SimpleNamespace(page_size=64)
        params = SimpleNamespace(
            tp_cache_group=None, token_to_kv_pool_allocator=object()
        )
        server_args = SimpleNamespace(
            hicache_mem_layout="page_first",
            hicache_storage_backend="mooncake",
        )
        host_pool = object()
        host_pool_group = SimpleNamespace(get_pool=MagicMock(return_value=host_pool))
        captured = {}

        def fake_build_anchor_sidecar_stack(**kwargs):
            captured.update(kwargs)
            return host_pool_group, object()

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "maybe_create_dsa_cp_shared_l2_allocator",
                return_value=object(),
            ) as create_allocator,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "build_anchor_sidecar_stack",
                side_effect=fake_build_anchor_sidecar_stack,
            ),
        ):
            result = _DsaStrategy().build(
                cache=cache,
                kvcache=kvcache,
                params=params,
                server_args=server_args,
                load_cache_event=None,
                storage_backend=None,
            )

        self.assertEqual(create_allocator.call_count, 1)
        self.assertEqual(create_allocator.call_args.kwargs["kind"], "dsa_l2_kv")
        self.assertIsNone(captured["sidecar_host_pool_factory"](host_pool))
        self.assertEqual(result.sidecars, [])
        self.assertEqual(result.pools_desc, "KV")


if __name__ == "__main__":
    unittest.main()
