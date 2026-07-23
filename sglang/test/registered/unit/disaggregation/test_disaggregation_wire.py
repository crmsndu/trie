import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVTransferMetric
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    filter_kv_indices_for_cp_rank,
    get_dsv4_c128_state_indices,
    localize_shared_kv_indices_for_cp_rank,
    select_shared_kv_transfer_pairs,
    should_transfer_replicated_cp_payload,
)
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestGroupConcurrentContiguous(unittest.TestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_both_empty(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([])), ([], [])
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


class TestSharedKvCpDisaggregation(unittest.TestCase):
    @staticmethod
    def _prefill_manager(cp_rank, cp_size=4):
        return SimpleNamespace(
            attn_cp_rank=cp_rank,
            attn_cp_size=cp_size,
            server_args=SimpleNamespace(enable_dsa_cp_shared_kv_cache=True),
        )

    @staticmethod
    def _decode_manager(cp_rank, cp_size=4, all_cp=True, shared=True):
        mgr = object.__new__(CommonKVManager)
        mgr.attn_tp_size = 1
        mgr.attn_cp_size = cp_size
        mgr.attn_cp_rank = cp_rank
        mgr.enable_all_cp_ranks_for_transfer = all_cp
        mgr.is_mla_backend = True
        mgr.kv_args = SimpleNamespace(engine_rank=cp_rank)
        mgr.server_args = SimpleNamespace(enable_dsa_cp_shared_kv_cache=shared)
        mgr.pp_size = 1
        mgr.pp_rank = 0
        return mgr

    @staticmethod
    def _prefill_info(cp_size=4, all_cp=True, shared=True):
        return PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=cp_size,
            dp_size=1,
            pp_size=1,
            page_size=64,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            enable_dsa_cp_shared_kv_cache=shared,
            all_cp_ranks_transfer=all_cp,
        )

    def test_independent_allocators_form_complete_owner_cross_product(self):
        cp_size = 4
        # Same request positions, deliberately unrelated P and D logical pages.
        src_global = np.array([8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32)
        dst_global = np.array([13, 8, 15, 10, 9, 14, 11, 12], dtype=np.int32)
        observed = {}

        for src_rank in range(cp_size):
            src_local, positions = filter_kv_indices_for_cp_rank(
                self._prefill_manager(src_rank, cp_size),
                src_global,
                slice(0, len(src_global)),
            )
            for dst_rank in range(cp_size):
                dst_local_full = localize_shared_kv_indices_for_cp_rank(
                    dst_global, cp_rank=dst_rank, cp_size=cp_size
                )
                dst_candidates = dst_local_full[positions]
                src_pair, dst_pair = select_shared_kv_transfer_pairs(
                    src_local, dst_candidates
                )
                valid_positions = positions[dst_candidates >= 0]
                for pos, src_page, dst_page in zip(valid_positions, src_pair, dst_pair):
                    pos = int(pos)
                    self.assertNotIn(pos, observed)
                    observed[pos] = (
                        src_rank,
                        dst_rank,
                        int(src_page),
                        int(dst_page),
                    )

        self.assertEqual(sorted(observed), list(range(len(src_global))))
        self.assertTrue(
            any(src_rank != dst_rank for src_rank, dst_rank, _, _ in observed.values())
        )
        for pos, (src_rank, dst_rank, src_page, dst_page) in observed.items():
            self.assertEqual(src_rank, int(src_global[pos] % cp_size))
            self.assertEqual(dst_rank, int(dst_global[pos] % cp_size))
            self.assertEqual(src_page, int(src_global[pos] // cp_size))
            self.assertEqual(dst_page, int(dst_global[pos] // cp_size))

    def test_matching_cp_shared_kv_maps_all_to_all(self):
        mgr = self._decode_manager(cp_rank=2)
        info = self._prefill_info()

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_cp_ranks, [0, 1, 2, 3])
        self.assertEqual(info.required_dst_info_num, 4)
        self.assertEqual(info.required_prefill_response_num, 4)

    def test_shared_kv_bounds_owner_positions_for_short_decode_metadata(self):
        # The second owner position belongs to a later chunk page that is absent
        # from decode metadata (the existing page_size>1 one-page drift case).
        src_local = np.array([20, 21], dtype=np.int32)
        src_positions = np.array([4, 6], dtype=np.int64)
        dst_local_full = np.array([-1, -1, -1, -1, 30, -1], dtype=np.int32)

        src_pair, dst_pair = select_shared_kv_transfer_pairs(
            src_local,
            dst_local_full,
            src_positions=src_positions,
        )

        np.testing.assert_array_equal(src_pair, np.array([20], dtype=np.int32))
        np.testing.assert_array_equal(dst_pair, np.array([30], dtype=np.int32))

    def test_matching_cp_shared_kv_requires_all_rank_transfer(self):
        mgr = self._decode_manager(cp_rank=0, all_cp=False)
        with self.assertRaisesRegex(RuntimeError, "ALL_CP_RANKS_TRANSFER"):
            mgr._resolve_rank_mapping(self._prefill_info())

    def test_decode_cp1_keeps_legacy_prefill_fanout(self):
        mgr = self._decode_manager(cp_rank=0, cp_size=1, shared=False)
        info = self._prefill_info(cp_size=4)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_cp_ranks, [0, 1, 2, 3])
        self.assertEqual(info.required_dst_info_num, 1)
        self.assertEqual(info.required_prefill_response_num, 4)

    def test_decode_cp1_shared_prefill_requires_local_all_rank_transfer(self):
        mgr = self._decode_manager(cp_rank=0, cp_size=1, all_cp=False, shared=False)

        with self.assertRaisesRegex(RuntimeError, "on both roles"):
            mgr._resolve_rank_mapping(self._prefill_info(cp_size=4))

    def test_only_cp0_sends_replicated_indexer_and_aux(self):
        self.assertTrue(
            should_transfer_replicated_cp_payload(enable_shared_kv=True, cp_rank=0)
        )
        for cp_rank in range(1, 4):
            self.assertFalse(
                should_transfer_replicated_cp_payload(
                    enable_shared_kv=True, cp_rank=cp_rank
                )
            )


class TestMooncakeTransferTiming(unittest.TestCase):
    @staticmethod
    def _manager():
        mgr = object.__new__(MooncakeKVManager)
        mgr.engine = MagicMock()
        mgr.engine.batch_transfer_sync.return_value = 0
        mgr._room_transfer_timings = {}
        mgr._room_transfer_generations = {}
        mgr._transfer_metric_generation_counter = 0
        mgr._room_transfer_timings_lock = threading.Lock()
        return mgr

    def test_tracks_actual_batch_span_and_bytes_per_room(self):
        mgr = self._manager()
        mgr.open_transfer_metric(7)

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.perf_counter",
            side_effect=[10.0, 12.0, 11.0, 14.0],
        ):
            mgr._batch_transfer_sync(7, "session", [1], [2], [100])
            mgr._batch_transfer_sync(7, "session", [3, 4], [5, 6], [20, 30])

        metric = mgr.pop_transfer_metric(7)
        self.assertIsNotNone(metric)
        self.assertEqual(metric.transfer_latency_s, 4.0)
        self.assertEqual(metric.transfer_total_bytes, 150)
        self.assertFalse(metric.allow_latency_fallback)
        self.assertIsNone(mgr.pop_transfer_metric(7))

    def test_close_and_reopen_rejects_late_transfer_metric(self):
        mgr = self._manager()
        transfer_started = threading.Event()
        allow_transfer_finish = threading.Event()

        def blocking_transfer(*_args):
            transfer_started.set()
            allow_transfer_finish.wait(timeout=5)
            return 0

        mgr.engine.batch_transfer_sync.side_effect = blocking_transfer
        mgr.open_transfer_metric(7)
        old_transfer = threading.Thread(
            target=mgr._batch_transfer_sync,
            args=(7, "old-session", [1], [2], [100]),
        )
        old_transfer.start()
        self.assertTrue(transfer_started.wait(timeout=5))

        mgr.close_transfer_metric(7)
        mgr.open_transfer_metric(7)
        allow_transfer_finish.set()
        old_transfer.join(timeout=5)

        self.assertFalse(old_transfer.is_alive())
        self.assertNotIn(7, mgr._room_transfer_timings)

        mgr.engine.batch_transfer_sync.side_effect = None
        mgr._batch_transfer_sync(7, "new-session", [3], [4], [50])
        self.assertEqual(mgr.pop_transfer_metric(7).transfer_total_bytes, 50)
        self.assertNotIn(7, mgr._room_transfer_generations)

    def test_close_transfer_metric_releases_room_bookkeeping(self):
        mgr = self._manager()
        mgr.open_transfer_metric(7)
        mgr._batch_transfer_sync(7, "session", [1], [2], [100])

        mgr.close_transfer_metric(7)

        self.assertNotIn(7, mgr._room_transfer_timings)
        self.assertNotIn(7, mgr._room_transfer_generations)

    def test_dummy_cp_sender_does_not_open_transfer_metric(self):
        mgr = SimpleNamespace(
            is_dummy_cp_rank=True,
            open_transfer_metric=MagicMock(),
            update_status=MagicMock(),
        )

        with patch.object(MooncakeKVSender, "_init_trace_ctx"):
            MooncakeKVSender(mgr, "prefill:8998", 7, [0], 0)

        mgr.open_transfer_metric.assert_not_called()

    def test_disables_queue_to_completion_latency_fallback(self):
        stats = object.__new__(SchedulerReqTimeStats)
        stats.prefill_transfer_queue_entry_time = 10.0
        stats.completion_time = 20.0
        metric = KVTransferMetric(
            transfer_total_bytes=1024,
            allow_latency_fallback=False,
        )

        self.assertIsNone(stats.compute_and_observe_kv_transfer_metrics(metric))

    def test_req_time_stats_exposes_actual_transfer_latency(self):
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.PREFILL)
        metric = KVTransferMetric(
            transfer_latency_s=0.012345,
            transfer_total_bytes=1024 * 1024,
            allow_latency_fallback=False,
        )

        result = stats.compute_and_observe_kv_transfer_metrics(metric)

        self.assertEqual(result["latency_ms"], 12.345)
        self.assertEqual(stats.transfer_latency_ms, 12.345)
        self.assertIn("transfer_latency=12.345ms", stats.convert_to_duration())


class TestEagleDsaSeedTransfer(unittest.TestCase):
    @staticmethod
    def _make_req(seed, metadata_buffer_index=0):
        return SimpleNamespace(
            metadata_buffer_index=metadata_buffer_index,
            output_ids=[101],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            multimodal_inputs=None,
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=torch.tensor([1.0, 2.0]),
            output_topk_p=torch.tensor([1.0]),
            output_topk_index=torch.tensor([7]),
            output_dsa_topk_indices=seed,
            bootstrap_room=9,
        )

    def test_metadata_buffer_copies_seed_and_uses_invalid_sentinel(self):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=2,
            hidden_states_dtype=torch.float32,
            output_dsa_topk_indices_dim=3,
        )
        seed = torch.tensor([4, 5, 6], dtype=torch.int32)
        buffers.set_buf(self._make_req(seed))
        buffers.set_buf(self._make_req(None, metadata_buffer_index=1))

        self.assertTrue(torch.equal(buffers.output_dsa_topk_indices[0], seed))
        self.assertEqual(buffers.output_dsa_topk_indices[1].tolist(), [-1, -1, -1])
        ptrs, data_lens, item_lens = buffers.get_buf_infos()
        self.assertEqual(ptrs[-2], buffers.output_dsa_topk_indices.data_ptr())
        self.assertEqual(data_lens[-2], buffers.output_dsa_topk_indices.nbytes)
        self.assertEqual(item_lens[-2], buffers.output_dsa_topk_indices[0].nbytes)

    def test_decode_input_requires_valid_seed_for_every_request(self):
        seeds = (
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5, 6], dtype=torch.int32),
        )
        batch = SimpleNamespace(
            reqs=[self._make_req(seed) for seed in seeds],
            device="cpu",
            enable_overlap=False,
        )
        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            enable_multi_layer_eagle=False,
        )
        last_tokens = torch.tensor([11, 12], dtype=torch.int64)

        draft_input = build_eagle_disagg_draft_input(
            batch, server_args, last_tokens, None
        )
        self.assertTrue(torch.equal(draft_input.dsa_topk_indices, torch.stack(seeds)))

        for invalid_seed in (None, torch.full((3,), -1, dtype=torch.int32)):
            batch.reqs[1].output_dsa_topk_indices = invalid_seed
            draft_input = build_eagle_disagg_draft_input(
                batch, server_args, last_tokens, None
            )
            self.assertIsNone(draft_input.dsa_topk_indices)

    def test_future_map_initializes_seed_buffer_after_seedless_payload(self):
        future_map = object.__new__(FutureMap)
        future_map.dsa_topk_indices_buf = None
        future_map.req_pool_size = 4
        future_map.device = "cpu"
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(bonus_tokens=torch.zeros((2,), dtype=torch.int64))
        )
        self.assertIsNone(future_map.dsa_topk_indices_buf)

        seeds = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(
                bonus_tokens=torch.zeros((2,), dtype=torch.int64),
                dsa_topk_indices=seeds,
            )
        )
        self.assertEqual(future_map.dsa_topk_indices_buf.shape, (4, 3))
        self.assertEqual(future_map.dsa_topk_indices_buf.dtype, torch.int32)


class TestDSV4C128StateIndices(unittest.TestCase):
    def test_online_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=True, ring_size=1),
            np.empty((0,), dtype=np.int32),
        )

    def test_online_partial_boundary_uses_request_slot(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 257, online=True, ring_size=1),
            np.array([7], dtype=np.int32),
        )

    def test_offline_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=False, ring_size=128),
            np.empty((0,), dtype=np.int32),
        )

    def test_offline_partial_boundary_uses_request_local_page(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 129, online=False, ring_size=256),
            np.array([15], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
