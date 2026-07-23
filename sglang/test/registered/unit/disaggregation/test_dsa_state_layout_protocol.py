import struct
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.utils import pack_int_lists, pack_str_list
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo as NixlKVArgsRegisterInfo,
    NixlKVManager,
)
from sglang.srt.disaggregation.utils import compute_state_layout_hash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSAStateLayoutProtocol(unittest.TestCase):
    @staticmethod
    def _layouts():
        kv_args = KVArgs()
        kv_args.state_types = [StateType.DSA]
        kv_args.state_data_ptrs = [[10, 20]]
        kv_args.state_item_lens = [[64, 64]]
        kv_args.state_dim_per_tensor = [[]]
        kv_args.state_layer_ids = [[2, 5]]
        kv_args.state_layout_hashes = [
            compute_state_layout_hash(StateType.DSA, [2, 5], [64, 64])
        ]
        remote_ids = [[0, 2, 4, 5]]
        remote_lens = [[8, 64, 8, 64]]
        remote_hashes = [
            compute_state_layout_hash(StateType.DSA, remote_ids[0], remote_lens[0])
        ]
        return kv_args, remote_ids, remote_lens, remote_hashes

    def test_mooncake_registration_round_trip(self):
        state_layer_ids = [[0, 1, 2, 6, 10, 78]]
        state_layout_hashes = ["a" * 64]
        msg = [
            b"None",
            b"127.0.0.1",
            b"1234",
            b"decode-session",
            struct.pack("Q", 100),
            struct.pack("Q", 200),
            pack_int_lists([[300, 400]], "Q"),
            b"0",
            b"8",
            b"4096",
            pack_int_lists([[8448, 8448]], "I"),
            pack_int_lists([[]], "I"),
            b"",
            b"",
            pack_int_lists(state_layer_ids, "I"),
            pack_str_list(state_layout_hashes),
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.dst_state_layer_ids, state_layer_ids)
        self.assertEqual(info.dst_state_layout_hashes, state_layout_hashes)
        self.assertIsNone(info.staging)

    def test_old_registration_has_no_layout_metadata(self):
        msg = [
            b"None",
            b"127.0.0.1",
            b"1234",
            b"decode-session",
            struct.pack("Q", 100),
            b"",
            pack_int_lists([[300]], "Q"),
            b"0",
            b"8",
            b"4096",
            pack_int_lists([[8448]], "I"),
            pack_int_lists([[]], "I"),
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.dst_state_layer_ids, [])
        self.assertEqual(info.dst_state_layout_hashes, [])

    def test_nixl_registration_round_trip(self):
        state_layer_ids = [[0, 1, 2, 6, 10, 78]]
        state_layout_hashes = ["b" * 64]
        msg = [
            b"None",
            b"127.0.0.1",
            b"1234",
            b"decode-agent",
            b"agent-metadata",
            struct.pack("Q", 100),
            struct.pack("Q", 200),
            pack_int_lists([[300]], "Q"),
            b"0",
            b"8",
            b"0",
            b"4096",
            pack_int_lists([[8448]], "I"),
            pack_int_lists([[]], "I"),
            b"",
            b"",
            b"64",
            b"VRAM",
            struct.pack("Q", 4096),
            pack_int_lists(state_layer_ids, "I"),
            pack_str_list(state_layout_hashes),
        ]
        info = NixlKVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.dst_state_layer_ids, state_layer_ids)
        self.assertEqual(info.dst_state_layout_hashes, state_layout_hashes)

    def test_mooncake_pp2_prefill_maps_into_pp1_decode_by_layer_id(self):
        kv_args, remote_ids, remote_lens, remote_hashes = self._layouts()
        mgr = object.__new__(MooncakeKVManager)
        mgr.kv_args = kv_args
        mgr.attn_tp_size = 8
        mgr.is_mla_backend = True
        mgr.pp_size = 2
        mgr._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="decode",
            dst_state_indices=[[7]],
            room=11,
        )
        registration = SimpleNamespace(
            dst_state_data_ptrs=[[100, 200, 300, 400]],
            dst_state_item_lens=remote_lens,
            dst_state_dim_per_tensor=[[]],
            dst_state_layer_ids=remote_ids,
            dst_state_layout_hashes=remote_hashes,
            dst_attn_tp_size=8,
            dst_tp_rank=0,
        )

        self.assertEqual(
            mgr.maybe_send_extra(req, [[3]], None, registration),
            0,
        )

        kwargs = mgr._send_kvcache_generic.call_args.kwargs
        self.assertEqual(kwargs["src_data_ptrs"], [10, 20])
        self.assertEqual(kwargs["dst_data_ptrs"], [200, 400])
        self.assertEqual(kwargs["item_lens"], [64, 64])

    def test_nixl_pp2_prefill_maps_into_pp1_decode_by_layer_id(self):
        kv_args, remote_ids, remote_lens, remote_hashes = self._layouts()
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = kv_args
        mgr.attn_tp_size = 8
        mgr.is_mla_backend = True
        mgr._send_kvcache_generic = MagicMock(return_value="handle")

        handles = mgr.maybe_send_extra(
            "decode",
            [[3]],
            [[100, 200, 300, 400]],
            [[7]],
            0,
            "state",
            8,
            dst_state_item_lens=remote_lens,
            dst_state_dim_per_tensor=[[]],
            dst_state_layer_ids=remote_ids,
            dst_state_layout_hashes=remote_hashes,
        )

        self.assertEqual(handles, ["handle"])
        kwargs = mgr._send_kvcache_generic.call_args.kwargs
        self.assertEqual(kwargs["src_data_ptrs"], [10, 20])
        self.assertEqual(kwargs["dst_data_ptrs"], [200, 400])
        self.assertEqual(kwargs["item_lens"], [64, 64])

    def test_mooncake_zero_producer_stage_is_noop_without_registration(self):
        kv_args = KVArgs()
        kv_args.state_types = [StateType.DSA]
        kv_args.state_data_ptrs = [[]]
        kv_args.state_item_lens = [[]]
        kv_args.state_dim_per_tensor = [[]]
        kv_args.state_layer_ids = [[]]
        kv_args.state_layout_hashes = [compute_state_layout_hash(StateType.DSA, [], [])]
        mgr = object.__new__(MooncakeKVManager)
        mgr.kv_args = kv_args
        mgr.attn_tp_size = 8
        mgr.is_mla_backend = True
        mgr._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="decode",
            dst_state_indices=[[7]],
            room=11,
        )

        self.assertEqual(mgr.maybe_send_extra(req, [[3]], None, None), 0)
        mgr._send_kvcache_generic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
