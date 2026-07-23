# ruff: noqa: E402

import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def _install_mori_stubs_if_needed():
    try:
        import mori  # noqa: F401

        return
    except ImportError:
        pass

    mori_module = types.ModuleType("mori")
    cpp_module = types.ModuleType("mori.cpp")
    io_module = types.ModuleType("mori.io")

    class TransferStatus:
        pass

    class EngineDesc:
        key = "stub-engine"

        @classmethod
        def unpack(cls, payload):
            return cls()

    class MemoryDesc:
        @classmethod
        def unpack(cls, payload):
            return cls()

    class StatusCode:
        SUCCESS = 0
        IN_PROGRESS = 1

    class BackendType:
        RDMA = 0

    class MemoryLocationType:
        CPU = 0
        GPU = 1

    class PollCqMode:
        POLLING = SimpleNamespace(name="POLLING")

    cpp_module.TransferStatus = TransferStatus
    io_module.BackendType = BackendType
    io_module.EngineDesc = EngineDesc
    io_module.IOEngine = type("IOEngine", (), {})
    io_module.IOEngineConfig = type("IOEngineConfig", (), {})
    io_module.MemoryDesc = MemoryDesc
    io_module.MemoryLocationType = MemoryLocationType
    io_module.PollCqMode = PollCqMode
    io_module.RdmaBackendConfig = type("RdmaBackendConfig", (), {})
    io_module.StatusCode = StatusCode
    sys.modules["mori"] = mori_module
    sys.modules["mori.cpp"] = cpp_module
    sys.modules["mori.io"] = io_module


_install_mori_stubs_if_needed()

from sglang.srt.disaggregation.mori import conn as mori_conn
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.mori.conn import (
    KVArgsRegisterInfo,
    MoriKVManager,
    TransferInfo,
    _pack_mem_desc_list,
    _pack_mem_desc_lists,
)
from sglang.srt.disaggregation.common.utils import pack_int_lists, pack_str_list
from sglang.srt.disaggregation.utils import compute_state_layout_hash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _local_kv_args():
    kv_args = KVArgs()
    kv_args.state_types = [StateType.DSA]
    kv_args.state_data_ptrs = [[10, 20]]
    kv_args.state_item_lens = [[64, 64]]
    kv_args.state_dim_per_tensor = [[]]
    kv_args.state_layer_ids = [[2, 5]]
    kv_args.state_layout_hashes = [
        compute_state_layout_hash(StateType.DSA, [2, 5], [64, 64])
    ]
    return kv_args


class TestMoriDSAStateProtocol(unittest.TestCase):
    def test_registration_wire_round_trip_preserves_dsa_layout(self):
        class PackedDesc:
            def __init__(self, payload):
                self.payload = payload
                self.key = payload.decode("ascii")

            def pack(self):
                return self.payload

            @classmethod
            def unpack(cls, payload):
                return cls(payload)

        layer_ids = [[0, 2, 4, 5]]
        item_lens = [[8, 64, 8, 64]]
        layout_hashes = [
            compute_state_layout_hash(StateType.DSA, layer_ids[0], item_lens[0])
        ]
        kv_descs = [PackedDesc(b"kv-0"), PackedDesc(b"kv-1")]
        aux_descs = [PackedDesc(b"aux")]
        state_descs = [[PackedDesc(f"state-{i}".encode()) for i in layer_ids[0]]]
        payload = [
            b"None",
            b"10.0.0.2",
            b"19191",
            PackedDesc(b"decode-engine").pack(),
            _pack_mem_desc_list(kv_descs),
            _pack_mem_desc_list(aux_descs),
            _pack_mem_desc_lists(state_descs),
            b"3",
            b"8",
            b"5",
            b"8192",
            pack_int_lists(item_lens, "I"),
            pack_int_lists([[]], "I"),
            pack_int_lists(layer_ids, "I"),
            pack_str_list(layout_hashes),
        ]

        with (
            patch.object(mori_conn, "EngineDesc", PackedDesc),
            patch.object(mori_conn, "MemoryDesc", PackedDesc),
        ):
            decoded = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(decoded.engine_key, "decode-engine")
        self.assertEqual(
            [desc.payload for desc in decoded.dst_kv_mem_descs], [b"kv-0", b"kv-1"]
        )
        self.assertEqual([desc.payload for desc in decoded.dst_aux_mem_descs], [b"aux"])
        self.assertEqual(
            [[desc.payload for desc in comp] for comp in decoded.dst_state_mem_descs],
            [[b"state-0", b"state-2", b"state-4", b"state-5"]],
        )
        self.assertEqual(decoded.dst_state_item_lens, item_lens)
        self.assertEqual(decoded.dst_state_dim_per_tensor, [[]])
        self.assertEqual(decoded.dst_state_layer_ids, layer_ids)
        self.assertEqual(decoded.dst_state_layout_hashes, layout_hashes)

    def test_pp2_prefill_maps_into_pp1_decode_by_layer_id(self):
        mgr = object.__new__(MoriKVManager)
        mgr.kv_args = _local_kv_args()
        mgr.state_mem_descs = [["src-2", "src-5"]]
        mgr._send_swa_dsa_state = MagicMock(return_value=["status"])
        remote_ids = [[0, 2, 4, 5]]
        remote_lens = [[8, 64, 8, 64]]
        peer = SimpleNamespace(
            engine_key="decode",
            dst_state_mem_descs=[["dst-0", "dst-2", "dst-4", "dst-5"]],
            dst_state_item_lens=remote_lens,
            dst_state_dim_per_tensor=[[]],
            dst_state_layer_ids=remote_ids,
            dst_state_layout_hashes=[
                compute_state_layout_hash(StateType.DSA, remote_ids[0], remote_lens[0])
            ],
            decode_tp_size=8,
            decode_tp_rank=0,
        )

        statuses = mgr.send_state(
            peer,
            [np.array([3], dtype=np.int32)],
            [np.array([7], dtype=np.int32)],
        )

        self.assertEqual(statuses, ["status"])
        args = mgr._send_swa_dsa_state.call_args.args
        self.assertEqual(args[3], ["src-2", "src-5"])
        self.assertEqual(args[4], [64, 64])
        self.assertEqual(args[5], ["dst-2", "dst-5"])

    def test_rejected_peer_metadata_is_failed_and_not_accepted(self):
        mgr = object.__new__(MoriKVManager)
        mgr.kv_args = _local_kv_args()
        mgr.protocol_rejected_peers = {}
        mgr.rejected_room_failures = {}
        mgr.peer_protocol_lock = threading.Lock()
        mgr.decode_kv_args_table = {}
        mgr.transfer_infos = {}
        mgr.transfer_lock = threading.Lock()
        mgr.request_status = {91: KVPoll.Bootstrapping}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.notify_decode_status = MagicMock()
        bad_registration = SimpleNamespace(
            engine_key="bad-peer",
            dst_state_layer_ids=[[2]],
            dst_state_layout_hashes=[
                compute_state_layout_hash(StateType.DSA, [2], [64])
            ],
            dst_state_mem_descs=[["dst-2"]],
            dst_state_item_lens=[[64]],
        )

        with patch.object(
            KVArgsRegisterInfo, "from_zmq", return_value=bad_registration
        ):
            mgr._handle_register_message([b"ignored"])

        self.assertIn("bad-peer", mgr.protocol_rejected_peers)
        transfer_info = SimpleNamespace(
            engine_key="bad-peer",
            room=91,
            endpoint="127.0.0.1",
            dst_port=23456,
        )
        with patch.object(TransferInfo, "from_zmq", return_value=transfer_info):
            mgr._handle_transfer_message([b"ignored"])

        self.assertEqual(mgr.request_status[91], KVPoll.Failed)
        self.assertNotIn(91, mgr.transfer_infos)
        mgr.notify_decode_status.assert_called_once()
        self.assertEqual(mgr.notify_decode_status.call_args.args[2], KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()
