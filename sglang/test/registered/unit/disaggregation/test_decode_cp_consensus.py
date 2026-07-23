import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.utils import (
    poll_and_all_reduce_attn_cp_tp_group,
    poll_and_all_reduce_with_staging,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _decode_req(room: int, metadata_buffer_index: int = 0):
    req = SimpleNamespace(
        bootstrap_host="127.0.0.1",
        bootstrap_room=room,
        rid=f"rid-{room}",
        time_stats=MagicMock(),
    )
    receiver = MagicMock(require_staging=False, conclude_state=KVPoll.Bootstrapping)
    receiver.poll.return_value = KVPoll.Success
    return SimpleNamespace(
        req=req,
        kv_receiver=receiver,
        waiting_for_input=False,
        metadata_buffer_index=metadata_buffer_index,
    )


class TestDecodeCPConsensus(CustomTestCase):
    def test_helper_preserves_non_cp_poll_result(self):
        tp_group = object()
        singleton_cp_group = object()
        poller = MagicMock()
        poller.poll.return_value = KVPoll.WaitingForInput

        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce"
        ) as mock_all_reduce:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [poller], singleton_cp_group, tp_group
            )

        self.assertEqual(polls, [KVPoll.WaitingForInput])
        self.assertEqual(
            [
                all_reduce.kwargs["group"]
                for all_reduce in mock_all_reduce.call_args_list
            ],
            [tp_group, singleton_cp_group],
        )

    def test_helper_applies_metadata_gate_then_tp_and_cp_min(self):
        tp_group = object()
        cp_group = object()
        decode_reqs = [_decode_req(11, 0), _decode_req(22, 1)]
        metadata_buffers = SimpleNamespace(bootstrap_room=torch.tensor([[0], [22]]))
        server_args = SimpleNamespace(disaggregation_transfer_backend="mooncake")
        reductions = []

        def fake_all_reduce(tensor, op, group):
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            reductions.append((group, tensor.clone()))
            if group is tp_group:
                tensor[1] = KVPoll.Transferring
            else:
                tensor[0] = KVPoll.WaitingForInput

        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=fake_all_reduce,
        ):
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [dr.kv_receiver for dr in decode_reqs],
                cp_group,
                tp_group,
                decode_reqs=decode_reqs,
                metadata_buffers=metadata_buffers,
                server_args=server_args,
            )

        self.assertEqual([group for group, _ in reductions], [tp_group, cp_group])
        self.assertEqual(
            reductions[0][1].tolist(), [KVPoll.Transferring, KVPoll.Success]
        )
        self.assertEqual(
            reductions[1][1].tolist(),
            [KVPoll.Transferring, KVPoll.Transferring],
        )
        self.assertEqual(polls, [KVPoll.WaitingForInput, KVPoll.Transferring])

    def test_staging_gate_reduces_tp_before_cp(self):
        tp_group = object()
        cp_group = object()
        decode_req = _decode_req(11)
        decode_req.kv_receiver.require_staging = True
        metadata_buffers = SimpleNamespace(bootstrap_room=torch.tensor([[11]]))
        server_args = SimpleNamespace(disaggregation_transfer_backend="mooncake")
        staging_handler = MagicMock()
        staging_handler.is_done.return_value = False
        reductions = []

        def fake_all_reduce(tensor, op, group):
            reductions.append((group, tensor.clone()))
            if group is cp_group:
                tensor[0] = KVPoll.WaitingForInput

        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=fake_all_reduce,
        ):
            polls = poll_and_all_reduce_with_staging(
                [decode_req],
                staging_handler,
                cp_group,
                tp_group,
                metadata_buffers=metadata_buffers,
                server_args=server_args,
            )

        staging_handler.advance_scatter.assert_called_once_with(decode_req)
        self.assertEqual([group for group, _ in reductions], [tp_group, cp_group])
        self.assertEqual(reductions[0][1].tolist(), [KVPoll.Transferring])
        self.assertEqual(reductions[1][1].tolist(), [KVPoll.Transferring])
        self.assertEqual(polls, [KVPoll.WaitingForInput])

    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce_attn_cp_tp_group")
    def test_prealloc_handshake_uses_both_attention_groups(self, mock_poll):
        tp_group = object()
        cp_group = object()
        decode_req = _decode_req(11)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.attn_cp_cpu_group = cp_group
        queue.attn_tp_cpu_group = tp_group
        mock_poll.return_value = [KVPoll.WaitingForInput]

        queue._update_handshake_waiters()

        mock_poll.assert_called_once_with([decode_req.kv_receiver], cp_group, tp_group)
        self.assertTrue(decode_req.waiting_for_input)
        decode_req.req.time_stats.set_bootstrap_done_time.assert_called_once_with()

    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce_attn_cp_tp_group")
    def test_transfer_metadata_gate_uses_both_attention_groups(self, mock_poll):
        tp_group = object()
        cp_group = object()
        decode_req = _decode_req(11)
        metadata_buffers = object()
        server_args = object()
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.attn_cp_cpu_group = cp_group
        queue.attn_tp_cpu_group = tp_group
        queue.metadata_buffers = metadata_buffers
        queue.scheduler = SimpleNamespace(
            enable_decode_hicache=False, server_args=server_args
        )
        mock_poll.return_value = [KVPoll.Success]

        polls = queue._poll_with_metadata_gate()

        self.assertEqual(polls, [KVPoll.Success])
        mock_poll.assert_called_once_with(
            [decode_req.kv_receiver],
            cp_group,
            tp_group,
            decode_reqs=[decode_req],
            metadata_buffers=metadata_buffers,
            server_args=server_args,
        )

    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce_with_staging")
    def test_transfer_staging_gate_uses_both_attention_groups(self, mock_poll):
        tp_group = object()
        cp_group = object()
        decode_req = _decode_req(11)
        metadata_buffers = object()
        server_args = object()
        staging_handler = object()
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.staging_handler = staging_handler
        queue.attn_cp_cpu_group = cp_group
        queue.attn_tp_cpu_group = tp_group
        queue.metadata_buffers = metadata_buffers
        queue.scheduler = SimpleNamespace(server_args=server_args)
        mock_poll.return_value = [KVPoll.Success]

        polls = queue._poll_with_staging()

        self.assertEqual(polls, [KVPoll.Success])
        mock_poll.assert_has_calls(
            [
                call(
                    [decode_req],
                    staging_handler,
                    cp_group,
                    tp_group,
                    metadata_buffers=metadata_buffers,
                    server_args=server_args,
                )
            ]
        )


if __name__ == "__main__":
    unittest.main()
