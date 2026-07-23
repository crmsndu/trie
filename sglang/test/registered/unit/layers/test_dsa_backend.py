import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa_backend import (
    _offset_and_pad_ragged_topk_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRaggedTopkIndicesOffset(unittest.TestCase):
    def test_cp_padded_indexer_rows_remain_sentinels(self):
        topk_indices = torch.tensor(
            [
                [0, 3, -1],
                [1, -1, 4],
                [2, 5, 7],
                [-1, 0, 6],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            dtype=torch.int32,
        )
        offsets = torch.tensor([0, 10, 20, 30], dtype=torch.int32)
        original_topk_indices = topk_indices.clone()

        actual = _offset_and_pad_ragged_topk_indices(
            topk_indices, offsets, output_num_tokens=8
        )

        expected = torch.tensor(
            [
                [0, 3, -1],
                [11, -1, 14],
                [22, 25, 27],
                [-1, 30, 36],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(topk_indices, original_topk_indices, rtol=0, atol=0)

    def test_offsets_real_rows_before_padding_to_query_layout(self):
        topk_indices = torch.tensor(
            [[0, -1], [1, 2], [3, 4], [-1, 5]], dtype=torch.int32
        )
        offsets = torch.tensor([[0], [10], [20], [30]], dtype=torch.int32)

        actual = _offset_and_pad_ragged_topk_indices(
            topk_indices, offsets, output_num_tokens=8
        )

        expected = torch.tensor(
            [
                [0, -1],
                [11, 12],
                [23, 24],
                [-1, 35],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_inconsistent_row_counts(self):
        topk_indices = torch.zeros((4, 2), dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "must be 2D"):
            _offset_and_pad_ragged_topk_indices(
                topk_indices[:, 0],
                torch.zeros(4, dtype=torch.int32),
                output_num_tokens=4,
            )
        with self.assertRaisesRegex(AssertionError, "offset rows"):
            _offset_and_pad_ragged_topk_indices(
                topk_indices,
                torch.zeros(5, dtype=torch.int32),
                output_num_tokens=8,
            )
        with self.assertRaisesRegex(AssertionError, "output rows"):
            _offset_and_pad_ragged_topk_indices(
                topk_indices,
                torch.zeros(4, dtype=torch.int32),
                output_num_tokens=3,
            )

    def test_async_guard_rejects_non_sentinel_padding(self):
        topk_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32)
        offsets = torch.tensor([0, 10], dtype=torch.int32)
        with patch.object(
            envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", return_value=True
        ), self.assertRaisesRegex(RuntimeError, "has no ragged offset"):
            _offset_and_pad_ragged_topk_indices(
                topk_indices, offsets, output_num_tokens=4
            )


if __name__ == "__main__":
    unittest.main()
