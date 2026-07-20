import pytest

from trie.prefix import block_aligned_prefix_length, common_prefix_length


def test_common_prefix_length() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 4, 5]) == 2
    assert common_prefix_length([], [1]) == 0
    assert common_prefix_length([1, 2], [1, 2]) == 2


def test_block_aligned_prefix_length() -> None:
    assert block_aligned_prefix_length(35, 16) == 32
    assert block_aligned_prefix_length(15, 16) == 0
    with pytest.raises(ValueError, match="block_size"):
        block_aligned_prefix_length(10, 0)
