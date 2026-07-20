def common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def block_aligned_prefix_length(prefix_tokens: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")
    return prefix_tokens - (prefix_tokens % block_size)
