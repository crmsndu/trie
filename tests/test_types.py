import pytest

from trie.types import AppendMessage, Generate, ReplayTrace, Wait


def test_replay_trace_round_trip() -> None:
    trace = ReplayTrace(
        trace_id="session-1",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[
            Generate(
                max_tokens=4,
                recorded_assistant={"role": "assistant", "content": "hi"},
                recorded_output_token_ids=[11, 12, 13, 14],
            ),
            Wait(seconds=1.5, actor="user"),
            AppendMessage(message={"role": "user", "content": "again"}),
            Generate(max_tokens=2),
        ],
        metadata={"agent": "Claude Code"},
    )

    parsed = ReplayTrace.from_jsonl_row(trace.to_jsonl_row())

    assert parsed.trace_id == trace.trace_id
    assert parsed.initial_messages == trace.initial_messages
    assert parsed.events == trace.events
    assert parsed.metadata == trace.metadata


def test_recorded_output_token_ids_must_match_generation_length() -> None:
    with pytest.raises(ValueError, match="length must equal max_tokens"):
        Generate(max_tokens=2, recorded_output_token_ids=[1])


def test_recorded_output_token_ids_must_be_non_negative_integers() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        Generate(max_tokens=1, recorded_output_token_ids=[-1])


def test_replay_trace_requires_generate() -> None:
    with pytest.raises(ValueError, match="generate event"):
        ReplayTrace(
            trace_id="session-1",
            initial_messages=[],
            events=[AppendMessage(message={"role": "user", "content": "hello"})],
        )
