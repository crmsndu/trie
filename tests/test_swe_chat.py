import json
from pathlib import Path

from trie.swe_chat import convert_claude_transcript
from trie.types import AppendMessage, Generate, ReplaceContext, Wait


class CharacterTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_convert_claude_transcript_preserves_tool_and_user_waits(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "fix it"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "a.py"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:05Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "file body",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:07Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "done"}],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:17Z",
                "message": {"role": "user", "content": "one more thing"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:19Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "ok"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    assert trace.metadata["source_path"] == "trace.jsonl"
    assert trace.initial_messages == [{"role": "user", "content": "fix it"}]
    assert [type(event) for event in trace.events] == [
        Generate,
        Wait,
        AppendMessage,
        Generate,
        Wait,
        AppendMessage,
        Generate,
    ]
    waits = [event for event in trace.events if isinstance(event, Wait)]
    assert [(wait.actor, wait.seconds) for wait in waits] == [
        ("tool", 3.0),
        ("user", 10.0),
    ]
    first_generate = trace.events[0]
    assert isinstance(first_generate, Generate)
    assert first_generate.recorded_assistant is not None
    assert first_generate.recorded_assistant["tool_calls"][0]["id"] == "tool-1"


def test_convert_claude_transcript_prefers_explicit_tool_duration(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "inspect"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "a.py"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:08Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "file body",
                        }
                    ],
                },
                "toolUseResult": {"durationMs": 1250},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:09Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    waits = [event for event in trace.events if isinstance(event, Wait)]
    assert [(wait.actor, wait.seconds) for wait in waits] == [("tool", 1.25)]


def test_convert_claude_transcript_marks_interactive_tool_wait_as_user(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "plan it"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "AskUserQuestion",
                            "input": {"questions": []},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:12Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "approved",
                        }
                    ],
                },
                "toolUseResult": {"durationMs": 10000},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:13Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "continuing"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    waits = [event for event in trace.events if isinstance(event, Wait)]
    assert [(wait.actor, wait.seconds) for wait in waits] == [("user", 10.0)]


def test_convert_claude_transcript_trims_events_after_last_generation(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "make a plan"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "ExitPlanMode",
                            "input": {},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:10:52Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "rejected",
                        }
                    ],
                },
                "toolUseResult": {"durationMs": 650000},
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    assert [type(event) for event in trace.events] == [Generate]
    assert trace.metadata["trailing_events_trimmed"] == 2


def test_convert_claude_transcript_uses_wall_time_for_parallel_tools(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "inspect"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "a.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "tool-2",
                            "name": "Read",
                            "input": {"file_path": "b.py"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:03Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "a",
                        }
                    ],
                },
                "toolUseResult": {"durationMs": 800},
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:03.100000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-2",
                            "content": "b",
                        }
                    ],
                },
                "toolUseResult": {"durationMs": 700},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:04Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    waits = [event for event in trace.events if isinstance(event, Wait)]
    assert [(wait.actor, wait.seconds) for wait in waits] == [
        ("tool", 1.0),
        ("tool", 0.1),
    ]


def test_convert_claude_transcript_replaces_compacted_context(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "original request"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "working"}],
                },
            },
            {
                "type": "system",
                "subtype": "compact_boundary",
                "timestamp": "2026-01-01T00:01:00Z",
                "content": "Conversation compacted",
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:01:00Z",
                "isCompactSummary": True,
                "message": {
                    "role": "user",
                    "content": "This session continues from a compacted context.",
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:01:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "continued"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    assert trace.initial_messages == [
        {"role": "user", "content": "original request"}
    ]
    assert [type(event) for event in trace.events] == [
        Generate,
        Wait,
        ReplaceContext,
        Generate,
    ]
    compact_wait = trace.events[1]
    assert isinstance(compact_wait, Wait)
    assert (compact_wait.actor, compact_wait.seconds) == ("other", 58.0)
    replacement = trace.events[2]
    assert isinstance(replacement, ReplaceContext)
    assert replacement.reason == "compaction"
    assert replacement.messages == [
        {
            "role": "user",
            "content": "This session continues from a compacted context.",
        }
    ]
    assert trace.metadata["compactions"] == 1


def test_convert_claude_transcript_skips_microcompaction(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "inspect"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "working"}],
                },
            },
            {
                "type": "system",
                "subtype": "microcompact_boundary",
                "timestamp": "2026-01-01T00:00:03Z",
                "content": "Context microcompacted",
            },
        ],
    )

    assert convert_claude_transcript(transcript, CharacterTokenizer()) is None


def test_convert_claude_transcript_skips_unmarked_compact_summary(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "start"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "working"}],
                },
            },
            {
                "type": "system",
                "subtype": "compact_boundary",
                "timestamp": "2026-01-01T00:00:02Z",
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {"role": "user", "content": "unmarked summary"},
            },
        ],
    )

    assert convert_claude_transcript(transcript, CharacterTokenizer()) is None


def test_convert_claude_transcript_deduplicates_delivered_queue_message(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "start"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "requestId": "request-1",
                "message": {
                    "id": "message-1",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "test"},
                        }
                    ],
                },
            },
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "timestamp": "2026-01-01T00:00:03Z",
                "content": "follow up",
            },
            {
                "type": "queue-operation",
                "operation": "remove",
                "timestamp": "2026-01-01T00:00:04Z",
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:05Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "done",
                        }
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:06Z",
                "message": {"role": "user", "content": "follow up"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:07Z",
                "requestId": "request-2",
                "message": {
                    "id": "message-2",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    appended_users = [
        event.message["content"]
        for event in trace.events
        if isinstance(event, AppendMessage) and event.message["role"] == "user"
    ]
    assert appended_users == ["follow up"]
    waits = [event for event in trace.events if isinstance(event, Wait)]
    assert [(wait.actor, wait.seconds) for wait in waits] == [
        ("tool", 3.0),
        ("other", 1.0),
    ]
    assert trace.metadata["queue_consumed"] == 1
    assert trace.metadata["queue_injected"] == 0
    assert trace.metadata["queue_delivered_deduplicated"] == 1
    assert trace.metadata["human_user_messages"] == 2
    assert trace.metadata["automatic_user_messages"] == 0


def test_convert_claude_transcript_injects_queue_after_tool_result(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "start"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "requestId": "request-1",
                "uuid": "assistant-1",
                "message": {
                    "id": "message-1",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "test"},
                        }
                    ],
                },
            },
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "timestamp": "2026-01-01T00:00:03Z",
                "content": "follow up",
            },
            {
                "type": "queue-operation",
                "operation": "remove",
                "timestamp": "2026-01-01T00:00:04Z",
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:05Z",
                "sourceToolAssistantUUID": "assistant-1",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "done",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:06Z",
                "requestId": "request-2",
                "message": {
                    "id": "message-2",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    assert [type(event) for event in trace.events] == [
        Generate,
        Wait,
        AppendMessage,
        AppendMessage,
        Generate,
    ]
    tool_result = trace.events[2]
    queued_user = trace.events[3]
    assert isinstance(tool_result, AppendMessage)
    assert tool_result.message["role"] == "tool"
    assert isinstance(queued_user, AppendMessage)
    assert queued_user.message == {"role": "user", "content": "follow up"}
    assert trace.metadata["queue_injected"] == 1
    assert trace.metadata["human_user_messages"] == 2
    assert trace.metadata["automatic_user_messages"] == 0


def test_convert_claude_transcript_collapses_adjacent_queue_duplicates(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "start"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "requestId": "request-1",
                "message": {
                    "id": "message-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "working"}],
                },
            },
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "timestamp": "2026-01-01T00:00:03Z",
                "content": "same message",
            },
            {
                "type": "queue-operation",
                "operation": "dequeue",
                "timestamp": "2026-01-01T00:00:04Z",
            },
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "timestamp": "2026-01-01T00:00:05Z",
                "content": "same message",
            },
            {
                "type": "queue-operation",
                "operation": "dequeue",
                "timestamp": "2026-01-01T00:00:06Z",
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:07Z",
                "requestId": "request-2",
                "message": {
                    "id": "message-2",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    appended_users = [
        event.message["content"]
        for event in trace.events
        if isinstance(event, AppendMessage) and event.message["role"] == "user"
    ]
    assert appended_users == ["same message"]
    assert trace.metadata["queue_adjacent_duplicates"] == 1
    assert trace.metadata["queue_orphan_operations"] == 0


def test_convert_claude_transcript_marks_automatic_user_wait_as_other(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "start"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "working"}],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:12Z",
                "message": {
                    "role": "user",
                    "content": "<task-notification>done</task-notification>",
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:13Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "finished"}],
                },
            },
        ],
    )

    trace = convert_claude_transcript(transcript, CharacterTokenizer())

    assert trace is not None
    waits = [event for event in trace.events if isinstance(event, Wait)]
    assert [(wait.actor, wait.seconds) for wait in waits] == [("other", 10.0)]
    assert trace.metadata["human_user_messages"] == 1
    assert trace.metadata["automatic_user_messages"] == 1


def test_convert_claude_transcript_skips_result_before_use(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "start"},
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "impossible",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "test"},
                        }
                    ],
                },
            },
        ],
    )

    assert convert_claude_transcript(transcript, CharacterTokenizer()) is None


def test_convert_claude_transcript_skips_ambiguous_assistant_requests(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "trace.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "start"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:01Z",
                "requestId": "request-1",
                "message": {
                    "id": "message-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "first"}],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:02Z",
                "requestId": "request-2",
                "message": {
                    "id": "message-2",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "second"}],
                },
            },
        ],
    )

    assert convert_claude_transcript(transcript, CharacterTokenizer()) is None
