import argparse
import copy
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from trie.tokenizer_manager import TokenizerManager
from trie.types import (
    AppendMessage,
    Generate,
    ReplaceContext,
    ReplayEvent,
    ReplayTrace,
    Wait,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _QueuedMessage:
    content: str
    enqueue_timestamp: datetime | None
    consume_timestamp: datetime | None
    actor: str

    @property
    def normalized_content(self) -> str:
        return self.content.strip()


_AUTOMATIC_USER_PREFIXES = (
    "<local-command-caveat>",
    "<command-name>",
    "<local-command-stdout>",
    "<task-notification>",
    "<system-reminder>",
)

_INTERACTIVE_TOOL_NAMES = {"AskUserQuestion", "ExitPlanMode"}


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _latest_timestamp(
    previous: datetime | None, current: datetime | None
) -> datetime | None:
    if previous is None:
        return current
    if current is None:
        return previous
    return max(previous, current)


def _assistant_request_identity(
    entry: dict[str, Any], message: dict[str, Any]
) -> tuple[str | None, str | None]:
    message_id = message.get("id")
    request_id = entry.get("requestId")
    return (
        message_id if isinstance(message_id, str) and message_id else None,
        request_id if isinstance(request_id, str) and request_id else None,
    )


def _assistant_requests_conflict(
    previous: tuple[str | None, str | None],
    current: tuple[str | None, str | None],
) -> bool:
    previous_message, previous_request = previous
    current_message, current_request = current
    return bool(
        previous_message
        and current_message
        and previous_message != current_message
    ) or bool(
        previous_request
        and current_request
        and previous_request != current_request
    )


def _merge_assistant_identity(
    previous: tuple[str | None, str | None],
    current: tuple[str | None, str | None],
) -> tuple[str | None, str | None]:
    return (previous[0] or current[0], previous[1] or current[1])


def _is_automatic_user_content(content: str) -> bool:
    stripped = content.lstrip()
    if stripped.startswith(_AUTOMATIC_USER_PREFIXES):
        return True
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and (
        "task_id" in parsed or "task_type" in parsed
    )


def _user_wait_actor(entry: dict[str, Any], content: str) -> str:
    if (
        entry.get("isMeta") is True
        or entry.get("isVisibleInTranscriptOnly") is True
        or _is_automatic_user_content(content)
    ):
        return "other"
    user_type = entry.get("userType")
    if user_type not in (None, "external"):
        return "other"
    return "user"


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text") or "")
        elif block_type == "thinking":
            reasoning_parts.append(block.get("thinking") or "")
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "unknown_tool",
                        "arguments": block.get("input") or {},
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(part for part in text_parts if part),
    }
    if reasoning_parts:
        message["reasoning_content"] = "\n".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _assistant_token_length(
    tokenizer: TokenizerManager, message: dict[str, Any]
) -> int:
    parts = [message.get("reasoning_content") or "", message.get("content") or ""]
    if message.get("tool_calls"):
        parts.append(
            json.dumps(
                message["tool_calls"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return max(tokenizer.count_tokens("\n".join(parts)), 1)


def _append_wait(
    events: list[ReplayEvent],
    *,
    actor: str,
    previous: datetime | None,
    current: datetime | None,
    explicit_seconds: float | None = None,
) -> None:
    if explicit_seconds is not None:
        seconds = explicit_seconds
    elif previous is not None and current is not None:
        seconds = (current - previous).total_seconds()
    else:
        return
    if seconds > 0:
        events.append(Wait(seconds=seconds, actor=actor))


def _tool_duration_seconds(entry: dict[str, Any]) -> float | None:
    tool_result = entry.get("toolUseResult")
    if not isinstance(tool_result, dict):
        return None
    duration_ms = tool_result.get("durationMs")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
        return None
    return max(float(duration_ms), 0.0) / 1000.0


def convert_claude_transcript(
    path: str | Path,
    tokenizer: TokenizerManager,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> ReplayTrace | None:
    path = Path(path)
    events: list[ReplayEvent] = []
    pending_blocks: list[dict[str, Any]] = []
    pending_model: str | None = None
    pending_timestamp_raw: str | None = None
    pending_timestamp: datetime | None = None
    pending_identity: tuple[str | None, str | None] = (None, None)
    pending_tool_calls: dict[str, str | None] = {}
    pending_tool_names: dict[str, str] = {}
    open_tool_calls: dict[str, str | None] = {}
    open_tool_names: dict[str, str] = {}
    parallel_tool_calls: set[str] = set()
    timeline_cursor: datetime | None = None
    queued_messages: deque[_QueuedMessage] = deque()
    ready_queued_messages: deque[_QueuedMessage] = deque()
    injected_awaiting_delivery: deque[_QueuedMessage] = deque()
    user_messages = 0
    human_user_messages = 0
    automatic_user_messages = 0
    tool_results = 0
    generations = 0
    compactions = 0
    sidechain_entries_skipped = 0
    queue_enqueued = 0
    queue_consumed = 0
    queue_injected = 0
    queue_delivered_deduplicated = 0
    queue_adjacent_duplicates = 0
    queue_empty_entries = 0
    queue_orphan_operations = 0
    queue_human_arrivals = 0
    queue_residence_seconds = 0.0
    compact_boundary_pending = False

    def skip_trace(reason: str, **fields: Any) -> None:
        logger.warning(
            "skipping Claude transcript",
            source_path=str(path),
            reason=reason,
            **fields,
        )

    def append_user_message(
        content: str,
        *,
        actor: str,
        timestamp: datetime | None,
        wait_timestamp: datetime | None = None,
        origin_actor: str | None = None,
    ) -> None:
        nonlocal timeline_cursor
        nonlocal user_messages
        nonlocal human_user_messages
        nonlocal automatic_user_messages
        _append_wait(
            events,
            actor=actor,
            previous=timeline_cursor,
            current=wait_timestamp if wait_timestamp is not None else timestamp,
        )
        events.append(
            AppendMessage(message={"role": "user", "content": content})
        )
        timeline_cursor = _latest_timestamp(timeline_cursor, timestamp)
        user_messages += 1
        if (origin_actor or actor) == "user":
            human_user_messages += 1
        else:
            automatic_user_messages += 1

    def mark_queue_consumed(
        queued: _QueuedMessage, consume_timestamp: datetime | None
    ) -> _QueuedMessage:
        nonlocal queue_consumed
        nonlocal queue_residence_seconds
        queue_consumed += 1
        if queued.enqueue_timestamp is not None and consume_timestamp is not None:
            residence = (consume_timestamp - queued.enqueue_timestamp).total_seconds()
            if residence > 0:
                queue_residence_seconds += residence
        return _QueuedMessage(
            content=queued.content,
            enqueue_timestamp=queued.enqueue_timestamp,
            consume_timestamp=consume_timestamp,
            actor=queued.actor,
        )

    def take_matching_queue(
        content: str, delivered_timestamp: datetime | None
    ) -> tuple[_QueuedMessage | None, bool]:
        normalized = content.strip()
        if (
            injected_awaiting_delivery
            and injected_awaiting_delivery[0].normalized_content == normalized
        ):
            return injected_awaiting_delivery.popleft(), True
        if (
            ready_queued_messages
            and ready_queued_messages[0].normalized_content == normalized
        ):
            return ready_queued_messages.popleft(), False
        if queued_messages and queued_messages[0].normalized_content == normalized:
            queued = queued_messages.popleft()
            return mark_queue_consumed(queued, delivered_timestamp), False
        return None, False

    def move_next_queue_to_ready(consume_timestamp: datetime | None) -> None:
        nonlocal queue_adjacent_duplicates
        queued = mark_queue_consumed(queued_messages.popleft(), consume_timestamp)
        if (
            ready_queued_messages
            and ready_queued_messages[-1].normalized_content
            == queued.normalized_content
        ):
            queue_adjacent_duplicates += 1
            return
        ready_queued_messages.append(queued)

    def inject_ready_queue(request_timestamp: datetime | None) -> None:
        nonlocal queue_injected
        while ready_queued_messages:
            queued = ready_queued_messages.popleft()
            delivery_timestamp = queued.consume_timestamp or request_timestamp
            append_user_message(
                queued.content,
                actor="other",
                timestamp=delivery_timestamp,
                wait_timestamp=delivery_timestamp,
                origin_actor=queued.actor,
            )
            injected_awaiting_delivery.append(queued)
            queue_injected += 1

    def handle_text_user(
        entry: dict[str, Any], content: str, timestamp: datetime | None
    ) -> None:
        nonlocal timeline_cursor
        nonlocal queue_delivered_deduplicated
        queued, already_injected = take_matching_queue(content, timestamp)
        if queued is not None:
            queue_delivered_deduplicated += 1
            if already_injected:
                timeline_cursor = _latest_timestamp(timeline_cursor, timestamp)
                return
            append_user_message(
                content,
                actor="other",
                timestamp=timestamp,
                wait_timestamp=timestamp,
                origin_actor=queued.actor,
            )
            return
        append_user_message(
            content,
            actor=_user_wait_actor(entry, content),
            timestamp=timestamp,
        )

    def replace_compacted_context(
        content: str, timestamp: datetime | None
    ) -> None:
        nonlocal timeline_cursor
        nonlocal compactions
        _append_wait(
            events,
            actor="other",
            previous=timeline_cursor,
            current=timestamp,
        )
        events.append(
            ReplaceContext(
                messages=[{"role": "user", "content": content}],
                reason="compaction",
            )
        )
        timeline_cursor = _latest_timestamp(timeline_cursor, timestamp)
        open_tool_calls.clear()
        open_tool_names.clear()
        parallel_tool_calls.clear()
        queued_messages.clear()
        ready_queued_messages.clear()
        injected_awaiting_delivery.clear()
        compactions += 1

    def flush_assistant() -> bool:
        nonlocal pending_blocks
        nonlocal pending_model
        nonlocal pending_timestamp_raw
        nonlocal pending_timestamp
        nonlocal pending_identity
        nonlocal pending_tool_calls
        nonlocal pending_tool_names
        nonlocal timeline_cursor
        nonlocal generations
        if not pending_blocks:
            pending_identity = (None, None)
            pending_tool_calls = {}
            pending_tool_names = {}
            return True
        duplicate_tool_ids = open_tool_calls.keys() & pending_tool_calls.keys()
        if duplicate_tool_ids:
            skip_trace(
                "duplicate tool_use id",
                tool_use_id=sorted(duplicate_tool_ids)[0],
            )
            return False
        recorded = _assistant_message(pending_blocks)
        events.append(
            Generate(
                max_tokens=_assistant_token_length(tokenizer, recorded),
                recorded_assistant=recorded,
                source_model=pending_model,
                source_timestamp=pending_timestamp_raw,
            )
        )
        generations += 1
        timeline_cursor = _latest_timestamp(timeline_cursor, pending_timestamp)
        if len(pending_tool_calls) > 1:
            parallel_tool_calls.update(pending_tool_calls)
        open_tool_calls.update(pending_tool_calls)
        open_tool_names.update(pending_tool_names)
        pending_blocks = []
        pending_model = None
        pending_timestamp_raw = None
        pending_timestamp = None
        pending_identity = (None, None)
        pending_tool_calls = {}
        pending_tool_names = {}
        return True

    with path.open("r", encoding="utf-8") as transcript:
        for line_number, line in enumerate(transcript, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc

            if not isinstance(entry, dict):
                skip_trace("non-object JSON row", line_number=line_number)
                return None

            if entry.get("isSidechain") is True:
                sidechain_entries_skipped += 1
                continue

            entry_type = entry.get("type")
            message = entry.get("message")
            if (
                entry_type == "system"
                and entry.get("subtype") == "microcompact_boundary"
            ):
                skip_trace("microcompaction cannot be reconstructed exactly")
                return None
            if (
                entry_type == "system"
                and entry.get("subtype") == "compact_boundary"
            ):
                compact_boundary_pending = True
                continue

            if entry_type == "queue-operation":
                operation = entry.get("operation")
                queue_timestamp = _parse_timestamp(entry.get("timestamp"))
                if operation == "enqueue":
                    queue_content = entry.get("content")
                    if not isinstance(queue_content, str) or not queue_content.strip():
                        queue_empty_entries += 1
                        continue
                    actor = _user_wait_actor(entry, queue_content)
                    queued_messages.append(
                        _QueuedMessage(
                            content=queue_content,
                            enqueue_timestamp=queue_timestamp,
                            consume_timestamp=None,
                            actor=actor,
                        )
                    )
                    queue_enqueued += 1
                    if actor == "user":
                        queue_human_arrivals += 1
                    continue
                if operation in ("dequeue", "remove"):
                    if not queued_messages:
                        queue_orphan_operations += 1
                        continue
                    move_next_queue_to_ready(queue_timestamp)
                    continue
                continue

            if entry_type == "assistant" and isinstance(message, dict):
                if compact_boundary_pending:
                    skip_trace("compact boundary has no marked summary")
                    return None
                timestamp = _parse_timestamp(entry.get("timestamp"))
                identity = _assistant_request_identity(entry, message)
                if pending_blocks and _assistant_requests_conflict(
                    pending_identity, identity
                ):
                    if not ready_queued_messages:
                        skip_trace(
                            "assistant request changed without an input boundary",
                            previous_message_id=pending_identity[0],
                            previous_request_id=pending_identity[1],
                            current_message_id=identity[0],
                            current_request_id=identity[1],
                            line_number=line_number,
                        )
                        return None
                    if not flush_assistant():
                        return None
                    inject_ready_queue(timestamp)
                elif not pending_blocks:
                    inject_ready_queue(timestamp)

                pending_identity = _merge_assistant_identity(
                    pending_identity, identity
                )
                content = message.get("content")
                if isinstance(content, list):
                    source_uuid = entry.get("uuid")
                    source_uuid = (
                        source_uuid
                        if isinstance(source_uuid, str) and source_uuid
                        else None
                    )
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "tool_use":
                            tool_use_id = block.get("id")
                            if not isinstance(tool_use_id, str) or not tool_use_id:
                                skip_trace(
                                    "tool_use has no id", line_number=line_number
                                )
                                return None
                            if (
                                tool_use_id in pending_tool_calls
                                or tool_use_id in open_tool_calls
                            ):
                                skip_trace(
                                    "duplicate tool_use id",
                                    tool_use_id=tool_use_id,
                                    line_number=line_number,
                                )
                                return None
                            pending_tool_calls[tool_use_id] = source_uuid
                            tool_name = block.get("name")
                            if isinstance(tool_name, str):
                                pending_tool_names[tool_use_id] = tool_name
                        pending_blocks.append(block)
                pending_model = message.get("model") or pending_model
                pending_timestamp_raw = entry.get("timestamp") or pending_timestamp_raw
                pending_timestamp = _latest_timestamp(pending_timestamp, timestamp)
                continue

            if entry_type != "user" or not isinstance(message, dict):
                continue

            if not flush_assistant():
                return None
            timestamp = _parse_timestamp(entry.get("timestamp"))
            content = message.get("content")
            if isinstance(content, str):
                if entry.get("isCompactSummary") is True:
                    replace_compacted_context(content, timestamp)
                    compact_boundary_pending = False
                    continue
                if compact_boundary_pending:
                    skip_trace("compact boundary has no marked summary")
                    return None
                handle_text_user(entry, content, timestamp)
                continue

            if not isinstance(content, list):
                if compact_boundary_pending:
                    skip_trace("compact boundary has no marked summary")
                    return None
                continue
            text_blocks: list[str] = []
            result_blocks: list[dict[str, Any]] = []
            result_ids: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if (
                        not isinstance(tool_use_id, str)
                        or tool_use_id not in open_tool_calls
                        or tool_use_id in result_ids
                    ):
                        skip_trace(
                            "tool_result does not match a prior tool_use",
                            tool_use_id=tool_use_id,
                            line_number=line_number,
                        )
                        return None
                    result_blocks.append(block)
                    result_ids.append(tool_use_id)
                elif block.get("type") == "text":
                    text_blocks.append(block.get("text") or "")

            source_uuid = entry.get("sourceToolAssistantUUID")
            if isinstance(source_uuid, str) and result_ids:
                expected_sources = {
                    open_tool_calls[tool_use_id] for tool_use_id in result_ids
                }
                if source_uuid not in expected_sources:
                    skip_trace(
                        "tool_result source does not match tool_use",
                        source_tool_assistant_uuid=source_uuid,
                        line_number=line_number,
                    )
                    return None

            explicit_tool_duration = _tool_duration_seconds(entry)
            for index, block in enumerate(result_blocks):
                tool_use_id = result_ids[index]
                _append_wait(
                    events,
                    actor=(
                        "user"
                        if open_tool_names.get(tool_use_id)
                        in _INTERACTIVE_TOOL_NAMES
                        else "tool"
                    ),
                    previous=timeline_cursor,
                    current=timestamp,
                    explicit_seconds=(
                        explicit_tool_duration
                        if index == 0 and tool_use_id not in parallel_tool_calls
                        else (0.0 if index > 0 else None)
                    ),
                )
                events.append(
                    AppendMessage(
                        message={
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": _content_text(block.get("content")),
                        }
                    )
                )
                timeline_cursor = _latest_timestamp(timeline_cursor, timestamp)
                open_tool_calls.pop(tool_use_id)
                open_tool_names.pop(tool_use_id, None)
                parallel_tool_calls.discard(tool_use_id)
                tool_results += 1

            if text_blocks:
                text_content = "\n".join(text_blocks)
                if entry.get("isCompactSummary") is True:
                    replace_compacted_context(text_content, timestamp)
                    compact_boundary_pending = False
                    continue
                if compact_boundary_pending:
                    skip_trace("compact boundary has no marked summary")
                    return None
                handle_text_user(entry, text_content, timestamp)

    if compact_boundary_pending:
        skip_trace("compact boundary has no marked summary")
        return None
    if not flush_assistant():
        return None
    if generations == 0 or user_messages == 0:
        return None

    trailing_events_trimmed = 0
    while events and not isinstance(events[-1], Generate):
        events.pop()
        trailing_events_trimmed += 1

    initial_messages: list[dict[str, Any]] = []
    while events and not isinstance(events[0], Generate):
        event = events.pop(0)
        if isinstance(event, AppendMessage):
            initial_messages.append(copy.deepcopy(event.message))
        elif isinstance(event, ReplaceContext):
            initial_messages = copy.deepcopy(event.messages)

    return ReplayTrace(
        trace_id=path.stem,
        initial_messages=initial_messages,
        events=events,
        tools=tools,
        metadata={
            "source": "SALT-NLP/SWE-chat",
            "source_format": "claude_code_jsonl",
            "source_path": path.name,
            "user_messages": user_messages,
            "human_user_messages": human_user_messages,
            "automatic_user_messages": automatic_user_messages,
            "tool_results": tool_results,
            "model_requests": generations,
            "compactions": compactions,
            "sidechain_entries_skipped": sidechain_entries_skipped,
            "queue_enqueued": queue_enqueued,
            "queue_human_arrivals": queue_human_arrivals,
            "queue_consumed": queue_consumed,
            "queue_injected": queue_injected,
            "queue_delivered_deduplicated": queue_delivered_deduplicated,
            "queue_adjacent_duplicates": queue_adjacent_duplicates,
            "queue_empty_entries": queue_empty_entries,
            "queue_orphan_operations": queue_orphan_operations,
            "queue_residence_seconds": queue_residence_seconds,
            "queue_unconsumed": len(queued_messages) + len(ready_queued_messages),
            "causal_policy": "strict_skip",
            "microcompaction_policy": "skip",
            "trailing_events_trimmed": trailing_events_trimmed,
        },
    )


def convert_transcripts(
    paths: Iterable[Path],
    *,
    output_path: Path,
    tokenizer_model: str,
    tools: list[dict[str, Any]] | None,
    limit: int | None,
) -> tuple[int, int]:
    tokenizer = TokenizerManager(tokenizer_model)
    converted = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for path in paths:
            if limit is not None and converted >= limit:
                break
            try:
                trace = convert_claude_transcript(path, tokenizer, tools=tools)
            except ValueError as exc:
                logger.warning(
                    "skipping invalid transcript",
                    source_path=str(path),
                    error=str(exc),
                )
                skipped += 1
                continue
            if trace is None:
                skipped += 1
                continue
            output.write(json.dumps(trace.to_jsonl_row(), ensure_ascii=False) + "\n")
            converted += 1
    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SWE-chat Claude Code transcripts to trie schema v2."
    )
    parser.add_argument("transcripts_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tools-json", type=Path)
    parser.add_argument("--trace-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    tools = None
    if args.tools_json is not None:
        with args.tools_json.open("r", encoding="utf-8") as tools_file:
            tools = json.load(tools_file)
        if not isinstance(tools, list):
            raise ValueError("tools JSON must contain a list")

    if args.trace_ids_file is None:
        paths = sorted(args.transcripts_dir.glob("*.jsonl"))
    else:
        trace_ids = [
            line.strip()
            for line in args.trace_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        paths = [args.transcripts_dir / f"{trace_id}.jsonl" for trace_id in trace_ids]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"transcript not found: {missing[0]}")
    converted, skipped = convert_transcripts(
        paths,
        output_path=args.output_path,
        tokenizer_model=args.tokenizer_model,
        tools=tools,
        limit=args.limit,
    )
    logger.info(
        "SWE-chat conversion complete",
        converted=converted,
        skipped=skipped,
        output_path=str(args.output_path),
    )


if __name__ == "__main__":
    main()
