from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


@dataclass
class Turn:
    assistant_response_length: int
    tool_call_output_length: int
    tool_call_latency: float

    def __post_init__(self) -> None:
        if self.assistant_response_length < 0:
            raise ValueError("assistant_response_length must be non-negative")
        if self.tool_call_output_length < 0:
            raise ValueError("tool_call_output_length must be non-negative")
        if self.tool_call_latency < 0:
            raise ValueError("tool_call_latency must be non-negative")


@dataclass
class Workload:
    input_prompt_length: int
    turns: list[Turn]
    final_assistant_response_length: int

    def cumulative_prompt_tokens(self, through_turn: int) -> int:
        return self.input_prompt_length + sum(
            self.turns[i].assistant_response_length
            + self.turns[i].tool_call_output_length
            for i in range(through_turn)
        )

    @classmethod
    def from_jsonl_row(cls, d: dict[str, Any]) -> "Workload":
        num_turns = d["num_turns"]
        resp = d["assistant_response_length"]
        out = d["tool_call_output_length"]
        lat = d["tool_call_latency"]
        for name, values in (
            ("assistant_response_length", resp),
            ("tool_call_output_length", out),
            ("tool_call_latency", lat),
        ):
            if not isinstance(values, list):
                raise ValueError(f"{name} must be a list of length {num_turns}")
            if len(values) != num_turns:
                raise ValueError(
                    f"{name} length {len(values)} != num_turns {num_turns}"
                )
        turns = [Turn(resp[i], out[i], lat[i]) for i in range(num_turns)]
        return cls(
            input_prompt_length=d["input_prompt_length"],
            turns=turns,
            final_assistant_response_length=d["final_assistant_response_length"],
        )


@dataclass
class AppendMessage:
    message: dict[str, Any]

    def __post_init__(self) -> None:
        role = self.message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("append_message.message.role must be a non-empty string")


@dataclass
class Wait:
    seconds: float
    actor: Literal["user", "tool", "other"] = "other"

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("wait seconds must be non-negative")
        if self.actor not in ("user", "tool", "other"):
            raise ValueError("wait actor must be user, tool, or other")


@dataclass
class Generate:
    max_tokens: int
    recorded_assistant: dict[str, Any] | None = None
    recorded_output_token_ids: list[int] | None = None
    source_model: str | None = None
    source_timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("generate max_tokens must be greater than 0")
        if self.recorded_assistant is not None:
            role = self.recorded_assistant.get("role", "assistant")
            if role != "assistant":
                raise ValueError("recorded_assistant role must be assistant")
        if self.recorded_output_token_ids is not None:
            if len(self.recorded_output_token_ids) != self.max_tokens:
                raise ValueError(
                    "recorded_output_token_ids length must equal max_tokens"
                )
            if any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in self.recorded_output_token_ids
            ):
                raise ValueError(
                    "recorded_output_token_ids must contain non-negative integers"
                )


@dataclass
class ReplaceContext:
    messages: list[dict[str, Any]]
    reason: Literal["compaction", "reset", "branch", "other"] = "other"

    def __post_init__(self) -> None:
        for message in self.messages:
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("replace_context messages must have non-empty roles")


ReplayEvent: TypeAlias = AppendMessage | Wait | Generate | ReplaceContext


@dataclass
class ReplayTrace:
    trace_id: str
    events: list[ReplayEvent]
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset_s: float = 0.0
    schema_version: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")
        if self.start_offset_s < 0:
            raise ValueError("start_offset_s must be non-negative")
        if not any(isinstance(event, Generate) for event in self.events):
            raise ValueError("replay trace must contain at least one generate event")
        for message in self.initial_messages:
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("initial messages must have non-empty roles")

    @classmethod
    def from_jsonl_row(cls, d: dict[str, Any]) -> "ReplayTrace":
        if d.get("schema_version") != 2:
            raise ValueError("replay trace schema_version must be 2")

        events: list[ReplayEvent] = []
        for index, raw_event in enumerate(d.get("events", [])):
            event_type = raw_event.get("type")
            if event_type == "append_message":
                events.append(AppendMessage(message=raw_event["message"]))
            elif event_type == "wait":
                events.append(
                    Wait(
                        seconds=float(raw_event["seconds"]),
                        actor=raw_event.get("actor", "other"),
                    )
                )
            elif event_type == "generate":
                events.append(
                    Generate(
                        max_tokens=int(raw_event["max_tokens"]),
                        recorded_assistant=raw_event.get("recorded_assistant"),
                        recorded_output_token_ids=raw_event.get(
                            "recorded_output_token_ids"
                        ),
                        source_model=raw_event.get("source_model"),
                        source_timestamp=raw_event.get("source_timestamp"),
                    )
                )
            elif event_type == "replace_context":
                events.append(
                    ReplaceContext(
                        messages=raw_event["messages"],
                        reason=raw_event.get("reason", "other"),
                    )
                )
            else:
                raise ValueError(
                    f"unknown replay event type at index {index}: {event_type!r}"
                )

        return cls(
            trace_id=d["trace_id"],
            initial_messages=d.get("initial_messages", []),
            events=events,
            tools=d.get("tools"),
            metadata=d.get("metadata", {}),
            start_offset_s=float(d.get("start_offset_s", 0.0)),
        )

    def to_jsonl_row(self) -> dict[str, Any]:
        raw_events: list[dict[str, Any]] = []
        for event in self.events:
            if isinstance(event, AppendMessage):
                raw_events.append({"type": "append_message", "message": event.message})
            elif isinstance(event, Wait):
                raw_events.append(
                    {"type": "wait", "seconds": event.seconds, "actor": event.actor}
                )
            elif isinstance(event, Generate):
                raw_event: dict[str, Any] = {
                    "type": "generate",
                    "max_tokens": event.max_tokens,
                }
                if event.recorded_assistant is not None:
                    raw_event["recorded_assistant"] = event.recorded_assistant
                if event.recorded_output_token_ids is not None:
                    raw_event["recorded_output_token_ids"] = (
                        event.recorded_output_token_ids
                    )
                if event.source_model is not None:
                    raw_event["source_model"] = event.source_model
                if event.source_timestamp is not None:
                    raw_event["source_timestamp"] = event.source_timestamp
                raw_events.append(raw_event)
            else:
                raw_events.append(
                    {
                        "type": "replace_context",
                        "messages": event.messages,
                        "reason": event.reason,
                    }
                )
        return {
            "schema_version": 2,
            "trace_id": self.trace_id,
            "start_offset_s": self.start_offset_s,
            "initial_messages": self.initial_messages,
            "tools": self.tools,
            "events": raw_events,
            "metadata": self.metadata,
        }


TraceWorkload: TypeAlias = Workload | ReplayTrace
