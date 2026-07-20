import asyncio
import json
import random

from trie.client import Client, GenerationResult
from trie.results import BenchmarkResult
from trie.types import AppendMessage, Generate, ReplayTrace, ReplaceContext, Wait


class SimpleTokenizer:
    def render_chat(self, messages, *, tools=None, add_generation_prompt=True):
        rendered = "".join(
            (
                f"<{message['role']}>{message.get('content') or ''}"
                + (
                    json.dumps(message["tool_calls"], sort_keys=True)
                    if message.get("tool_calls")
                    else ""
                )
            )
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode()


class FakeClient(Client):
    def __init__(self, outputs: list[str | tuple[str, list[int]]]) -> None:
        self._model = "fake"
        self._rng = random.Random(0)
        self._result = None
        self._benchmark_start = None
        self._tokenizer_manager = SimpleTokenizer()
        self._outputs = iter(outputs)
        self.prompts: list[str] = []
        self.cache_salts: list[str | None] = []

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        self.prompts.append(prompt)
        self.cache_salts.append(kwargs.get("cache_salt"))
        output = next(self._outputs)
        if isinstance(output, tuple):
            text, generated_token_ids = output
        else:
            text = output
            generated_token_ids = list(text.encode())
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt),
            completion_tokens=len(text),
            cached_tokens=0,
            prompt_token_ids=self._tokenizer_manager.encode(prompt),
            generated_token_ids=generated_token_ids,
        )


def _run_context_source(
    context_source: str,
    *,
    prefix_block_size: int = 16,
) -> tuple[FakeClient, BenchmarkResult]:
    client = FakeClient(["live-one", "live-two"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-1",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[
            Generate(
                max_tokens=8,
                recorded_assistant={"role": "assistant", "content": "recorded-one"},
            ),
            AppendMessage(message={"role": "user", "content": "continue"}),
            Generate(
                max_tokens=8,
                recorded_assistant={"role": "assistant", "content": "recorded-two"},
            ),
        ],
    )
    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source=context_source,
            prefix_block_size=prefix_block_size,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )
    return client, result


def test_generated_context_uses_live_output() -> None:
    client, _ = _run_context_source("generated")
    assert "live-one" in client.prompts[1]
    assert "recorded-one" not in client.prompts[1]


def test_generated_context_does_not_duplicate_recorded_tool_calls() -> None:
    client = FakeClient(["abcdefghij", "done"])
    result = BenchmarkResult()
    tool_calls = [
        {
            "id": "tool-1",
            "type": "function",
            "function": {"name": "Read", "arguments": {"path": "a.py"}},
        }
    ]
    trace = ReplayTrace(
        trace_id="trace-tools",
        initial_messages=[{"role": "user", "content": "inspect"}],
        events=[
            Generate(
                max_tokens=10,
                recorded_assistant={
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": tool_calls,
                },
            ),
            AppendMessage(
                message={
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": "result",
                }
            ),
            Generate(max_tokens=4),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert "<assistant>ab" in client.prompts[1]
    assert "abcdefghij" not in client.prompts[1]
    assert '"id": "tool-1"' in client.prompts[1]
    first_prompt_tokens = len(client.prompts[0].encode())
    assert result.client_prefix_tokens == [0, first_prompt_tokens + len("ab")]


def test_generated_tool_only_turn_reuses_only_the_previous_prompt() -> None:
    client = FakeClient(["generated-but-discarded", "done"])
    result = BenchmarkResult()
    tool_calls = [
        {
            "id": "tool-1",
            "type": "function",
            "function": {"name": "Bash", "arguments": {"command": "true"}},
        }
    ]
    trace = ReplayTrace(
        trace_id="trace-tool-only",
        initial_messages=[{"role": "user", "content": "run it"}],
        events=[
            Generate(
                max_tokens=23,
                recorded_assistant={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                },
            ),
            AppendMessage(
                message={
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": "ok",
                }
            ),
            Generate(max_tokens=4),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert "generated-but-discarded" not in client.prompts[1]
    assert '"id": "tool-1"' in client.prompts[1]
    assert result.client_prefix_tokens == [0, len(client.prompts[0].encode())]


def test_recorded_context_uses_recorded_output() -> None:
    client, _ = _run_context_source("recorded")
    assert "recorded-one" in client.prompts[1]
    assert "live-one" not in client.prompts[1]


def test_event_delay_applies_actor_caps_before_scaling() -> None:
    assert Client._event_delay(
        Wait(seconds=10.0, actor="user"),
        delay_scale=0.5,
        max_user_delay=4.0,
        max_tool_delay=None,
    ) == 2.0
    assert Client._event_delay(
        Wait(seconds=10.0, actor="tool"),
        delay_scale=0.25,
        max_user_delay=None,
        max_tool_delay=6.0,
    ) == 1.5
    assert Client._event_delay(
        Wait(seconds=10.0, actor="other"),
        delay_scale=0.1,
        max_user_delay=1.0,
        max_tool_delay=1.0,
    ) == 1.0


def test_replay_trace_executes_scaled_start_and_event_waits(monkeypatch) -> None:
    sleeps: list[float] = []
    clock = [100.0]

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    monkeypatch.setattr("trie.client.time.perf_counter", lambda: clock[0])
    client = FakeClient(["done"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-waits",
        start_offset_s=4.0,
        events=[
            Wait(seconds=10.0, actor="user"),
            Wait(seconds=8.0, actor="tool"),
            Wait(seconds=3.0, actor="other"),
            Generate(max_tokens=4),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.5,
            max_user_delay=2.0,
            max_tool_delay=4.0,
            context_source="generated",
            prefix_block_size=16,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert sleeps == [2.0, 1.0, 2.0, 1.5]
    assert result.completed_requests == 1
    assert result.latencies == [4.5]


def test_prefix_uses_exact_generated_token_ids() -> None:
    client, result = _run_context_source("generated", prefix_block_size=1)

    previous_sequence_tokens = len(client.prompts[0].encode()) + len("live-one")
    assert result.client_prefix_tokens == [0, previous_sequence_tokens]


def test_prefix_does_not_reencode_generated_text() -> None:
    client = FakeClient([("live-one", [999]), "live-two"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-exact-ids",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[Generate(max_tokens=8), Generate(max_tokens=8)],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert result.client_prefix_tokens == [0, len(client.prompts[0].encode())]


def test_eligible_prefix_is_block_aligned() -> None:
    client, result = _run_context_source("generated", prefix_block_size=16)

    previous_sequence_tokens = len(client.prompts[0].encode()) + len("live-one")
    expected = previous_sequence_tokens - (previous_sequence_tokens % 16)
    assert result.block_aligned_prefix_tokens == [0, expected]


def test_eligible_prefix_excludes_uncached_terminal_token() -> None:
    initial_messages = [{"role": "user", "content": "hello"}]
    initial_prompt = SimpleTokenizer().render_chat(initial_messages)
    first_output = "x" * (32 - len(initial_prompt.encode()))
    client = FakeClient([first_output, "done"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-block-boundary",
        initial_messages=initial_messages,
        events=[Generate(max_tokens=len(first_output)), Generate(max_tokens=4)],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=16,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert result.client_prefix_tokens == [0, 32]
    assert result.block_aligned_prefix_tokens == [0, 16]


def test_replace_context_drops_pre_compaction_history() -> None:
    client = FakeClient(["live-before-compact", "live-after-compact"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-compact",
        initial_messages=[{"role": "user", "content": "old history"}],
        events=[
            Generate(max_tokens=8),
            ReplaceContext(
                messages=[{"role": "user", "content": "compact summary"}],
                reason="compaction",
            ),
            Generate(max_tokens=8),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=16,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert "compact summary" in client.prompts[1]
    assert "old history" not in client.prompts[1]
    assert "live-before-compact" not in client.prompts[1]
    assert result.block_aligned_prefix_tokens == [0, 0]


def test_session_cache_salt_is_stable_within_trace() -> None:
    client, _ = _run_context_source("generated")

    assert client.cache_salts[0] == client.cache_salts[1]
    assert client.cache_salts[0] is not None
    assert client.cache_salts[0].startswith("run-1:trace-1:")


def test_session_cache_salt_changes_between_runs() -> None:
    client = FakeClient(["one", "two"])
    trace = ReplayTrace(trace_id="same-trace", events=[Generate(max_tokens=1)])

    for _ in range(2):
        asyncio.run(
            client.run(
                [trace],
                concurrency=1,
                    duration=1.0,
                    replay_once=True,
                    drain_timeout=1.0,
                    cache_salt_mode="session",
            )
        )

    assert client.cache_salts[0] != client.cache_salts[1]
    assert all(salt and ":same-trace:" in salt for salt in client.cache_salts)


def test_session_cache_salt_changes_between_trace_invocations() -> None:
    client = FakeClient(["one", "two"])
    trace = ReplayTrace(trace_id="same-trace", events=[Generate(max_tokens=1)])

    asyncio.run(
        client.run(
            [trace, trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
            cache_salt_mode="session",
        )
    )

    assert len(client.cache_salts) == 2
    assert client.cache_salts[0] != client.cache_salts[1]


def test_global_cache_salt_remains_none() -> None:
    client = FakeClient(["one"])
    trace = ReplayTrace(trace_id="trace-global", events=[Generate(max_tokens=1)])

    asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
            cache_salt_mode="global",
        )
    )

    assert client.cache_salts == [None]


class SchedulerClient(Client):
    def __init__(self) -> None:
        self._model = "fake"
        self._rng = random.Random(0)
        self._result = None
        self._benchmark_start = None

    async def _run_workload(self, workload, *, result, **kwargs) -> None:
        await asyncio.sleep(0.01)
        result.record_trace_success(0.01)


class SlowSchedulerClient(SchedulerClient):
    async def _run_workload(self, workload, *, result, **kwargs) -> None:
        await asyncio.sleep(0.2)
        result.record_trace_success(0.2)


def test_replay_once_drains_after_all_traces_are_admitted() -> None:
    client = SchedulerClient()
    traces = [
        ReplayTrace(trace_id=f"trace-{index}", events=[Generate(max_tokens=1)])
        for index in range(2)
    ]

    result = asyncio.run(
        client.run(
            traces,
            concurrency=4,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
        )
    )

    assert result.completed_requests == len(traces)
    assert result.expected_requests == len(traces)


def test_replay_once_refills_concurrency_until_all_traces_run() -> None:
    client = SchedulerClient()
    traces = [
        ReplayTrace(trace_id=f"trace-{index}", events=[Generate(max_tokens=1)])
        for index in range(3)
    ]

    result = asyncio.run(
        client.run(
            traces,
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
        )
    )

    assert result.completed_requests == len(traces)
    assert result.expected_requests == len(traces)


def test_replay_once_zero_drain_timeout_cancels_active_traces() -> None:
    client = SchedulerClient()
    trace = ReplayTrace(trace_id="trace", events=[Generate(max_tokens=1)])

    result = asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=0.0,
        )
    )

    assert result.completed_requests == 0


def test_arrival_rate_replay_once_does_not_wait_for_extra_arrival() -> None:
    client = SchedulerClient()
    trace = ReplayTrace(trace_id="trace", events=[Generate(max_tokens=1)])

    result = asyncio.run(
        client.run(
            [trace],
            arrival_rate=2.0,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
        )
    )

    assert result.completed_requests == 1
    assert result.wall_time < 0.25


def test_duration_cancels_full_concurrency_wait() -> None:
    client = SlowSchedulerClient()
    trace = ReplayTrace(trace_id="trace", events=[Generate(max_tokens=1)])

    result = asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=0.05,
        )
    )

    assert result.completed_requests == 0
    assert result.wall_time < 0.15
