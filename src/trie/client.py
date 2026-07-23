import asyncio
import copy
import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Literal

import structlog
from openai import AsyncOpenAI, OpenAIError
from openai.types.completion_usage import CompletionUsage
from rich.live import Live

from trie.reporting import build_progress_lines, console, log_summary
from trie.prefix import block_aligned_prefix_length, common_prefix_length
from trie.results import (
    BenchmarkResult,
    ServerMetrics,
    StreamAccumulator,
)
from trie.tokenizer_manager import TokenizerManager
from trie.types import (
    AppendMessage,
    Generate,
    ReplayTrace,
    ReplaceContext,
    TraceWorkload,
    Wait,
    Workload,
)

logger = structlog.get_logger(__name__)


def load_workloads(
    workload: list[TraceWorkload] | str,
) -> list[TraceWorkload]:
    if not isinstance(workload, str):
        return workload
    with open(workload, "r", encoding="utf-8") as f:
        rows: list[TraceWorkload] = []
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            try:
                parsed = (
                    ReplayTrace.from_jsonl_row(row)
                    if row.get("schema_version") == 2
                    else Workload.from_jsonl_row(row)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid workload row {line_number}: {exc}") from exc
            rows.append(parsed)
        return rows


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    prompt_token_ids: list[int] | None = None
    generated_token_ids: list[int] | None = None
    usage: CompletionUsage | None = None


def _extension_field(value: object, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    direct = getattr(value, field, None)
    if direct is not None:
        return direct
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(field)
    return None


def _extension_token_ids(value: object, field: str) -> list[int] | None:
    raw_ids = _extension_field(value, field)
    if raw_ids is None:
        return None
    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError(f"response field {field} must be a list of token IDs")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in raw_ids):
        raise ValueError(f"response field {field} contains a non-integer token ID")
    return list(raw_ids)


def _cached_prompt_tokens(usage: CompletionUsage | None) -> int:
    if usage is None or usage.prompt_tokens_details is None:
        return 0
    return usage.prompt_tokens_details.cached_tokens or 0


def _validate_token_id_counts(
    *,
    prompt_token_ids: list[int] | None,
    generated_token_ids: list[int] | None,
    usage: CompletionUsage | None,
) -> None:
    if usage is None:
        return
    if prompt_token_ids is not None and len(prompt_token_ids) != usage.prompt_tokens:
        raise ValueError(
            "response prompt_token_ids length does not match usage.prompt_tokens"
        )
    if (
        generated_token_ids is not None
        and len(generated_token_ids) != usage.completion_tokens
    ):
        raise ValueError(
            "response token_ids length does not match usage.completion_tokens"
        )


class Client:
    """Runs synthetic multi-turn workloads against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        tokenizer_model: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        max_retries: int = 2,
        timeout: float = 600.0,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._tokenizer_manager = TokenizerManager(tokenizer_model or model, seed=seed)
        self._seed = seed
        self._rng = random.Random(seed)
        self._result: BenchmarkResult | None = None
        self._benchmark_start: float | None = None

    async def _execute_stream_request(
        self,
        prompt: str,
        max_tokens: int,
        *,
        trace_start: float,
        stream_acc: StreamAccumulator,
        cache_salt: str | None = None,
        return_token_ids: bool = False,
        deterministic_sampling: bool = False,
    ) -> GenerationResult:
        request_start = time.perf_counter()
        extra_body: dict[str, object] = {"ignore_eos": True}
        if return_token_ids:
            extra_body["return_token_ids"] = True
        if deterministic_sampling:
            extra_body["top_k"] = 1
        if cache_salt is not None:
            extra_body["cache_salt"] = cache_salt
        request_kwargs: dict[str, object] = {}
        if deterministic_sampling:
            request_kwargs.update(temperature=0.0, top_p=1.0)
        stream = await self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
            **request_kwargs,
        )
        text_parts: list[str] = []
        last_token_at_s: float | None = None
        inter_token_latencies_ms: list[float] = []
        ttft_s: float | None = None
        first_token_offset_s: float | None = None
        usage: CompletionUsage | None = None
        prompt_token_ids: list[int] | None = None
        generated_token_ids: list[int] = []
        saw_generated_token_ids = False
        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                text = choice.text or ""
                if text:
                    text_parts.append(text)
                chunk_generated_token_ids = _extension_token_ids(choice, "token_ids")
                if chunk_generated_token_ids is None:
                    chunk_generated_token_ids = _extension_token_ids(
                        chunk, "token_ids"
                    )
                token_count = (
                    len(chunk_generated_token_ids)
                    if chunk_generated_token_ids is not None
                    else self._tokenizer_manager.count_tokens(text)
                )
                if token_count > 0:
                    chunk_at_s = time.perf_counter()
                    if ttft_s is None:
                        ttft_s = chunk_at_s - request_start
                        first_token_offset_s = chunk_at_s - trace_start
                    if last_token_at_s is not None:
                        gap_ms = (
                            (chunk_at_s - last_token_at_s) * 1000.0 / token_count
                        )
                        inter_token_latencies_ms.extend([gap_ms] * token_count)
                    last_token_at_s = chunk_at_s
                chunk_prompt_token_ids = _extension_token_ids(
                    choice, "prompt_token_ids"
                )
                if chunk_prompt_token_ids is None:
                    chunk_prompt_token_ids = _extension_token_ids(
                        chunk, "prompt_token_ids"
                    )
                if chunk_prompt_token_ids is not None:
                    if (
                        prompt_token_ids is not None
                        and chunk_prompt_token_ids != prompt_token_ids
                    ):
                        raise ValueError(
                            "stream returned inconsistent prompt_token_ids"
                        )
                    prompt_token_ids = chunk_prompt_token_ids

                if chunk_generated_token_ids is not None:
                    generated_token_ids.extend(chunk_generated_token_ids)
                    saw_generated_token_ids = True
            if chunk.usage is not None:
                usage = chunk.usage
        latency_s = time.perf_counter() - request_start
        if ttft_s is None or first_token_offset_s is None:
            raise ValueError("stream produced no generated tokens")
        exact_generated_token_ids = (
            generated_token_ids if saw_generated_token_ids else None
        )
        _validate_token_id_counts(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=exact_generated_token_ids,
            usage=usage,
        )
        completion_tokens = (
            usage.completion_tokens
            if usage is not None
            else (
                len(exact_generated_token_ids)
                if exact_generated_token_ids is not None
                else max_tokens
            )
        )
        stream_acc.record_turn_stream_metrics(
            ttft_s,
            latency_s,
            completion_tokens,
            first_token_offset_s=first_token_offset_s,
            inter_token_latencies_ms=inter_token_latencies_ms,
        )
        return GenerationResult(
            text="".join(text_parts),
            prompt_tokens=(
                usage.prompt_tokens
                if usage is not None
                else (
                    len(prompt_token_ids)
                    if prompt_token_ids is not None
                    else self._tokenizer_manager.count_tokens(prompt)
                )
            ),
            completion_tokens=completion_tokens,
            cached_tokens=_cached_prompt_tokens(usage),
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=exact_generated_token_ids,
            usage=usage,
        )

    async def _execute_request(
        self,
        prompt: str,
        max_tokens: int,
        *,
        stream: bool,
        trace_start: float,
        stream_acc: StreamAccumulator,
        cache_salt: str | None = None,
        return_token_ids: bool = False,
        deterministic_sampling: bool = False,
    ) -> GenerationResult:
        if stream:
            return await self._execute_stream_request(
                prompt,
                max_tokens,
                trace_start=trace_start,
                stream_acc=stream_acc,
                cache_salt=cache_salt,
                return_token_ids=return_token_ids,
                deterministic_sampling=deterministic_sampling,
            )

        extra_body: dict[str, object] = {"ignore_eos": True}
        if return_token_ids:
            extra_body["return_token_ids"] = True
        if deterministic_sampling:
            extra_body["top_k"] = 1
        if cache_salt is not None:
            extra_body["cache_salt"] = cache_salt
        request_kwargs: dict[str, object] = {}
        if deterministic_sampling:
            request_kwargs.update(temperature=0.0, top_p=1.0)
        response = await self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=max_tokens,
            extra_body=extra_body,
            **request_kwargs,
        )
        if response.usage is None:
            raise ValueError("response.usage must not be None")
        choice = response.choices[0]
        prompt_token_ids = _extension_token_ids(choice, "prompt_token_ids")
        if prompt_token_ids is None:
            prompt_token_ids = _extension_token_ids(response, "prompt_token_ids")
        generated_token_ids = _extension_token_ids(choice, "token_ids")
        if generated_token_ids is None:
            generated_token_ids = _extension_token_ids(response, "token_ids")
        _validate_token_id_counts(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            usage=response.usage,
        )
        return GenerationResult(
            text=choice.text or "",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cached_tokens=_cached_prompt_tokens(response.usage),
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            usage=response.usage,
        )

    async def _run_legacy_workload(
        self,
        workload: Workload,
        *,
        benchmark_start: float,
        result: BenchmarkResult,
        stream: bool,
        refresh: Callable[[], None],
    ) -> None:
        request_start = time.perf_counter()
        turns_completed = 0
        metrics = ServerMetrics()
        stream_acc = StreamAccumulator()
        try:
            # Generate one fewer token than requested to compensate for BOS token
            # which is added by vLLM, SGLang, and TensorRT-LLM.
            prompt = self._tokenizer_manager.get_prompt(
                max(workload.input_prompt_length - 1, 0)
            )
            for turn_index, turn in enumerate(workload.turns):
                response = await self._execute_request(
                    prompt,
                    turn.assistant_response_length,
                    stream=stream,
                    trace_start=request_start,
                    stream_acc=stream_acc,
                )
                if response.usage is not None:
                    metrics.record_usage(response.usage)
                result.record_turn(
                    workload,
                    turn_index,
                    timestamp=time.perf_counter() - benchmark_start,
                )
                turns_completed = turn_index + 1
                refresh()
                prompt += response.text
                await asyncio.sleep(turn.tool_call_latency)
                prompt += self._tokenizer_manager.get_prompt(
                    turn.tool_call_output_length
                )

            final_response = await self._execute_request(
                prompt,
                workload.final_assistant_response_length,
                stream=stream,
                trace_start=request_start,
                stream_acc=stream_acc,
            )
            if final_response.usage is not None:
                metrics.record_usage(final_response.usage)
            if not stream or metrics.prompt_tokens > 0:
                result.server_metrics.append(metrics)
            else:
                logger.warning("stream response usage missing; omitting server metrics")
            event_timestamp = time.perf_counter()
            result.record_success(
                workload,
                latency=event_timestamp - request_start,
                timestamp=event_timestamp - benchmark_start,
            )
            if stream:
                stream_acc.commit(result)
        except asyncio.CancelledError:
            return
        except (OpenAIError, ValueError) as e:
            result.record_failure()
            logger.warning("request failed", error=str(e), turn=turns_completed)
        refresh()

    @staticmethod
    def _event_delay(
        event: Wait,
        *,
        delay_scale: float,
        max_user_delay: float | None,
        max_tool_delay: float | None,
    ) -> float:
        delay = event.seconds
        if event.actor == "user" and max_user_delay is not None:
            delay = min(delay, max_user_delay)
        elif event.actor == "tool" and max_tool_delay is not None:
            delay = min(delay, max_tool_delay)
        return delay * delay_scale

    def _generated_assistant_content(
        self,
        event: Generate,
        generation: GenerationResult,
    ) -> str:
        recorded = event.recorded_assistant
        if recorded is None or not recorded.get("tool_calls"):
            return generation.text

        text_parts = [
            recorded.get("reasoning_content") or "",
            recorded.get("content") or "",
        ]
        text_budget = len(
            self._tokenizer_manager.encode(
                "\n".join(part for part in text_parts if part)
            )
        )
        if text_budget == 0:
            return ""
        assert generation.generated_token_ids is not None
        return self._tokenizer_manager.decode(
            generation.generated_token_ids[:text_budget]
        )

    async def _run_replay_trace(
        self,
        trace: ReplayTrace,
        *,
        benchmark_start: float,
        result: BenchmarkResult,
        stream: bool,
        refresh: Callable[[], None],
        delay_scale: float,
        max_user_delay: float | None,
        max_tool_delay: float | None,
        context_source: Literal["generated", "recorded"],
        prefix_block_size: int,
        cache_salt_mode: Literal["global", "session"],
        run_cache_salt: str | None,
        strict_replay: bool = False,
    ) -> None:
        metrics = ServerMetrics()
        stream_acc = StreamAccumulator()
        messages = copy.deepcopy(trace.initial_messages)
        previous_actual_ids: list[int] | None = None
        request_index = 0
        if cache_salt_mode == "session":
            if run_cache_salt is None:
                raise ValueError("session cache salt requires a run UUID")
            cache_salt = f"{run_cache_salt}:{trace.trace_id}:{uuid.uuid4().hex}"
        else:
            cache_salt = None

        try:
            if trace.start_offset_s:
                await asyncio.sleep(trace.start_offset_s * delay_scale)
            trace_start = time.perf_counter()

            for event in trace.events:
                if isinstance(event, AppendMessage):
                    messages.append(copy.deepcopy(event.message))
                    continue
                if isinstance(event, ReplaceContext):
                    messages = copy.deepcopy(event.messages)
                    continue
                if isinstance(event, Wait):
                    delay = self._event_delay(
                        event,
                        delay_scale=delay_scale,
                        max_user_delay=max_user_delay,
                        max_tool_delay=max_tool_delay,
                    )
                    if delay:
                        await asyncio.sleep(delay)
                    continue

                assert isinstance(event, Generate)
                prompt = self._tokenizer_manager.render_chat(
                    messages,
                    tools=trace.tools,
                    add_generation_prompt=True,
                )
                generation = await self._execute_request(
                    prompt,
                    event.max_tokens,
                    stream=stream,
                    trace_start=trace_start,
                    stream_acc=stream_acc,
                    cache_salt=cache_salt,
                    return_token_ids=True,
                    deterministic_sampling=strict_replay,
                )
                prompt_token_ids = generation.prompt_token_ids
                generated_token_ids = generation.generated_token_ids
                if prompt_token_ids is None:
                    raise ValueError(
                        "return_token_ids response missing prompt_token_ids"
                    )
                if generated_token_ids is None:
                    raise ValueError("return_token_ids response missing token_ids")
                client_prefix = (
                    common_prefix_length(previous_actual_ids, prompt_token_ids)
                    if previous_actual_ids is not None
                    else 0
                )
                reusable_prefix = (
                    min(
                        client_prefix,
                        max(len(previous_actual_ids) - 1, 0),
                        max(len(prompt_token_ids) - 1, 0),
                    )
                    if previous_actual_ids is not None
                    else 0
                )
                block_prefix = block_aligned_prefix_length(
                    reusable_prefix, prefix_block_size
                )
                cached_tokens = generation.cached_tokens
                if generation.usage is not None:
                    cached_tokens = metrics.record_usage(
                        generation.usage,
                        eligible_prompt_tokens=block_prefix,
                    )
                timestamp = time.perf_counter() - benchmark_start
                result.record_replay_request(
                    timestamp=timestamp,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    client_prefix_tokens=client_prefix,
                    block_aligned_prefix_tokens=block_prefix,
                    server_cached_tokens=cached_tokens,
                )
                request_index += 1
                refresh()

                previous_actual_ids = prompt_token_ids + generated_token_ids
                if context_source == "recorded":
                    if event.recorded_assistant is None:
                        raise ValueError(
                            "recorded context requested but generate event has no "
                            f"recorded_assistant: trace={trace.trace_id!r} "
                            f"request={request_index - 1}"
                        )
                    assistant_message = copy.deepcopy(event.recorded_assistant)
                else:
                    assistant_message = {
                        "role": "assistant",
                        "content": self._generated_assistant_content(
                            event, generation
                        ),
                    }
                    if (
                        event.recorded_assistant is not None
                        and event.recorded_assistant.get("tool_calls")
                    ):
                        assistant_message["tool_calls"] = copy.deepcopy(
                            event.recorded_assistant["tool_calls"]
                        )
                assistant_message["role"] = "assistant"
                messages.append(assistant_message)

            if not stream or metrics.prompt_tokens > 0:
                result.server_metrics.append(metrics)
            else:
                logger.warning("stream response usage missing; omitting server metrics")
            result.record_trace_success(time.perf_counter() - trace_start)
            if stream:
                stream_acc.commit(result)
        except asyncio.CancelledError:
            return
        except (OpenAIError, ValueError) as exc:
            result.record_failure()
            logger.warning(
                "replay trace failed",
                error=str(exc),
                trace_id=trace.trace_id,
                request=request_index,
            )
        refresh()

    async def _run_workload(
        self,
        workload: TraceWorkload,
        *,
        benchmark_start: float,
        result: BenchmarkResult,
        stream: bool,
        refresh: Callable[[], None],
        delay_scale: float,
        max_user_delay: float | None,
        max_tool_delay: float | None,
        context_source: Literal["generated", "recorded"],
        prefix_block_size: int,
        cache_salt_mode: Literal["global", "session"],
        run_cache_salt: str | None,
        strict_replay: bool,
    ) -> None:
        if isinstance(workload, ReplayTrace):
            await self._run_replay_trace(
                workload,
                benchmark_start=benchmark_start,
                result=result,
                stream=stream,
                refresh=refresh,
                delay_scale=delay_scale,
                max_user_delay=max_user_delay,
                max_tool_delay=max_tool_delay,
                context_source=context_source,
                prefix_block_size=prefix_block_size,
                cache_salt_mode=cache_salt_mode,
                run_cache_salt=run_cache_salt,
                strict_replay=strict_replay,
            )
            return
        await self._run_legacy_workload(
            workload,
            benchmark_start=benchmark_start,
            result=result,
            stream=stream,
            refresh=refresh,
        )

    def _order_workloads(
        self,
        workloads: list[TraceWorkload],
        *,
        workload_order: Literal["natural", "shuffle", "prefix"],
        prefix_sort_tokens: int,
    ) -> list[TraceWorkload]:
        ordered = list(workloads)
        if workload_order == "shuffle":
            self._rng.shuffle(ordered)
            return ordered
        if workload_order != "prefix":
            return ordered

        def prefix_key(workload: TraceWorkload) -> tuple[int, tuple[int, ...]]:
            if not isinstance(workload, ReplayTrace):
                return (1, ())
            prompt = self._tokenizer_manager.render_chat(
                workload.initial_messages,
                tools=workload.tools,
                add_generation_prompt=True,
            )
            return (
                0,
                tuple(self._tokenizer_manager.encode(prompt)[:prefix_sort_tokens]),
            )

        ordered.sort(key=prefix_key)
        return ordered

    async def run(
        self,
        workload: list[TraceWorkload] | str,
        concurrency: int | None = None,
        duration: float = 300.0,
        arrival_rate: float | None = None,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
        delay_scale: float = 1.0,
        max_user_delay: float | None = None,
        max_tool_delay: float | None = None,
        context_source: Literal["generated", "recorded"] = "generated",
        prefix_block_size: int = 16,
        cache_salt_mode: Literal["global", "session"] = "session",
        workload_order: Literal["natural", "shuffle", "prefix"] = "natural",
        prefix_sort_tokens: int = 256,
        replay_once: bool = False,
        drain_timeout: float = 0.0,
        strict_replay: bool = False,
    ) -> BenchmarkResult:
        workload_list = self._order_workloads(
            load_workloads(workload),
            workload_order=workload_order,
            prefix_sort_tokens=prefix_sort_tokens,
        )
        if not workload_list:
            raise ValueError("workload must contain at least one trace")
        if (concurrency is None) == (arrival_rate is None):
            raise ValueError("specify exactly one of concurrency or arrival_rate")
        if concurrency is not None and concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")
        if arrival_rate is not None and arrival_rate <= 0:
            raise ValueError("arrival_rate must be greater than 0")
        if duration_update_interval is not None and duration_update_interval <= 0:
            raise ValueError("duration_update_interval must be greater than 0")
        if delay_scale < 0:
            raise ValueError("delay_scale must be non-negative")
        if max_user_delay is not None and max_user_delay < 0:
            raise ValueError("max_user_delay must be non-negative")
        if max_tool_delay is not None and max_tool_delay < 0:
            raise ValueError("max_tool_delay must be non-negative")
        if context_source not in ("generated", "recorded"):
            raise ValueError("context_source must be generated or recorded")
        if prefix_block_size <= 0:
            raise ValueError("prefix_block_size must be greater than 0")
        if cache_salt_mode not in ("global", "session"):
            raise ValueError("cache_salt_mode must be global or session")
        if workload_order not in ("natural", "shuffle", "prefix"):
            raise ValueError("workload_order must be natural, shuffle, or prefix")
        if prefix_sort_tokens <= 0:
            raise ValueError("prefix_sort_tokens must be greater than 0")
        if drain_timeout < 0:
            raise ValueError("drain_timeout must be non-negative")
        if strict_replay:
            if any(not isinstance(item, ReplayTrace) for item in workload_list):
                raise ValueError("strict_replay only supports schema-v2 replay traces")
            context_source = "recorded"
        run_cache_salt = uuid.uuid4().hex if cache_salt_mode == "session" else None
        benchmark_log_fields = {
            "model": self._model,
            "workload_templates": len(workload_list),
            "duration": duration,
            "num_gpus": num_gpus,
            "delay_scale": delay_scale,
            "context_source": context_source,
            "cache_salt_mode": cache_salt_mode,
            "workload_order": workload_order,
            "replay_once": replay_once,
            "drain_timeout": drain_timeout,
            "strict_replay": strict_replay,
        }
        if concurrency is not None:
            benchmark_log_fields["concurrency"] = concurrency
        if arrival_rate is not None:
            benchmark_log_fields["arrival_rate"] = arrival_rate
        if duration_update_interval is not None:
            benchmark_log_fields["duration_update_interval"] = duration_update_interval
        logger.info(
            "starting benchmark",
            **benchmark_log_fields,
        )
        result = BenchmarkResult()
        if replay_once:
            result.expected_requests = len(workload_list)
        benchmark_start = time.perf_counter()
        self._result = result
        self._benchmark_start = benchmark_start

        with Live(
            build_progress_lines(result, 0.0, duration),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:
            next_duration_update = duration_update_interval

            def refresh() -> None:
                live.update(
                    build_progress_lines(
                        result,
                        time.perf_counter() - benchmark_start,
                        duration,
                    ),
                    refresh=True,
                )

            def log_duration_update() -> None:
                nonlocal next_duration_update
                if next_duration_update is None:
                    return
                elapsed = time.perf_counter() - benchmark_start
                if elapsed < next_duration_update:
                    return
                logger.info(
                    "benchmark in progress",
                    remaining_duration=round(max(duration - elapsed, 0.0), 1),
                )
                next_duration_update = elapsed + duration_update_interval

            active: set[asyncio.Task[None]] = set()
            workload_iter = iter(workload_list) if replay_once else cycle(workload_list)
            next_arrival = benchmark_start
            arrival_interval = 1.0 / arrival_rate if arrival_rate is not None else None
            exhausted = False
            admitted_workloads = 0

            while time.perf_counter() - benchmark_start < duration:
                log_duration_update()
                if arrival_interval is not None:
                    done = {task for task in active if task.done()}
                    active -= done
                    for task in done:
                        task.result()

                if concurrency is not None and len(active) == concurrency:
                    elapsed = time.perf_counter() - benchmark_start
                    timeout = max(duration - elapsed, 0.0)
                    if next_duration_update is not None:
                        timeout = min(
                            timeout,
                            max(next_duration_update - elapsed, 0.0),
                        )
                    done, active = await asyncio.wait(
                        active,
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        task.result()
                    if not done:
                        continue

                if arrival_interval is not None:
                    wait_until = min(next_arrival, benchmark_start + duration)
                    if next_duration_update is not None:
                        wait_until = min(
                            wait_until,
                            benchmark_start + next_duration_update,
                        )
                    timeout = wait_until - time.perf_counter()
                    if timeout > 0:
                        await asyncio.sleep(timeout)
                        continue

                try:
                    next_workload = next(workload_iter)
                except StopIteration:
                    exhausted = True
                    break
                active.add(
                    asyncio.create_task(
                        self._run_workload(
                            next_workload,
                            benchmark_start=benchmark_start,
                            result=result,
                            stream=stream,
                            refresh=refresh,
                            delay_scale=delay_scale,
                            max_user_delay=max_user_delay,
                            max_tool_delay=max_tool_delay,
                            context_source=context_source,
                            prefix_block_size=prefix_block_size,
                            cache_salt_mode=cache_salt_mode,
                            run_cache_salt=run_cache_salt,
                            strict_replay=strict_replay,
                        )
                    )
                )
                admitted_workloads += 1
                if replay_once and admitted_workloads == len(workload_list):
                    exhausted = True
                    break
                if arrival_interval is not None:
                    next_arrival += arrival_interval
            if exhausted:
                logger.info("all replay traces admitted", active_traces=len(active))
            if active and drain_timeout > 0:
                done, pending = await asyncio.wait(active, timeout=drain_timeout)
                for task in done:
                    task.result()
                active = pending
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
        result.wall_time = time.perf_counter() - benchmark_start
        log_summary(result, num_gpus=num_gpus)
        return result

    def sync_run(
        self,
        workload: list[TraceWorkload] | str,
        concurrency: int | None = None,
        duration: float = 300.0,
        arrival_rate: float | None = None,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
        delay_scale: float = 1.0,
        max_user_delay: float | None = None,
        max_tool_delay: float | None = None,
        context_source: Literal["generated", "recorded"] = "generated",
        prefix_block_size: int = 16,
        cache_salt_mode: Literal["global", "session"] = "session",
        workload_order: Literal["natural", "shuffle", "prefix"] = "natural",
        prefix_sort_tokens: int = 256,
        replay_once: bool = False,
        drain_timeout: float = 0.0,
        strict_replay: bool = False,
    ) -> BenchmarkResult:
        try:
            return asyncio.run(
                self.run(
                    workload,
                    concurrency,
                    duration,
                    arrival_rate=arrival_rate,
                    duration_update_interval=duration_update_interval,
                    num_gpus=num_gpus,
                    stream=stream,
                    delay_scale=delay_scale,
                    max_user_delay=max_user_delay,
                    max_tool_delay=max_tool_delay,
                    context_source=context_source,
                    prefix_block_size=prefix_block_size,
                    cache_salt_mode=cache_salt_mode,
                    workload_order=workload_order,
                    prefix_sort_tokens=prefix_sort_tokens,
                    replay_once=replay_once,
                    drain_timeout=drain_timeout,
                    strict_replay=strict_replay,
                )
            )
        except KeyboardInterrupt:
            logger.warning("benchmark interrupted")
            assert self._result is not None and self._benchmark_start is not None
            result = self._result
            result.wall_time = time.perf_counter() - self._benchmark_start
            log_summary(result, num_gpus=num_gpus)
            return result
        finally:
            self._result = None
            self._benchmark_start = None
