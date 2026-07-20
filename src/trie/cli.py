import logging
from typing import Literal

import chz
import structlog

from trie.client import Client


@chz.chz
class RunArgs:
    workload_path: str = chz.field(doc="Path to the JSONL workload file.")
    endpoint: str = chz.field(doc="OpenAI-compatible endpoint base URL.")
    model: str = chz.field(doc="Model name to send to the completion API.")
    concurrency: int | None = chz.field(default=None, doc="Maximum concurrent traces.")
    arrival_rate: float | None = chz.field(default=None, doc="Trace starts per second.")
    tokenizer_model: str | None = chz.field(
        default=None,
        doc="Optional model name override for loading tokenizer.",
    )
    duration: float = chz.field(
        default=300.0,
        doc="Duration in seconds to run the benchmark.",
    )
    duration_update_interval: float | None = chz.field(
        default=None,
        doc="Optional interval in seconds for logging remaining benchmark duration.",
    )
    seed: int | None = chz.field(
        default=None,
        doc="Optional random seed for reproducible workload resolution and prompt generation.",
    )
    api_key: str | None = chz.field(
        default=None,
        doc="API key for the endpoint. Falls back to OPENAI_API_KEY env var if not set.",
    )
    max_retries: int = chz.field(
        default=2,
        doc="Number of retries for the endpoint, openai default is 2.",
    )
    timeout: float = chz.field(
        default=600.0,
        doc="Timeout for the endpoint, openai default is 600 seconds.",
    )
    num_gpus: int | None = chz.field(
        default=None,
        doc="Optional GPU count used for the steady-state per-GPU throughput column.",
    )
    stream: bool = chz.field(
        default=False,
        doc="Use streaming completions and report TTFT, TTFAT, and TPS per trace.",
    )
    delay_scale: float = chz.field(
        default=1.0,
        doc="Scale recorded user/tool waits; 0 disables waits and 1 replays real time.",
    )
    max_user_delay: float | None = chz.field(
        default=None,
        doc="Optional cap in seconds applied to each recorded user wait.",
    )
    max_tool_delay: float | None = chz.field(
        default=None,
        doc="Optional cap in seconds applied to each recorded tool wait.",
    )
    context_source: Literal["generated", "recorded"] = chz.field(
        default="generated",
        doc="Use live model output or recorded assistant output in the next prompt.",
    )
    prefix_block_size: int = chz.field(
        default=16,
        doc="Token block size used for client-side prefix-cache overlap metrics.",
    )
    cache_salt_mode: Literal["global", "session"] = chz.field(
        default="session",
        doc="Share vLLM prefix cache globally or isolate it by replay session.",
    )
    workload_order: Literal["natural", "shuffle", "prefix"] = chz.field(
        default="natural",
        doc="Trace ordering; prefix is a cache-locality upper-bound experiment.",
    )
    prefix_sort_tokens: int = chz.field(
        default=256,
        doc="Initial prompt tokens used when workload_order=prefix.",
    )
    replay_once: bool = chz.field(
        default=False,
        doc="Admit each trace once instead of cycling the workload until duration.",
    )
    drain_timeout: float = chz.field(
        default=0.0,
        doc="Seconds to drain in-flight traces after admission stops; 0 cancels immediately.",
    )

    @chz.validate
    def _validate_fields(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")
        if (
            self.duration_update_interval is not None
            and self.duration_update_interval <= 0
        ):
            raise ValueError("duration_update_interval must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.num_gpus is not None and self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        if self.delay_scale < 0:
            raise ValueError("delay_scale must be non-negative")
        if self.max_user_delay is not None and self.max_user_delay < 0:
            raise ValueError("max_user_delay must be non-negative")
        if self.max_tool_delay is not None and self.max_tool_delay < 0:
            raise ValueError("max_tool_delay must be non-negative")
        if self.prefix_block_size <= 0:
            raise ValueError("prefix_block_size must be greater than 0")
        if self.prefix_sort_tokens <= 0:
            raise ValueError("prefix_sort_tokens must be greater than 0")
        if self.drain_timeout < 0:
            raise ValueError("drain_timeout must be non-negative")


def main() -> None:
    args: RunArgs = chz.entrypoint(RunArgs)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    Client(
        endpoint=args.endpoint,
        model=args.model,
        tokenizer_model=args.tokenizer_model,
        api_key=args.api_key,
        seed=args.seed,
        max_retries=args.max_retries,
        timeout=args.timeout,
    ).sync_run(
        args.workload_path,
        concurrency=args.concurrency,
        arrival_rate=args.arrival_rate,
        duration=args.duration,
        duration_update_interval=args.duration_update_interval,
        num_gpus=args.num_gpus,
        stream=args.stream,
        delay_scale=args.delay_scale,
        max_user_delay=args.max_user_delay,
        max_tool_delay=args.max_tool_delay,
        context_source=args.context_source,
        prefix_block_size=args.prefix_block_size,
        cache_salt_mode=args.cache_salt_mode,
        workload_order=args.workload_order,
        prefix_sort_tokens=args.prefix_sort_tokens,
        replay_once=args.replay_once,
        drain_timeout=args.drain_timeout,
    )


if __name__ == "__main__":
    main()
