# trie

`trie` stands for trace replay inference evaluation and is a lightweight benchmarking harness that exercises OpenAI-compatible inference servers with synthetic workloads derived from production traces. It targets backends like vLLM, SGLang, and TensorRT-LLM.

Most inference benchmarks test prefill-heavy or decode-heavy workloads (e.g. 1k/8k, 1k/1k, 8k/1k). However, real agentic workloads look very different: they're multi-turn, have high per-turn prefill from tool outputs, and put increasing pressure on KV cache management as context grows.

## Quick start

From the CLI:

```bash
uv run trie \
  workload_path=workload.jsonl \
  endpoint=http://localhost:8000/v1 \
  model=deepseek-ai/DeepSeek-R1 \
  concurrency=8 \
  duration=300 \
  stream=True \
  num_gpus=8
```

CLI arguments use the `RunArgs` field names directly, so multiword arguments
should be passed with underscores such as `tokenizer_model=...`.

`model` is the name sent to the inference endpoint. `tokenizer_model` is
loaded separately via `transformers.AutoTokenizer.from_pretrained(...)` to
generate synthetic prompts with the requested token lengths. If `model` is
not a valid Hugging Face model ID or local checkpoint/tokenizer path, pass
`tokenizer_model=...` explicitly.

From Python:

```python
from trie import Client

client = Client(
    endpoint="http://localhost:8000/v1",
    model="deepseek-ai/DeepSeek-R1",
)
client.sync_run("workload.jsonl", concurrency=8, duration=300, stream=True)
# run() is async if you're already in an event loop
```

`duration` is the deadline for launching new traces. Once the benchmark
reaches that limit, the client stops admitting new work and cancels all
in-flight traces immediately.

For event replay, set `replay_once=True` and a positive `drain_timeout` to
admit each recorded trace once and let in-flight sessions finish for up to the
requested timeout.

Before starting a benchmark, make sure the engine is idle and not serving
leftover traffic from earlier runs. Starting from a non-idle state can skew
cache behavior, warmup, and steady-state throughput measurements.

## Backend requirements

- `extra_body={"ignore_eos": True}`: the harness uses this to force fixed-length
  generations (effectively `min_tokens = max_tokens`). Backends must support
  this extension.
- Cache-hit metrics require the server to return
  `usage.prompt_tokens_details.cached_tokens` on completion responses:
  - **SGLang**: launch with `--enable-cache-report`.
  - **vLLM**: launch with `--enable-prompt-tokens-details`.
- The client-side `transformers` tokenizer should match what the server uses.
  Mismatches can miscount synthetic prompts and cause context-length errors.

For vLLM prefix-cache experiments, launch the server explicitly with:

```bash
vllm serve MODEL \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --generation-config vllm
```

vLLM matches exact token blocks; the replay client does not approximate or
semantically match prefixes.

### vLLM 3P1D smoke platform

The repository includes a same-host vLLM disaggregated-prefill setup for the
eight-B200 machine. It runs three TP=2 prefill instances and one TP=2 decode
instance with NIXL:

```bash
uv venv ../.venv-vllm --python 3.12
uv pip install \
  --python ../.venv-vllm/bin/python \
  'vllm==0.25.1' \
  'nixl==1.3.0'

scripts/vllm_pd/launch_3p1d.sh
scripts/vllm_pd/smoke_3p1d.sh
```

The default topology is P0 on GPUs 0-1 (`8100`), P1 on GPUs 2-3 (`8101`),
P2 on GPUs 4-5 (`8102`), D on GPUs 6-7 (`8200`), and the OpenAI-compatible
proxy on `http://127.0.0.1:9000/v1`. The MiniMax M2.7 NVFP4 server defaults to
its 192K context limit; override `MODEL`, `SERVED_MODEL_NAME`,
`MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, or `MAX_NUM_SEQS` when launching.
For same-host NIXL, the launcher pins `UCX_TLS=tcp,cuda_copy,cuda_ipc` and
`UCX_NET_DEVICES=lo`: TCP provides the reliable control transport NIXL needs,
while CUDA IPC carries GPU data between the local processes.

The proxy forwards the prefill response's `kv_transfer_params` to the decoder
and relays the decoder response unchanged, so trie still receives streaming
usage, token IDs, and cached-token details. Requests carrying trie's session
`cache_salt` are pinned to one prefill instance, preserving per-trace P-side
prefix-cache locality; requests without a salt use round-robin scheduling.

The smoke test sends one direct request through each P, runs the one-trace
`workloads/swe_chat_smoke.jsonl` replay, and checks that the decoder reports
positive `external_kv_transfer` tokens, successful NIXL transfers, local prefix
reuse during both the direct and replay phases, and zero NIXL
transfer/notification failures. It is a correctness smoke, not a serving
pressure benchmark. Artifacts are written under `/tmp/trie-vllm-3p1d.smoke.*`.

Stop the platform with:

```bash
scripts/vllm_pd/stop_3p1d.sh
```

## Workload format

Each JSONL row defines one trace:

- `num_turns` — number of tool-use turns
- `input_prompt_length` — initial user prompt token length
- `assistant_response_length` — per-turn assistant tokens (list of length `num_turns`)
- `tool_call_output_length` — per-turn tool result tokens (list of length `num_turns`)
- `tool_call_latency` — per-turn simulated delay in seconds (list of length `num_turns`)
- `final_assistant_response_length` — final assistant response token length after all tool-use turns

Example:

```json
{"num_turns": 2, "input_prompt_length": 32, "assistant_response_length": [16, 20], "tool_call_output_length": [8, 12], "tool_call_latency": [0.0, 0.0], "final_assistant_response_length": 64}
```

A trace produces `num_turns + 1` completion requests: one per tool-use turn plus a final turn after the last tool result.

### Event replay format

Schema version 2 stores real messages and causal events. It supports tool
waits, user think time, and context replacement after compaction or reset.

```json
{
  "schema_version": 2,
  "trace_id": "session-123",
  "initial_messages": [{"role": "user", "content": "Inspect the failure"}],
  "events": [
    {
      "type": "generate",
      "max_tokens": 64,
      "recorded_assistant": {
        "role": "assistant",
        "content": "I will inspect the logs."
      }
    },
    {"type": "wait", "seconds": 1.2, "actor": "tool"},
    {
      "type": "append_message",
      "message": {"role": "tool", "content": "...", "tool_call_id": "call-1"}
    },
    {"type": "generate", "max_tokens": 128},
    {"type": "wait", "seconds": 18.0, "actor": "user"},
    {
      "type": "append_message",
      "message": {"role": "user", "content": "Can you also fix it?"}
    },
    {"type": "generate", "max_tokens": 256}
  ]
}
```

The client renders structured messages with the target tokenizer's chat
template, then sends the rendered text to `/v1/completions`. Two context modes
are available:

- `context_source=generated` appends live model text. When the recorded turn
  contains structured tool calls, the client keeps only the live prefix matching
  the recorded text/reasoning budget and then attaches the recorded tool calls.
  The request still generates the full recorded token budget, but the next
  prompt does not count the tool-call portion twice.
- `context_source=recorded` appends the dataset's assistant output. Inputs are
  deterministic. On a trace's first pass, reuse from the preceding request ends
  where live and recorded continuations diverge; repeated identical prompts can
  still be fully reused from the server cache.

For replay requests, the client asks vLLM to return the authoritative prompt
and generated token IDs. It computes the exact LCP between the previous
request's `prompt_token_ids + token_ids` and the next request's
`prompt_token_ids`, without decoding and re-encoding the generated text.
`prefix_block_size=16` reports the block-aligned portion eligible for vLLM's
default cache. Eligibility excludes the final token of each sequence because
vLLM must recompute that token to obtain logits. The server-reported
`cached_tokens` value remains the ground truth. `cache_salt_mode=session` uses a
fresh salt for each trace invocation, preserving intra-session reuse without
leaking cache state between sessions or benchmark runs; `global` also allows
common prefixes to be shared across sessions.

Recorded waits can be controlled with `delay_scale`, `max_user_delay`, and
`max_tool_delay`. `workload_order=prefix` groups similar initial prefixes, but
should be treated only as a cache-locality upper-bound experiment.

### SWE-chat conversion

The first converter targets Claude Code raw JSONL transcripts:

```bash
uv run trie-swe-chat \
  /path/to/SWE-chat/transcripts \
  workloads/swe_chat_claude.jsonl \
  --tokenizer-model /path/to/MiniMax-M2.7-NVFP4
```

For a reproducible subset, pass a text file containing one transcript ID per
line with `--trace-ids-file workloads/swe_chat_long_batch_ids.txt`.

Conversion and cleaning are a one-time offline step. Persist the schema-v2
JSONL and replay that artifact directly; the serving client does not reopen or
repair raw SWE-chat transcripts on each benchmark run.

It groups assistant content blocks into model requests and pairs tool results
with the preceding tool calls. Tool waits use `toolUseResult.durationMs` when it
is present and otherwise fall back to timestamp gaps. When one model response
launches tools in parallel, timestamp gaps are used so their durations are not
summed as serial work. User waits use timestamp gaps. Queued user messages are
injected immediately before the next model
request, after any intervening tool results. Their enqueue time is retained as
metadata rather than treated as user think time. Automatic/meta messages such
as task notifications are preserved but timed as `other`.

Claude Code compact summaries become `ReplaceContext` events so the old context
is not retained after compaction. Traces containing microcompaction are skipped
because the exact rewritten context is not present in the log. The converter
also skips structurally ambiguous traces, including tool results without a
prior tool call and assistant request identity changes without an input
boundary. Sidechain entries are excluded from the main trace. SWE-chat does not
guarantee that hidden provider system prompts, tool schemas, or sampling
parameters are present, so this is an application-trace replay rather than a
byte-exact provider request replay.

## Example output

```
[info     ] starting benchmark             concurrency=24 duration=300.0 model=/data/models/DeepSeek-R1 num_gpus=8 workload_templates=8192
[warning  ] benchmark interrupted
[info     ] benchmark complete             completed_requests=8 failed_requests=0 wall_time_s=109.25 ...

                                                 Per-trace metrics
┏━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        ┃             ┃          ┃           ┃                    ┃                    ┃ Eligible cache hit rate ┃
┃ Metric ┃ Latency (s) ┃ TTFT (s) ┃ TTFAT (s) ┃ Decode TPS (tok/s) ┃ Cache hit rate (%) ┃                     (%) ┃
┡━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ mean   │      74.572 │    2.231 │    28.510 │              23.69 │               53.0 │                    96.6 │
│ min    │      37.151 │    2.105 │     4.666 │              19.27 │               23.3 │                    93.7 │
│ p50    │      77.915 │    2.181 │    16.585 │              20.71 │               43.9 │                    96.9 │
│ p90    │      99.497 │    2.450 │    63.775 │              30.72 │               85.7 │                    98.6 │
│ p95    │     103.044 │    2.452 │    66.554 │              32.30 │               85.8 │                    98.7 │
│ p99    │     105.882 │    2.454 │    68.778 │              33.57 │               85.9 │                    98.8 │
│ max    │     106.592 │    2.455 │    69.334 │              33.89 │               86.0 │                    98.8 │
└────────┴─────────────┴──────────┴───────────┴────────────────────┴────────────────────┴─────────────────────────┘

                                     Workload metrics
                                completed=8/8  trace/s=0.07
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                ┃  Overall ┃ Last 30s Window ┃ Steady State ┃ Steady State / GPU ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ total prompt tok/s    │ 24952.61 │        30315.66 │     30263.17 │            3782.90 │
│ cached prompt tok/s   │ 19720.40 │        25550.18 │     24628.21 │            3078.53 │
│ uncached prompt tok/s │  5232.21 │         4765.47 │      5634.96 │             704.37 │
│ completion tok/s      │   278.97 │          280.88 │       281.12 │              35.14 │
└───────────────────────┴──────────┴─────────────────┴──────────────┴────────────────────┘
```

## Metrics

### Per-trace

- `Latency (s)` — end-to-end latency from the first request of a trace to the final response.
- `TTFT (s)` — (streaming) time to the first streamed token of the first request.
- `TTFAT (s)` — (streaming) time from trace start to the first streamed token of the *final* request. The user-visible first token in an agent that hides intermediate tool turns.
- `Decode TPOT (ms/tok)` — (streaming) inverse of mean post-TTFT decode throughput across the trace's requests. Higher percentile rows are the slow tail, so p95/p99 represent worse decode cases.
- `ITL (ms)` — (streaming) client-observed inter-token latency, reported across observed token intervals.
- `Cache hit rate (%)` — server-reported `cached_prompt_tokens / prompt_tokens` over all requests in a trace.
- `Eligible cache hit rate (%)` — same numerator, but the denominator is the
  block-aligned exact token-ID LCP between consecutive requests. The first
  request has zero intra-session eligible tokens.
- `Client prefix overlap (tok)` — exact server-token-ID LCP between the
  previous actual request sequence and the next rendered prompt.
- `Block-aligned prefix (tok)` — client overlap rounded down to the configured
  cache block size.
- `Server cached prompt (tok)` — request-level cached prompt tokens reported by
  the inference server.

### Workload

- `trace/s` — completed traces per wall-clock second.
- `total prompt tok/s`, `cached prompt tok/s`, `uncached prompt tok/s` — aggregate prompt-token throughput, split by what the synthetic workload accounting expects to be cached vs. new.
- `completion tok/s` — aggregate completion-token throughput.

Each is reported under four columns:

- `Overall` — totals over the full benchmark wall time.
- `Last 30s Window` — slope of cumulative token counts over the most recent 30 seconds.
- `Steady State` — **the headline throughput metric**. Slope after dropping the first 20% of wall time as warmup. Avoids dilution from ramp-up and drain when fewer than `concurrency` traces are in flight. With `concurrency > 1` the completion curve depends on finish order, so the metric has small run-to-run variance even at fixed seed.
- `Steady State / GPU` — `Steady State / num_gpus` when `num_gpus` is provided.

Prompt-token throughputs use the synthetic workload accounting; cache-hit metrics use server-reported usage. Divergence implies a tokenizer mismatch between client and server.

## Known limitations

- Synthetic prompts are freshly random per trace, so cross-trace prefix sharing (e.g. a common system prompt or tool definition block) is not modeled and cache hit rates can be lower than in a deployment where such prefixes are shared.
- `Decode TPOT (ms/tok)` and `ITL (ms)` are client-observed streaming metrics. If a backend buffers multiple tokens into one streamed chunk, `ITL (ms)` spreads the elapsed time since the prior chunk evenly across newly observed tokens in that chunk, and intervals inside the first streamed chunk are not observable.
- The pinned `transformers==4.57.6` is required, at least for DeepSeek-R1, to match the tokenizer used by vLLM and SGLang as of April 2026. Other `transformers` versions can produce subtly different token counts and cause prompt-accounting drift.
