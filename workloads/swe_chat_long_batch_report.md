# SWE-chat MiniMax 3P1D grouped replay verification

This is an end-to-end correctness verification of the cleaned four-trace
Claude Code workload. It is not a serving-pressure benchmark.

## Setup

- 8x NVIDIA B200: three TP=2 prefill instances on GPUs 0-5 and one TP=2
  decode instance on GPUs 6-7
- MiniMax M2.7 NVFP4 snapshot
  `e79701cb1f9dce8fe5395b9ed2b20170beebecde`, with a 196,608-token context
  limit and FP8 KV cache
- vLLM `0.25.1` and NIXL `1.3.0`
- Generated context, session cache salt, concurrency 4, natural workload order,
  one replay pass, and recorded delays disabled

The platform and replay can be reproduced with:

```bash
SESSION=trie-vllm-3p1d-tcp PROXY_PORT=9001 \
  scripts/vllm_pd/launch_3p1d.sh

NO_COLOR=1 OPENAI_API_KEY=EMPTY \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/trie \
  workload_path=workloads/swe_chat_long_batch.jsonl \
  endpoint=http://127.0.0.1:9001/v1 \
  model=minimax-m2.7-fp4 \
  tokenizer_model=/data/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/snapshots/e79701cb1f9dce8fe5395b9ed2b20170beebecde \
  concurrency=4 duration=300 stream=True delay_scale=0 \
  context_source=generated cache_salt_mode=session workload_order=natural \
  replay_once=True drain_timeout=7200 max_retries=0 num_gpus=8
```

## Workload coverage

| Trace | Requests | Waits (tool/user/other) | Tool results | Human messages | Queue enqueued/injected | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| `610152f7-58cb-4735-bd6b-c0e9bf2e785e` | 38 | 31/6/0 | 31 | 7 | 5/4 | 0 |
| `72a7f9a7-deff-4d51-9d26-38d8fa3fe93c` | 79 | 75/5/0 | 78 | 3 | 2/1 | 0 |
| `aba4a2c6-c75e-48a5-af54-ea4f168c6006` | 63 | 49/13/1 | 49 | 16 | 2/0 | 0 |
| `46f96dbb-70c0-4b03-a8ff-ba815081bc6f` | 61 | 46/15/2 | 47 | 16 | 0/0 | 1 |

Aggregate coverage is 4 traces, 241 model requests, 243 waits (201 tool,
39 user, and 3 other), 205 tool results, 42 human messages, 9 queue enqueue
operations, 5 injected queue messages, and 1 full compaction.

## Result

- The replay completed 4/4 traces and 241/241 model requests with 0 failures
  in 1,164.8 seconds.
- The decoder request-count delta was exactly 241, and the three prefill
  deltas summed to exactly 241.
- Prefill deltas were `[124, 0, 117]`. The nonzero values are whole-trace
  subset sums (`63+61` and `38+79`), verifying that session cache salts kept
  every trace on one prefill instance. With only four salts, none happened to
  map to the second prefill instance.
- Decoder prompt-token deltas were 2,590,608 local-cache hits and 130,304
  external-KV-transfer tokens, with 0 locally computed prompt tokens. Local
  decoder reuse therefore covered 95.211% of the 2,720,912 prompt tokens.
- NIXL recorded 482 transfer observations (`TP=2` x 241 requests) and
  16,762,863,616 transferred bytes (15.612 GiB), with 0 failed transfers or
  notifications.
- Trie reported a 100% eligible prefix-cache hit rate and a mean block-aligned
  reusable prefix of 10,749 tokens.
- No HTTP 500, traceback, UCX error, or NIXL error appeared in the prefill,
  decode, proxy, or replay logs.
