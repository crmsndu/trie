# Strict replay for SGLang P/D + MTP

## Scope

This implementation makes repeated runs follow the same recorded conversation
without changing trie's schema-v2 trace format. It aims to reproduce the system
workload closely enough for P/D, prefill CP, MLA KV-cache, and MTP experiments;
it does not claim bit-exact kernel scheduling or fully deterministic MTP
acceptance.

Use `strict_replay=True` (CLI: `strict_replay=true`). The mode:

1. accepts only schema-v2 `ReplayTrace` workloads;
2. always builds later prompts from `recorded_assistant`;
3. keeps each event's recorded `max_tokens` and sends `ignore_eos=true`;
4. uses greedy target sampling (`temperature=0`, `top_p=1`, `top_k=1`);
5. still runs the real target model, P/D transfer, cache lookup, and MTP path;
6. asks SGLang for its authoritative prompt and generated token IDs.

The trace data structure is unchanged.

## SGLang extension

SGLang's OpenAI-compatible `/v1/completions` endpoint accepts:

```json
{"return_token_ids": true}
```

For a non-streaming response, each choice includes:

```json
{
  "prompt_token_ids": [1, 2, 3],
  "token_ids": [4, 5]
}
```

For streaming responses, `prompt_token_ids` appears on the first choice chunk
and `token_ids` contains only the IDs emitted by that chunk. The request is
translated to SGLang's existing `return_prompt_token_ids` internal field, so no
new scheduler or KV-cache data structure is introduced. The fields also pass
through the P/D gateway as ordinary JSON.

Trie requires these IDs during replay. It uses the server-tokenized prompt and
the actual generated IDs to calculate exact and block-aligned reusable-prefix
lengths. Missing or inconsistent IDs fail that trace rather than silently using
a local token-count approximation.

## Why MTP is intentionally approximate

Strict mode does not inject recorded target tokens and does not force an MTP
draft/accept schedule. With greedy target sampling, repeated target outputs
should normally be stable, while MTP continues to perform natural drafting and
verification. Hardware-level nondeterminism can still change an output or an
accepted draft length.

When that happens, the current request still measures the real execution. The
next request returns to the recorded conversation, and the client computes cache
reuse against the actual preceding token IDs. This bounds model-output drift
without replacing the system path being measured.

## Expected reproducibility

Stable across repeats:

- rendered prompts after every recorded event;
- requested generation lengths;
- target sampling policy;
- trace timing inputs, subject to the configured delay scaling;
- client-side prefix accounting based on server token IDs.

Allowed to vary:

- latency and concurrent scheduling;
- P/D routing unless separately pinned;
- cache residency under contention;
- floating-point tie behavior;
- MTP proposals and accepted lengths.

For low-noise comparison, use `replay_once=true`, a fixed concurrency/arrival
schedule, session cache salts, pinned P/D routing when available, and at least
one untimed warm-up run.

## Verification

- Trie unit suite: `uv run --extra test pytest -q tests`
- SGLang completion unit:
  `PYTHONPATH=sglang/python:sglang <python> sglang/test/registered/unit/entrypoints/openai/test_serving_completions.py -v`

An end-to-end run should verify:

1. every replay request returns prompt and output IDs;
2. completion length equals the event's recorded `max_tokens`;
3. repeat runs have identical request counts and prompt-token counts;
4. any remaining output/cache/latency variance is reported rather than hidden.

## 2026-07-23 B200 smoke result

The implementation was exercised on a two-node 1P1D deployment:

- prefill: `b200-dev-2`, TP8, Mooncake RDMA;
- decode: `b200-dev`, TP8;
- target and draft MoE: FlashInfer TRT-LLM;
- MTP: EAGLE, 3 steps, top-k 1, 4 draft tokens;
- workload: `swe_chat_smoke.jsonl`, one schema-v2 trace and two generate
  events (`max_tokens` 80 and 99).

The router's OpenAI completion response returned 6 authoritative prompt IDs and
4 output IDs for a 6+4 token probe; both lengths matched `usage`.

Two independent strict replay runs both completed 1/1 traces, 2/2 model
requests, with zero failures. Both reported 113 total prompt tokens, 179
completion tokens, the same client prefix lengths (0 and 31), and the same
block-aligned prefix lengths (0 and 16). End-to-end trace latency differed
(4.848 s versus 2.609 s), as expected after warm-up.

Server logs confirmed real P/D transfers for all four replay requests (31 and
82 prompt tokens in each repeat) and no transfer failures. MTP remained natural:
observed decode-batch acceptance varied (for example, accept length 3.15/rate
0.72 and 2.90/0.63), which is within this mode's documented tolerance.
