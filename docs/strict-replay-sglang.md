# Strict replay for SGLang P/D + MTP

## Scope

Strict replay makes repeated benchmark runs commit the same output token stream
to SGLang's scheduler, KV cache, and subsequent prompts. It is intended for
controlled P/D, prefill CP, MLA KV-cache, DSA, and MTP comparisons.

Use `strict_replay=true`. Before the benchmark timer starts, the client:

1. accepts only schema-v2 `ReplayTrace` workloads;
2. builds every later prompt from `recorded_assistant`;
3. resolves one fixed output-token sequence for every generate event;
4. sends that sequence as `forced_output_token_ids`;
5. uses greedy target sampling (`temperature=0`, `top_p=1`, `top_k=1`);
6. verifies that SGLang returned exactly the requested token IDs.

SGLang still executes the target model, scheduler, P/D transfer, cache lookup,
DSA path, and MTP draft model. Strict replay controls which generated tokens
are committed; it does not bypass those system paths.

## Trace format

Schema v2 generate events can store the original output IDs:

```json
{
  "type": "generate",
  "max_tokens": 3,
  "recorded_assistant": {
    "role": "assistant",
    "content": "Done."
  },
  "recorded_output_token_ids": [123, 456, 2]
}
```

`recorded_output_token_ids` is optional for compatibility, but when present it
must contain exactly `max_tokens` non-negative integer IDs. New trace collectors
should populate it: this is the only way to reproduce the original generation
bit-for-bit without depending on reversible text parsing.

For an old text-only trace, Trie deterministically constructs an equal-length
canonical sequence before timing starts. It tokenizes the recorded assistant
continuation, truncates it if necessary, and otherwise pads it to the recorded
`max_tokens`. The final missing token is inferred as EOS or, for a tool turn,
from the separator at the start of the next recorded prompt. Trie logs how many
events required this ambiguous reconstruction.

That fallback gives every compared run the same token IDs and generation
lengths, but it cannot recover IDs that were discarded by the old collector.
It is therefore suitable for stable A/B system measurements, not for claiming
bit-exact reproduction of the original collection run.

## SGLang extension

SGLang's OpenAI-compatible `/v1/completions` endpoint accepts:

```json
{
  "forced_output_token_ids": [123, 456, 2],
  "return_token_ids": true
}
```

`forced_output_token_ids` sets `max_new_tokens` to the sequence length and
disables early EOS termination. For a non-streaming response, each choice
includes:

```json
{
  "prompt_token_ids": [1, 2, 3],
  "token_ids": [123, 456, 2]
}
```

For streaming responses, `prompt_token_ids` appears on the first choice chunk
and `token_ids` contains only the IDs emitted by that chunk. These fields pass
through the P/D gateway as ordinary JSON.

Trie uses the authoritative IDs to calculate the exact and block-aligned
reusable prefix. A missing, malformed, or mismatched sequence fails the trace
instead of silently using a local token-count approximation.

## MTP semantics

For greedy strict replay, SGLang computes target logits and the natural greedy
target prediction as usual. Immediately before tree verification, it substitutes
the trace token at each active forced-output position:

- a draft token equal to the trace token is accepted;
- the first unequal draft token rejects that draft suffix;
- the trace token is emitted through the verifier's bonus-token path.

The normal strict replay path calls the tree verifier once per decode step.
This preserves meaningful MTP hit/miss work while guaranteeing that the tokens
committed to the request and KV cache match the trace. MTP acceptance rates may
differ from the source collection run because the draft model and runtime are
still live.

## Expected reproducibility

Stable across repeats:

- rendered prompts after every recorded event;
- committed output token IDs and generation lengths;
- exact and block-aligned client prefix accounting;
- request count and causal trace structure;
- trace timing inputs, subject to configured delay scaling.

Allowed to vary:

- latency and concurrent scheduling;
- P/D routing unless separately pinned;
- cache residency under contention;
- kernel timing and floating-point behavior;
- MTP proposals and accepted lengths.

For low-noise comparison, use `replay_once=true`, a fixed concurrency/arrival
schedule, session cache salts, pinned P/D routing when available, and at least
one untimed warm-up run.

## Verification

- Trie: `uv run --extra test pytest -q tests`
- SGLang: run the forced replay, sampling, and completion unit suites.

An end-to-end run should verify:

1. every request returns prompt and output IDs;
2. every output-ID sequence exactly matches the prepared trace sequence;
3. completion length equals the event's recorded `max_tokens`;
4. repeat runs have identical request counts, prompts, outputs, and prefix
   lengths;
5. latency, routing, cache-residency, and MTP-acceptance variance remains visible.

## 2026-07-27 B200 smoke result

The current implementation was exercised with GLM-5.2-FP8 on one eight-GPU B200
node using TP8, attention CP8, EP8, FP8 KV cache, DSA prefill/decode kernels,
DSA CP shared KV, full decode CUDA graphs, and EAGLE MTP3.

A two-turn strict replay trace was executed twice. Both runs returned exactly
the forced token IDs at lengths 164 and 114. The second turn reported 832 cached
prompt tokens in both runs, with the same 896-token client block-aligned prefix.
End-to-end trace latency was 4.441 s and 4.380 s. Server logs showed live MTP
acceptance and no replay mismatch.
