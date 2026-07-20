# SWE-chat long-context MiniMax workload

Tokenizer: `/data/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/snapshots/e79701cb1f9dce8fe5395b9ed2b20170beebecde`

All four traces pass the current strict Claude Code converter. The context figures below are the exact MiniMax chat-template token counts using recorded assistant context, including each request's recorded output budget.

| Trace ID | Requests | Max prompt+output | Prompt p50 / p90 | Recorded output tokens | Compactions | Tool results | Human user messages | Queue human arrivals / injected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2026-01-19-c2b51eb0-d0e9-43cf-b431-42c05d49450b` | 73 | 64,084 | 27,900 / 61,990 | 30,913 | 0 | 52 | 30 | 0 / 0 |
| `2026-01-29-b88df712-e4c6-40ac-8217-80952475f370` | 62 | 67,306 | 37,709 / 63,260 | 31,076 | 0 | 69 | 4 | 0 / 0 |
| `2292db0c-9045-4f70-949f-1869f36ae08c` | 136 | 109,485 | 58,188 / 77,369 | 59,202 | 2 (before requests 42, 110) | 135 | 26 | 12 / 11 |
| `9b76417d-a646-4737-a9ec-8aa94abe0a1c` | 96 | 94,931 | 40,515 / 80,701 | 45,764 | 1 (before request 78) | 89 | 9 | 1 / 5 |

Aggregate:

- 4 traces, 367 model requests
- 16,625,269 prompt tokens across recorded-context requests
- 166,955 recorded output tokens
- 3 compactions, 345 tool results, 69 human user messages
- 13 human queue arrivals and 16 injected queue messages (injected count also includes automatic queued messages)
- 3 interactive-tool waits classified as user time: 2 `AskUserQuestion` and 1 `ExitPlanMode`
- Maximum single request context: 109,485 tokens

Timing and tail cleanup:

- `AskUserQuestion` and `ExitPlanMode` wait for operator input, so their result waits use `actor=user` rather than `actor=tool`. In this batch, `2026-01-29-b88df712-e4c6-40ac-8217-80952475f370` contains one of each, and `9b76417d-a646-4737-a9ec-8aa94abe0a1c` contains one `AskUserQuestion` wait.
- Events after the final `Generate` are omitted because no later serving request consumes them. Only `2292db0c-9045-4f70-949f-1869f36ae08c` required this cleanup, with 3 trailing events removed; the other three traces removed 0.

One-time cleaned artifacts:

- Manifest: `workloads/swe_chat_long_context_batch_ids.txt`
- Converted schema-v2 workload: `workloads/swe_chat_long_context_batch.jsonl`

Validation:

- Converter result: 4 converted, 0 skipped
- Workload loader result: all 4 trace IDs load successfully
- Every converted trace ends with a `Generate` event
- Every selected trace is below MiniMax M2.7's 196,608-token model limit

Additional traces considered for this first four-trace batch:

- `4949fdf7-f610-4d6f-8910-6f41aff09dc6` (max 136,204) and `815083b0-f629-4a24-90d4-47187cb61197` (max 132,660) are valid longer candidates for scale runs; they were omitted only to keep this first batch at four diverse traces.
- `ad3ade73-25df-457e-a41d-262c2cb4a3bf` reaches 167,087 tokens but has low-frequency orphan/empty queue metadata, so it is left out of the cleaned batch.
- `8bf95c26-d9d3-4f78-990c-9784125b69b8` has 32 orphan queue operations and is left out for the same data-quality reason.
- `46f96dbb-70c0-4b03-a8ff-ba815081bc6f` is already represented in `swe_chat_long_batch.jsonl`.
- Other valid profiled traces were omitted only to keep the first workload small, not because conversion failed or their contexts were too long.
