#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trie_repo=$(cd "${script_dir}/../.." && pwd)
root=$(cd "${trie_repo}/.." && pwd)
session=${SESSION:-trie-vllm-3p1d}
proxy_url=${PROXY_URL:-http://127.0.0.1:9000}
decoder_url=${DECODER_URL:-http://127.0.0.1:8200}
wait_timeout=${WAIT_TIMEOUT:-3600}
artifact_prefix=${ARTIFACT_PREFIX:-/tmp/${session}.smoke}
python=${root}/.venv-vllm/bin/python
model=${SERVED_MODEL_NAME:-minimax-m2.7-fp4}
tokenizer_model=${TOKENIZER_MODEL:-${MODEL:-/data/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/snapshots/e79701cb1f9dce8fe5395b9ed2b20170beebecde}}

wait_for_health() {
  local url=$1
  local deadline=$((SECONDS + wait_timeout))
  local next_update=${SECONDS}
  while ! curl -fsS "${url}/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${url}/health" >&2
      return 1
    fi
    if (( SECONDS >= next_update )); then
      echo "Waiting for ${url}/health ..."
      next_update=$((SECONDS + 30))
    fi
    sleep 5
  done
  echo "Healthy: ${url}"
}

wait_for_health http://127.0.0.1:8100
wait_for_health http://127.0.0.1:8101
wait_for_health http://127.0.0.1:8102
wait_for_health "${decoder_url}"
wait_for_health "${proxy_url}"

curl -fsS "${decoder_url}/metrics" > "${artifact_prefix}.before.metrics"
curl -fsS "${proxy_url}/status" > "${artifact_prefix}.before.status.json"

smoke_nonce=$("${python}" -c 'import time; print(time.time_ns())')
for index in 0 1 2; do
  request_json=$(SMOKE_INDEX="${index}" SMOKE_NONCE="${smoke_nonce}" SMOKE_MODEL="${model}" "${python}" -c '
import json
import os

index = os.environ["SMOKE_INDEX"]
nonce = os.environ["SMOKE_NONCE"]
prompt = (
    f"pd-smoke-{nonce} "
    + "shared-prefix " * 2048
    + f"route-{index} "
    + f"distinct-{index} " * 256
)
print(json.dumps({
    "model": os.environ["SMOKE_MODEL"],
    "prompt": prompt,
    "max_tokens": 2,
    "stream": False,
    "ignore_eos": True,
    "return_token_ids": True,
}))
')
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d "${request_json}" \
    "${proxy_url}/v1/completions" \
    > "${artifact_prefix}.direct-${index}.json"
done

curl -fsS "${proxy_url}/status" > "${artifact_prefix}.after-direct.status.json"
curl -fsS "${decoder_url}/metrics" > "${artifact_prefix}.after-direct.metrics"
"${python}" - \
  "${artifact_prefix}.before.status.json" \
  "${artifact_prefix}.after-direct.status.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    before = json.load(f)["prefill_request_counts"]
with open(sys.argv[2]) as f:
    after_status = json.load(f)
    after = after_status["prefill_request_counts"]
deltas = [new - old for old, new in zip(before, after)]
if deltas != [1, 1, 1]:
    raise SystemExit(f"expected one direct request per P, got {deltas}")
engine_ids = after_status["prefill_engine_ids"]
if engine_ids != ["trie-p0", "trie-p1", "trie-p2"]:
    raise SystemExit(f"unexpected P engine IDs: {engine_ids}")
print(f"Direct routing covered all three P instances: {engine_ids}")
PY

cd "${trie_repo}"
env \
  NO_COLOR=1 \
  OPENAI_API_KEY=EMPTY \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  "${trie_repo}/.venv/bin/trie" \
  workload_path=workloads/swe_chat_smoke.jsonl \
  endpoint="${proxy_url}/v1" \
  model="${model}" \
  tokenizer_model="${tokenizer_model}" \
  concurrency=1 \
  duration=300 \
  stream=True \
  delay_scale=0 \
  context_source=generated \
  cache_salt_mode=session \
  workload_order=natural \
  replay_once=True \
  drain_timeout=600 \
  max_retries=0 \
  num_gpus=8 \
  2>&1 | tee "${artifact_prefix}.replay.log"

"${python}" - "${artifact_prefix}.replay.log" <<'PY'
import re
import sys

ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
with open(sys.argv[1]) as f:
    summary_lines = [
        ansi_escape.sub("", line) for line in f if "benchmark complete" in line
    ]

if not summary_lines:
    raise SystemExit("replay log is missing the benchmark complete summary")

summary = summary_lines[-1]
expected = {
    "completed_requests": 1,
    "failed_requests": 0,
    "completed_model_requests": 2,
}
for field, expected_value in expected.items():
    match = re.search(rf"\b{field}\s*=\s*(\d+)\b", summary)
    if not match:
        raise SystemExit(f"replay summary is missing {field}: {summary.strip()}")
    actual_value = int(match.group(1))
    if actual_value != expected_value:
        raise SystemExit(
            f"expected {field}={expected_value}, got {actual_value}: {summary.strip()}"
        )

print("SWE-chat replay completed one two-turn trace without failures")
PY

curl -fsS "${decoder_url}/metrics" > "${artifact_prefix}.after.metrics"
curl -fsS "${proxy_url}/status" > "${artifact_prefix}.after.status.json"

"${python}" - \
  "${artifact_prefix}.after-direct.status.json" \
  "${artifact_prefix}.after.status.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    before = json.load(f)["prefill_request_counts"]
with open(sys.argv[2]) as f:
    after = json.load(f)["prefill_request_counts"]
deltas = [new - old for old, new in zip(before, after)]
if sorted(deltas) != [0, 0, 2]:
    raise SystemExit(f"one two-turn trace should stay on one P, got {deltas}")
print(f"SWE-chat trace kept P-side affinity across both turns: {deltas}")
PY

"${python}" - \
  "${artifact_prefix}.before.metrics" \
  "${artifact_prefix}.after-direct.metrics" \
  "${artifact_prefix}.after.metrics" <<'PY'
import re
import sys


def metric_sum(path: str, name: str, labels: dict[str, str] | None = None) -> float:
    total = 0.0
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            sample, value = line.rsplit(None, 1)
            metric_name = sample.split("{", 1)[0]
            if metric_name != name:
                continue
            if labels:
                match = re.search(r"\{(.*)\}", sample)
                text = match.group(1) if match else ""
                if any(f'{key}="{expected}"' not in text for key, expected in labels.items()):
                    continue
            total += float(value)
    return total


before_path, after_direct_path, after_path = sys.argv[1:]
external_name = "vllm:prompt_tokens_by_source_total"
external_labels = {"source": "external_kv_transfer"}


def metric_delta(before: str, after: str, name: str, labels=None) -> float:
    return metric_sum(after, name, labels) - metric_sum(before, name, labels)


def phase_metrics(before: str, after: str) -> dict[str, float]:
    return {
        "external": metric_delta(before, after, external_name, external_labels),
        "local": metric_delta(
            before, after, external_name, {"source": "local_cache_hit"}
        ),
        "transfers": metric_delta(
            before, after, "vllm:nixl_xfer_time_seconds_count"
        ),
        "bytes_count": metric_delta(
            before, after, "vllm:nixl_bytes_transferred_count"
        ),
        "bytes": metric_delta(before, after, "vllm:nixl_bytes_transferred_sum"),
        "external_hits": metric_delta(
            before, after, "vllm:external_prefix_cache_hits_total"
        ),
    }


direct = phase_metrics(before_path, after_direct_path)
replay = phase_metrics(after_direct_path, after_path)
failed_transfers = metric_sum(
    after_path, "vllm:nixl_num_failed_transfers_total"
) - metric_sum(before_path, "vllm:nixl_num_failed_transfers_total")
failed_notifications = metric_sum(
    after_path, "vllm:nixl_num_failed_notifications_total"
) - metric_sum(before_path, "vllm:nixl_num_failed_notifications_total")

for phase_name, phase in (("direct", direct), ("replay", replay)):
    if phase["external"] <= 0:
        raise SystemExit(
            f"D did not report {phase_name} external KV tokens: {phase['external']}"
        )
    if phase["local"] <= 0:
        raise SystemExit(
            f"D did not report {phase_name} local prefix reuse: {phase['local']}"
        )
    if phase["transfers"] <= 0:
        raise SystemExit(f"D did not report {phase_name} NIXL transfers")
    if phase["bytes_count"] <= 0 or phase["bytes"] <= 0:
        raise SystemExit(f"D did not report {phase_name} transferred NIXL bytes")
    if phase["external_hits"] <= 0:
        raise SystemExit(f"D did not report {phase_name} external prefix hits")
if failed_transfers or failed_notifications:
    raise SystemExit(
        "NIXL failures observed: "
        f"transfers={failed_transfers}, notifications={failed_notifications}"
    )

print(
    "3P1D smoke passed: "
    f"direct_external/local=+{direct['external']:.0f}/+{direct['local']:.0f}, "
    f"replay_external/local=+{replay['external']:.0f}/+{replay['local']:.0f}, "
    f"replay_transfers=+{replay['transfers']:.0f}, "
    f"replay_bytes=+{replay['bytes']:.0f}, failures=0"
)
PY

echo "Smoke artifacts: ${artifact_prefix}.*"
