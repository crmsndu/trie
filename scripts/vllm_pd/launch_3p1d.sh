#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${script_dir}/../../.." && pwd)
session=${SESSION:-trie-vllm-3p1d}
proxy_port=${PROXY_PORT:-9000}

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 1
fi

"${root}/.venv-vllm/bin/python" -c \
  'import importlib.metadata as m; assert m.version("nixl") == "1.3.0", m.version("nixl")'

launch_engine() {
  local window=$1
  local role=$2
  local gpu_ids=$3
  local http_port=$4
  local side_port=$5
  local engine_id=$6
  local log=/tmp/${session}.${window}.log
  local -a command=(
    env
    REPLAY_ROOT="${root}"
    MODEL="${MODEL:-/data/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/snapshots/e79701cb1f9dce8fe5395b9ed2b20170beebecde}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-minimax-m2.7-fp4}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-196608}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    "${script_dir}/serve_instance.sh"
    "${role}"
    "${gpu_ids}"
    "${http_port}"
    "${side_port}"
    "${engine_id}"
  )
  local shell_command
  printf -v shell_command '%q ' "${command[@]}"
  shell_command+=">$(printf '%q' "${log}") 2>&1"

  if [[ "${window}" == p0 ]]; then
    tmux new-session -d -s "${session}" -n "${window}" "${shell_command}"
    tmux set-option -t "${session}" remain-on-exit on >/dev/null
  else
    tmux new-window -d -t "${session}" -n "${window}" "${shell_command}"
  fi
}

launch_engine p0 producer 0,1 8100 5600 trie-p0
launch_engine p1 producer 2,3 8101 5601 trie-p1
launch_engine p2 producer 4,5 8102 5602 trie-p2
launch_engine d consumer 6,7 8200 5603 trie-d0

proxy_log=/tmp/${session}.proxy.log
proxy_command=(
  "${root}/.venv-vllm/bin/python"
  "${script_dir}/pd_proxy.py"
  --host 127.0.0.1
  --port "${proxy_port}"
  --prefill http://127.0.0.1:8100 http://127.0.0.1:8101 http://127.0.0.1:8102
  --decode http://127.0.0.1:8200
)
printf -v proxy_shell_command '%q ' "${proxy_command[@]}"
proxy_shell_command+=">$(printf '%q' "${proxy_log}") 2>&1"
tmux new-window -d -t "${session}" -n proxy "${proxy_shell_command}"

cat <<EOF
Started ${session} in tmux.
  P0:    GPUs 0,1  http://127.0.0.1:8100
  P1:    GPUs 2,3  http://127.0.0.1:8101
  P2:    GPUs 4,5  http://127.0.0.1:8102
  D:     GPUs 6,7  http://127.0.0.1:8200
  Proxy:           http://127.0.0.1:${proxy_port}

Run ${script_dir}/smoke_3p1d.sh after the four model servers become healthy.
Logs: /tmp/${session}.{p0,p1,p2,d,proxy}.log
EOF
