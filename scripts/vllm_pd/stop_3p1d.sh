#!/usr/bin/env bash

set -euo pipefail

session=${SESSION:-trie-vllm-3p1d}

if tmux has-session -t "${session}" 2>/dev/null; then
  mapfile -t pane_pids < <(
    tmux list-panes -t "${session}" -F '#{pane_pid}'
  )
  for pid in "${pane_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done

  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    running=false
    for pid in "${pane_pids[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        running=true
        break
      fi
    done
    if [[ "${running}" == false ]]; then
      break
    fi
    sleep 1
  done

  for pid in "${pane_pids[@]}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || true
  done
  tmux kill-session -t "${session}"
  echo "Stopped ${session}. Logs remain in /tmp/${session}.*.log"
else
  echo "No tmux session named ${session}."
fi
