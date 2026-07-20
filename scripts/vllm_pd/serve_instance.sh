#!/usr/bin/env bash

set -euo pipefail

role=${1:?usage: serve_instance.sh ROLE GPU_IDS HTTP_PORT SIDE_PORT ENGINE_ID}
gpu_ids=${2:?}
http_port=${3:?}
side_port=${4:?}
engine_id=${5:?}

root=${REPLAY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
model=${MODEL:-/data/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/snapshots/e79701cb1f9dce8fe5395b9ed2b20170beebecde}
served_model_name=${SERVED_MODEL_NAME:-minimax-m2.7-fp4}
max_model_len=${MAX_MODEL_LEN:-196608}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.85}
max_num_seqs=${MAX_NUM_SEQS:-8}

case "${role}" in
  producer) kv_role=kv_producer ;;
  consumer) kv_role=kv_consumer ;;
  *) echo "role must be producer or consumer" >&2; exit 2 ;;
esac

kv_transfer_config=$(printf \
  '{"kv_connector":"NixlConnector","kv_role":"%s","kv_load_failure_policy":"fail","engine_id":"%s"}' \
  "${kv_role}" "${engine_id}")

exec env \
  PATH="${root}/.venv-vllm/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  CUDA_VISIBLE_DEVICES="${gpu_ids}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  FLASHINFER_WORKSPACE_BASE="${root}/.runtime" \
  VLLM_CACHE_ROOT="${root}/.runtime/.cache/vllm" \
  TORCH_EXTENSIONS_DIR="${root}/.runtime/.cache/torch_extensions" \
  VLLM_KV_CACHE_LAYOUT=HND \
  VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
  UCX_TLS="${UCX_TLS:-tcp,cuda_copy,cuda_ipc}" \
  UCX_NET_DEVICES="${UCX_NET_DEVICES:-lo}" \
  UCX_RCACHE_MAX_UNRELEASED=1024 \
  "${root}/.venv-vllm/bin/vllm" serve "${model}" \
  --served-model-name "${served_model_name}" \
  --host 127.0.0.1 \
  --port "${http_port}" \
  --trust-remote-code \
  --generation-config vllm \
  --tensor-parallel-size 2 \
  --max-model-len "${max_model_len}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-num-seqs "${max_num_seqs}" \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --kv-transfer-config "${kv_transfer_config}"
