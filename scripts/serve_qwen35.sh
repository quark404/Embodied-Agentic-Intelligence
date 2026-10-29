#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_ENV="${QWEN_ENV:-$HOME/miniconda3/envs/qwen35-vllm}"
export PATH="$QWEN_ENV/bin:$PATH"
QWEN_GPU="${QWEN_GPU:?Set QWEN_GPU to a GPU allocated to you; no default GPU is selected.}"
QWEN_PORT="${QWEN_PORT:-18000}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/embodied-agentic/vllm}"
export TMPDIR="${TMPDIR:-/tmp}"

if [[ ! -x "$QWEN_ENV/bin/vllm" ]]; then
    echo "Missing vLLM environment: $QWEN_ENV" >&2
    exit 1
fi
if [[ ! "$QWEN_GPU" =~ ^[0-9]+$ ]]; then
    echo "This single-GPU preset requires one numeric QWEN_GPU index." >&2
    exit 1
fi
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    case ",${CUDA_VISIBLE_DEVICES}," in
        *,"${QWEN_GPU}",*) ;;
        *) echo "QWEN_GPU is outside the inherited CUDA_VISIBLE_DEVICES allocation; refusing to override it." >&2
           exit 1 ;;
    esac
fi
used_mb="$(nvidia-smi -i "$QWEN_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)"
if [[ ! "$used_mb" =~ ^[0-9]+$ ]]; then
    echo "Could not verify GPU memory usage; refusing to launch." >&2
    exit 1
fi
if (( used_mb > 1024 )); then
    echo "GPU $QWEN_GPU is already using ${used_mb} MiB; refusing to share or stop another job." >&2
    exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$QWEN_GPU"

cd "$PROJECT_ROOT"
exec "$QWEN_ENV/bin/vllm" serve Qwen/Qwen3.5-27B \
    --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
    --served-model-name Qwen/Qwen3.5-27B \
    --download-dir "$HF_HUB_CACHE" \
    --host 127.0.0.1 --port "$QWEN_PORT" \
    --dtype bfloat16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --limit-mm-per-prompt '{"image":12,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1048576}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --generation-config vllm \
    --enforce-eager
