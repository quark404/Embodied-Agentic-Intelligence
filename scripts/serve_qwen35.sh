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
if [[ ! "$QWEN_GPU" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))?$ ]]; then
    echo "QWEN_GPU requires one or two explicit numeric GPU indices." >&2
    exit 1
fi
IFS=',' read -r -a selected_gpus <<< "$QWEN_GPU"
if (( ${#selected_gpus[@]} == 2 )) && [[ "${selected_gpus[0]}" == "${selected_gpus[1]}" ]]; then
    echo 'QWEN_GPU contains a duplicate GPU; refusing to launch.' >&2
    exit 1
fi
if (( ${#selected_gpus[@]} == 2 )); then
    memory_utilization=0.90
    memory_percent=90
else
    memory_utilization=0.85
    memory_percent=85
fi
for selected_gpu in "${selected_gpus[@]}"; do
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        case ",${CUDA_VISIBLE_DEVICES}," in
            *,"${selected_gpu}",*) ;;
            *) echo "QWEN_GPU is outside the inherited CUDA_VISIBLE_DEVICES allocation; refusing to override it." >&2
               exit 1 ;;
        esac
    fi
    used_mb="$(nvidia-smi -i "$selected_gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if [[ ! "$used_mb" =~ ^[0-9]+$ ]]; then
        echo "Could not verify GPU memory usage; refusing to launch." >&2
        exit 1
    fi
    if (( used_mb > 1024 )); then
        echo "GPU $selected_gpu is already using ${used_mb} MiB; refusing to share or stop another job." >&2
        exit 1
    fi
    total_mb="$(nvidia-smi -i "$selected_gpu" --query-gpu=memory.total --format=csv,noheader,nounits)"
    if [[ ! "$total_mb" =~ ^[0-9]+$ ]]; then
        echo "Could not verify GPU memory capacity; refusing to launch." >&2
        exit 1
    fi
    if (( total_mb * memory_percent / 100 < 54000 / ${#selected_gpus[@]} + 1024 )); then
        echo "GPU $selected_gpu has insufficient memory for this BF16 Qwen3.5-27B topology; no automatic quantization." >&2
        exit 1
    fi
done
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-2}"
if [[ ! "$QWEN_MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ || ! "$QWEN_MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Qwen context and sequence limits must be positive integers.' >&2
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
    --dtype bfloat16 --tensor-parallel-size "${#selected_gpus[@]}" \
    --gpu-memory-utilization "$memory_utilization" \
    --max-model-len "$QWEN_MAX_MODEL_LEN" --max-num-seqs "$QWEN_MAX_NUM_SEQS" \
    --max-num-batched-tokens 4096 \
    --limit-mm-per-prompt '{"image":12,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1048576}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --generation-config vllm \
    --enforce-eager
