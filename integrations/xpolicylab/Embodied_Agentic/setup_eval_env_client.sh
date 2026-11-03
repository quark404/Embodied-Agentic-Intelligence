#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env=${8}
additional_info=${9}
port=${10}
host=${11:-localhost}
script_dir="$(cd "$(dirname "$0")" && pwd)"
xpl_root="$(cd "$script_dir/../.." && pwd)"
bench_root="$(cd "$xpl_root/.." && pwd)"

sim_bin=""
if [[ -x "$eval_env/bin/python" && -x "$eval_env/bin/python3" ]]; then
    sim_bin="$(cd "$eval_env/bin" && pwd)"
elif [[ -x "$eval_env" && ! -d "$eval_env" ]]; then
    sim_bin="$(cd "$(dirname "$eval_env")" && pwd)"
    if [[ ! -x "$sim_bin/python" || ! -x "$sim_bin/python3" ]]; then
        echo 'The explicit simulator environment must provide both python and python3.' >&2
        exit 2
    fi
elif [[ "$eval_env" == */* ]]; then
    echo 'The explicit simulator environment does not exist; refusing Conda fallback.' >&2
    exit 2
fi

if [[ -n "$sim_bin" ]]; then
    if [[ "$bench_name" != RoboDojo || "${EVAL_ENV_TYPE:-sim}" != sim ]]; then
        echo 'The isolated environment entry supports native RoboDojo simulation only.' >&2
        exit 2
    fi
    "$sim_bin/python3" -c 'import sys,yaml; data=yaml.safe_load(open(sys.argv[1])); assert data.get("eval_batch") is False and data.get("protocol") == "ws"' "$script_dir/deploy.yml"
    exec env PATH="$sim_bin:$PATH" PYTHONNOUSERSITE=1 \
        ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-0}" \
        bash "$bench_root/scripts/eval_policy.sh" \
        --bench_name "$bench_name" --task_name "$task_name" \
        --env_cfg_type "$env_cfg_type" --policy_name Embodied_Agentic \
        --host "$host" --port "$port" --protocol ws --eval_batch false \
        --root_dir "$bench_root" --device_id "$env_gpu_id" \
        --additional_info "$additional_info" --seed "$seed"
fi

exec bash "$xpl_root/utils/setup_env_client.sh" \
    "$xpl_root/utils" "$script_dir/deploy.yml" "$eval_env" "$port" \
    "$bench_name" "$task_name" "$env_cfg_type" Embodied_Agentic \
    "$additional_info" "$bench_root" "$seed" "$env_gpu_id" "$host"
