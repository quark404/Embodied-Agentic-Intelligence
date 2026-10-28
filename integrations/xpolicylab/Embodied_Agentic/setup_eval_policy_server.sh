#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_env=${8}
port=${9}
host=${10:-localhost}
script_dir="$(cd "$(dirname "$0")" && pwd)"
xpl_root="$(cd "$script_dir/../.." && pwd)"
bench_root="$(cd "$xpl_root/.." && pwd)"

if [[ -x "$policy_env" && ! -d "$policy_env" ]]; then
    policy_python="$policy_env"
elif [[ -x "$policy_env/bin/python" ]]; then
    policy_python="$policy_env/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$policy_env"
    policy_python=python
fi

exec env PYTHONPATH="$bench_root${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$policy_gpu_id" "$policy_python" "$xpl_root/setup_policy_server.py" \
    --config_path "$script_dir/deploy.yml" --overrides \
    port="$port" host="$host" bench_name="$bench_name" task_name="$task_name" \
    ckpt_name="$ckpt_name" env_cfg_type="$env_cfg_type" seed="$seed" \
    policy_name=Embodied_Agentic action_type="$action_type"
