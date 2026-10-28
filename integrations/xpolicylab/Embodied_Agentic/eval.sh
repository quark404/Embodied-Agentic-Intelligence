#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_env=${9}
eval_env=${10}

mode="${EMBODIED_ROBODOJO_CONTROL_MODE:-ee_delta}"
if [[ "$bench_name" != RoboDojo || "$env_cfg_type" != arx_x5 ]] || \
   { [[ "$mode" == ee_delta && "$action_type" != ee ]] || \
     [[ "$mode" == joint && "$action_type" != joint ]] || \
     [[ "$mode" != ee_delta && "$mode" != joint ]]; }; then
    echo 'Embodied_Agentic requires RoboDojo arx_x5, with ee for ee_delta or joint for joint mode' >&2
    exit 2
fi
script_dir="$(cd "$(dirname "$0")" && pwd)"
xpl_root="$(cd "$script_dir/../.." && pwd)"
port="$(bash "$xpl_root/utils/get_free_port.sh")"

cleanup() {
    if [[ -n "${server_pid:-}" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

bash "$script_dir/setup_eval_policy_server.sh" \
    "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" \
    "$seed" "$policy_gpu_id" "$policy_env" "$port" &
server_pid=$!
bash "$xpl_root/utils/wait_for_policy_server.sh" localhost "$port" "$server_pid" 'Policy server' 1200
bash "$script_dir/setup_eval_env_client.sh" \
    "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" \
    "$seed" "$env_gpu_id" "$eval_env" "ckpt_name=$ckpt_name,action_type=$action_type" \
    "$port" localhost
