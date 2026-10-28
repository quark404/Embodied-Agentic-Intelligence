"""Create or execute a frozen, serial RoboDojo condition/task matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from .experiment import CONDITIONS, load_experiment_config


def load_task_manifest(path: str | Path) -> list[dict]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Task manifest requires a nonempty tasks list")
    keys = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or
                not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", row["task_id"]) or
                type(row.get("seed")) is not int or row["seed"] < 0 or
                type(row.get("eval_num")) is not int or row["eval_num"] < 1):
            raise ValueError("Each task needs a safe task_id, nonnegative seed and positive eval_num")
        key = (row["task_id"], row["seed"])
        if key in keys:
            raise ValueError("Duplicate task/seed in manifest")
        keys.add(key)
    return rows


def make_schedule(*, robodojo_root: Path, experiment_config: Path,
                  task_manifest: Path, policy_env: str, sim_env: str,
                  run_id: str, action_mode: str, conditions: tuple[str, ...],
                  policy_gpu: int = 0, sim_gpu: int = 0) -> list[dict]:
    load_experiment_config(experiment_config)
    tasks = load_task_manifest(task_manifest)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("run_id must use only letters, numbers, _ or -")
    if action_mode not in ("joint", "ee_delta"):
        raise ValueError("action_mode must be joint or ee_delta")
    if not conditions or len(set(conditions)) != len(conditions) or any(
            name not in CONDITIONS for name in conditions):
        raise ValueError("Unknown or duplicate condition")
    if policy_gpu < 0 or sim_gpu < 0:
        raise ValueError("GPU indices cannot be negative")
    runner = robodojo_root / "scripts" / "robodojo.sh"
    policy = robodojo_root / "XPolicyLab" / "policy" / "Embodied_Agentic"
    if not runner.is_file() or not (policy / "eval.sh").is_file():
        raise FileNotFoundError("Install RoboDojo and link Embodied_Agentic into XPolicyLab first")
    config_hash = hashlib.sha256(experiment_config.read_bytes()).hexdigest()
    task_hash = hashlib.sha256(task_manifest.read_bytes()).hexdigest()
    schedule = []
    for task in tasks:
        for condition in conditions:
            native_run_id = f"{run_id}_{condition}_{task['task_id']}_s{task['seed']}"
            command = ["bash", str(runner), "eval", "--policy-dir", str(policy),
                       "--task", task["task_id"], "--ckpt", condition,
                       "--env-cfg", "arx_x5", "--action-type",
                       "joint" if action_mode == "joint" else "ee",
                       "--seed", str(task["seed"]), "--eval-num", str(task["eval_num"]),
                       "--policy-gpu", str(policy_gpu), "--env-gpu", str(sim_gpu),
                       "--policy-env", policy_env, "--eval-env", sim_env]
            schedule.append({"task_id": task["task_id"], "seed": task["seed"],
                             "eval_num": task["eval_num"], "condition": condition,
                             "native_run_id": native_run_id, "action_mode": action_mode,
                             "config_sha256": config_hash, "task_manifest_sha256": task_hash,
                             "command": command})
    return schedule


def native_result_dir(root: Path, row: dict) -> Path:
    action_type = "ee" if row["action_mode"] == "ee_delta" else "joint"
    return (root / "eval_result" / "RoboDojo" / row["task_id"] /
            "Embodied_Agentic" / "arx_x5" /
            f"{row['seed']}_ckpt_name={row['condition']},action_type={action_type}" /
            row["native_run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboDojo mixed-agent matrix")
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--policy-env", required=True)
    parser.add_argument("--sim-env", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--action-mode", choices=("joint", "ee_delta"), default="ee_delta")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--sim-gpu", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="Actually launch the native simulator; default only writes schedule")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse an identical frozen schedule and skip fully scored native runs")
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    robodojo_root = args.robodojo_root.resolve(strict=True)
    rows = make_schedule(robodojo_root=robodojo_root,
        experiment_config=config, task_manifest=args.tasks.resolve(strict=True),
        policy_env=args.policy_env, sim_env=args.sim_env, run_id=args.run_id,
        action_mode=args.action_mode, conditions=tuple(args.conditions),
        policy_gpu=args.policy_gpu, sim_gpu=args.sim_gpu)
    output = args.output or Path("runs") / args.run_id / "schedule.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Schedule already exists: {output}; use --resume for identical runs")
        if json.loads(output.read_text(encoding="utf-8")) != rows:
            raise ValueError("Existing schedule differs from current config, tasks or options")
        print(f"Resuming identical {len(rows)}-run schedule at {output}")
    else:
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"Wrote {len(rows)} native runs to {output}")
    if args.execute:
        for row in rows:
            if hashlib.sha256(config.read_bytes()).hexdigest() != row["config_sha256"]:
                raise RuntimeError("Experiment config changed during the frozen matrix")
            native_dir = native_result_dir(robodojo_root, row)
            result_file = native_dir / "_result.json"
            if args.resume and result_file.is_file():
                result = json.loads(result_file.read_text(encoding="utf-8"))
                if len(result.get("details", {})) >= row["eval_num"]:
                    print(f"Skipping fully scored run {row['native_run_id']}")
                    continue
            if args.resume and native_dir.exists():
                raise RuntimeError(f"Incomplete native run needs RoboDojo's own episode resume: {native_dir}")
            env = os.environ.copy()
            env.update({"EMBODIED_EXPERIMENT_CONFIG": str(config),
                        "EMBODIED_CONDITION": row["condition"],
                        "EMBODIED_ROBODOJO_CONTROL_MODE": args.action_mode,
                        "ROBODOJO_RUN_ID": row["native_run_id"]})
            subprocess.run(row["command"], cwd=robodojo_root, env=env, check=True)


if __name__ == "__main__":
    main()
