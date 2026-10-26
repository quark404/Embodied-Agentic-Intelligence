import argparse
import json

from .core import CabinetWorld, Task, run_episode
from .memory import MemoryBank
from .policies import DirectPolicy, PlannerMemoryPolicy, ToolPolicy


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic embodied capability smoke test")
    parser.add_argument("--output", default="", help="Optional JSON results path")
    args = parser.parse_args()
    bank = MemoryBank()
    train = Task("train-1", "train", "cabinet-blue", "cup", "table", "4172")
    heldout = Task("test-1", "test", "cabinet-blue", "mug", "counter", "4172")
    training = run_episode(CabinetWorld(), ToolPolicy(), train)
    if training["success"]:
        observed_code = next(item["receipt"]["code"] for item in training["trace"]
                             if item["receipt"]["action"] == "lookup_code")
        bank.add(task_id=train.task_id, split=train.split, cabinet=train.cabinet,
                 code=observed_code, success=True)
    results = {
        "protocol": "synthetic-smoke-v1",
        "train_episode": training,
        "heldout": {
            name: run_episode(CabinetWorld(), policy, heldout)
            for name, policy in (
                ("direct", DirectPolicy()),
                ("tool", ToolPolicy()),
                ("planner_memory", PlannerMemoryPolicy(bank)),
            )
        },
    }
    report = json.dumps(results, indent=2)
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n")
    print(json.dumps({name: {key: row[key] for key in ("success", "steps", "tool_calls")}
                      for name, row in results["heldout"].items()}, indent=2))


if __name__ == "__main__":
    main()
