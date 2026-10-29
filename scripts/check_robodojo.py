"""Read-only deployment inventory. Does not reserve GPUs or launch Isaac Sim."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=20)
        return {"ok": result.returncode == 0, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(), "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}


def classify_gpu(name: str) -> str:
    if re.search(r"\b(?:H20|H100|H200|H800|A100)\b", name, re.IGNORECASE):
        return "unsupported_no_rt_cores"
    return "requires_isaac_compatibility_test"


def readiness_summary(blockers: list[str]) -> dict:
    return {"metadata_preflight_passed": not blockers,
            "native_ready": False, "blockers": list(blockers),
            "remaining_verification": ["Explicit resource allocation", "Native Isaac camera smoke",
                                       "Native RoboDojo episode and scoring"]}


def check_sources(root: Path, lock: dict) -> dict:
    checks = {}
    for relative, key in ((".", "robodojo"), ("XPolicyLab", "xpolicylab")):
        result = command(["git", "-C", str(root / relative), "rev-parse", "HEAD"])
        revision = result.get("stdout") if result["ok"] else None
        checks[key] = {"expected": lock[key]["revision"], "actual": revision,
                       "ok": revision == lock[key]["revision"]}
    expected = PROJECT_ROOT / "integrations/xpolicylab/Embodied_Agentic"
    policy = root / "XPolicyLab/policy/Embodied_Agentic"
    checks["policy_link"] = {"ok": policy.is_symlink() and policy.resolve() == expected,
                             "path": str(policy)}
    for relative in ("scripts/robodojo.sh", "src/eval_client/ik_continuity.py"):
        checks[relative] = {"ok": (root / relative).is_file()}
    return {"ok": all(item["ok"] for item in checks.values()), "checks": checks}


def check_endpoint(base_url: str, model: str) -> dict:
    parsed = urlparse(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        return {"ok": False, "error": "Use a loopback HTTP endpoint or an SSH tunnel"}
    try:
        headers = {}
        if os.environ.get("MODEL_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["MODEL_API_KEY"]
        request = Request(base_url.rstrip("/") + "/models", headers=headers)
        with urlopen(request, timeout=10) as response:
            data = json.load(response)
        models = [item["id"] for item in data["data"]]
        return {"ok": model in models, "models": models,
                "inference_requested": False}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"ok": False, "error": str(error)}


def check_sim_python(python: Path) -> dict:
    probe = (
        "import importlib.metadata as metadata, json, sys; "
        "packages={item.metadata['Name'].lower():item.version "
        "for item in metadata.distributions()}; "
        "print(json.dumps({'python':list(sys.version_info[:3]),"
        "'packages':{name:packages.get(name) for name in "
        "['isaacsim','isaaclab','torch','numpy','pillow','websockets','pydantic']}}))"
    )
    result = command([str(python), "-I", "-B", "-c", probe])
    if not result["ok"]:
        return result
    try:
        data = json.loads(result["stdout"])
        packages = data["packages"]
        websocket_major = int((packages.get("websockets") or "0").split(".")[0])
        data["ok"] = (data["python"][:2] == [3, 11]
                      and packages.get("isaacsim") == "5.1.0"
                      and all(packages.get(name) for name in
                              ("isaaclab", "torch", "numpy", "pillow", "pydantic"))
                      and websocket_major >= 14)
        data["note"] = "Metadata only, not an Isaac import/render test; websocket 12 is incompatible with this XPolicyLab"
        return data
    except (ValueError, KeyError, TypeError) as error:
        return {"ok": False, "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path)
    parser.add_argument("--sim-gpu", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--output", type=Path, default=Path("runs/robodojo-preflight/report.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path to retain previous evidence")
    lock = json.loads((PROJECT_ROOT / "configs/robodojo_versions.json").read_text())
    gpu_query = command(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                         "--format=csv,noheader,nounits"])
    gpus = []
    if gpu_query["ok"]:
        for line in gpu_query["stdout"].splitlines():
            index, name, used, total = (value.strip() for value in line.split(","))
            gpus.append({"index": int(index), "name": name, "memory_used_mib": int(used),
                         "memory_total_mib": int(total), "isaac_support": classify_gpu(name)})
    loader = ctypes.util.find_library("vulkan")
    try:
        ctypes.CDLL(loader or "libvulkan.so.1")
        vulkan = {"loader_found": True, "render_tested": False}
    except OSError as error:
        vulkan = {"loader_found": False, "render_tested": False, "error": str(error)}
    source = check_sources(args.robodojo_root, lock)
    endpoint = check_endpoint(args.base_url, args.model)
    sim_python = check_sim_python(args.sim_python) if args.sim_python else {
        "ok": False, "error": "No isolated simulator Python provided"}
    assets = {name: (args.robodojo_root / "Assets" / name).is_dir()
              for name in ("Robots", "Object", "Material", "Eval_Layout")}
    blockers = []
    if not source["ok"]:
        blockers.append("Pinned source/policy link check failed")
    if not endpoint["ok"]:
        blockers.append("Qwen model endpoint unavailable or wrong model")
    if not sim_python["ok"]:
        blockers.append("Simulator environment absent or metadata requirements unmet")
    if not all(assets.values()):
        blockers.append("Assets incomplete; directory checks alone do not verify asset hashes")
    selected = [gpu for gpu in gpus if gpu["index"] == args.sim_gpu]
    if not selected:
        blockers.append("No explicit simulator GPU selected/found; inventory is not allocation")
    elif selected[0]["isaac_support"] == "unsupported_no_rt_cores":
        blockers.append("Selected GPU lacks RT cores; do not install Isaac Sim on this GPU host")
    elif selected[0]["memory_used_mib"] > 1024:
        blockers.append("Selected GPU is occupied; do not stop or share another job")
    if gpus and all(gpu["isaac_support"] == "unsupported_no_rt_cores" for gpu in gpus):
        blockers.append("All enumerated GPUs lack supported RT hardware; use a separate RTX simulator host")
    if not vulkan["loader_found"]:
        blockers.append("Vulkan loader unavailable; presence of an ICD JSON is insufficient")
    disk = shutil.disk_usage(PROJECT_ROOT)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "os": platform.platform(), "sources": source, "qwen_endpoint": endpoint,
              "gpu_inventory": gpus, "gpu_query": gpu_query, "vulkan": vulkan,
              "sim_python": sim_python, "asset_directories": assets,
              "workspace_free_bytes": disk.free, "asset_payload_bytes": lock["assets"]["total_bytes"],
              **readiness_summary(blockers),
              "side_effects": "Only this JSON report; no install, GPU allocation, or simulator launch"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(2 if blockers else 0)


if __name__ == "__main__":
    main()
