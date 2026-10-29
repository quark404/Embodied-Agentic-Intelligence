"""Cluster inventory only: no installation, rendering, model load or training."""

from __future__ import annotations

import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def check_context() -> Path:
    if os.environ.get("EAI_CLUSTER_CONFIRMED") != "1":
        raise ValueError("EAI_CLUSTER_CONFIRMED=1 is required; offline previews cannot run")
    working = os.environ.get("WORKING_PATH")
    if not working or Path(working).resolve() != ROOT:
        raise ValueError("Must run inside the uploaded WORKING_PATH")
    result = Path(os.environ.get("EAI_RESULT_DIR", "")).resolve()
    if not result.is_relative_to(Path("/jfs-public")) or not os.path.ismount("/jfs-public"):
        raise ValueError("Real JFS output is required; no local fallback")
    origin_bytes = (result / "origin.json").read_bytes()
    if hashlib.sha256(origin_bytes).hexdigest() != os.environ.get("EAI_ORIGIN_SHA256"):
        raise ValueError("JFS origin digest differs from the development-host marker")
    origin = json.loads(origin_bytes)
    if origin.get("run_id") != os.environ.get("EAI_RUN_ID") or origin.get("writer") != "development_host":
        raise ValueError("JFS origin does not belong to this run")
    return result


def run_command(arguments: list[str]) -> dict:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "error": type(error).__name__}


def inspect_gpus(output: str) -> list[dict]:
    devices = []
    for row in output.splitlines():
        index, name, driver, memory = (value.strip() for value in row.split(","))
        devices.append({"index": int(index), "name": name, "driver": driver,
                        "memory_mib": int(memory)})
    return devices


def library_status(name: str) -> dict:
    path = ctypes.util.find_library(name)
    candidates = [path] if path else []
    if name == "vulkan":
        candidates.append("libvulkan.so.1")
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate)
            return {"loadable": True, "path": candidate}
        except OSError:
            continue
    return {"loadable": False}


def main() -> None:
    result_dir = check_context()
    output_path = result_dir / "inventory.json"
    if output_path.exists():
        raise FileExistsError("This run already has an inventory; refusing duplicate execution")
    query = run_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                         "--format=csv,noheader,nounits"])
    devices = inspect_gpus(query["stdout"]) if query["returncode"] == 0 else []
    topology_ok = len(devices) == 1 and "RTX 5090" in devices[0]["name"]
    packages = {}
    for name in ("isaacsim", "isaaclab", "torch", "websockets", "numpy", "pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "run_id": os.environ["EAI_RUN_ID"], "host": platform.node(),
              "os": platform.platform(), "python": platform.python_version(),
              "gpu_query": query, "gpus": devices, "single_5090_visible": topology_ok,
              "libraries": {name: library_status(name) for name in ("vulkan", "GLX_nvidia", "EGL")},
              "vulkaninfo_available": shutil.which("vulkaninfo") is not None,
              "driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
              "packages": packages, "jfs_origin_verified": True,
              "jfs_free_bytes": shutil.disk_usage(result_dir).free,
              "inventory_passed": topology_ok, "native_evaluation_ready": False,
              "not_tested": ["Vulkan rendering", "Isaac camera frames", "Qwen endpoint connectivity",
                             "Benchmark assets", "Native RoboDojo episode"],
              "side_effects": "Only inventory JSON; no install, model load, GPU computation, or robot action"}
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"inventory_passed": topology_ok, "output": str(output_path),
                      "native_evaluation_ready": False}))
    if not topology_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
