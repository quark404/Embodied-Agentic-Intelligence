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
import sys

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from cluster_eval.storage import create_local_results, export_inventory, network_inventory
from cluster_eval.resource_probe import resource_inventory
from cluster_eval.image_inventory import inspect_image_runtime


def check_execution_context() -> None:
    if os.environ.get("EAI_CLUSTER_CONFIRMED") != "1":
        raise ValueError("EAI_CLUSTER_CONFIRMED=1 is required; offline previews cannot run")
    working = os.environ.get("WORKING_PATH")
    if not working or Path(working).resolve() != ROOT:
        raise ValueError("Must run inside the uploaded WORKING_PATH")


def check_context() -> Path:
    check_execution_context()
    mode = os.environ.get("EAI_STORAGE_MODE", "jfs")
    if mode == "job_local":
        return create_local_results(ROOT, Path(os.environ.get("EAI_RESULT_DIR", "")),
                                    os.environ.get("EAI_RUN_ID", ""),
                                    os.environ.get("EAI_ORIGIN_SHA256", ""))
    if mode != "jfs":
        raise ValueError("Unrecognized explicit storage mode")
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


def inspect_mounts(mountinfo: str, result: Path) -> list[dict]:
    mounts = []
    for line in mountinfo.splitlines():
        sections = line.split(" - ", 1)
        if len(sections) != 2:
            continue
        fields, filesystem = sections[0].split(), sections[1].split()
        if len(fields) < 6 or not filesystem:
            continue
        mountpoint = Path(fields[4])
        if not mountpoint.is_absolute():
            continue
        if result.is_relative_to(mountpoint) or mountpoint.is_relative_to(Path("/jfs-public")):
            mounts.append({"mountpoint": str(mountpoint), "filesystem": filesystem[0]})
    return mounts


def context_diagnostics() -> dict:
    check_execution_context()
    result = Path(os.environ.get("EAI_RESULT_DIR", "")).resolve()
    under_jfs = result.is_relative_to(Path("/jfs-public"))
    report = {"jfs_root_exists": Path("/jfs-public").is_dir(),
              "jfs_root_is_mount": os.path.ismount("/jfs-public"),
              "result_under_jfs": under_jfs,
              "inventory_passed": False, "native_evaluation_ready": False}
    report.update(storage_mode=os.environ.get('EAI_STORAGE_MODE', 'jfs'),
                  requested_result_path=os.environ.get('EAI_RESULT_DIR'),
                  resolved_result_path=str(result),
                  job_data_is_symlink=Path('/job_data').is_symlink(),
                  job_data_resolved=str(Path('/job_data').resolve()),
                  platform_data_root_is_mount=os.path.ismount('/running_root'))
    if under_jfs:
        report["result_directory_exists"] = result.is_dir()
        origin_path = result / "origin.json"
        report["origin_regular_file"] = origin_path.is_file() and not origin_path.is_symlink()
        if report["origin_regular_file"]:
            try:
                origin_bytes = origin_path.read_bytes()
                origin = json.loads(origin_bytes)
                report["origin_sha256_matches"] = (hashlib.sha256(origin_bytes).hexdigest()
                                                    == os.environ.get("EAI_ORIGIN_SHA256"))
                report["origin_identity_matches"] = (origin.get("run_id") == os.environ.get("EAI_RUN_ID")
                                                      and origin.get("writer") == "development_host")
            except (OSError, ValueError, AttributeError) as error:
                report["origin_error_type"] = type(error).__name__
    try:
        report["relevant_mounts"] = inspect_mounts(Path("/proc/self/mountinfo").read_text(), result)
    except OSError as error:
        report["mountinfo_error_type"] = type(error).__name__
    report["gpu_query"] = run_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                       "--format=csv,noheader,nounits"])
    report["libraries"] = {name: library_status(name) for name in ("vulkan", "GLX_nvidia", "EGL")}
    return report


def main(*, check_network: bool = True) -> None:
    check_execution_context()
    requested_gpus = os.environ.get("EAI_EXPECTED_GPUS", "1")
    if requested_gpus not in ("1", "3"):
        raise ValueError("Only reviewed one- or three-GPU topologies are allowed")
    expected_gpus = int(requested_gpus)
    try:
        result_dir = check_context()
    except (OSError, ValueError) as error:
        try:
            diagnostics = context_diagnostics()
        except (OSError, ValueError) as diagnostic_error:
            diagnostics = {"error_type": type(diagnostic_error).__name__}
        print(json.dumps({"event": "preflight_context_failed", "error_type": type(error).__name__,
                          "inventory_passed": False, "native_evaluation_ready": False,
                          "diagnostics": diagnostics}), flush=True)
        raise
    output_path = result_dir / "inventory.json"
    if output_path.exists():
        raise FileExistsError("This run already has an inventory; refusing duplicate execution")
    query = run_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                         "--format=csv,noheader,nounits"])
    devices = inspect_gpus(query["stdout"]) if query["returncode"] == 0 else []
    topology_ok = len(devices) == expected_gpus and all("RTX 5090" in device["name"] for device in devices)
    packages = {}
    for name in ("isaacsim", "isaaclab", "torch", "torchvision", "vllm", "transformers",
                 "websockets", "cuda-core", "cuda-bindings", "numpy", "pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "run_id": os.environ["EAI_RUN_ID"], "host": platform.node(),
              "os": platform.platform(), "python": platform.python_version(),
              "gpu_query": query, "gpus": devices,
              "single_5090_visible": topology_ok and expected_gpus == 1,
              "expected_gpus": expected_gpus, "gpu_topology_verified": topology_ok,
              "libraries": {name: library_status(name) for name in ("vulkan", "GLX_nvidia", "EGL")},
              "vulkaninfo_available": shutil.which("vulkaninfo") is not None,
              "driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
              "packages": packages,
              "storage_mode": os.environ.get("EAI_STORAGE_MODE", "jfs"),
              "jfs_origin_verified": os.environ.get("EAI_STORAGE_MODE", "jfs") == "jfs",
              "storage_origin_verified": True,
              "storage_free_bytes": shutil.disk_usage(result_dir).free,
              "jfs_free_bytes": (shutil.disk_usage(result_dir).free
                                 if os.environ.get("EAI_STORAGE_MODE", "jfs") == "jfs" else None),
              "inventory_passed": topology_ok, "native_evaluation_ready": False,
              "not_tested": ["Vulkan rendering", "Isaac camera frames", "Qwen endpoint connectivity",
                             "Benchmark assets", "Native RoboDojo episode"],
              "side_effects": "Only inventory JSON; no install, model load, GPU computation, or robot action"}
    if report["storage_mode"] == "job_local":
        if check_network:
            print(json.dumps({'event': 'inventory_stage', 'stage': 'network_endpoints'}), flush=True)
            report["network"] = network_inventory()
            print(json.dumps({'event': 'inventory_stage', 'stage': 'actual_resource_paths'}), flush=True)
            report["resource_routes"] = resource_inventory()
        else:
            report['network_checked'] = False
        print(json.dumps({'event': 'inventory_stage', 'stage': 'baked_image_and_read_only_inputs'}), flush=True)
        report["baked_image"] = inspect_image_runtime()
        report["storage_path"] = str(result_dir)
        report["storage_ephemeral"] = True
        report["inventory_passed"] = topology_ok and all(
            entry.get("loadable") is True for entry in report["libraries"].values())
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    if report["storage_mode"] == "job_local":
        print(export_inventory(report, os.environ["EAI_ORIGIN_SHA256"]), flush=True)
    print(json.dumps({"inventory_passed": report["inventory_passed"], "output": str(output_path),
                      "native_evaluation_ready": False}))
    if not report["inventory_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
