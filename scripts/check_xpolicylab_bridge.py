"""Exercise the real XPolicyLab handshake/reset in isolated CPU-only processes."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)


async def serve(output: Path) -> None:
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.policy.Embodied_Agentic.model import Model

    model = Model({"action_type": "ee"})
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
    await server.start()
    write_json(output, {"url": server.url, "pid": os.getpid(),
                       "websockets": version("websockets")})
    try:
        await asyncio.Future()
    finally:
        await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("check", "server", "client"), default="check")
    parser.add_argument("--robodojo-root", type=Path)
    parser.add_argument("--server-python", type=Path)
    parser.add_argument("--client-python", type=Path)
    parser.add_argument("--client-site", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url")
    args = parser.parse_args()
    os.umask(0o077)
    if args.role == "server":
        asyncio.run(serve(args.output))
        return
    if args.role == "client":
        from client_server.ws.model_client import WsModelClient

        with WsModelClient(url=args.url, evaluation_id="eai-bridge-smoke",
                           trial_id="eai-reset-smoke", max_connect_attempts=1,
                           connect_timeout_s=10, handshake_timeout_s=10, request_timeout_s=10) as client:
            client.call("reset")
            client.call("trial_end", {"infrastructure_smoke": True})
        write_json(args.output, {"handshake_reset_passed": True, "websockets": version("websockets"),
                               "native_evaluation_ready": False})
        return
    if not args.robodojo_root or not args.server_python or not args.client_python:
        parser.error("Specify the pinned source tree and both isolated Python paths")
    args.output.mkdir(mode=0o700)
    root = args.robodojo_root.resolve()
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                       EMBODIED_ROBODOJO_CONTROL_MODE="ee_delta", PYTHONPATH=f"{root}:{root / 'XPolicyLab'}",
                       NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1")
    script = str(Path(__file__).resolve())
    server_output = args.output / "server-ready.json"
    with (args.output / "server.private.log").open("x") as log:
        server = subprocess.Popen([str(args.server_python), "-B", script, "--role", "server",
                                   "--output", str(server_output.resolve())], env=environment,
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while not server_output.exists():
                if server.poll() is not None:
                    raise RuntimeError("Our bridge server exited; inspect its private log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Our bridge server did not become ready")
                time.sleep(0.25)
            ready = json.loads(server_output.read_text())
            if ready["pid"] != server.pid:
                raise ValueError("Unexpected bridge process identity")
            client_environment = dict(environment)
            if args.client_site:
                if not args.client_site.is_dir():
                    raise ValueError("Client-only dependency directory does not exist")
                client_environment["PYTHONPATH"] = str(args.client_site.resolve()) + ":" + environment["PYTHONPATH"]
            with (args.output / "client.private.log").open("x") as client_log:
                subprocess.run([str(args.client_python), "-B", script, "--role", "client", "--url", ready["url"],
                                "--output", str((args.output / "client-result.json").resolve())],
                               env=client_environment, stdout=client_log, stderr=subprocess.STDOUT, timeout=120, check=True)
            client = json.loads((args.output / "client-result.json").read_text())
            report = {"checked_at": datetime.now(timezone.utc).isoformat(),
                      "server_websockets": ready["websockets"], "client_websockets": client["websockets"],
                      "handshake_reset_passed": client["handshake_reset_passed"],
                      "native_evaluation_ready": False, "gpu_used": False}
            write_json(args.output / "result.json", report)
            print(json.dumps(report, indent=2))
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
