"""Private JFS transport to an existing loopback Qwen service; no public listener."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener
import uuid


MAX_BYTES = 16 * 1024 * 1024
PAYLOAD_KEYS = {"model", "messages", "max_tokens", "temperature", "top_p", "response_format",
                "tools", "tool_choice", "chat_template_kwargs", "stream"}


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Relay file exceeds size limit")
    return json.loads(data)


def validate_payload(payload: dict, model: str) -> None:
    if not isinstance(payload, dict) or set(payload) - PAYLOAD_KEYS:
        raise ValueError("Unsupported request fields")
    if payload.get("model") != model or payload.get("stream", False) is not False:
        raise ValueError("Only the fixed model and non-streaming requests are allowed")
    limit = payload.get("max_tokens")
    if type(limit) is not int or not 1 <= limit <= 8192:
        raise ValueError("max_tokens must be 1..8192")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise ValueError("A bounded messages list is required")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Invalid message")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("Only text and inline image content is allowed")
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Invalid content block")
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                continue
            if item.get("type") != "image_url":
                raise ValueError("Unsupported content block")
            if not isinstance(item.get("image_url"), dict):
                raise ValueError("Invalid image block")
            image = item["image_url"].get("url", "")
            if not isinstance(image, str):
                raise ValueError("Invalid image URL")
            if not image.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                raise ValueError("Remote images and filesystem URLs are forbidden")
            base64.b64decode(image.split(",", 1)[1], validate=True)


def validate_session_root(root: Path) -> None:
    if (root.is_symlink() or not root.resolve().is_relative_to("/jfs-public")
            or not os.path.ismount("/jfs-public")):
        raise ValueError("A private session on the real JFS mount is required")
    if root.exists() and root.stat().st_mode & 0o077:
        raise PermissionError("Relay session directory must be private (0700)")


def process_request(path: Path, *, model: str, base_url: str, timeout: int) -> None:
    request_id = path.stem
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ValueError("Invalid request ID")
    claim = path.parent.parent / "claims" / (request_id + ".json")
    try:
        write_json(claim, {"claimed_at": datetime.now(timezone.utc).isoformat()})
    except FileExistsError:
        return
    started = time.monotonic()
    try:
        request = read_json(path)
        if request["expires_at"] < time.time():
            raise TimeoutError("Request expired before inference")
        payload = request["payload"]
        validate_payload(payload, model)
        headers = {"Content-Type": "application/json"}
        if os.environ.get("QWEN_UPSTREAM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["QWEN_UPSTREAM_API_KEY"]
        upstream = Request(base_url.rstrip("/") + "/chat/completions",
                           data=json.dumps(payload).encode(), headers=headers)
        with build_opener(ProxyHandler({})).open(upstream, timeout=timeout) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("Upstream response exceeds limit")
        result = {"status": 200, "body": json.loads(body)}
    except Exception as error:
        status = error.code if isinstance(error, HTTPError) else 502
        result = {"status": status, "body": {"error": {"type": type(error).__name__,
                  "message": "Inference failed; no automatic retry"}}}
    result.update(request_id=request_id, elapsed_seconds=time.monotonic() - started)
    write_json(path.parent.parent / "responses" / (request_id + ".json"), result)


def make_server(root: Path, port: int, timeout: int) -> ThreadingHTTPServer:
    session = read_json(root / "session.json")
    admission_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def authorized(self):
            expected = "Bearer " + session["token"]
            return secrets.compare_digest(self.headers.get("Authorization", ""), expected)

        def respond(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.authorized():
                self.respond(401, {"error": "Unauthorized"})
            elif self.path == "/v1/models":
                self.respond(200, {"data": [{"id": session["model"]}]})
            else:
                self.respond(404, {"error": "Unsupported endpoint"})

        def do_POST(self):
            if not self.authorized():
                self.respond(401, {"error": "Unauthorized"})
                return
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "Unsupported endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= MAX_BYTES - 4096:
                    raise ValueError("Request size is invalid")
                self.connection.settimeout(30)
                payload = json.loads(self.rfile.read(length))
                validate_payload(payload, session["model"])
                with admission_lock:
                    if len(list((root / "requests").glob("*.json"))) >= session["max_requests"]:
                        self.respond(429, {"error": "Relay session request budget exhausted"})
                        return
                    request_id = uuid.uuid4().hex
                    write_json(root / "requests" / (request_id + ".json"),
                               {"payload": payload, "expires_at": time.time() + timeout})
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, {"error": type(error).__name__})
                return
            deadline = time.monotonic() + timeout
            response = root / "responses" / (request_id + ".json")
            while time.monotonic() < deadline:
                if response.is_file():
                    result = read_json(response)
                    self.respond(result["status"], result["body"])
                    return
                time.sleep(0.25)
            self.respond(504, {"error": "Relay response timeout; do not automatically retry"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("worker", "server"))
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=650)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    root = args.session_dir.absolute()
    validate_session_root(root)
    if not 1 <= args.max_requests <= 200 or not 30 <= args.timeout <= 900 or not 0 < args.max_hours <= 24:
        parser.error("Relay budgets exceed permitted bounds")
    os.umask(0o077)
    if args.role == "server":
        server = make_server(root, args.port, args.timeout)
        write_json(root / "server.json", {"pid": os.getpid(), "port": server.server_address[1],
                   "host": "127.0.0.1", "started_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"port": server.server_address[1], "loopback_only": True}), flush=True)
        server.timeout = 1
        deadline = time.monotonic() + args.max_hours * 3600
        try:
            while time.monotonic() < deadline:
                server.handle_request()
        finally:
            server.server_close()
        return
    parsed = urlparse(args.base_url)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path != "/v1"):
        parser.error("Upstream must be the existing loopback /v1 endpoint")
    root.mkdir(mode=0o700)
    for directory in ("requests", "responses", "claims"):
        (root / directory).mkdir(mode=0o700)
    write_json(root / "session.json", {"token": secrets.token_urlsafe(32), "model": args.model,
               "max_requests": args.max_requests, "worker_pid": os.getpid(),
               "started_at": datetime.now(timezone.utc).isoformat()})
    deadline = time.monotonic() + args.max_hours * 3600
    print(json.dumps({"worker_started": True, "session_dir": str(root), "max_requests": args.max_requests}), flush=True)
    while time.monotonic() < deadline:
        for path in sorted((root / "requests").glob("*.json")):
            if len(list((root / "claims").glob("*.json"))) >= args.max_requests:
                return
            if not (root / "claims" / path.name).exists():
                process_request(path, model=args.model, base_url=args.base_url, timeout=args.timeout - 10)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
