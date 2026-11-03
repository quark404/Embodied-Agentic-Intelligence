import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from scripts.qwen_jfs_relay import make_server, process_request, read_json, validate_payload, write_json
from scripts.download_robodojo_assets import validate_manifest


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"model": "Qwen-test", "max_tokens": 64,
                        "messages": [{"role": "user", "content": "Return ready"}]}

    def test_fixed_model_and_budgets(self):
        validate_payload(self.payload, "Qwen-test")
        for key, value in (("model", "another"), ("max_tokens", 9000),
                           ("max_tokens", True), ("stream", True), ("endpoint", "external")):
            payload = {**self.payload, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_payload(payload, "Qwen-test")

    def test_no_remote_or_filesystem_image_fetch(self):
        for image in ("https://external.invalid/image.png", "file:///etc/passwd", "data:image/png;base64,%%"):
            payload = copy.deepcopy(self.payload)
            payload["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": image}}]
            with self.subTest(image=image), self.assertRaises(ValueError):
                validate_payload(payload, "Qwen-test")

    def test_atomic_private_files_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            write_json(path, self.payload)
            self.assertEqual(read_json(path), self.payload)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_json(path, {})
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                read_json(link)

    def test_expired_requests_never_infer_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("requests", "responses", "claims"):
                (root / name).mkdir()
            path = root / "requests" / (uuid.uuid4().hex + ".json")
            write_json(path, {"payload": self.payload, "expires_at": time.time() - 1})
            with patch("scripts.qwen_jfs_relay.build_opener") as opener:
                for _ in range(2):
                    process_request(path, model="Qwen-test", base_url="http://127.0.0.1:18000/v1", timeout=30)
                opener.assert_not_called()
            result = read_json(root / "responses" / path.name)
            self.assertEqual(result["status"], 502)
            self.assertEqual(result["body"]["error"]["type"], "TimeoutError")

    def test_loopback_requires_session_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "session.json", {"token": "test-session-only", "model": "Qwen-test", "max_requests": 1})
            for name in ("requests", "responses", "claims"):
                (root / name).mkdir()
            server = make_server(root, 0, 30)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            opener = build_opener(ProxyHandler({}))
            endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/models"
            try:
                with self.assertRaises(HTTPError) as error:
                    opener.open(endpoint, timeout=5)
                self.assertEqual(error.exception.code, 401)
                request = Request(endpoint, headers={"Authorization": "Bearer test-session-only"})
                with opener.open(request, timeout=5) as response:
                    self.assertEqual(json.load(response)["data"][0]["id"], "Qwen-test")
                self.assertEqual(list((root / "requests").iterdir()), [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_asset_scope_is_pinned_and_bounded(self):
        lock = {"revision": "pinned", "file_count": 1, "total_bytes": 4}
        manifest = {**lock, "complete": True, "files": [{"path": "Assets/file", "size": 4}]}
        validate_manifest(manifest, lock)
        for path in ("outside/file", "Assets/../private", "/Assets/file"):
            changed = copy.deepcopy(manifest)
            changed["files"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_manifest(changed, lock)


if __name__ == "__main__":
    unittest.main()
