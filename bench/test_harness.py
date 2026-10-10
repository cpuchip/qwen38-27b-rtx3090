#!/usr/bin/env python3
"""CPU-only checks for bench/harness.py: the key chain matches resolve_api_key.sh, one URL convention, post()."""
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import harness

REPO = Path(__file__).resolve().parents[1]
KEY_VARS = ("OPENAI_API_KEY", "VLLM_API_KEY")


def bash_client_key(env, repo):
    script = f'REPO="$1"; . "{REPO}/resolve_api_key.sh"; resolve_client_key; printf %s "$OPENAI_API_KEY"'
    clean = {k: v for k, v in os.environ.items() if k not in KEY_VARS}
    return subprocess.run(["bash", "-c", script, "bash", str(repo)], env={**clean, **env},
                          capture_output=True, check=True).stdout.decode()


class ClientKeyTests(unittest.TestCase):
    def test_matches_resolve_client_key(self):
        cases = [
            ({"OPENAI_API_KEY": "o", "VLLM_API_KEY": "v"}, "f\n", "o"),
            ({"VLLM_API_KEY": "v"}, "f\n", "v"),
            ({"OPENAI_API_KEY": "", "VLLM_API_KEY": "v"}, None, "v"),
            ({}, "f\n\n", "f"),
            ({}, "f \r\n", "f \r"),
            ({}, None, "EMPTY"),
            ({"VLLM_API_KEY": ""}, None, "EMPTY"),
        ]
        for env, file, want in cases:
            with self.subTest(env=env, file=file), tempfile.TemporaryDirectory() as tmp:
                if file is not None:
                    Path(tmp, "api_key.txt").write_text(file)
                with mock.patch.dict(os.environ, env), mock.patch.object(harness, "REPO", tmp):
                    for k in KEY_VARS:
                        if k not in env:
                            os.environ.pop(k, None)
                    self.assertEqual(harness.client_key(), want)
                self.assertEqual(bash_client_key(env, tmp), want)


class BaseUrlTests(unittest.TestCase):
    def test_one_convention(self):
        cases = [
            ({}, "http://127.0.0.1:18020"),
            ({"PORT": "18021"}, "http://127.0.0.1:18021"),
            ({"VLLM_API": "http://h:1/v1", "PORT": "18021"}, "http://h:1"),
            ({"VLLM_API": "http://h:1/v1/"}, "http://h:1"),
            ({"VLLM_API": "http://h:1"}, "http://h:1"),
            ({"VLLM_API": "http://h:1/proxy/v1"}, "http://h:1/proxy"),
        ]
        for env, want in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                for k in ("VLLM_API", "PORT"):
                    if k not in env:
                        os.environ.pop(k, None)
                self.assertEqual(harness.base_url(), want)


class PostTests(unittest.TestCase):
    def test_post_sends_key_and_json(self):
        seen = {}

        class Stub(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.update(path=self.path, auth=self.headers["Authorization"], ctype=self.headers["Content-Type"],
                            body=json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                out = json.dumps({"ok": 1}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, format, *args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            env = {"VLLM_API": f"http://127.0.0.1:{srv.server_port}/v1", "OPENAI_API_KEY": "k"}
            with mock.patch.dict(os.environ, env):
                got = harness.post("/v1/chat/completions", {"model": "m"}, timeout=10)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(got, {"ok": 1})
        self.assertEqual(seen, {"path": "/v1/chat/completions", "auth": "Bearer k",
                                "ctype": "application/json", "body": {"model": "m"}})

    def test_get_request_has_no_body(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", "PORT": "1"}):
            os.environ.pop("VLLM_API", None)
            req = harness.request("/metrics")
        self.assertEqual((req.full_url, req.get_method(), req.data), ("http://127.0.0.1:1/metrics", "GET", None))
        self.assertEqual(req.get_header("Authorization"), "Bearer k")


if __name__ == "__main__":
    unittest.main()
