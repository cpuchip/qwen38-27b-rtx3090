#!/usr/bin/env python3
"""Send one exact request to populate vLLM's in-memory hybrid prefix cache.

This is deliberately a warm-start helper, not a disk KV serializer. It is
safe for the hybrid GDN model because vLLM itself computes and stores the KV
and recurrent state; after a restart the helper simply pays the prefill once
before the service is considered ready by the wrapper.

The request file is JSON with this shape:

  {"endpoint": "/v1/chat/completions", "payload": { ... }}

The payload should contain the same long prefix that future requests will
reuse. The final question/output can be a one-token warm-up request.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request


def api_key() -> str:
    key = os.environ.get("VLLM_API_KEY", "")
    if key:
        return key
    try:
        with open("/app/api_key.txt", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: warmup_prefix.py REQUEST_JSON", file=sys.stderr)
        return 2

    request_path = sys.argv[1]
    with open(request_path, encoding="utf-8") as f:
        spec = json.load(f)

    endpoint = spec.get("endpoint", "/v1/chat/completions")
    payload = dict(spec["payload"])
    payload.setdefault("max_tokens", 1)
    payload.setdefault("temperature", 0)
    payload["stream"] = False

    base = os.environ.get(
        "WARMUP_API", f"http://127.0.0.1:{os.environ.get('PORT', '18020')}"
    )
    url = base.rstrip("/") + "/" + endpoint.lstrip("/")
    headers = {"Content-Type": "application/json"}
    if key := api_key():
        headers["Authorization"] = f"Bearer {key}"

    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(
        request, timeout=float(os.environ.get("WARMUP_REQUEST_TIMEOUT", "3600"))
    ) as response:
        body = json.load(response)

    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    elapsed = time.perf_counter() - started
    print(
        "warmup prefix complete: "
        f"prompt_tokens={usage.get('prompt_tokens', '?')} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
