"""How the bench/ scripts reach the server: the key, the URL and the request.

Every script runs as `python bench/<script>.py`, which puts bench/ on sys.path, so `import harness` needs no
path setup. Stdlib only: the scripts run on the venv's bare Python.
"""
import json
import os
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def client_key():
    """resolve_client_key in resolve_api_key.sh: OPENAI_API_KEY, else VLLM_API_KEY, else api_key.txt, else
    "EMPTY" (a server that bound no key ignores it). test_harness.py runs both and compares."""
    for name in ("OPENAI_API_KEY", "VLLM_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    path = os.path.join(REPO, "api_key.txt")
    if os.path.isfile(path):
        with open(path, newline="") as f:
            return f.read().rstrip("\n")
    return "EMPTY"


def base_url():
    """The server root: VLLM_API if set, else http://127.0.0.1:${PORT:-18020}. A trailing /v1 on VLLM_API is
    dropped, because five scripts documented VLLM_API with it."""
    api = os.environ.get("VLLM_API", "").rstrip("/")
    if api:
        return api.removesuffix("/v1")
    return "http://127.0.0.1:" + os.environ.get("PORT", "18020")


def request(path, payload=None):
    """A request to base_url() + path with the key; a payload makes it a JSON POST."""
    headers = {"Authorization": "Bearer " + client_key()}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    return urllib.request.Request(base_url() + path, data=data, headers=headers)


def post(path, payload, timeout=1200):
    """JSON POST to base_url() + path; returns the decoded JSON body."""
    with urllib.request.urlopen(request(path, payload), timeout=timeout) as r:
        return json.load(r)
