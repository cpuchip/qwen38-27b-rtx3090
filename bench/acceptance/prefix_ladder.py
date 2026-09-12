"""Prefix-cache reuse ladder: for prompt lengths L, send the prompt (greedy, 8 out), replay it at once with a short
follow-up appended, and report cached_tokens on the replay. python ladder.py TAG PORT"""
import glob, json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
TAG, PORT = sys.argv[1], int(sys.argv[2])
KEY = os.environ.get("VLLM_API_KEY", ""); H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
TEXT = "\n\n".join(open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.environ["DEPTH_CORPUS_GLOB"], recursive=True)))
def post(path, body): return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers=H), timeout=900))
def tok(t): return post("/tokenize", {"model": "qwen3.8-27b", "prompt": t})["tokens"]
def comp(ids, n): r = post("/v1/completions", {"model": "qwen3.8-27b", "prompt": ids, "max_tokens": n, "temperature": 0}); u = r["usage"]; return u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens")
FOLLOW = tok(" One more sentence.")
for i, L in enumerate([512, 1024, 2048, 4096, 8192, 16384, 32768]):
    ids = tok(TEXT[400000 + i * 200000: 400000 + i * 200000 + L * 4])[:L]
    pt1, c1 = comp(ids, 8)
    pt2, c2 = comp(ids + FOLLOW, 8)
    pt3, c3 = comp(ids + FOLLOW, 8)
    print(json.dumps({"tag": TAG, "L": L, "prompt_tokens": pt1, "cached_first": c1, "cached_replay": c2, "cached_replay2": c3, "reuse_frac": round((c2 or 0) / pt1, 3)}), flush=True)
