"""Depth curve, persano's shape (#87): fresh prefill and decode at 25K / 50K / 90K tokens, 256 out, greedy, median of 3.

  python depth.py <tag> <port> [depths=25000,50000,90000] [reps=3]

Prompts are real prose (public-domain scripture text from the workspace corpus), a different slice per repeat
and a salt on every request, so nothing is served from the prefix cache: every row is a fresh prefill.
Prefill tok/s = prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (total - TTFT), both from the
server's own usage counts. Thinking off. Prints one ROW per request and one MEDIAN line per depth.
"""
import glob, json, os, statistics, sys, time, urllib.request

TAG = sys.argv[1]; PORT = sys.argv[2]
DEPTHS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "25000,50000,90000").split(",")]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
BASE = f"http://127.0.0.1:{PORT}"
KEY = os.environ.get("VLLM_API_KEY", "")  # the server key, by env only; keyless servers ignore the header
CORPUS_GLOB = os.environ["DEPTH_CORPUS_GLOB"]  # any >= 2M chars of varied prose; set by acceptance.sh
CHARS_PER_TOKEN = 2.7  # the corpus runs 2.5-2.9 chars per token depending on the slice; the count printed is always the server's

def corpus():
    files = sorted(glob.glob(CORPUS_GLOB, recursive=True))
    out = []
    for f in files:
        try:
            out.append(open(f, encoding="utf-8", errors="ignore").read())
        except OSError:
            pass
    text = "\n\n".join(out)
    if len(text) < 2_000_000:
        raise SystemExit(f"corpus too small: {len(text)} chars from {len(files)} files")
    return text

TEXT = corpus()

ASK = os.environ.get("DEPTH_ASK") or "Write a short original commentary on the main themes of the document above, in your own words, without quoting it."

def prompt(tokens, rep, salt):
    n = int(tokens * CHARS_PER_TOKEN)
    start = (rep * 1_000_003 + tokens * 7) % max(1, len(TEXT) - n - 1)
    return (f"[depth probe {salt}] Read the following document, then answer.\n\n" + TEXT[start:start + n]
            + "\n\n" + ASK)

def one(tokens, rep):
    salt = f"depth-{tokens}-{rep}"  # deterministic: rep i is the same prompt on every image, run and box; it differs per rep so prefix caching never hits
    slice_rep = int(os.environ["FIX_REP"]) if os.environ.get("FIX_REP") else rep
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt(tokens, slice_rep, salt)}],
                       "temperature": 0, "max_tokens": 256, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            ev = json.loads(payload)
            if ev.get("usage"): usage = ev["usage"]
            ch = ev.get("choices") or []
            if first is None and ch and (ch[0].get("delta") or {}).get("content"):
                first = time.perf_counter() - t0
    total = time.perf_counter() - t0
    pt = usage["prompt_tokens"]; ct = usage["completion_tokens"]
    pre = pt / first; dec = (ct - 1) / (total - first) if ct > 1 else float("nan")
    print(f"ROW {TAG} depth={tokens} rep={rep} prompt={pt} out={ct} ttft={first:.2f}s prefill={pre:.0f} tok/s decode={dec:.1f} tok/s total={total:.1f}s", flush=True)
    import subprocess
    try:
        g = subprocess.run(["nvidia-smi", "-i", os.environ.get("ACC_GPU_SMI", os.environ.get("GPU_IDX", "0")), "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        g = f"nvidia-smi failed: {e}"
    print(f"GPU {TAG} depth={tokens} rep={rep} slice={slice_rep} {g}", flush=True)
    return pt, pre, dec

for d in DEPTHS:
    rows = []
    for i in range(REPS):
        try:
            rows.append(one(d, i))
        except Exception as e:
            print(f"ROW {TAG} depth={d} rep={i} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
    if rows:
        print(f"MEDIAN {TAG} depth={d} prompt~{int(statistics.median(r[0] for r in rows))} prefill={statistics.median(r[1] for r in rows):.0f} tok/s decode={statistics.median(r[2] for r in rows):.1f} tok/s n={len(rows)}", flush=True)
