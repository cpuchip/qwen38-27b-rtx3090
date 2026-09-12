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
CORPUS_GLOB = os.environ.get("DEPTH_CORPUS_GLOB", os.path.expanduser("~/gospel-library/eng/scriptures/**/*.md"))  # any >= 2M chars of varied prose
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


def first_deltas(tokens, rep, n_show=8):
    salt = f"firsttok-{tokens}-{rep}"
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt(tokens, rep, salt)}],
                       "temperature": 0, "max_tokens": 64, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); shown = 0; n_deltas = 0; n_content = 0; n_reason = 0; first_content = None; first_reason = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            ev = json.loads(payload)
            ch = ev.get("choices") or []
            if not ch: continue
            delta = ch[0].get("delta") or {}
            n_deltas += 1
            t = time.perf_counter() - t0
            if delta.get("content"):
                n_content += 1
                if first_content is None: first_content = t
            if delta.get("reasoning_content") or delta.get("reasoning"):
                n_reason += 1
                if first_reason is None: first_reason = t
            if shown < n_show:
                shown += 1
                print(f"DELTA {TAG} t={t:.2f}s keys={sorted(k for k, v in delta.items() if v)} content={str(delta.get('content'))[:30]!r}", flush=True)
    print(f"FIRSTTOK {TAG} depth={tokens} deltas={n_deltas} content_deltas={n_content} reasoning_deltas={n_reason} first_content={first_content} first_reasoning={first_reason}", flush=True)

first_deltas(DEPTHS[0], 0)
