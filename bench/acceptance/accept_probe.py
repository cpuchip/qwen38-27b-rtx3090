"""accept_probe.py - decode rate WITH its acceptance counters, per request."""
import glob, json, os, re, sys, time, urllib.request
TAG, PORT, CHARS, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
KEY = os.environ.get("VLLM_API_KEY", "")
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
BASE = f"http://127.0.0.1:{PORT}"

def counters():
    """Whatever this build calls them: discover rather than assume the metric names."""
    out = {}
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or not line.strip(): continue
                if re.search(r"accept|draft|spec", line):
                    name, _, val = line.rpartition(" ")
                    try: out[name.split("{")[0]] = out.get(name.split("{")[0], 0.0) + float(val)
                    except ValueError: pass
    except Exception: pass
    return out

files = sorted(glob.glob(os.environ["DEPTH_CORPUS_GLOB"], recursive=True))
TEXT = "".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)
prompt = "[accept probe] Summarise the following.\n\n" + TEXT[:CHARS]

names = sorted(counters().keys())
print(json.dumps({"tag": TAG, "counters_found": names}), flush=True)
for i in range(N):
    a = counters(); t0 = time.time()
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 256, "temperature": 0, "stream": False})
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions",
                                    data=body.encode(), headers=H), timeout=1200) as r:
            d = json.load(r)
    except Exception as e:
        print(json.dumps({"tag": TAG, "rep": i, "error": f"{type(e).__name__}: {e}"}), flush=True); continue
    dt = time.time() - t0; b = counters(); u = d["usage"]
    ct = u["completion_tokens"]
    acc = {k: round(b.get(k, 0) - a.get(k, 0), 1) for k in set(a) | set(b) if b.get(k, 0) != a.get(k, 0)}
    # exact-suffix selection (threadchip, 2026-09-12): "draft" matched both num_drafts and
    # num_draft_tokens and set iteration order is hash-dependent, so the old line picked
    # a different counter per process; report both named quantities instead.
    def pick(suffix):
        return next((v for k, v in acc.items() if k.endswith(suffix)), None)
    accepted = pick("num_accepted_tokens_total")
    draft_tokens = pick("num_draft_tokens_total")
    n_drafts = pick("num_drafts_total")
    row = {"tag": TAG, "rep": i, "prompt_tokens": u["prompt_tokens"], "completion_tokens": ct,
           "seconds": round(dt, 2), "decode_tok_s": round(ct / dt, 1),
           "accepted": accepted, "draft_tokens": draft_tokens, "n_drafts": n_drafts,
           "accepted_per_draft_step": round(accepted / n_drafts, 3) if (accepted and n_drafts) else None,
           "accept_frac_of_draft_tokens": round(accepted / draft_tokens, 3) if (accepted and draft_tokens) else None,
           "delta": acc}
    print(json.dumps(row), flush=True)
