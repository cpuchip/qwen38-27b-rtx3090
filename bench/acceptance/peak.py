"""Peak KV pool usage during one long prefill (vllm #55450 real-path oracle).
Sends one ~N-token prompt (greedy, 16 out) and samples vllm:kv_cache_usage_perc every 0.25 s until the response
returns; prints the peak and the settle value. Align-mode Mamba retirement that stops at null gaps shows up as a
higher peak for the same prompt.    python peak.py TAG PORT CHARS
"""
import glob, json, os, sys, threading, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
TAG, PORT, CHARS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
KEY = os.environ.get("VLLM_API_KEY", ""); H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
TEXT = "\n\n".join(open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.environ["DEPTH_CORPUS_GLOB"], recursive=True)))
def usage():
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers=H), timeout=10).read().decode()
    for line in r.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc"):
            return float(line.rsplit(" ", 1)[1])
    return -1.0
samples = []; stop = False
def sampler():
    while not stop:
        try: samples.append((time.time(), usage()))
        except Exception: pass
        time.sleep(0.25)
t = threading.Thread(target=sampler, daemon=True); t.start()
prompt = "Summarize the following text in one sentence.\n\n" + TEXT[300000:300000 + CHARS]
body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0})
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body.encode(), headers=H), timeout=1800))
dt = time.time() - t0; time.sleep(1.5); stop = True; t.join(timeout=2)
u = r["usage"]; peak = max(v for _, v in samples) if samples else -1; settle = usage()
print(json.dumps({"tag": TAG, "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"], "seconds": round(dt, 1), "samples": len(samples), "peak_usage": peak, "settle_usage": settle}), flush=True)
