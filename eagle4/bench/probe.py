#!/usr/bin/env python3
"""Single-user decode speed and outputs, the probe behind eagle4/README.md's table.

    python eagle4/bench/probe.py <label> [--port 8000] [--passage FILE]

One untimed warm-up, then (if --passage) a verbatim copy of FILE's text and five short prompts, greedy and then
temperature 1 (top_p 1, top_k off, seed 0), one request at a time, streaming, thinking off. Decode tok/s =
(completion tokens - 1) / (last content chunk - first content chunk). Tokens per step from the server's
vllm:spec_decode_* counters when speculation is on. Texts go to <label>.json for score.py.
The README's runs used Lectures on Faith, Lecture First (public domain), cut to its first 8,000 characters.
"""
import argparse
import json
import os
import time
import urllib.request

PROMPTS = ["Explain how a hash map handles collisions, in about 150 words.",
           "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
           "List ten countries in Europe and their capitals, one per line.",
           "Summarize the causes of the French Revolution in five bullet points.",
           "Describe the water cycle to a ten-year-old in about 120 words."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--passage")
    ap.add_argument("--model", default="qwen3.8-27b")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    hdr = {"Content-Type": "application/json"}
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY")
    if key:
        hdr["Authorization"] = f"Bearer {key}"

    def counters():
        try:
            with urllib.request.urlopen(urllib.request.Request(base + "/metrics", headers=hdr), timeout=10) as r:
                txt = r.read().decode()
        except OSError:
            return None
        tot = lambda name: sum(float(l.rsplit(" ", 1)[1]) for l in txt.splitlines()
                               if l.startswith(name) and not l.startswith("#"))
        return tot("vllm:spec_decode_num_drafts_total"), tot("vllm:spec_decode_num_accepted_tokens_total")

    def run(content, max_tokens, sampled):
        body = dict(model=args.model, stream=True, max_tokens=max_tokens, seed=0,
                    stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False},
                    messages=[{"role": "user", "content": content}])
        body.update(dict(temperature=1.0, top_p=1.0, top_k=0) if sampled else dict(temperature=0))
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), hdr)
        c0, t0, tf, tl, text, usage = counters(), time.time(), None, None, [], None
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                c = json.loads(line[6:])
                usage = c.get("usage") or usage
                for ch in c.get("choices") or []:
                    d = ch.get("delta", {}).get("content")
                    if d:
                        tl = time.time()
                        tf = tf or tl
                        text.append(d)
        c1 = counters()
        n = usage["completion_tokens"]
        step = (c1[1] - c0[1]) / (c1[0] - c0[0]) + 1 if c0 and c1 and c1[0] > c0[0] else None
        return dict(text="".join(text), completion_tokens=n, prompt_tokens=usage["prompt_tokens"],
                    ttft=tf - t0 if tf else None, decode_tps=(n - 1) / (tl - tf) if tf and tl > tf else float("nan"),
                    tok_per_step=step)

    cases = []
    if args.passage:
        passage = open(args.passage, encoding="utf-8").read().strip()
        cases.append(("copy", "Repeat the following passage exactly, word for word, with no commentary before or "
                      "after it.\n\n" + passage, 2600))
    cases += [(f"p{i}", q, 400) for i, q in enumerate(PROMPTS)]
    run("Say hello in five words.", 32, False)
    out = {}
    for sampled in (False, True):
        mode = "temp1" if sampled else "greedy"
        toks = secs = 0.0
        for name, content, max_tokens in cases:
            r = run(content, max_tokens, sampled)
            out[f"{mode}/{name}"] = r
            toks += r["completion_tokens"] - 1
            secs += (r["completion_tokens"] - 1) / r["decode_tps"]
            extra = f" tok/step {r['tok_per_step']:.2f}" if r["tok_per_step"] else ""
            if name == "copy":
                extra += f" verbatim {' '.join(r['text'].split()) == ' '.join(passage.split())}"
            print(f"[{args.label}] {mode:6s} {name:4s} out {r['completion_tokens']:5d} "
                  f"decode {r['decode_tps']:7.1f} tok/s{extra}", flush=True)
        print(f"[{args.label}] {mode} pooled decode {toks / secs:.1f} tok/s over {toks:.0f} tokens", flush=True)
    json.dump(out, open(f"{args.label}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
