#!/usr/bin/env python3
"""First greedy divergence of each arm vs the reference arm, in tokens (run where `tokenizers` exists, e.g. the image).
  python eagle4/bench/score.py <tokenizer.json> <ref.json> <arm.json> [<arm.json> ...]
Counts the prompts (copy excluded) that diverge inside the first 32 tokens.
"""
import json
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
ref = json.load(open(sys.argv[2], encoding="utf-8"))


def ids(t):
    return tok.encode(t, add_special_tokens=False).ids


for path in sys.argv[3:]:
    arm = json.load(open(path, encoding="utf-8"))
    early = 0
    print(f"== {path} vs {sys.argv[2]}")
    for key in sorted(k for k in ref if k.startswith("greedy/")):
        a, b = ids(arm[key]["text"]), ids(ref[key]["text"])
        n = min(len(a), len(b))
        div = next((i for i in range(n) if a[i] != b[i]), None)
        if div is None and len(a) != len(b):
            div = n
        if key != "greedy/copy" and div is not None and div < 32:
            early += 1
        where = "identical" if div is None else f"first divergence at token {div} of {len(b)}"
        ctx = "" if div is None else f"  ref {tok.decode(b[max(0, div - 3):div + 3])!r} | arm {tok.decode(a[max(0, div - 3):div + 3])!r}"
        print(f"  {key:12s} {where}{ctx}")
    print(f"  prompts diverging inside 32 tokens: {early} of 5 ")
