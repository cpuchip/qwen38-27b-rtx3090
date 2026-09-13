#!/usr/bin/env python3
"""The reply gate: run on every public comment before it is posted (docs/pr-standard.md, "Public replies").

    python scripts/pr-gate/reply_gate.py comment.md

Exit 1 on a FAIL (em-dash, attribution line, retracted phrase); WARNs are for a person to read.
"""
import os, re, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OVERCLAIM = ["rules out", "proves that", "proof that", "identical output", "always", "never", "must have been", "cannot be", "impossible", "guarantee"]
JARGON = {"the track": "the branch that rides vLLM main", "whale": "one large prompt", "minnow": "a small request", "dead-man": "a watchdog", "falsifier": "a check that could have failed", "oracle": "a deterministic check", "the fork route": "building from our vLLM fork", "series-check": "scripts/series-check.sh"}
CONFIG_TOKENS = re.compile(r"(k=\d|CTX=|MAX_LEN|3090|4090|5060|WSL2|native|n=\d|DFLASH_TOKENS|SPEC=|PREFIX_CACHE|INT8_ACT|q=\d|kv=\d|box|card|arm|pool)", re.I)
NUMBER_LINE = re.compile(r"(\d+(\.\d+)?\s*(tok/s|%|ms|s\b|GiB|GB|x\b))")


def main():
    path = sys.argv[1]
    text = open(path, encoding="utf-8").read()
    fails, warns = [], []
    if "—" in text:
        fails.append(f"{text.count(chr(8212))} em-dash(es)")
    if re.search(r"claude\.ai/code|Co-Authored-By|Claude-Session", text, re.I):
        fails.append("attribution line or session URL")
    for l in open(os.path.join(HERE, "retracted.txt"), encoding="utf-8"):
        if l.strip() and not l.startswith("#"):
            ph = l.split("\t")[0].strip()
            if ph.lower() in text.lower():
                fails.append(f"retracted phrase '{ph}'")
    low = text.lower()
    for w in OVERCLAIM:
        if w in low:
            warns.append(f"overclaim word '{w}': does a control back it?")
    for j, plain in JARGON.items():
        if j in low and plain.split()[0] not in low:
            warns.append(f"shorthand '{j}' without its plain definition ({plain})")
    lines = text.splitlines()
    has_numbers = False
    for i, l in enumerate(lines):
        if NUMBER_LINE.search(l):
            has_numbers = True
            if not CONFIG_TOKENS.search(l) and not (i and CONFIG_TOKENS.search(lines[i - 1])):
                warns.append(f"measurement without configuration: '{l.strip()[:70]}'")
    if has_numbers and not re.search(r"(unproven|not measured|did not (measure|run|test)|n=\d|single run|one boot|caveat)", text, re.I):
        warns.append("numbers present and no caveat or 'unproven' statement")
    words = len(text.split())
    if words > 600:
        warns.append(f"{words} words: the reader pays for every one; cut to what changes their next action")
    for f in fails:
        print(f"  FAIL {f}")
    for w in warns:
        print(f"  WARN {w}")
    print(f"reply-gate: {len(fails)} FAIL, {len(warns)} WARN, {words} words")
    print("  shape: decision first; what was checked and how; what is still unproven; what would unblock; credit")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
