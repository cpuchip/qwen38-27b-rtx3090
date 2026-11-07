"""score.py OUT_DIR CASES_FILE: totals, the non-OK cases grouped by verdict and message, and a tally of the
distinct vLLM warnings across cases (read from each case's launcher.out, where vLLM logs; digits shown as N).

Verdicts: OK, RAISE, PARSE_ERROR (from build_config.py); REFUSED (the launcher exited non-zero before starting
vLLM, last stderr line shown); NO_RESULT (exit 0 and no JSON: the harness broke, never a pass).
"""
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
out, cases = sys.argv[1], sys.argv[2]
rows, warn = [], collections.Counter()
WARN = re.compile(r" WARNING \S+ \S+ (.*)")
for line in open(cases):
    cid = line.split()[0]
    rc_path, js = os.path.join(out, cid + ".rc"), os.path.join(out, cid + ".json")
    rc = open(rc_path).read().strip() if os.path.exists(rc_path) else "missing"
    log = os.path.join(out, cid + ".launcher.out")
    if os.path.exists(log):
        seen = set()
        for m in WARN.finditer(open(log, errors="replace").read()):
            seen.add(re.sub(r"\d+", "N", m.group(1))[:200])
        warn.update(seen)
    if os.path.exists(js):
        d = json.load(open(js))
        v, msg = d["verdict"], d.get("error", "")
    elif rc not in ("0", "missing"):
        err = open(os.path.join(out, cid + ".launcher.err"), errors="replace").read().strip().splitlines()
        v, msg = "REFUSED", err[-1] if err else ""
    else:
        v, msg = "NO_RESULT", f"rc={rc}"
    rows.append((cid, v, " ".join(msg.split())))

print("totals:", dict(collections.Counter(v for _, v, _ in rows)), "of", len(rows))
groups = collections.defaultdict(list)
for cid, v, msg in rows:
    if v != "OK":
        groups[(v, re.sub(r"input_value=.*", "", msg)[:300])].append(cid)
for (v, msg), ids in sorted(groups.items(), key=lambda g: -len(g[1])):
    prefixes = collections.Counter("_".join(i.split("_")[:2]) for i in ids)
    print(f"\n{v} x{len(ids)} ({', '.join(f'{p} {n}' for p, n in sorted(prefixes.items()))})\n  {msg}")
print("\nwarnings (cases emitting each):")
for w, n in warn.most_common():
    print(f"{n:4}  {w}")
