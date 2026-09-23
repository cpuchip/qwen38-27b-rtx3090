#!/usr/bin/env python3
"""Move every mechanical vLLM pin in this repo from one release to the next, in one pass.

    python scripts/pin-bump.py --from 0.29.0 --to 0.30.0 --fork ../../vllm.git [--repo .] [--dry-run]

What it changes (and only this; narrative docs that describe a past pin are left for a person):
  - docker/requirements.txt  vllm==<old>          -> vllm==<new>
  - verify.sh                the version check, the kvarn patch names
  - Dockerfile               the header comment's "vLLM <old>"
  - kvarn/install.sh         kvarn-<old>.patch / kvarn-v2-runner-<old>.patch and the "vLLM <old> venv" comment
  - docs/install.md          flashinfer-cubin==<x>, read from vLLM's own requirements/cuda.txt at the NEW tag
    (the cubin package is not on PyPI and must match flashinfer-python exactly; 0.30.0 moved it to 0.6.18.post1)
Every file it touches is listed with its line count changed; anything it expected and did not find is an error.
"""
import argparse, re, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def flashinfer_at(fork, tag):
    txt = subprocess.run(["git", "-C", fork, "show", f"{tag}:requirements/cuda.txt"], capture_output=True, text=True,
                         encoding="utf-8").stdout
    m = re.search(r"^flashinfer-cubin==(\S+)", txt, re.M)
    if not m:
        sys.exit(f"no flashinfer-cubin pin in {tag}:requirements/cuda.txt")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--fork", required=True, help="vLLM fork repo (bare or checkout) with the release tags")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    old, new, root = a.old, a.new, Path(a.repo)
    fi_old, fi_new = flashinfer_at(a.fork, f"v{old}"), flashinfer_at(a.fork, f"v{new}")
    o = re.escape(old)
    edits = {
        "docker/requirements.txt": [(rf"^vllm=={o}(?=\r?$)", f"vllm=={new}"), (rf"# vllm=={o} ", f"# vllm=={new} ")],
        "verify.sh": [(rf'\[ "\$VER" = "{o}" \]', f'[ "$VER" = "{new}" ]'), (rf"written against {o}\)", f"written against {new})"),
                      (rf"kvarn-{o}\.patch", f"kvarn-{new}.patch"), (rf"kvarn-v2-runner-{o}\.patch", f"kvarn-v2-runner-{new}.patch")],
        "Dockerfile": [(rf"# vLLM {o} ", f"# vLLM {new} ")],
        "kvarn/install.sh": [(rf"kvarn-{o}\.patch", f"kvarn-{new}.patch"), (rf"kvarn-v2-runner-{o}\.patch", f"kvarn-v2-runner-{new}.patch"),
                             (rf"vLLM {o} venv", f"vLLM {new} venv")],
        "docs/install.md": [(rf"flashinfer-cubin=={re.escape(fi_old)}\b", f"flashinfer-cubin=={fi_new}")],
    }
    errors = 0
    for rel, subs in edits.items():
        p = root / rel
        s = p.read_bytes().decode("utf-8")  # bytes, not read_text: keep each file's own line endings
        before, changed = s, 0
        for pat, rep in subs:
            s, n = re.subn(pat, rep, s, flags=re.M)
            if n == 0:
                print(f"  MISSING  {rel}: /{pat}/"); errors += 1
            changed += n
        if s != before and not a.dry_run:
            p.write_bytes(s.encode("utf-8"))
        print(f"  {'would edit' if a.dry_run else 'edited'} {rel}: {changed} replacement(s)")
    print(f"pin-bump {old} -> {new}; flashinfer-cubin {fi_old} -> {fi_new}; {errors} expected pattern(s) not found")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
