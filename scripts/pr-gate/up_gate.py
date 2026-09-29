#!/usr/bin/env python3
"""The up/ gate: vLLM's contribution rules for a PR from cpuchip/vllm up/<topic> to vllm-project/vllm.

    python scripts/pr-gate/up_gate.py --branch up/<topic> --body body.md [--fork <vllm checkout or bare repo>]
    add --no-ai when no AI produced or ported any of the content (the co-author and disclosure checks become WARNs)

The integration branch's native topics carry no trailers, because export-patch.sh makes each one's message a syv-ai
patch header (option A, Michael, 2026-09-29). vLLM's attribution goes on the up/ commit instead, and this checks it is
there: vLLM's docs/contributing/README.md, "AI Assisted Contributions", asks to disclose AI help in the PR and to mark
commits with a trailer such as Co-authored-by, and its DCO check wants every commit signed off by its author.
bash scripts/pr-gate/test_up_gate.sh exercises every check. Exit code is 1 on any FAIL.
"""
import argparse, os, re, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CLAUDE_COAUTHOR = re.compile(r"^co-authored-by:\s*claude\b.*$", re.I | re.M)
# The harness writes "Claude <model> (1M context)"; the ruled form is "Co-authored-by: Claude <noreply@anthropic.com>".
MODEL_FORM = re.compile(r"^co-authored-by:\s*claude\s+(opus|sonnet|haiku|fable)\b|^co-authored-by:.*\(\d+[mk] context\)", re.I | re.M)
SESSION = re.compile(r"claude\.ai/code|Claude-Session", re.I)
DISCLOSURE = re.compile(r"\bAI[- ](assisted|generated)\b|\bassisted by (an )?AI\b|\bAI tools?\b|\bClaude\b|\bCopilot\b", re.I)
results = []


def report(level, name, detail=""):
    results.append((level, name, detail))


def git(args, gitdir):
    p = subprocess.run(["git", "-C", gitdir] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{p.stderr}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--fork", default=os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "vllm.git")))
    ap.add_argument("--base", default="upstream/main")
    ap.add_argument("--no-ai", action="store_true")
    a = ap.parse_args()
    body = open(a.body, encoding="utf-8").read()
    ai = "WARN" if a.no_ai else "FAIL"

    if not a.branch.startswith("up/"):
        report("WARN", "branch", f"{a.branch} is not an up/<topic> branch")
    base = git(["merge-base", a.base, a.branch], a.fork).strip()
    commits = git(["rev-list", "--reverse", f"{base}..{a.branch}"], a.fork).split()
    if not commits:
        report("FAIL", "commits", f"no commits on {a.branch} over {a.base}")
    for c in commits:
        short = c[:9]
        subject = git(["log", "-1", "--format=%s", c], a.fork).strip()
        msg = git(["log", "-1", "--format=%B", c], a.fork)
        name, email = git(["log", "-1", "--format=%an%n%ae", c], a.fork).strip().split("\n")
        signoffs = re.findall(r"^Signed-off-by:\s*.+?\s*<([^>]+)>\s*$", msg, re.M)
        if any(e.lower() == email.lower() for e in signoffs):
            report("OK", "dco", f"{short} signed off by its author <{email}>")
        else:
            report("FAIL", "dco", f"{short} has no Signed-off-by from its author {name} <{email}>")
        if MODEL_FORM.search(msg):
            report("FAIL", "co-author", f"{short}: '{MODEL_FORM.search(msg).group(0).strip()[:50]}' is the harness's "
                   "model-name form; write 'Co-authored-by: Claude <noreply@anthropic.com>'")
        elif CLAUDE_COAUTHOR.search(msg):
            report("OK", "co-author", f"{short}: {CLAUDE_COAUTHOR.search(msg).group(0).strip()}")
        else:
            report(ai, "co-author", f"{short} has no 'Co-authored-by: Claude' line"
                   + (" (--no-ai)" if a.no_ai else "; vLLM asks AI-assisted commits to carry one (--no-ai if none was)"))
        if SESSION.search(msg):
            report("FAIL", "session", f"{short}: session URL or Claude-Session line")
        if subject.startswith("[qwen38]"):
            report("FAIL", "subject", f"{short}: '[qwen38]' is the fork's integration-branch prefix, not a vLLM subject")

    if SESSION.search(body):
        report("FAIL", "session", "body: session URL or Claude-Session line")
    if DISCLOSURE.search(body):
        report("OK", "disclosure", f"the body says AI helped ('{DISCLOSURE.search(body).group(0)}')")
    else:
        report(ai, "disclosure", "the body has no note that AI helped; vLLM: 'Always mention when a pull request "
               "includes AI-generated code'" + (" (--no-ai)" if a.no_ai else ""))
    if "—" in body:
        report("FAIL", "voice", f"{body.count(chr(8212))} em-dash(es) in the body")

    width = max(len(n) for _, n, _ in results)
    fails = sum(1 for l, _, _ in results if l == "FAIL")
    for level, name, detail in results:
        print(f"  {level:4} {name:{width}}  {detail}")
    print(f"\nup-gate: {fails} FAIL, {sum(1 for l, _, _ in results if l == 'WARN')} WARN ({len(commits)} commit(s) on {a.branch})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
