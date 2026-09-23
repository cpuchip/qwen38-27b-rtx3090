#!/usr/bin/env python3
"""The PR gate: the mechanical half of docs/pr-standard.md.

    python scripts/pr-gate/pr_gate.py --pr 100 [--repo syv-ai/qwen38-27b-rtx3090] [--base syv-main]
    python scripts/pr-gate/pr_gate.py --branch offload-mtp-serve --body body.md [--title "..."]
    add --pristine /path/to/vllm-<pin> to run patches/check_vllm_series.sh at the head as well

The diff is read from the local git repository this script lives in (a worktree of the ops repo, or pass
--git <path> to a checkout or bare repo). The body comes from GitHub (--pr) or a file (--body). Exit code is
1 on any FAIL. WARNs are for a person to look at; they do not block.
"""
import argparse, json, os, re, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHERS = ["single-user/start_qwen.sh", "batch/start_qwen.sh", "single-user/alternative.sh"]
CONFIG_TOKENS = re.compile(r"(k=\d|CTX=|MAX_LEN|3090|4090|5060|WSL2|native|n=\d|DFLASH_TOKENS|SPEC=|PREFIX_CACHE|INT8_ACT|q=\d|kv=\d|box|card|arm)", re.I)
NUMBER_LINE = re.compile(r"(\d+(\.\d+)?\s*(tok/s|%|ms|s\b|GiB|GB|x\b))")
HW_WORDS = re.compile(r"(fp8|e4m3|sm89|sm90|sm86|capability|cuda\s*graph|flashinfer)", re.I)
GUARD_WORDS = re.compile(r"(get_device_capability|SKIP|skip\(|pytest\.skip|by design)", re.I)
# Any spelling of a raw read: os.environ.get / os.getenv / os.environ[...], an aliased `_os`, and
# `__import__("os").environ.get`; the 09-14 census missed the last two and left eight raw reads in the 0.29 tree.
ENV_READ = re.compile(r"(?:os|__import__\(\s*[\"']os[\"']\s*\))\.(?:environ\.get|getenv|environ)\s*[\(\[]\s*[\"'](VLLM_[A-Z0-9_]+)")
results = []


def report(level, name, detail=""):
    results.append((level, name, detail))


def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def git(args, gitdir):
    return run(["git", "-C", gitdir] + args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", type=int)
    ap.add_argument("--repo", default="syv-ai/qwen38-27b-rtx3090")
    ap.add_argument("--branch")
    ap.add_argument("--body")
    ap.add_argument("--title", default="")
    ap.add_argument("--base", default="syv-main")
    ap.add_argument("--git", default=os.path.abspath(os.path.join(HERE, "..", "..")))
    ap.add_argument("--pristine")
    ap.add_argument("--local-head", action="store_true", help="with --pr: gate the local branch head even though it differs from the PR head (the pre-push case)")
    a = ap.parse_args()

    if a.pr:
        pr = json.loads(run(["gh", "pr", "view", str(a.pr), "-R", a.repo, "--json", "title,body,headRefName,headRefOid,commits"]))
        title, body, branch = pr["title"], pr["body"], a.branch or pr["headRefName"]
        # A --body file is the text about to replace the live one, so it is what gets gated. Until 2026-09-23 --pr
        # silently gated the LIVE body and ignored --body, so every "gate the new body" run checked the old text.
        if a.body:
            body = open(a.body, encoding="utf-8").read()
        lead_commit = pr["commits"][-1]["messageHeadline"] + "\n" + pr["commits"][-1].get("messageBody", "")
        # The diff is read from the local branch. If that is not the PR's head, every objective check below
        # (counts, series, env registration) describes a commit the reviewer will never see; say so and stop
        # (2026-09-13: a worktree one commit behind reported 6 hunks for a 7-hunk head, green).
        local = git(["rev-parse", branch], a.git).strip()
        if local != pr["headRefOid"] and not a.local_head:
            raise SystemExit(f"FAIL stale-diff: local {branch} is at {local[:7]}, PR #{a.pr} head is {pr['headRefOid'][:7]}. "
                             "Fetch or check out the PR head, or pass --local-head to gate a local head that is meant to replace it.")
    else:
        if not (a.branch and a.body):
            raise SystemExit("need --pr, or --branch and --body")
        title, branch = a.title, a.branch
        body = open(a.body, encoding="utf-8").read()
        lead_commit = git(["log", "-1", "--format=%s%n%b", branch], a.git)
    base = git(["merge-base", a.base, branch], a.git).strip()
    files = git(["diff", "--name-only", base, branch], a.git).split()
    added_lines = {}
    # For a patch file, its own section headers ("diff --git a/envs.py ...") are usually unchanged context in
    # this diff when the file was modified, not created; check 3 needs them to know which section an added line
    # is in, so patch files keep their headers (as context) beside the added lines.
    patch_sections = {}
    for f in files:
        d = git(["diff", base, branch, "--", f], a.git)
        added_lines[f] = [l[1:] for l in d.splitlines() if l.startswith("+") and not l.startswith("+++")]
        if f.endswith(".patch"):
            full = git(["diff", "--unified=1000000", base, branch, "--", f], a.git).splitlines()
            body_start = next((i for i, l in enumerate(full) if l.startswith("@@")), len(full)) + 1
            patch_sections[f] = [(l[0], l[1:]) for l in full[body_start:] if l[:1] in "+ "]
    head_tree = lambda path: run(["git", "-C", a.git, "show", f"{branch}:{path}"], check=False)
    text_all = body + "\n" + lead_commit

    # 1. counts in body/lead commit vs the patch files in the diff
    patch_files = [f for f in files if f.startswith("patches/") and f.endswith(".patch")]
    claimed = re.findall(r"\b(\w+|\d+)\s+hunks?\b(?!-)", text_all, re.I)  # "hunk-body" is not a count
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    claimed_n = sorted({words.get(c.lower(), int(c) if c.isdigit() else None) for c in claimed} - {None})
    if patch_files:
        actual = {f: head_tree(f).count("\n@@ ") + (1 if head_tree(f).startswith("@@ ") else 0) for f in patch_files}
        if claimed_n and not any(n in actual.values() for n in claimed_n):
            report("FAIL", "counts", f"body/commit say {claimed_n} hunks; the patch files have {actual}")
        else:
            report("OK", "counts", f"hunks in diff: {actual}" + (f"; claimed {claimed_n}" if claimed_n else "; no hunk count claimed"))
    else:
        report("OK", "counts", "no patch files in the diff")

    # 2. series line for every patch file in the diff
    series = head_tree("patches/series")
    if patch_files:
        if not series.strip():
            report("FAIL", "series", "patches/series is absent at the head; main applies from it since #54 (rebase, then add the lines)")
        else:
            listed = {l.split("#")[0].strip() for l in series.splitlines()}
            missing = [os.path.basename(f) for f in patch_files if os.path.basename(f) not in listed and head_tree(f)]
            report("FAIL" if missing else "OK", "series", f"not in patches/series: {missing}" if missing else "every patch file in the diff has a series line")

    # 3. env knobs read but not registered
    # Only reads that land in vLLM code (patch files, kvarn/) count: a bench or launcher reading an env var is
    # not a kernel knob outside the torch.compile cache key.
    # The registry itself (envs.py hunks) is the one place a raw read belongs, so a patch's envs.py section is
    # skipped: track the patch's own "diff --git" / "--- a/" headers, added or context, while walking it.
    reads = set()
    for f, lines in added_lines.items():
        if not (f.endswith(".patch") or f.startswith("kvarn/files/")):
            continue
        in_envs = f.endswith("envs.py")
        for sign, l in patch_sections.get(f, [("+", l) for l in lines]):
            if l.startswith("diff --git ") or l.startswith("--- a/") or l.startswith("--- /dev/null"):
                in_envs = l.rstrip().endswith("envs.py")
                continue
            if in_envs or sign != "+":
                continue
            for m in ENV_READ.finditer(l):
                reads.add(m.group(1))
    # Bar item 7 says never read a knob with os.environ inside vLLM code, registered or not: the registered
    # bool and a raw string read disagree at "0" (the string is truthy), so a knob read both ways has two
    # senses (review of #90, 2026-09-14). Any such read fails; the fix is `envs.<NAME>`.
    if reads:
        all_patches = git(["ls-tree", "-r", "--name-only", branch, "--", "patches"], a.git).split()
        envs_text = "\n".join(head_tree(p) for p in all_patches if p.endswith(".patch"))
        envs_text += head_tree("kvarn/kvarn-0.29.0.patch") + head_tree("kvarn/kvarn-0.28.0.patch")
        unregistered = sorted(k for k in reads if not re.search(rf"^\+.*[\"']{k}[\"']\s*:\s*lambda|^\+\s*{k}\s*:", envs_text, re.M))
        report("FAIL", "env-registration",
               f"knobs read with os.environ in vLLM code (read them through envs.<NAME>): {sorted(reads)}"
               + (f"; also unregistered: {unregistered}" if unregistered else ""))
    else:
        report("OK", "env-registration", "no os.environ reads of VLLM_ knobs in the diff")

    # 4. retracted phrases
    phrases = []
    for l in open(os.path.join(HERE, "retracted.txt"), encoding="utf-8"):
        if l.strip() and not l.startswith("#"):
            phrases.append(l.split("\t")[0].strip())
    hits = []
    for ph in phrases:
        if ph.lower() in text_all.lower():
            hits.append(f"body/commit: '{ph}'")
        for f, lines in added_lines.items():
            if any(ph.lower() in l.lower() for l in lines):
                hits.append(f"{f}: '{ph}'")
    report("FAIL" if hits else "OK", "retracted", "; ".join(hits) if hits else f"{len(phrases)} retracted phrases absent")

    # 5. bench guards
    for f in files:
        if f.startswith("bench/test_") and f.endswith(".py"):
            src = head_tree(f)
            if HW_WORDS.search(src) and not GUARD_WORDS.search(src):
                report("FAIL", "bench-guard", f"{f} needs hardware it does not guard for (no capability check or skip)")
            elif HW_WORDS.search(src):
                report("OK", "bench-guard", f"{f} guards its hardware need")

    # 6. sibling launchers
    touched = [f for f in files if f in LAUNCHERS]
    for f in touched:
        keys = [l.strip() for l in added_lines[f] if len(l.strip()) > 30 and re.search(r"(export |echo |ALLOC_DEFAULT|EXTRA_ARGS)", l)]
        for sib in LAUNCHERS:
            if sib == f:
                continue
            sib_src = head_tree(sib)
            missing = [k for k in keys if k not in sib_src]
            if missing:
                report("WARN", "siblings", f"{f} gained {len(keys)} launcher line(s); {len(missing)} absent from {sib}, e.g. '{missing[0][:70]}'")
    if not touched:
        report("OK", "siblings", "no launcher touched")

    # 7. promised docs
    if re.search(r"gotcha", text_all, re.I) and "docs/gotchas.md" not in files:
        report("FAIL", "promised-docs", "the body or commit mentions a gotcha and docs/gotchas.md is not in the diff")
    for doc in set(re.findall(r"docs/[\w.-]+\.md", text_all)):
        if doc not in files:
            report("WARN", "promised-docs", f"{doc} is named in the body and not in the diff (fine if it is only cited)")

    # 8. unproven section, numbers with configuration, voice
    if not re.search(r"(unproven|not measured|did not (measure|run|test)|not (yet )?verified|what I did not)", body, re.I):
        report("WARN", "unproven-section", "the body has no 'what is still unproven' statement")
    # A table row inherits its configuration from the header or the sentence above the table.
    bare = []
    lines = body.splitlines()
    for i, l in enumerate(lines):
        if not NUMBER_LINE.search(l) or CONFIG_TOKENS.search(l):
            continue
        context = " ".join(lines[max(0, i - 8):i]) if l.lstrip().startswith("|") else (lines[i - 1] if i else "")
        if not CONFIG_TOKENS.search(context):
            bare.append(l.strip()[:80])
    if bare:
        report("WARN", "numbers-config", f"{len(bare)} line(s) carry a measurement with no configuration token, e.g. '{bare[0]}'")
    if "—" in body:
        report("FAIL", "voice", f"{body.count(chr(8212))} em-dash(es) in the body")
    if re.search(r"claude\.ai/code|Co-Authored-By|Claude-Session", text_all, re.I):
        report("FAIL", "voice", "attribution line or session URL present")
    if title:
        tw = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", title)}
        cw = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", lead_commit.splitlines()[0])}
        if len(tw & cw) < 2:
            report("WARN", "title-commit", "title and lead commit subject share fewer than two words")

    # 9. integrity at the head
    if a.pristine:
        tmp = run(["git", "-C", a.git, "worktree", "add", "--detach", os.path.join(os.path.dirname(a.pristine), "gate-head"), branch], check=False)
        wt = os.path.join(os.path.dirname(a.pristine), "gate-head")
        p = subprocess.run(["bash", "patches/check_vllm_series.sh", os.path.join(a.pristine, "vllm")], cwd=wt, capture_output=True, text=True)
        report("OK" if p.returncode == 0 else "FAIL", "integrity", (p.stdout.strip().splitlines() or ["(no output)"])[-1])
        run(["git", "-C", a.git, "worktree", "remove", "--force", wt], check=False)

    width = max(len(n) for _, n, _ in results)
    fails = 0
    for level, name, detail in results:
        fails += level == "FAIL"
        print(f"  {level:4} {name:{width}}  {detail}")
    print("\nManual, before the push (docs/pr-standard.md items 2, 3, 10):")
    print("  - verified on the box: which card, which configuration, which command; numbers with their conditions")
    print("  - 'What is still unproven' written, and every claim narrower than its measurement")
    print("  - credit named; asks specific; title, body, lead commit and diff re-read as one thing after the last commit")
    print(f"\npr-gate: {fails} FAIL, {sum(1 for l, _, _ in results if l == 'WARN')} WARN")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
