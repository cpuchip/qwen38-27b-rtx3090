#!/usr/bin/env python3
"""Refresh docs/upstream-tracker.md from GitHub.

Answers one question per thread: is the ball in our court? A thread needs a reply
when the last comment is not ours and lands after our last one. Everything else is
either waiting on them or done.

    python scripts/track_upstream.py            # rewrite the tracker
    python scripts/track_upstream.py --check    # exit 1 if anything needs a reply
"""
import json, subprocess, sys, datetime, os

REPO = "syv-ai/qwen38-27b-rtx3090"
US = "cpuchip"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "upstream-tracker.md")


def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else ""


def threads():
    """Every issue or PR we have touched or that names us."""
    seen = {}
    for kind in ("issue", "pr"):
        raw = gh(kind, "list", "--repo", REPO, "--state", "all", "--limit", "100",
                 "--json", "number,title,state,updatedAt,author")
        for it in json.loads(raw or "[]"):
            seen[it["number"]] = {**it, "kind": kind}
    raw = gh("search", "issues", "--repo", REPO, US, "--limit", "60",
             "--json", "number,title,state,updatedAt")
    for it in json.loads(raw or "[]"):
        seen.setdefault(it["number"], {**it, "kind": "issue", "author": {"login": "?"}})
    return seen


def detail(num, kind):
    raw = gh(kind, "view", str(num), "--repo", REPO, "--json", "number,title,state,updatedAt,comments,author")
    return json.loads(raw) if raw else None


def main():
    check = "--check" in sys.argv
    rows = []
    for num, meta in sorted(threads().items()):
        d = detail(num, meta["kind"])
        if not d:
            continue
        cs = d.get("comments", [])
        ours = [c for c in cs if c["author"]["login"] == US]
        mine_authored = d.get("author", {}).get("login") == US
        if not ours and not mine_authored:
            continue                      # we were never involved
        last_ours = ours[-1]["createdAt"] if ours else ""
        after = [c for c in cs if c["createdAt"] > last_ours and c["author"]["login"] != US]
        rows.append({
            "num": num, "kind": meta["kind"], "state": d["state"], "title": d["title"],
            "updated": d["updatedAt"][:16], "comments": len(cs), "ours": len(ours),
            "after": [(c["author"]["login"], c["createdAt"][:16],
                       " ".join(c["body"].split())[:220]) for c in after],
        })
    needs = [r for r in rows if r["after"] and r["state"] in ("OPEN", "open")]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    if check:
        for r in needs:
            print(f"NEEDS REPLY #{r['num']}: {r['title'][:60]}")
        sys.exit(1 if needs else 0)

    L = [f"# Upstream tracker: {REPO}", "",
         f"Generated {stamp} by `scripts/track_upstream.py`. Regenerate rather than hand-edit;",
         "the notes section below the table is the part that is written by hand.", "",
         "## Ball in our court", ""]
    if needs:
        for r in needs:
            L.append(f"- **#{r['num']} {r['title'][:70]}**")
            for who, when, body in r["after"]:
                L.append(f"  - {who}, {when}: {body}")
    else:
        L.append("Nothing open is waiting on a reply from us.")
    L += ["", "## Every thread we are on", "",
          "| # | kind | state | ours | unanswered | updated | title |",
          "|---|---|---|---:|---:|---|---|"]
    for r in sorted(rows, key=lambda x: x["updated"], reverse=True):
        L.append(f"| {r['num']} | {r['kind']} | {r['state'].lower()} | {r['ours']} | "
                 f"{len(r['after'])} | {r['updated']} | {r['title'][:58]} |")
    L.append("")
    body = "\n".join(L)
    prev = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
    marker = "<!-- HAND-WRITTEN BELOW -->"
    tail = prev.split(marker, 1)[1] if marker in prev else "\n\n## Notes\n\n(none yet)\n"
    open(OUT, "w", encoding="utf-8", newline="\n").write(body + marker + tail)
    print(f"wrote {OUT}: {len(rows)} threads, {len(needs)} needing a reply")


if __name__ == "__main__":
    main()
