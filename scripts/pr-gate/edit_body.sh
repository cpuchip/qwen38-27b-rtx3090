#!/usr/bin/env bash
# Update a PR body only if the gate passes against the PR's LIVE head, then read the body back.
# The gate runs first and the edit is chained on its exit code (2026-09-23: an unchained `gate; gh pr edit` updated
# #148's body while the gate was refusing a stale local head).
#   bash scripts/pr-gate/edit_body.sh <pr> <body.md> [--repo syv-ai/HyperQwen]
#   DRY=1 stops after the gate. Test this script ONLY with DRY=1: a test run against a live PR is a live edit
#   (2026-09-23: a falsification run of this script overwrote #148's body for a minute).
set -euo pipefail
PR=$1; BODY=$2; REPO=${4:-syv-ai/HyperQwen}; [ "${3:-}" = --repo ] || REPO=syv-ai/HyperQwen
HERE=$(cd "$(dirname "$0")" && pwd); GIT=$(cd "$HERE/../.." && pwd)
HEAD=$(gh pr view "$PR" -R "$REPO" --json headRefOid --jq .headRefOid)
REF=pr-gate-live-$PR; git -C "$GIT" fetch -q origin "$HEAD" 2>/dev/null || true
git -C "$GIT" branch -f "$REF" "$HEAD"; trap 'git -C "$GIT" branch -D "$REF" -q' EXIT
python "$HERE/pr_gate.py" --pr "$PR" --repo "$REPO" --branch "$REF" --body "$BODY" --git "$GIT"
[ "${DRY:-0}" = 1 ] && { echo "edit_body: DRY=1, gate passed against live head ${HEAD:0:9}; no edit made"; exit 0; }
gh pr edit "$PR" -R "$REPO" --body-file "$BODY" >/dev/null
n=$(gh pr view "$PR" -R "$REPO" --json body --jq .body | tr -d '\r' | diff - <(tr -d '\r' < "$BODY") | grep -c '^[<>] .' || true)
echo "edit_body: PR #$PR body updated against live head ${HEAD:0:9}; read-back differing lines: $n"
