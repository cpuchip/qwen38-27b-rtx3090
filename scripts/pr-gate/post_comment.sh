#!/usr/bin/env bash
# Post a comment on a PR or issue only if reply_gate.py passes, then read the posted body back and count the lines
# that differ WITH content (a trailing blank line that `--jq` adds is not a difference; 2026-09-24 a plain diff went
# red on every post for that reason, and a check that is always red gets read past).
#   bash scripts/pr-gate/post_comment.sh pr|issue <number> <body.md> [--repo syv-ai/HyperQwen]
#   DRY=1 stops after the gate. READBACK_URL=<comment url> skips the post and only runs the read-back against that
#   existing comment (the way to test this script without posting). Never test it against a live thread otherwise.
set -euo pipefail
KIND=$1; NUM=$2; BODY=$3; REPO=${5:-syv-ai/HyperQwen}; [ "${4:-}" = --repo ] || REPO=syv-ai/HyperQwen
case "$KIND" in pr|issue) ;; *) echo "post_comment: kind must be pr or issue"; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
readback() { # $1 = comment id
  local n
  n=$(gh api "repos/$REPO/issues/comments/$1" --jq .body | tr -d '\r' | diff - <(tr -d '\r' < "$BODY") | grep -c '^[<>] .' || true)
  echo "post_comment: read-back differing lines with content: $n"
  [ "$n" = 0 ]
}
if [ -n "${READBACK_URL:-}" ]; then readback "${READBACK_URL##*-}"; exit $?; fi
python "$HERE/reply_gate.py" "$BODY"
[ "${DRY:-0}" = 1 ] && { echo "post_comment: DRY=1, gate passed; nothing posted"; exit 0; }
URL=$(gh "$KIND" comment "$NUM" -R "$REPO" --body-file "$BODY")
echo "post_comment: $URL"
readback "${URL##*-}"
