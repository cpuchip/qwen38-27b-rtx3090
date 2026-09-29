#!/usr/bin/env bash
# Self-test for up_gate.py: one throwaway commit per case on the fork's upstream/main (commit-tree, no worktree),
# a temporary up/gate-selftest-* ref for each, the gate run against it, the ref deleted. Nothing is pushed.
#   bash scripts/pr-gate/test_up_gate.sh [--fork <vllm checkout or bare repo>]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK="$(cd "$HERE/../../../../vllm.git" 2>/dev/null && pwd)"; [ "${1:-}" = --fork ] && FORK="$2"
G="git -C $FORK"; BASE=$($G rev-parse upstream/main); TREE=$($G rev-parse "upstream/main^{tree}")
T=$(mktemp -d); REFS=(); trap 'for r in "${REFS[@]}"; do $G update-ref -d "$r"; done; rm -rf "$T"' EXIT
ME="Michael Stufflebeam"; MAIL="cpuchip@gmail.com"; SOB="Signed-off-by: $ME <$MAIL>"; CO="Co-authored-by: Claude <noreply@anthropic.com>"
printf 'A fix.\n\nThis PR is AI-assisted: Claude wrote the first draft, and I reviewed and tested every line.\n' > "$T/good.md"
printf 'A fix.\n\nTested on an RTX 3090.\n' > "$T/silent.md"
printf 'A fix, AI-assisted.\n\nhttps://claude.ai/code/session_x\n' > "$T/session.md"
FAILS=0; n=0
# case <want exit> <check that must FAIL, or -> <label> <author name> <author email> <body file> <message> [gate flags...]
case_() { local want=$1 check=$2 label=$3 an=$4 ae=$5 bodyf=$6 msg=$7; shift 7; n=$((n+1))
  local c; c=$(printf '%s' "$msg" | GIT_AUTHOR_NAME="$an" GIT_AUTHOR_EMAIL="$ae" GIT_COMMITTER_NAME="$ME" GIT_COMMITTER_EMAIL="$MAIL" $G commit-tree "$TREE" -p "$BASE")
  local ref="refs/heads/up/gate-selftest-$n"; $G update-ref "$ref" "$c"; REFS+=("$ref")
  python "$HERE/up_gate.py" --fork "$FORK" --branch "up/gate-selftest-$n" --body "$bodyf" "$@" > "$T/out" 2>&1; local got=$?
  local hit=yes; [ "$check" != - ] && ! grep -qE "^  FAIL $check " "$T/out" && hit=no
  if [ "$got" = "$want" ] && [ $hit = yes ]; then echo "  PASS  $label (exit $got$( [ "$check" != - ] && echo ", FAIL on $check"))"; else echo "  FAIL  $label (exit $got, want $want; FAIL on '$check': $hit)"; sed 's/^/        /' "$T/out"; FAILS=$((FAILS+1)); fi
}
case_ 0 - "sign-off + Co-authored-by + disclosure"        "$ME" "$MAIL" "$T/good.md"    "$(printf '[Bugfix] x\n\nWhy.\n\n%s\n%s\n' "$CO" "$SOB")"
case_ 1 dco "no trailers at all"                             "$ME" "$MAIL" "$T/good.md"    "$(printf '[Bugfix] x\n\nWhy.\n')"
case_ 1 co-author "the harness's model-name co-author form"        "$ME" "$MAIL" "$T/good.md"    "$(printf '[Bugfix] x\n\nWhy.\n\nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n%s\n' "$SOB")"
case_ 1 subject "a [qwen38] subject"                             "$ME" "$MAIL" "$T/good.md"    "$(printf '[qwen38] x\n\nWhy.\n\n%s\n%s\n' "$CO" "$SOB")"
case_ 1 disclosure "no disclosure in the body"                      "$ME" "$MAIL" "$T/silent.md"  "$(printf '[Bugfix] x\n\nWhy.\n\n%s\n%s\n' "$CO" "$SOB")"
case_ 1 dco "signed off by someone other than the author"   "Mads Henrichsen" "mads@syv.ai" "$T/good.md" "$(printf '[Bugfix] x\n\nWhy.\n\n%s\n%s\n' "$CO" "$SOB")"
case_ 1 session "a session URL in the body"                      "$ME" "$MAIL" "$T/session.md" "$(printf '[Bugfix] x\n\nWhy.\n\n%s\n%s\n' "$CO" "$SOB")"
case_ 0 - "--no-ai: sign-off only, no disclosure"          "$ME" "$MAIL" "$T/silent.md"  "$(printf '[Bugfix] x\n\nWhy.\n\n%s\n' "$SOB")" --no-ai
echo; [ $FAILS = 0 ] && echo "up_gate self-test: all $n cases as expected" || { echo "up_gate self-test: $FAILS of $n cases wrong"; exit 1; }
