#!/usr/bin/env bash
# The drift check for the fork workflow (docs/fork-workflow.md). Exit 0 = every invariant it can see holds.
#
#   bash scripts/series-check.sh [--fork <path to a cpuchip/vllm checkout or bare repo>] [--image <tag>] [--mirror]
#
# Checks (numbers as in docs/fork-workflow.md):
#   3  one pin: docker/requirements.txt vllm==, verify.sh pin, and the base tag of the fork commit in Dockerfile.fork agree
#   6  ledger: every PATCHES.md row names a fork commit whose subject is "[qwen38] <topic>", and vice versa
#   2  patches == branch: the patch files applied to the release tag reproduce the fork branch's vllm/ tree (needs --fork)
#   5  no mixed commits: subjects on the branch are "[qwen38] <topic>" (unique) or "fixup!" (needs --fork)
#   1  the image is the branch: an image's installed vllm .py tree == the branch tree, version stamp excepted (needs --fork and --image)
#   4  the mirror is a mirror: syv-main == syv-ai/main and main contains syv-main (--mirror; needs remotes origin + upstream)
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
FORK=""; IMAGE=""; MIRROR=0
while [ $# -gt 0 ]; do case "$1" in --fork) FORK="$2"; shift 2 ;; --image) IMAGE="$2"; shift 2 ;; --mirror) MIRROR=1; shift ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
fail=0; ok(){ echo "  OK    $*"; }; bad(){ echo "  DRIFT $*"; fail=$((fail+1)); }

echo "== 3: one pin"
PIN=$(grep -E '^vllm==' docker/requirements.txt | cut -d= -f3); VPIN=$(grep -oE 'vllm==[0-9.]+|PIN=[0-9.]+|"0\.[0-9]+\.[0-9]+"' verify.sh | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
COMMIT=$(grep -oE '^ARG VLLM_FORK_COMMIT=[0-9a-f]+' Dockerfile.fork | cut -d= -f2); FVER=$(grep -oE '^ARG VLLM_VERSION=[0-9.]+' Dockerfile.fork | cut -d= -f2)
[ -n "$PIN" ] && [ "$PIN" = "$FVER" ] && ok "requirements vllm==$PIN, Dockerfile.fork VLLM_VERSION=$FVER" || bad "requirements vllm==$PIN vs Dockerfile.fork VLLM_VERSION=$FVER"
[ -n "$VPIN" ] && [ "$VPIN" = "$PIN" ] && ok "verify.sh pin $VPIN" || bad "verify.sh pin '$VPIN' vs requirements $PIN"
[ -n "$COMMIT" ] && ok "Dockerfile.fork pins $COMMIT" || bad "Dockerfile.fork has no VLLM_FORK_COMMIT"

echo "== 6: ledger rows vs fork commits named in them"
ROWS=$(grep -E '^\| [a-z]' PATCHES.md | grep -v '^| patch' | awk -F'|' '{t=$2; gsub(/^ +| +$/,"",t); print t}' | sed 's|^kvarn/||')
NROWS=$(echo "$ROWS" | wc -l); ok "$NROWS rows in PATCHES.md"
KV="kvarn/kvarn-$PIN.patch kvarn/kvarn-v2-runner-$PIN.patch"   # older kvarn-*.patch files are history, not the series
for p in patches/*.patch $KV; do n=$(basename "$p" .patch); echo "$ROWS" | grep -qx "$n" || bad "patch file without a row: $n"; done
[ "$fail" = 0 ] && ok "every patch file has a row"

if [ -n "$FORK" ]; then
  G="git -C $FORK"
  $G rev-parse --verify -q "v$PIN" >/dev/null || $G fetch -q --tags upstream "v$PIN" 2>/dev/null || true
  BR=$($G branch -a --contains "$COMMIT" 2>/dev/null | grep -oE 'qwen38/[0-9.]+' | head -1)
  echo "== 3b: the pinned commit is on a qwen38/<pin> branch based on v$PIN"
  [ -n "$BR" ] && ok "$COMMIT is on $BR" || bad "$COMMIT is on no qwen38/ branch in $FORK (fetch origin?)"
  $G merge-base --is-ancestor "v$PIN" "$COMMIT" 2>/dev/null && ok "v$PIN is an ancestor of $COMMIT" || bad "v$PIN is not an ancestor of $COMMIT"
  echo "== 5: one topic per commit on $COMMIT"
  SUBJ=$($G log --format='%s' "v$PIN..$COMMIT")
  BADS=$(echo "$SUBJ" | grep -vE '^\[qwen38\] [A-Za-z0-9._-]+|^fixup! ' || true)
  [ -z "$BADS" ] && ok "$(echo "$SUBJ" | wc -l) commits, all '[qwen38] <topic>' or fixup!" || bad "off-pattern subjects: $(echo "$BADS" | head -3 | tr '\n' ';')"
  DUP=$(echo "$SUBJ" | grep -oE '^\[qwen38\] [A-Za-z0-9._-]+' | sort | uniq -d)
  [ -z "$DUP" ] && ok "topics unique" || bad "duplicate topics: $DUP"
  echo "== 6b: every topic on the branch has a row, every row's commit is on the branch"
  for t in $(echo "$SUBJ" | grep -oE '^\[qwen38\] [A-Za-z0-9._-]+' | sed 's/^\[qwen38\] //'); do
    case "$t" in kvarn-modules) continue ;; esac
    echo "$ROWS" | grep -qx "$t" || bad "topic on the branch without a PATCHES.md row: $t"
  done
  for h in $(grep -E '^\| [a-z]' PATCHES.md | grep -v '^| patch' | awk -F'|' '{print $(NF-1)}' | grep -oE '\b[0-9a-f]{7,}\b'); do
    $G merge-base --is-ancestor "$h" "$COMMIT" 2>/dev/null || bad "PATCHES.md names $h, not on $COMMIT"
  done
  ok "row/commit cross-check done"
  echo "== 2: patches applied to v$PIN reproduce the branch tree"
  TMP=$(mktemp -d); $G worktree add -q "$TMP/tag" "v$PIN" 2>/dev/null && {
    for p in patches/*.patch; do case "$p" in *dflash2-backport*) continue ;; esac; patch -p1 -N -s -r /dev/null -d "$TMP/tag/vllm" < "$p" >/dev/null 2>&1 || bad "patch does not apply to v$PIN: $(basename $p)"; done
    cp -r kvarn/files/vllm/. "$TMP/tag/vllm/" 2>/dev/null
    for p in $KV; do patch -p1 -N -s -r /dev/null -d "$TMP/tag/vllm" < "$p" >/dev/null 2>&1 || bad "kvarn patch does not apply to v$PIN: $(basename $p)"; done
    find "$TMP/tag/vllm" -name '*.orig' -delete
    D=$(cd "$TMP/tag" && git add -A >/dev/null 2>&1 && git diff --cached --stat "$COMMIT" -- vllm | tail -1)
    [ -z "$D" ] && ok "patches + kvarn on v$PIN == $COMMIT tree (vllm/)" || bad "patch view differs from the branch: $D"
    $G worktree remove --force "$TMP/tag" >/dev/null 2>&1; rm -rf "$TMP"
  }
  if [ -n "$IMAGE" ]; then
    echo "== 1: the image is the branch ($IMAGE)"
    C=qwen-series-check-$$; docker create --name "$C" "$IMAGE" >/dev/null && docker cp "$C:/app/venv/lib/python3.12/site-packages/vllm" "$TMP-img" >/dev/null 2>&1; docker rm "$C" >/dev/null 2>&1
    $G worktree add -q "$TMP-br" "$COMMIT" 2>/dev/null
    N=$(diff -rq -x __pycache__ -x '*.so' -x '*.pyc' -x _version.py -x third_party "$TMP-br/vllm" "$TMP-img" 2>/dev/null | grep -c '^Files .* differ$')
    ONLY=$(diff -rq -x __pycache__ -x '*.so' -x '*.pyc' -x _version.py -x third_party "$TMP-br/vllm" "$TMP-img" 2>/dev/null | grep -c '^Only in')
    [ "$N" = 0 ] && ok "image .py tree == branch tree (differing files: 0; only-in-one-side entries: $ONLY, expected wheel-generated)" || bad "image differs from the branch in $N files"
    $G worktree remove --force "$TMP-br" >/dev/null 2>&1; rm -rf "$TMP-img"
  fi
fi

if [ "$MIRROR" = 1 ]; then
  echo "== 4: syv-main mirrors syv-ai/main, main contains it"
  git fetch -q upstream 2>/dev/null; git fetch -q origin 2>/dev/null
  U=$(git rev-parse upstream/main 2>/dev/null); S=$(git rev-parse origin/syv-main 2>/dev/null)
  [ -n "$U" ] && [ "$U" = "$S" ] && ok "syv-main == syv-ai/main ($(git rev-parse --short "$U"))" || bad "syv-main ($(git rev-parse --short "${S:-0000000}" 2>/dev/null)) != syv-ai/main ($(git rev-parse --short "${U:-0000000}" 2>/dev/null)); fast-forward it"
  git merge-base --is-ancestor origin/syv-main origin/main 2>/dev/null && ok "main contains syv-main" || bad "main does not contain syv-main; merge it"
fi

echo; [ "$fail" = 0 ] && echo "series-check: OK (0 drift)" || echo "series-check: $fail drift item(s)"
exit $fail
