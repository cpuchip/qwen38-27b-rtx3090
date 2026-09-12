#!/usr/bin/env bash
set -euo pipefail

# Validate the patch series against a pristine checkout of the pinned vLLM (docker/requirements.txt).
#
# Two passes, because two tools are in play and they answer different questions:
#
#   1. Every patch, in the glob order the Dockerfile and the README use, applied
#      with GNU `patch` -- the tool that actually installs this stack. This is
#      the pass that says "a clone of this repo still builds". It was missing
#      entirely until 2026-09-07; the job checked five of thirty files.
#   2. The five DFlash patches whose order and hunk metadata are part of the
#      0.28.0 contract, applied with `git apply`, which is strict about offsets
#      and catches a hand-edited hunk header immediately.
#
# Five patches are accepted by GNU patch and rejected by git apply once the
# patches before them have moved their context. That is why pass 1 uses GNU
# patch: the installed tree is produced by GNU patch, so GNU patch defines
# "applies", and git apply stays on the five where hunk metadata is contractual.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE=${1:?usage: bash patches/check_vllm_series.sh /path/to/vllm-<pinned tag>}
VLLM_SOURCE=$(cd -- "$VLLM_SOURCE" && pwd)

git -C "$VLLM_SOURCE" rev-parse --is-inside-work-tree >/dev/null

# The patches address files relative to the installed vllm package, not to the
# checkout root. `git apply` resolves a patch path against the REPOSITORY root
# and silently skips ("Skipped patch '...'") anything outside the subdirectory
# it runs in -- so running it from the package directory checked nothing at all
# and reported OK. Run from the repository root and name the prefix explicitly.
GIT_ROOT=$(git -C "$VLLM_SOURCE" rev-parse --show-toplevel)
PREFIX=${VLLM_SOURCE#"$GIT_ROOT"/}
if [ "$PREFIX" = "$VLLM_SOURCE" ]; then PREFIX=.; fi

# DFlash2 is native in 0.28.0; the backport patch is kept for older pins.
SKIP=(dflash2-backport.patch)

echo "== pass 1: the whole series, GNU patch, glob order"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
count=0; offset=0
for p in "$HERE"/patches/*.patch; do
  name=$(basename "$p")
  for s in "${SKIP[@]}"; do [ "$name" = "$s" ] && continue 2; done
  out=$(patch -p1 --forward --no-backup-if-mismatch -d "$VLLM_SOURCE" < "$p" 2>&1) || {
    echo "FAILED: $name"; echo "$out" | sed 's/^/    /'; exit 1
  }
  n=$(printf '%s\n' "$out" | grep -c "offset\|with fuzz" || true)
  [ "$n" -gt 0 ] && { echo "   $name (applied, $n hunk(s) with offset)"; offset=$((offset+1)); }
  count=$((count+1))
done
if git -C "$GIT_ROOT" diff --quiet; then
  echo "ERROR: the series applied but changed nothing -- the paths did not resolve." >&2
  exit 1
fi
echo "   $count patches applied, $offset with an offset"

echo "== pass 2: the ordered DFlash patches, git apply --check"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
PATCHES=(
  vllm-pr50021-gdn-spec-bounds.patch
  dflash2-lookup-drafting.patch
  dflash2-ngram-chains.patch
  dflash2-prewarm.patch
  dflash2-z-adaptive-emitted.patch
)
for name in "${PATCHES[@]}"; do
  patch="$HERE/patches/$name"
  echo "   git apply --check $name"
  git -C "$GIT_ROOT" apply --check --whitespace=error -p1 --directory="$PREFIX" < "$patch"
  git -C "$GIT_ROOT" apply --whitespace=error -p1 --directory="$PREFIX" < "$patch"
done
if git -C "$GIT_ROOT" diff --quiet; then
  echo "ERROR: pass 2 applied but changed nothing." >&2
  exit 1
fi

git -C "$GIT_ROOT" diff --check
echo "patch integrity: OK"
