#!/bin/bash
# drive.sh IMAGE TAG MODELS_DIR [OVERLAY_DIR]: the launcher-matrix gate (README.md beside this file).
# Copies this folder to $WORK (default ~/launcher-matrix), writes cases.txt, runs every case in one container
# and scores it into $WORK/out-TAG/score.txt. OVERLAY_DIR, if given, is copied over /app first (launcher files
# from another branch). The card is visible for platform detection only; a watchdog kills the container if card
# memory passes MATRIX_VRAM_CAP MiB (default 2000), since nothing here should allocate.
set -u
IMG=$1; TAG=$2; MODELS=$3; OVL=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/launcher-matrix}
mkdir -p "$WORK" && cp "$HERE"/*.py "$HERE"/*.sh "$WORK"/ && chmod +x "$WORK"/*.sh
cd "$WORK" || exit 1
python3 cases.py > cases.txt
rm -rf "out-$TAG"; mkdir "out-$TAG"
# out must be the symlink: a real directory left there would take every result and mix two runs.
if [ -e out ] && [ ! -L out ]; then echo "drive.sh: $WORK/out is not a symlink; move it aside" >&2; exit 1; fi
rm -f out; ln -s "out-$TAG" out
(
  while sleep 5; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "$(date -u +%T) $v" >> out/vram.log
    if [ "$v" -gt "${MATRIX_VRAM_CAP:-2000}" ]; then
      echo "WATCHDOG kill at $v MiB" >> out/vram.log; docker kill launcher-matrix >/dev/null 2>&1
    fi
  done
) &
WD=$!
OV=()
[ -n "$OVL" ] && OV=(-v "$OVL:/ovl:ro")
docker run --rm --name launcher-matrix --gpus all --entrypoint bash -e MATRIX_JOBS="${MATRIX_JOBS:-16}" \
  -v "$MODELS:/app/models:ro" -v "$WORK:/m" "${OV[@]}" "$IMG" /m/inner.sh > out/drive.log 2>&1
echo "docker_rc=$?" >> out/drive.log
kill $WD
python3 score.py out cases.txt > out/score.txt
echo "@@ MATRIX DONE $TAG $(date -u +%T)" >> out/drive.log
sed -n '/^totals/p' out/score.txt
