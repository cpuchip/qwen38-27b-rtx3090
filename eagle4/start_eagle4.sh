#!/bin/bash
# start_eagle4.sh - serve Qwen3.8-27B W4A16 with an EAGLE-4 draft on one 24 GB card (experimental).
#
# Runs eagle-hq's published vLLM image (their vLLM with the speculator hooks, plus eagle-harness) and mounts
# patched copies of six files over it:
#   vllm   eagle/utils.py      a quantized lm_head is never "equal" to the draft's (patches/vllm/, upstream PR)
#   vllm   qwen3_5.py, _mtp    the int8 token embedding (../patches/qwen3_5-embed-quant.patch, as the main launchers)
#   eagle  models.py, gather_dot.py, model_rows.py   the int8 draft lm_head (eagle_harness/, Eagle-licensed)
# The originals come out of the image on each run (docker cp), so this script never ships Eagle's files.
#
#   MODELS_DIR=/path/to/models bash eagle4/start_eagle4.sh
#
# Inputs, under MODELS_DIR: the target (MODEL, default Qwen3.8-27B-W4A16-AutoRound) and the draft made by
# prepare_draft.py (DRAFT, default Qwen3.8-27B-EAGLE4-24g). README.md in this directory has the numbers.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

IMAGE=${IMAGE:-ghcr.io/eagle-hq/eagle-4-vllm@sha256:93cc820f7ace448804ae4732bbe630b8ced6eac6784286c7c190e01634dd78cc}
: "${MODELS_DIR:?set MODELS_DIR to the directory holding the target and the draft}"
MODEL=${MODEL:-Qwen3.8-27B-W4A16-AutoRound}
DRAFT=${DRAFT:-Qwen3.8-27B-EAGLE4-24g}
PORT=${PORT:-8000}
NAME=${NAME:-qwen-eagle4}
MAX_LEN=${MAX_LEN:-16384}
MAX_SEQS=${MAX_SEQS:-1}
GPU_UTIL=${GPU_UTIL:-0.93}
GPU=${GPU:-0}
LM_HEAD_INT8=${LM_HEAD_INT8:-1}
# their single-GPU vLLM shape: static tree, top-k 4, depth 14, verify width 16
TREE=${TREE:-'{"mode":"static","node_top_k":4,"tree_depth":14,"max_verify_width":16}'}
STATE=${STATE:-$HERE/.mounts}

# Git Bash on Windows: keep MSYS from rewriting container paths, and hand docker Windows-form host paths.
if command -v cygpath >/dev/null 2>&1; then
  export MSYS_NO_PATHCONV=1
  hostpath() { cygpath -m "$1"; }
else
  hostpath() { printf '%s' "$1"; }
fi

for f in "$MODELS_DIR/$MODEL/config.json" "$MODELS_DIR/$DRAFT/model.safetensors"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

SP=/opt/eagle/lib/python3.12/site-packages
FILES=(
  vllm/v1/worker/gpu/spec_decode/eagle/utils.py
  vllm/model_executor/models/qwen3_5.py
  vllm/model_executor/models/qwen3_5_mtp.py
  eagle_harness/vllm/models.py
  eagle_harness/kernels/gather_dot.py
  eagle_harness/vllm/model_rows.py
)

# Fresh copies of the originals from this exact image, then the patches. A patch that does not apply stops here.
rm -rf "$STATE" && mkdir -p "$STATE"
cid=$(docker create "$IMAGE")
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
for f in "${FILES[@]}"; do
  mkdir -p "$STATE/$(dirname "$f")"
  docker cp "$cid:$SP/$f" "$(hostpath "$STATE/$f")" >/dev/null
done
docker rm -f "$cid" >/dev/null; trap - EXIT
patch -s --no-backup-if-mismatch -p1 -d "$STATE" < "$HERE/patches/vllm/eagle-share-quantized-head.diff"
patch -s --no-backup-if-mismatch -p1 -d "$STATE/vllm" < "$REPO/patches/qwen3_5-embed-quant.patch"
PLUGIN=()
if [ "$LM_HEAD_INT8" = 1 ]; then
  for d in models gather_dot model_rows; do patch -s --no-backup-if-mismatch -p1 -d "$STATE" < "$HERE/eagle_harness/$d.diff"; done
  PLUGIN=(-e EAGLE4_LM_HEAD_INT8=1)
else
  patch -s --no-backup-if-mismatch -p1 -d "$STATE" < "$HERE/eagle_harness/model_rows.diff"
fi

if [ "${PREPARE_ONLY:-0}" = 1 ]; then echo "patched copies ready in $STATE"; exit 0; fi

MOUNTS=()
for f in "${FILES[@]}"; do MOUNTS+=(-v "$(hostpath "$STATE/$f"):$SP/$f:ro"); done

KEY=()
if [ -n "${VLLM_API_KEY:-}" ]; then KEY=(-e VLLM_API_KEY); fi

SPEC=$(printf '{"method":"eagle_harness","model":"/models/%s","method_config":%s}' "$DRAFT" "$TREE")

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "device=$GPU" --ipc host -p "127.0.0.1:$PORT:8000" \
  -v "$(hostpath "$MODELS_DIR"):/models:ro" "${MOUNTS[@]}" "${PLUGIN[@]}" "${KEY[@]}" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=4 \
  "$IMAGE" "/models/$MODEL" --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_UTIL" \
  --no-enable-prefix-caching --reasoning-parser qwen3 --language-model-only \
  --kv-cache-dtype auto --mamba-ssm-cache-dtype float16 \
  --attention-backend CUSTOM --speculative-config "$SPEC" >/dev/null
echo "started $NAME on 127.0.0.1:$PORT (logs: docker logs -f $NAME)"
