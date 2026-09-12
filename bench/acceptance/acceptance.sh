#!/usr/bin/env bash
# Acceptance run for one image of this repo: boot each profile in a container, probe it, record the facts.
#
#   IMAGE=qwen38-27b-rtx3090:port-0.29 TAG=029 bash bench/acceptance/acceptance.sh
#
# Environment (all optional except IMAGE and TAG):
#   PROFILES   which profiles to run, space separated: A B C D H P (default "A B C D")
#              A fast   dflash2 k=7, prefix caching: boot facts, quality battery, depth rows at 25k/47k tokens
#              B long   mtp, align + prefix caching: prefix ladder, peak pool usage at 60k/160k chars
#              C int4   alternative.sh dflash2, int4 per-token-head KV, 120k: boot facts, mq3d oracle, depth rows
#              D offload mtp long + 12 GiB CPU tier: serve oracle (TIER_GIB guard)
#              H huge   dflash2 + KVarN (CTX=huge): quality, needle at 32k/90k/200k, first-delta probe, depth rows
#              P probe  fast profile, one fixed prompt five times, spec-decode counters per request
#   GPU        device for --gpus (index or GPU-UUID; default 0)          ACC_GPU_SMI  nvidia-smi index for the GPU lines (default 0)
#   MODELS     host path mounted at /app/models (default ./models)        QDATA        quality-battery data dir (default ./bench/quality-data)
#   ENVFILE    docker --env-file with VLLM_API_KEY (default ./single-user/.env if present)   VLLM_API_KEY  used if no ENVFILE
#   DEPTH_CORPUS_GLOB  plain-text files for the depth/ladder prompts (required for A B D H P; any >= 2M chars of prose)
#   PORT       host port for the container's 18020 (default 18031)       OUT          output dir (default bench/acceptance/out-$TAG)
#   GPU_UTIL   gpu_memory_utilization (default 0.90)                     TIER_GIB     offload tier size for D (default 12)
#
# What compares across two images (see docs/vllm-0.29.md): quality en/da and the geometry lines compare outright;
# prefill compares by paired row (rep i is the same prompt on both); decode at n=3 does not, report it as a band;
# the code-perplexity lane reads the image's own vLLM source and never compares across pins. Pair a port against
# a control image built from the same repo commit with the old pin, so the two differ by the pin alone.
set -u
IMAGE=${IMAGE:?set IMAGE}; TAG=${TAG:?set TAG}; PROFILES=${PROFILES:-A B C D}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$HERE/out-$TAG}; mkdir -p "$OUT"
GPU=${GPU:-0}; PORT=${PORT:-18031}; GPU_UTIL=${GPU_UTIL:-0.90}; TIER_GIB=${TIER_GIB:-12}
MODELS=${MODELS:-$REPO/models}; QDATA=${QDATA:-$REPO/bench/quality-data}
ENVFILE=${ENVFILE:-}; [ -z "$ENVFILE" ] && [ -f "$REPO/single-user/.env" ] && ENVFILE="$REPO/single-user/.env"
if [ -n "$ENVFILE" ]; then export VLLM_API_KEY="$(grep -E '^VLLM_API_KEY=' "$ENVFILE" | cut -d= -f2-)"; ENVARG=(--env-file "$ENVFILE"); else ENVARG=(-e "VLLM_API_KEY=${VLLM_API_KEY:?set VLLM_API_KEY or ENVFILE}"); fi
export ACC_GPU_SMI=${ACC_GPU_SMI:-0}
WSL=""; grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && WSL="-e VLLM_WSL2_ENABLE_PIN_MEMORY=1"
log(){ echo "$(date -u +%H:%M:%SZ) [$TAG] $*"; }
fails=0
boot(){ # $1 name, $2 extra docker -e args, $3 entry command
  NAME=qwen-acc-$1; docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --gpus "device=$GPU" --ipc host --shm-size 4g \
    -p "127.0.0.1:$PORT:18020" -v "qwen-cache-$TAG:/cache" -v "$MODELS:/app/models" -v "$QDATA:/qd:ro" \
    -e HOME=/cache -e PORT=18020 -e PREPARE=0 -e VERIFY=0 -e "GPU_UTIL=$GPU_UTIL" -e VLLM_NO_USAGE_STATS=1 $WSL $2 "${ENVARG[@]}" \
    --entrypoint bash "$IMAGE" -c "cd /app && echo \"\$VLLM_API_KEY\" > api_key.txt && export PATH=/app/venv/bin:\$PATH && $3 2>&1 | tee /tmp/server.log" >/dev/null \
    && log "started $NAME :: $3"
  local c=000 i
  for i in $(seq 1 240); do
    c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health"); [ "$c" = 200 ] && break
    docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { log "container died"; docker logs "$NAME" 2>&1 | grep -aE "rror|Traceback|KV cache" | tail -6 | cut -c1-240; break; }
    sleep 5
  done
  log "health $1 $c"; [ "$c" = 200 ] || { fails=$((fails+1)); return 1; }
  docker exec "$NAME" bash -c 'grep -aE "vLLM API server version|Using .* backend|GPU KV cache size|CUDAGraphMode|Graph capturing finished|attention block size|KV cache layout|Raising block size|Not promoting|kv_offload|3D scratch|Available KV|draft_logits" /tmp/server.log | grep -av "%|" | sed "s/.*INFO [0-9: -]*//" | cut -c1-180 | sort -u | head -16' | tee "$OUT/$1-boot.txt"
}
stop(){ docker cp "qwen-acc-$1:/tmp/server.log" "$OUT/$1-server.log" >/dev/null 2>&1; docker rm -f "qwen-acc-$1" >/dev/null 2>&1; return 0; }
quality(){ # $1 container name, $2 tag
  docker run --rm --network "container:$1" -v "$HERE/../quality_battery.py:/app/bench/q.py:ro" -v "$QDATA:/qd" \
    -e VLLM_API="http://127.0.0.1:18020/v1" -e QUALITY_DATA=/qd "${ENVARG[@]}" --entrypoint bash "$IMAGE" \
    -c 'export PATH=/app/venv/bin:$PATH && pip install -q pyarrow >/dev/null 2>&1; python /app/bench/q.py '"$2"' --gsm-n 100' 2>&1 | grep -E "PPL|GSM8K|rror" | cut -c1-220
}
needle(){ # $1 container name, tokens...
  local name=$1; shift
  for t in "$@"; do
    docker run --rm --network "container:$name" -v "$HERE/../needle_test.py:/app/bench/needle.py:ro" -e VLLM_API="http://127.0.0.1:18020/v1" "${ENVARG[@]}" \
      --entrypoint bash "$IMAGE" -c 'export PATH=/app/venv/bin:$PATH && python /app/bench/needle.py '"$t"' 0.9' 2>&1 | tail -2 | sed "s/^/NEEDLE $t: /"
  done
}
# ---- A: fast (production profile)
if [[ " $PROFILES " == *" A "* ]] && boot fast "-e SPEC=dflash2 -e CTX=fast -e DFLASH_TOKENS=7 -e PREFIX_CACHE=1 -e MAX_SEQS=4" "exec bash single-user/start_qwen.sh"; then
  log "=== A quality n=100 (en/da compare across pins; code does not) ==="; quality qwen-acc-fast "acc-$TAG" | tee "$OUT/A-quality.txt"
  log "=== A depth rows 25k/50k x3 ==="; python3 "$HERE/depth.py" "A-$TAG" "$PORT" 25000,50000 3 2>&1 | grep -E "^ROW|^GPU|^MEDIAN|rror" | tee "$OUT/A-ladder.txt" | cut -c1-200
fi; stop fast
# ---- B: long (mtp, align, prefix caching)
if [[ " $PROFILES " == *" B "* ]] && boot long "-e SPEC=mtp -e CTX=long -e MAX_LEN=65536 -e PREFIX_CACHE=1 -e MAX_SEQS=4" "exec bash single-user/start_qwen.sh"; then
  log "=== B prefix ladder ==="; python3 "$HERE/prefix_ladder.py" "B-$TAG" "$PORT" 2>&1 | grep -E '"L"|rror' | tee "$OUT/B-ladder.txt" | cut -c1-160
  log "=== B peak usage ==="; for chars in 60000 160000; do python3 "$HERE/peak.py" "B-$TAG-$chars" "$PORT" "$chars" 2>&1 | tail -1 | tee -a "$OUT/B-peak.txt" | cut -c1-200; done
fi; stop long
# ---- C: int4 per-token-head KV (alternative.sh dflash2)
if [[ " $PROFILES " == *" C "* ]] && boot int4 "-e MAX_LEN=120000 -e PREFIX_CACHE=0" "exec bash single-user/alternative.sh dflash2"; then
  log "=== C mq3d oracle (all rows) ==="; docker exec qwen-acc-int4 bash -c 'export PATH=/app/venv/bin:$PATH; test -f bench/mq3d_layer2_oracle.py && timeout 900 python bench/mq3d_layer2_oracle.py 2>&1 | grep -aE "^\[oracle\]|PASS|FAIL|rror" || echo "oracle script not in this image"' | tee "$OUT/C-oracle.txt" | cut -c1-200
  log "=== C depth rows 25k/90k x1 ==="; python3 "$HERE/depth.py" "C-$TAG" "$PORT" 25000,90000 1 2>&1 | grep -E "^ROW|^GPU|rror" | tee "$OUT/C-ladder.txt" | cut -c1-200
fi; stop int4
# ---- D: offload tier serve oracle
export EXTRA_ARGS="--kv-offloading-size=$TIER_GIB --kv-cache-memory-bytes 2000000000"
if [[ " $PROFILES " == *" D "* ]] && boot offload "-e SPEC=mtp -e CTX=long -e MAX_LEN=24000 -e PREFIX_CACHE=1 -e MAX_SEQS=4 -e EXTRA_ARGS" "exec bash single-user/start_qwen.sh"; then
  log "=== D serve oracle (TIER_GIB=$TIER_GIB, 3 evictors) ==="; TIER_GIB=$TIER_GIB python3 "$HERE/replay_offload_serve.py" "D-$TAG" "$PORT" 54000 3 2>&1 | grep -E '"verdict"|INVALID|rror' | tee "$OUT/D-oracle.txt" | cut -c1-300
fi; stop offload; unset EXTRA_ARGS
# ---- H: huge (KVarN)
if [[ " $PROFILES " == *" H "* ]] && boot huge "-e SPEC=dflash2 -e CTX=huge -e PREFIX_CACHE=1 -e MAX_SEQS=4" "exec bash single-user/start_qwen.sh"; then
  log "=== H quality n=100 ==="; quality qwen-acc-huge "huge-$TAG" | tee "$OUT/H-quality.txt"
  log "=== H needle 32k/90k/200k at 0.9 (thinking off) ==="; needle qwen-acc-huge 32000 90000 200000 | tee "$OUT/H-needle.txt"
  log "=== H first deltas at 25k ==="; python3 "$HERE/firsttok.py" "H-$TAG" "$PORT" 25000 1 2>&1 | grep -E "^DELTA|^FIRSTTOK|rror" | tee "$OUT/H-firsttok.txt" | cut -c1-200
  log "=== H depth rows 25k/90k x1 ==="; python3 "$HERE/depth.py" "H-$TAG" "$PORT" 25000,90000 1 2>&1 | grep -E "^ROW|^GPU|rror" | tee "$OUT/H-ladder.txt" | cut -c1-200
fi; stop huge
# ---- P: fixed-prompt acceptance probe
if [[ " $PROFILES " == *" P "* ]] && boot fast "-e SPEC=dflash2 -e CTX=fast -e DFLASH_TOKENS=7 -e PREFIX_CACHE=1 -e MAX_SEQS=4" "exec bash single-user/start_qwen.sh"; then
  log "=== P accept probe, one prompt x5 ==="; python3 "$HERE/accept_probe.py" "P-$TAG" "$PORT" 90000 5 2>&1 | tee "$OUT/P-accept.txt" | cut -c1-220
fi; stop fast
log "done: $fails profile(s) failed to boot; records in $OUT"
exit $fails
