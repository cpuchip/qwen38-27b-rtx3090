#!/bin/bash
# CPU test for the RTX 4090 memory defaults (single-user/start_qwen.sh, batch/start_qwen.sh, launcher_common.sh's
# qwen_gpu_name). Every launcher arm is run against a stub nvidia-smi that reports an RTX 4090 and one that reports
# another compute capability 8.9 card (an L4), and the two argv and VLLM_* environments may differ only where the
# 4090 defaults apply: --kv-cache-memory and --max-model-len on SPEC=dflash2 CTX=huge (any DFLASH_TOKENS) and on
# SPEC=dflash2 DFLASH_TOKENS=15, --gpu-memory-utilization on native batch KV=fp8. An explicit KV_MEM, DFLASH_MAX_LEN or
# GPU_UTIL wins on the 4090, an explicit KV_MEM keeps the 24 GiB MAX_LEN, TP>1 and two different GPUs change nothing,
# and the L4 stub gives the argv, environment and stderr of a host with no nvidia-smi answer. Copies the checkout
# without its venv to a temp dir and runs the launchers against a stub vllm (prints its argv, then "@@ENV" and its
# VLLM_* environment) and a stub venv python. No GPU, no model, no network.
#
#   bash bench/test_4090_defaults.sh
#   QWEN4090_BASE_TREE=<checkout without the 4090 defaults> bash bench/test_4090_defaults.sh
#       also requires every arm under an RTX 3090 stub (compute capability 8.6) to give the base tree's argv, VLLM_*
#       environment and stderr byte for byte (the negative control against the tree this change is cut from)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
stage() {
  mkdir -p "$2"
  (cd "$1" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$2"
  mkdir -p "$2/venv/bin" "$2/models/Qwen3.8-27B-DFlash2-W4A16"
  : > "$2/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"
  printf '#!/bin/sh\nprintf "%%s\\n" "$@"\necho @@ENV\nenv | grep "^VLLM_" | sort\n' > "$2/venv/bin/vllm"
  printf '#!/bin/sh\ncase "$*" in *pth_flash_attn*|*fa2_fp8kv*) exit 0 ;; esac\nexec python3 "$@"\n' > "$2/venv/bin/python"
  chmod +x "$2/venv/bin/vllm" "$2/venv/bin/python"
}
stage "$REPO" "$T/new"
# smi <tag> <compute_cap> <name>...: PATH=<dir>:$PATH, where <dir>/nvidia-smi answers the compute_cap and name
# queries, one line per GPU; tag "none" answers nothing and fails, as on a host without a GPU driver.
smi() { local tag=$1 cap=$2 d=$T/smi-$1; shift 2; mkdir -p "$d"
  if [ "$tag" = none ]; then printf '#!/bin/sh\nexit 9\n' > "$d/nvidia-smi"
  else
    printf '%s\n' "$@" > "$d/names"
    # -i N answers for GPU N only; with no -i, one line per GPU
    printf '#!/bin/sh\ni=; q=\nwhile [ $# -gt 0 ]; do case "$1" in -i) i=$2; shift ;; --query-gpu=*) q=${1#*=} ;; esac; shift; done\n' > "$d/nvidia-smi"
    printf 'if [ -n "$i" ]; then l=$(sed -n "$((i + 1))p" "%s/names"); [ -n "$l" ] || exit 6; else l=$(cat "%s/names"); fi\n' "$d" "$d" >> "$d/nvidia-smi"
    printf 'case "$q" in compute_cap) printf "%%s\\n" "$l" | sed "s/.*/%s/" ;; name) printf "%%s\\n" "$l" ;; *) exit 9 ;; esac\n' "$cap" >> "$d/nvidia-smi"
  fi
  chmod +x "$d/nvidia-smi"; echo "PATH=$d:$PATH"; }
S4090=$(smi 4090 8.9 "NVIDIA GeForce RTX 4090"); SL4=$(smi l4 8.9 "NVIDIA L4"); SNONE=$(smi none -)
S3090=$(smi 3090 8.6 "NVIDIA GeForce RTX 3090"); SMIX=$(smi mix 8.9 "NVIDIA GeForce RTX 4090" "NVIDIA L4")
# run <tree> <smi> <launcher> <env...>: argv and VLLM_* environment on stdout, stderr in $T/err, exit code in $T/rc.
run() { local tree=$1 s=$2 sc=$3; shift 3
  (cd "$tree" && env -u PRINT_ARGV -u PTH_FA -u KV_OFFLOAD_GB -u EXTRA_ARGS -u SPEC -u CTX -u PYTORCH_CUDA_ALLOC_CONF \
     -u WSL_DISTRO_NAME -u CUDA_HOME -u FA2_FP8KV -u DRAFT_METHOD -u DFLASH_TOKENS -u LOOKUP -u VISION -u DYNAMIC_K \
     -u DRAFT -u DRAFT_TOKENS -u CUDA_VISIBLE_DEVICES -u KV_MEM -u DFLASH_MAX_LEN -u MAX_LEN -u GPU_UTIL -u KV \
     -u MAX_SEQS -u CG \
     VLLM_API_KEY=k "$s" "$@" bash "$sc" 2>"$T/err" </dev/null); echo $? > "$T/rc"; }
rc() { cat "$T/rc"; }
after() { local v; v=$(sed -n "/^$1\$/{n;p;}" | tr '\n' ' ' | sed 's/ $//'); echo "${v:-none}"; }
kvm() { local v; v=$(sed -n 's/^--kv-cache-memory=//p' | tr '\n' ' ' | sed 's/ $//'); echo "${v:-none}"; }
# the argv and environment with the three 4090-dependent values blanked, and the launcher's resident-requests note
# (stdout), which reports the pool those values give
blank() { sed -E 's/^--kv-cache-memory=.*/--kv-cache-memory=X/; s/^\[start_qwen\] note: the pool holds .*/[pool note]/' \
  | awk 'skip { print "X"; skip = 0; next } /^--(max-model-len|gpu-memory-utilization)$/ { skip = 1 } { print }'; }
# stderr without the lines that name the compute capability (the FA2_FP8KV default's note says 8.9 or unknown)
nocap() { grep -v 'compute capability' "$1"; }
# both staged trees' paths as one, for comparing a run in one tree with a run in the other
norm() { sed "s#$T/new#@TREE#g; s#$T/base#@TREE#g" "$1"; }

SU=single-user/start_qwen.sh; BA=batch/start_qwen.sh; AL=single-user/alternative.sh
ARMS=("$SU|SPEC=mtp CTX=fast" "$SU|SPEC=mtp CTX=long" "$SU|SPEC=mtp CTX=huge" "$SU|SPEC=dflash2 CTX=fast"
      "$SU|SPEC=dflash2 CTX=long" "$SU|SPEC=dflash2 CTX=huge" "$SU|SPEC=dflash2 CTX=fast DFLASH_TOKENS=15"
      "$SU|SPEC=dflash2 CTX=huge DFLASH_TOKENS=15" "$SU|SPEC=off CTX=long" "$BA|KV=fp8" "$BA|KV=kvarn" "$BA|KV=int4pth"
      "$AL|SPEC=dflash2")
CHANGED=("$SU|SPEC=dflash2 CTX=huge" "$SU|SPEC=dflash2 CTX=fast DFLASH_TOKENS=15" "$SU|SPEC=dflash2 CTX=huge DFLASH_TOKENS=15" "$BA|KV=fp8")

echo "== an L4 (8.9, not a 4090) gives the argv, environment and stderr of no nvidia-smi answer, on every arm"
echo "   (stderr apart from the lines naming the compute capability)"
for a in "${ARMS[@]}"; do sc=${a%%|*}; arm=${a#*|}
  run "$T/new" "$SL4" $sc $arm > "$T/l4"; cp "$T/err" "$T/l4.err"; r=$(rc)
  run "$T/new" "$SNONE" $sc $arm > "$T/none"; cp "$T/err" "$T/none.err"
  same=1; cmp -s "$T/l4" "$T/none" && cmp -s <(nocap "$T/l4.err") <(nocap "$T/none.err") || same=0
  check "0 1" "$r $same" "$sc $arm"
done

echo "== a 4090 against an L4: only the 4090 values differ, and only on the arms that have them"
for a in "${ARMS[@]}"; do sc=${a%%|*}; arm=${a#*|}
  run "$T/new" "$SL4" $sc $arm > "$T/l4"
  run "$T/new" "$S4090" $sc $arm > "$T/g"; cp "$T/err" "$T/g.err"; r=$(rc)
  ch=0; for c in "${CHANGED[@]}"; do [ "$c" = "$a" ] && ch=1; done
  if [ $ch = 1 ]; then
    check "0 same differs" "$r $(cmp -s <(blank < "$T/g") <(blank < "$T/l4") && echo same || echo differs) $(cmp -s "$T/g" "$T/l4" && echo same || echo differs)" \
      "$sc $arm: boots; differs only in the 4090 values"
    grep -q '^\[start_qwen\] RTX 4090' "$T/g.err" && check 1 1 "  ... and says so on stderr" || check 1 0 "  ... and says so on stderr"
  else
    check "0 same" "$r $(cmp -s "$T/g" "$T/l4" && echo same || echo differs)" "$sc $arm: unchanged"
  fi
done

echo "== the 4090 values"
run "$T/new" "$S4090" $SU SPEC=dflash2 CTX=huge > "$T/a"
check "4348444672 193792" "$(kvm < "$T/a") $(after --max-model-len < "$T/a")" "CTX=huge k7"
run "$T/new" "$S4090" $SU SPEC=dflash2 CTX=huge DFLASH_TOKENS=15 > "$T/a"
check "4348444672 174336" "$(kvm < "$T/a") $(after --max-model-len < "$T/a")" "CTX=huge k15"
run "$T/new" "$S4090" $SU SPEC=dflash2 DFLASH_TOKENS=15 > "$T/a"
check "4833935360 43008" "$(kvm < "$T/a") $(after --max-model-len < "$T/a")" "CTX=fast k15"
run "$T/new" "$S4090" $BA > "$T/a"
check "0.93" "$(after --gpu-memory-utilization < "$T/a")" "batch KV=fp8"

echo "== explicit values win on the 4090"
run "$T/new" "$S4090" $SU SPEC=dflash2 CTX=huge KV_MEM=4000000000 > "$T/a"
run "$T/new" "$SL4" $SU SPEC=dflash2 CTX=huge KV_MEM=4000000000 > "$T/b"
check "4000000000 245760 same" "$(kvm < "$T/a") $(after --max-model-len < "$T/a") $(cmp -s "$T/a" "$T/b" && echo same || echo differs)" \
  "KV_MEM: its value, the 24 GiB MAX_LEN, the L4's argv"
run "$T/new" "$S4090" $SU SPEC=dflash2 CTX=huge KV_MEM= > "$T/a"
check "none 245760" "$(kvm < "$T/a") $(after --max-model-len < "$T/a")" "KV_MEM= (GPU_UTIL sizing): no pin, the 24 GiB MAX_LEN"
run "$T/new" "$S4090" $SU SPEC=dflash2 CTX=huge DFLASH_MAX_LEN=131072 > "$T/a"
check "4348444672 131072" "$(kvm < "$T/a") $(after --max-model-len < "$T/a")" "DFLASH_MAX_LEN: its value, the 4090 pin"
run "$T/new" "$S4090" $SU SPEC=dflash2 DFLASH_TOKENS=15 MAX_LEN=32768 > "$T/a"
check "4833935360 32768" "$(kvm < "$T/a") $(after --max-model-len < "$T/a")" "MAX_LEN: its value, the 4090 pin"
run "$T/new" "$S4090" $BA GPU_UTIL=0.94 > "$T/a"
check "0.94" "$(after --gpu-memory-utilization < "$T/a")" "batch GPU_UTIL=0.94"
run "$T/new" "$S4090" $BA WSL_DISTRO_NAME=Ubuntu > "$T/a"; run "$T/new" "$SL4" $BA WSL_DISTRO_NAME=Ubuntu > "$T/b"
check "0.91 same" "$(after --gpu-memory-utilization < "$T/a") $(cmp -s "$T/a" "$T/b" && echo same || echo differs)" "batch under WSL keeps its own 0.91"

echo "== TP>1 and a mixed host are not a single 4090"
run "$T/new" "$S4090" $SU SPEC=dflash2 CTX=huge EXTRA_ARGS="--tensor-parallel-size 2" > "$T/a"
run "$T/new" "$SL4" $SU SPEC=dflash2 CTX=huge EXTRA_ARGS="--tensor-parallel-size 2" > "$T/b"
check same "$(cmp -s "$T/a" "$T/b" && echo same || echo differs)" "TP=2 on 4090s: unchanged"
run "$T/new" "$SMIX" $SU SPEC=dflash2 CTX=huge > "$T/a"; run "$T/new" "$SNONE" $SU SPEC=dflash2 CTX=huge > "$T/b"
check same "$(cmp -s "$T/a" "$T/b" && echo same || echo differs)" "a 4090 and an L4, nothing selected: unchanged"
run "$T/new" "$SMIX" $SU SPEC=dflash2 CTX=huge CUDA_VISIBLE_DEVICES=0 > "$T/a"
check "4348444672" "$(kvm < "$T/a")" "the same host with CUDA_VISIBLE_DEVICES=0 (the stub's first GPU is the 4090)"

if [ -n "${QWEN4090_BASE_TREE:-}" ]; then
  echo "== negative control: an RTX 3090 gives the base tree's argv, environment and stderr on every arm"
  stage "$QWEN4090_BASE_TREE" "$T/base"
  for a in "${ARMS[@]}"; do sc=${a%%|*}; arm=${a#*|}
    run "$T/new" "$S3090" $sc $arm > "$T/n"; cp "$T/err" "$T/n.err"
    run "$T/base" "$S3090" $sc $arm > "$T/b"; cp "$T/err" "$T/b.err"
    check same "$(cmp -s <(norm "$T/n") <(norm "$T/b") && cmp -s <(norm "$T/n.err") <(norm "$T/b.err") && echo same || echo differs)" "$sc $arm"
  done
fi
echo "== $FAILS failure(s)"
[ $FAILS = 0 ]
