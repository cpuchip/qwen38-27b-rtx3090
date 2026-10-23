#!/bin/bash
# CPU test for the memory defaults that depend on the served model's lm_head and on the card (single-user/start_qwen.sh,
# batch/start_qwen.sh, launcher_common.sh's qwen_gpu_name and qwen_lm_head_extra_bytes). Two staged hosts: "fast", whose
# models/ has the fast variant (config.json with an int4 lm_head, so select_model.sh serves it), and "base", whose
# models/ has only the base dir (an int8 lm_head). Each runs against stub nvidia-smi answers for an RTX 4090, an RTX
# 3090, an L4 and no answer. Only --kv-cache-memory and --max-model-len on SPEC=dflash2 CTX=huge (any DFLASH_TOKENS) and
# on SPEC=dflash2 DFLASH_TOKENS=15, and --gpu-memory-utilization on native batch KV=fp8 on a 4090, may move; the fast
# host on a 3090 is the shipped tree; the base host on a 4090 gets the pins measured there. An explicit KV_MEM,
# DFLASH_MAX_LEN, MAX_LEN or GPU_UTIL wins, TP>1 and two different GPUs change nothing, and a config the helper cannot
# read, or one with an int4 or unquantized lm_head, counts as int4. Copies the checkout without its venv to a temp dir
# and runs the launchers against a stub vllm (prints its argv, then "@@ENV" and its VLLM_* environment) and a stub venv
# python. No GPU, no model, no network.
#
#   bash bench/test_memory_defaults.sh
#   QWEN_BASE_TREE=<checkout without these defaults> bash bench/test_memory_defaults.sh
#       also requires every arm of the fast host under an RTX 3090 stub to give the base tree's argv, VLLM_*
#       environment and stderr byte for byte (the negative control against the tree this change is cut from)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# cfg <dir> <lm_head bits>: a config.json shaped like the shipped model's (text_config, compressed-tensors groups);
# bits "none" leaves the lm_head out of the quantization config
cfg() { mkdir -p "$1"
  local lm=''
  [ "$2" != none ] && lm=", \"group_1\": {\"targets\": [\"re:.*lm_head\$\"], \"weights\": {\"num_bits\": $2, \"group_size\": 128}}"
  printf '{"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"vocab_size": 248320, "hidden_size": 5120},
 "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 4, "group_size": 128}}%s}}}\n' "$lm" > "$1/config.json"; }
# stage <src> <dst> <fast|base>
stage() {
  mkdir -p "$2"
  (cd "$1" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$2"
  mkdir -p "$2/venv/bin" "$2/models/Qwen3.8-27B-DFlash2-W4A16"
  : > "$2/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"
  cfg "$2/models/Qwen3.8-27B-W4A16-AutoRound" 8
  [ "$3" = fast ] && cfg "$2/models/Qwen3.8-27B-W4A16-AutoRound-fast" 4
  printf '#!/bin/sh\nprintf "%%s\\n" "$@"\necho @@ENV\nenv | grep "^VLLM_" | sort\n' > "$2/venv/bin/vllm"
  printf '#!/bin/sh\ncase "$*" in *pth_flash_attn*|*fa2_fp8kv*) exit 0 ;; esac\nexec python3 "$@"\n' > "$2/venv/bin/python"
  chmod +x "$2/venv/bin/vllm" "$2/venv/bin/python"
}
stage "$REPO" "$T/fast" fast
stage "$REPO" "$T/base" base
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
     -u MAX_SEQS -u CG -u MODEL \
     VLLM_API_KEY=k "$s" "$@" bash "$sc" 2>"$T/err" </dev/null); echo $? > "$T/rc"; }
rc() { cat "$T/rc"; }
after() { local v; v=$(sed -n "/^$1\$/{n;p;}" | tr '\n' ' ' | sed 's/ $//'); echo "${v:-none}"; }
kvm() { local v; v=$(sed -n 's/^--kv-cache-memory=//p' | tr '\n' ' ' | sed 's/ $//'); echo "${v:-none}"; }
# kv <tree> <smi> <env...>: "<KV_MEM> <MAX_LEN>" the single-user launcher gives
kv() { local tree=$1 s=$2; shift 2; run "$tree" "$s" $SU "$@" > "$T/kv"; echo "$(kvm < "$T/kv") $(after --max-model-len < "$T/kv")"; }
# the argv and environment with the memory values blanked, and the launcher's resident-requests note (stdout), which
# reports the pool those values give
blank() { sed -E 's/^--kv-cache-memory=.*/--kv-cache-memory=X/; s/^\[start_qwen\] note: the pool holds .*/[pool note]/' \
  | awk 'skip { print "X"; skip = 0; next } /^--(max-model-len|gpu-memory-utilization)$/ { skip = 1 } { print }'; }
# stderr without the lines that name the compute capability (the FA2_FP8KV default's note says 8.9 or unknown)
nocap() { grep -v 'compute capability' "$1"; }
# every staged tree's paths as one, for comparing runs across trees
norm() { sed "s#$T/[a-z]*#@TREE#g" "$1"; }

SU=single-user/start_qwen.sh; BA=batch/start_qwen.sh; AL=single-user/alternative.sh
ARMS=("$SU|SPEC=mtp CTX=fast" "$SU|SPEC=mtp CTX=long" "$SU|SPEC=mtp CTX=huge" "$SU|SPEC=dflash2 CTX=fast"
      "$SU|SPEC=dflash2 CTX=long" "$SU|SPEC=dflash2 CTX=huge" "$SU|SPEC=dflash2 CTX=fast DFLASH_TOKENS=15"
      "$SU|SPEC=dflash2 CTX=huge DFLASH_TOKENS=15" "$SU|SPEC=off CTX=long" "$BA|KV=fp8" "$BA|KV=kvarn" "$BA|KV=int4pth"
      "$AL|SPEC=dflash2")
SU_CHANGED=("$SU|SPEC=dflash2 CTX=huge" "$SU|SPEC=dflash2 CTX=fast DFLASH_TOKENS=15" "$SU|SPEC=dflash2 CTX=huge DFLASH_TOKENS=15")
in_list() { local x=$1 c; shift; for c in "$@"; do [ "$c" = "$x" ] && return 0; done; return 1; }

echo "== an L4 (8.9, not a 4090) gives the argv, environment and stderr of no nvidia-smi answer, on every arm, both hosts"
echo "   (stderr apart from the lines naming the compute capability)"
for h in fast base; do for a in "${ARMS[@]}"; do sc=${a%%|*}; arm=${a#*|}
  run "$T/$h" "$SL4" $sc $arm > "$T/l4"; cp "$T/err" "$T/l4.err"; r=$(rc)
  run "$T/$h" "$SNONE" $sc $arm > "$T/none"; cp "$T/err" "$T/none.err"
  same=1; cmp -s "$T/l4" "$T/none" && cmp -s <(nocap "$T/l4.err") <(nocap "$T/none.err") || same=0
  check "0 1" "$r $same" "$h: $sc $arm"
done; done

echo "== against an L4 on the same host, only the memory values move, and only on the arms that have them"
# fast host on a 4090: the three single-user arms and batch; base host on a 4090: the same; base host on an L4 against
# the fast host on an L4 is checked below
for h in fast base; do for a in "${ARMS[@]}"; do sc=${a%%|*}; arm=${a#*|}
  run "$T/$h" "$SL4" $sc $arm > "$T/l4"
  run "$T/$h" "$S4090" $sc $arm > "$T/g"; cp "$T/err" "$T/g.err"; r=$(rc)
  if in_list "$a" "${SU_CHANGED[@]}" "$BA|KV=fp8"; then
    check "0 same differs" "$r $(cmp -s <(blank < "$T/g") <(blank < "$T/l4") && echo same || echo differs) $(cmp -s "$T/g" "$T/l4" && echo same || echo differs)" \
      "$h 4090: $sc $arm: boots; differs only in the memory values"
    if [ "$sc" = "$SU" ]; then
      grep -q '^\[start_qwen\] KV_MEM=.*for an RTX 4090' "$T/g.err" && check 1 1 "  ... and says so on stderr" || check 1 0 "  ... and says so on stderr"
    fi
  else
    check "0 same" "$r $(cmp -s "$T/g" "$T/l4" && echo same || echo differs)" "$h 4090: $sc $arm: unchanged"
  fi
done; done

echo "== the base host against the fast host, both on an L4: only the model path, and the memory values on the two profiles"
for a in "${ARMS[@]}"; do sc=${a%%|*}; arm=${a#*|}
  run "$T/fast" "$SL4" $sc $arm > "$T/f"; cp "$T/err" "$T/f.err"
  run "$T/base" "$SL4" $sc $arm > "$T/b"; cp "$T/err" "$T/b.err"; r=$(rc)
  nf() { norm "$1" | sed 's#Qwen3.8-27B-W4A16-AutoRound-fast#Qwen3.8-27B-W4A16-AutoRound#g'; }
  if in_list "$a" "${SU_CHANGED[@]}"; then
    check "0 same differs" "$r $(cmp -s <(nf "$T/b" | blank) <(nf "$T/f" | blank) && echo same || echo differs) $(cmp -s <(nf "$T/b") <(nf "$T/f") && echo same || echo differs)" \
      "$sc $arm: differs only in the memory values"
    grep -q "^\[start_qwen\] KV_MEM=.* less 606 MiB for the model's int8 lm_head and 0 MiB" "$T/b.err" && check 1 1 "  ... and says so on stderr" || check 1 0 "  ... and says so on stderr"
  else
    check "0 same" "$r $(cmp -s <(nf "$T/b") <(nf "$T/f") && echo same || echo differs)" "$sc $arm: unchanged but for the model path"
  fi
done

echo "== the values: KV_MEM and MAX_LEN for each host and card"
for row in \
  "fast 3090 5261334938 245760 221184 5583457484 57344 the shipped pins" \
  "fast 4090 4984143872 236800 213120 5469634560 50432 a 4090's term only" \
  "base 3090 4625635738 212480 191232 4947758284 44416 the lm_head's term only" \
  "base 4090 4348444672 193792 174336 4833935360 43008 the pins measured on a 4090 (launcher-4090-memory's)"; do
  set -- $row; h=$1 card=$2; shift 2
  if [ "$card" = 4090 ]; then s=$S4090; else s=$S3090; fi
  check "$1 $2" "$(kv "$T/$h" "$s" SPEC=dflash2 CTX=huge)" "$h host on a $card: CTX=huge k7 (${*:6})"
  check "$1 $3" "$(kv "$T/$h" "$s" SPEC=dflash2 CTX=huge DFLASH_TOKENS=15)" "  CTX=huge k15"
  check "$4 $5" "$(kv "$T/$h" "$s" SPEC=dflash2 DFLASH_TOKENS=15)" "  CTX=fast k15"
done
run "$T/base" "$S4090" $BA > "$T/a"
check "0.93" "$(after --gpu-memory-utilization < "$T/a")" "batch KV=fp8 on a 4090"

echo "== what the lm_head helper reads (base host on a 3090; anything but int8 is the shipped pin)"
M=$T/base/models/Qwen3.8-27B-W4A16-AutoRound
cfg "$M" 4;    check "5261334938 245760" "$(kv "$T/base" "$S3090" SPEC=dflash2 CTX=huge)" "int4 lm_head in the base dir"
cfg "$M" none; check "5261334938 245760" "$(kv "$T/base" "$S3090" SPEC=dflash2 CTX=huge)" "lm_head not in the quantization config"
cfg "$M" 16;   check "5261334938 245760" "$(kv "$T/base" "$S3090" SPEC=dflash2 CTX=huge)" "a width other than 4 or 8"
echo '{not json' > "$M/config.json"
               check "5261334938 245760" "$(kv "$T/base" "$S3090" SPEC=dflash2 CTX=huge)" "a config.json that does not parse"
rm "$M/config.json"
               check "5261334938 245760" "$(kv "$T/base" "$S3090" SPEC=dflash2 CTX=huge)" "no config.json"
cfg "$M" 8;    check "4625635738 212480" "$(kv "$T/base" "$S3090" SPEC=dflash2 CTX=huge)" "int8 again"
check "5261334938 245760" "$(kv "$T/base" "$S3090" SPEC=dflash2 CTX=huge MODEL="$T/fast/models/Qwen3.8-27B-W4A16-AutoRound-fast")" \
  "an exported MODEL is the one read (the fast dir from the base host)"

echo "== explicit values win"
check "4000000000 245760" "$(kv "$T/base" "$S4090" SPEC=dflash2 CTX=huge KV_MEM=4000000000)" "KV_MEM: its value, the 24 GiB MAX_LEN"
run "$T/base" "$S4090" $SU SPEC=dflash2 CTX=huge KV_MEM=4000000000 > "$T/a"; run "$T/base" "$SL4" $SU SPEC=dflash2 CTX=huge KV_MEM=4000000000 > "$T/b"
check same "$(cmp -s "$T/a" "$T/b" && echo same || echo differs)" "  ... and the L4's argv"
check "none 245760" "$(kv "$T/base" "$S4090" SPEC=dflash2 CTX=huge KV_MEM=)" "KV_MEM= (GPU_UTIL sizing): no pin, the 24 GiB MAX_LEN"
check "4348444672 131072" "$(kv "$T/base" "$S4090" SPEC=dflash2 CTX=huge DFLASH_MAX_LEN=131072)" "DFLASH_MAX_LEN: its value, the pin"
check "4947758284 32768" "$(kv "$T/base" "$S3090" SPEC=dflash2 DFLASH_TOKENS=15 MAX_LEN=32768)" "MAX_LEN: its value, the pin"
run "$T/base" "$S4090" $BA GPU_UTIL=0.94 > "$T/a"
check "0.94" "$(after --gpu-memory-utilization < "$T/a")" "batch GPU_UTIL=0.94"
run "$T/base" "$S4090" $BA WSL_DISTRO_NAME=Ubuntu > "$T/a"; run "$T/base" "$SL4" $BA WSL_DISTRO_NAME=Ubuntu > "$T/b"
check "0.91 same" "$(after --gpu-memory-utilization < "$T/a") $(cmp -s "$T/a" "$T/b" && echo same || echo differs)" "batch under WSL keeps its own 0.91"

echo "== TP>1 and a mixed host are not one 4090, and TP>1 takes no lm_head term"
run "$T/base" "$S4090" $SU SPEC=dflash2 CTX=huge EXTRA_ARGS="--tensor-parallel-size 2" > "$T/a"
run "$T/fast" "$SL4" $SU SPEC=dflash2 CTX=huge EXTRA_ARGS="--tensor-parallel-size 2" > "$T/b"
check same "$(cmp -s <(norm "$T/a" | sed 's#-fast##') <(norm "$T/b" | sed 's#-fast##') && echo same || echo differs)" "TP=2 on 4090s, base host: the fast host's argv on an L4"
run "$T/fast" "$SMIX" $SU SPEC=dflash2 CTX=huge > "$T/a"; run "$T/fast" "$SNONE" $SU SPEC=dflash2 CTX=huge > "$T/b"
check same "$(cmp -s "$T/a" "$T/b" && echo same || echo differs)" "a 4090 and an L4, nothing selected: unchanged"
run "$T/fast" "$SMIX" $SU SPEC=dflash2 CTX=huge CUDA_VISIBLE_DEVICES=0 > "$T/a"
check "4984143872" "$(kvm < "$T/a")" "the same host with CUDA_VISIBLE_DEVICES=0 (the stub's first GPU is the 4090)"

if [ -n "${QWEN_BASE_TREE:-}" ]; then
  echo "== negative control: the fast host on an RTX 3090 gives the base tree's argv, environment and stderr on every arm"
  stage "$QWEN_BASE_TREE" "$T/ctrl" fast
  for a in "${ARMS[@]}"; do sc=${a%%|*}; arm=${a#*|}
    run "$T/fast" "$S3090" $sc $arm > "$T/n"; cp "$T/err" "$T/n.err"
    run "$T/ctrl" "$S3090" $sc $arm > "$T/b"; cp "$T/err" "$T/b.err"
    check same "$(cmp -s <(norm "$T/n") <(norm "$T/b") && cmp -s <(norm "$T/n.err") <(norm "$T/b.err") && echo same || echo differs)" "$sc $arm"
  done
fi
echo "== $FAILS failure(s)"
[ $FAILS = 0 ]
