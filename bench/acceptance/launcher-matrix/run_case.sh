#!/bin/bash
# run_case.sh ID LAUNCHER VAR=val...: run one launcher for real (inside the image, cwd /app) against the
# stub vllm, with only the case's knobs set; record its exit code, stdout and stderr under /m/out.
id=$1; launcher=$2; shift 2
cd /app || exit 9
env -u SPEC -u CTX -u PREFIX_CACHE -u REQ_METRICS -u REQ_METRICS_DETAILED -u VISION -u KV_OFFLOAD_GB \
    -u ASYNC_SCHED -u KV -u INT8_ACT -u PRINT_ARGV -u EXTRA_ARGS \
    VLLM_API_KEY=matrix CASE_ID="$id" MATRIX_OUT=/m/out "$@" \
    timeout 900 bash "$launcher" > "/m/out/$id.launcher.out" 2> "/m/out/$id.launcher.err" < /dev/null
echo $? > "/m/out/$id.rc"
