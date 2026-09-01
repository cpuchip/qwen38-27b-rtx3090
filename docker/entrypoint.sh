#!/bin/bash
# Container entrypoint. First argument selects what to run:
#   single   single-user/start_qwen.sh  (MTP speculative decoding, low latency)
#            or start_with_warmup.sh when WARMUP_REQUEST_FILE is set
#   batch    batch/start_qwen.sh        (throughput)
#   prepare  docker/prepare.sh          (download + requantize the model into /app/models)
#   verify   verify.sh [args]
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, docker/prepare.sh runs (idempotent: a state check and
# seconds when the model is already prepared, the download + requantization
# otherwise; PREPARE=0 skips it — this is what makes a bare `docker run` with
# an empty models volume work), then verify.sh --no-server, which aborts on
# FAIL (patches missing, ...); VERIFY=0 skips that.
set -e
cd /app

cmd=single
if [ "$#" -gt 0 ]; then
  cmd=$1
  shift
fi

case "$cmd" in
  single|batch)
    if [ "${PREPARE:-1}" != "0" ]; then
      bash docker/prepare.sh
    fi
    if [ "$(printenv VERIFY 2>/dev/null || true)" != "0" ]; then
      bash verify.sh --no-server || {
        echo "entrypoint: verify.sh FAILED — fix the above or set VERIFY=0"
        exit 1
      }
    fi
    if [ "$cmd" = single ]; then
      if printenv WARMUP_REQUEST_FILE >/dev/null 2>&1; then
        exec bash single-user/start_with_warmup.sh "$@"
      else
        exec bash single-user/start_qwen.sh "$@"
      fi
    else
      exec bash batch/start_qwen.sh "$@"
    fi
    ;;
  prepare)
    exec bash docker/prepare.sh "$@"
    ;;
  verify)
    exec bash verify.sh "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
