#!/bin/bash
# Optional warm-start wrapper for a known long prefix.
#
# WARMUP_REQUEST_FILE points to a JSON request consumed by warmup_prefix.py.
# The normal single-user service does not use this wrapper; docker/entrypoint.sh
# selects it only when WARMUP_REQUEST_FILE is set.
set -euo pipefail
cd /app

warmup_file=$(printenv WARMUP_REQUEST_FILE 2>/dev/null || true)
if [ -z "$warmup_file" ]; then
  echo "warm-start: WARMUP_REQUEST_FILE must be set" >&2
  exit 2
fi

bash single-user/start_qwen.sh "$@" &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

port=$(printenv PORT 2>/dev/null || true)
[ -n "$port" ] || port=18020
health_url=$(printenv WARMUP_HEALTH_URL 2>/dev/null || true)
if [ -z "$health_url" ]; then
  health_url="http://127.0.0.1:"$port"/health"
fi
timeout_s=$(printenv WARMUP_START_TIMEOUT 2>/dev/null || true)
[ -n "$timeout_s" ] || timeout_s=3600
started=$(date +%s)

while :; do
  if curl -fsS "$health_url" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "warm-start: vLLM exited before becoming healthy" >&2
    exit 1
  fi
  now=$(date +%s)
  if [ $((now - started)) -ge "$timeout_s" ]; then
    echo "warm-start: timed out waiting for $health_url" >&2
    exit 1
  fi
  sleep 2
done

/app/venv/bin/python /app/single-user/warmup_prefix.py "$warmup_file"
touch /tmp/warmup-prefix.done
echo "warm-start: prefix is resident in the in-memory KV/GDN cache" >&2

trap - EXIT INT TERM
wait "$server_pid"
