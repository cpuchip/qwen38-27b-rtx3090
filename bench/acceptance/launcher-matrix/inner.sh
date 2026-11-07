#!/bin/bash
# Inside the container: lay the overlay over /app if one is mounted, swap venv/bin/vllm for the stub,
# run every case, 16 at a time.
cd /app || exit 1
[ -d /ovl ] && cp -r /ovl/. /app/
mv venv/bin/vllm venv/bin/vllm.real
printf '#!/bin/bash\nexec /app/venv/bin/python /m/build_config.py "$@"\n' > venv/bin/vllm
chmod +x venv/bin/vllm
S=$(date +%s)
xargs -P "${MATRIX_JOBS:-16}" -L 1 /m/run_case.sh < /m/cases.txt
echo "matrix_wall=$(( $(date +%s) - S ))s"
