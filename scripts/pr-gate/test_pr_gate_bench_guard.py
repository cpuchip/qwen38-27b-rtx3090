#!/usr/bin/env python3
"""Self-test for pr_gate.py's bench-guard check (needs_hw_guard): a quoted KV dtype such as "fp8" is not a hardware
need, an fp8 kernel or a capability-gated path still is, and a guard still clears it.

    python scripts/pr-gate/test_pr_gate_bench_guard.py
"""
import importlib.util, os, sys

sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("pr_gate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr_gate.py"))
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

CASES = [
    # (name, source, needs a guard)
    ("KV dtype string, double quotes", 'for kv in ["bfloat16", "fp8"]:\n    page = {"fp8": 2048}[kv]\n', False),
    ("KV dtype string, single quotes, e4m3", "dtype = 'fp8_e4m3'\n", False),
    ("fp8 tensor dtype outside a string", "x = torch.empty(4, dtype=torch.float8_e4m3fn)\n", True),
    ("bare fp8 word in code", "if use_fp8:\n    run()\n", True),
    ("the word capability, no guard", "major, _ = torch.cuda.get_device_capability()\n".replace("get_device_capability", "device_capability"), True),
    ("CUDA graph without a guard", "g = torch.cuda.CUDAGraph()  # cuda graph capture\n", True),
    ("fp8 kernel with a skip", "if not ok:\n    raise unittest.SkipTest('needs sm89')\nx = torch.float8_e4m3fn\n", False),
    ("no hardware words", "assert 1 + 1 == 2\n", False),
]

fails = 0
for name, src, want in CASES:
    got = pg.needs_hw_guard(src)
    print(f"  {'PASS' if got == want else 'FAIL'}  {name}: needs a guard {got}")
    fails += got != want
print("all checks passed" if not fails else f"{fails} checks FAILED")
sys.exit(1 if fails else 0)
