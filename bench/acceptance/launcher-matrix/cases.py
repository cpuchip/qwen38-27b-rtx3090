"""cases.py > cases.txt: one line per case, "ID LAUNCHER VAR=val ...". The axes are the knobs that change the
argv's shape; each launcher crosses all of its own. Add an axis here when a launcher gains a knob that adds,
drops or rewrites a flag.
"""
import itertools

METRICS = [("m0", ["REQ_METRICS=0"]), ("m1", ["REQ_METRICS=1"]),
           ("m2", ["REQ_METRICS=1", "REQ_METRICS_DETAILED=1"])]
OFFLOAD = [("o0", []), ("o8", ["KV_OFFLOAD_GB=8"])]

rows = []
for s, c, p, (mk, mv), v, (ok, ov) in itertools.product(
        ["off", "mtp", "dflash2"], ["fast", "long", "huge"], ["0", "1"], METRICS, ["0", "1"], OFFLOAD):
    rows.append((f"single_{s}_{c}_p{p}_{mk}_v{v}_{ok}", "single-user/start_qwen.sh",
                 [f"SPEC={s}", f"CTX={c}", f"PREFIX_CACHE={p}", *mv, f"VISION={v}", *ov]))
for s, a, (mk, mv), v, (ok, ov) in itertools.product(["dflash2", "off"], ["0", "1"], METRICS, ["0", "1"], OFFLOAD):
    rows.append((f"alt_{s}_a{a}_{mk}_v{v}_{ok}", "single-user/alternative.sh",
                 [f"SPEC={s}", f"ASYNC_SCHED={a}", *mv, f"VISION={v}", *ov]))
for k, i, p, (mk, mv), (ok, ov) in itertools.product(["fp8", "kvarn"], ["int8", "none"], ["0", "1"], METRICS, OFFLOAD):
    rows.append((f"batch_{k}_i{i}_p{p}_{mk}_{ok}", "batch/start_qwen.sh",
                 [f"KV={k}", "INT8_ACT=" + ("" if i == "none" else i), f"PREFIX_CACHE={p}", *mv, *ov]))

for cid, launcher, env in rows:
    print(" ".join([cid, launcher, *env]))
