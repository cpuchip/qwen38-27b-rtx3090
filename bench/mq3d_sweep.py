"""Sweep the int4 per-token-head attention kernel: 3D leg vs 2D leg vs dequantized reference, over sequence
length, for the verify-batch shapes the collapse runs under, with two KV fills (random; a repeated pattern,
the shape of a polling loop's cached content). Phase 2 mimics the OLD production sizing (8 scratch rows,
the capture-size snap) with canaries around the scratch to catch out-of-bounds writes.
Built on bench/mq3d_layer2_oracle.py (mounted copy), which owns the cache layout and the reference."""
import importlib.util, inspect, json, os, sys, time
import torch
sys.stdout.reconfigure(encoding="utf-8")
os.environ["VLLM_INT4_MQ_3D"] = "1"
# old tree: unified_attention() lacks the two Patch A kwargs; filter to the live signature before the oracle binds it
import vllm.v1.attention.ops.triton_unified_attention as tua
_ua = tua.unified_attention; _params = set(inspect.signature(_ua).parameters)
def _ua_compat(*a, **kw): return _ua(*a, **{k: v for k, v in kw.items() if k in _params})
tua.unified_attention = _ua_compat
print("unified_attention accepts max_query_len_3d:", "max_query_len_3d" in _params, flush=True)
spec = importlib.util.spec_from_file_location("oracle", "/work/bench/mq3d_layer2_oracle.py")
oracle = importlib.util.module_from_spec(spec)
os.chdir("/work/bench")
try:
    spec.loader.exec_module(oracle)          # runs its own matrix once, then sys.exit()s
except BaseException as e:               # SystemExit at its end on the new tree; KeyError in its breach row on the old tree
    print("oracle matrix ended with:", repr(e)[:160], flush=True)
print("oracle loaded: capacity", oracle.CAPACITY, "thr", oracle.SEQ_THRESHOLD, "segments", oracle.SEGM, "qlen_cap", oracle.QLEN_CAP,
      "buf shapes", [tuple(b.shape) for b in oracle.BUFS], [str(b.dtype) for b in oracle.BUFS], flush=True)

# the oracle's cache holds 96 blocks (1,536 tokens); the sweep needs 4 x 16,384 + change
oracle.NUM_BLOCKS = 4480
oracle.kv_raw = torch.zeros(oracle.NUM_BLOCKS, oracle.NUM_HEADS_KV, oracle.BLOCK_SIZE, 2 * oracle.PACKED_HS, dtype=torch.uint8, device=oracle.DEV)
oracle.key_cache, oracle.value_cache = oracle.kv_raw.transpose(1, 2).split(oracle.PACKED_HS, dim=-1)
oracle.k_scale_cache = torch.zeros(oracle.NUM_BLOCKS, oracle.BLOCK_SIZE, oracle.NUM_HEADS_KV, dtype=torch.float32, device=oracle.DEV)
oracle.v_scale_cache = torch.zeros_like(oracle.k_scale_cache)
print("cache rebuilt:", oracle.NUM_BLOCKS, "blocks", flush=True)

_orig_populate = oracle.populate_seq
def populate_repeat(seq_idx, kv_len, block_table_row):
    """The oracle's populate_seq with the K/V rows a short pattern repeated down the sequence plus small noise."""
    torch.manual_seed(1000 + seq_idx)
    period = 24
    base_k = torch.randn(period, oracle.NUM_HEADS_KV, oracle.HEAD_DIM, dtype=torch.float16, device=oracle.DEV)
    base_v = torch.randn(period, oracle.NUM_HEADS_KV, oracle.HEAD_DIM, dtype=torch.float16, device=oracle.DEV)
    reps = kv_len // period + 1
    k = base_k.repeat(reps, 1, 1)[:kv_len] + 0.02 * torch.randn(kv_len, oracle.NUM_HEADS_KV, oracle.HEAD_DIM, dtype=torch.float16, device=oracle.DEV)
    v = base_v.repeat(reps, 1, 1)[:kv_len] + 0.02 * torch.randn(kv_len, oracle.NUM_HEADS_KV, oracle.HEAD_DIM, dtype=torch.float16, device=oracle.DEV)
    src = inspect.getsource(_orig_populate).split("\n")
    src = [l for l in src if not l.strip().startswith("k = torch.randn") and not l.strip().startswith("v = torch.randn")]
    code = "\n".join(src).replace("def populate_seq(seq_idx, kv_len, block_table_row):", "def populate_seq(seq_idx, kv_len, block_table_row, k, v):")
    ns = dict(_orig_populate.__globals__); exec(code, ns)
    return ns["populate_seq"](seq_idx, kv_len, block_table_row, k, v)

CANARY = {}
def ran3d():  # the oracle NaN-fills the scratch before every call; 2D leaves it NaN, 3D writes rows
    return not torch.isnan(oracle.BUFS[1]).all().item()
def canary_ok():
    return all(bool((c[0] == 12345.0).all()) and bool((c[1] == 12345.0).all()) for c in CANARY.values()) if CANARY else None

def one(mode, L, q_lens, tag):
    rows = {}
    for leg, force in (("3d", True), ("2d", False)):
        try:
            r = oracle.run_case(q_lens, force, seed=7); r["ran3d"] = ran3d(); rows[leg] = r
        except Exception as e:
            rows[leg] = {"error": repr(e)[:140]}
    rec = {"phase": tag, "mode": mode, "L": L, "q_lens": q_lens}
    if "error" in rows["3d"] or "error" in rows["2d"]:
        rec["err"] = {k: v.get("error") for k, v in rows.items()}; print(json.dumps(rec), flush=True); return
    rec.update({"ref_err_3d": round(rows["3d"]["ref_max_abs"], 5), "ref_err_2d": round(rows["2d"]["ref_max_abs"], 5),
                "diff_3d_2d": round(float((rows["3d"]["out"] - rows["2d"]["out"]).abs().max().item()), 5),
                "ran3d": [rows["3d"]["ran3d"], rows["2d"]["ran3d"]], "finite_3d": rows["3d"]["out_finite"], "finite_2d": rows["2d"]["out_finite"],
                "out_max_3d": round(rows["3d"]["out_max_abs"], 4), "canary_ok": canary_ok()})
    print(json.dumps(rec), flush=True)

def sweep(tag, shapes, Ls):
    for mode in ("random", "repeat"):
        oracle.populate_seq = _orig_populate if mode == "random" else populate_repeat
        for L in Ls:
            oracle.KV_HISTORY = L
            for q_lens in shapes:
                one(mode, L, list(q_lens), tag)

t0 = time.time()
LS = (48, 256, 1000, 2048, 4096, 8192, 12288, 16384)
print("===== phase 1: builder sizing, verify shapes =====", flush=True)
sweep("builder", [(1,), (8,), (8, 8), (8, 8, 8, 8), (1, 1, 1, 1)], LS)

print("===== phase 2: OLD production sizing (8 rows, capture snap) with canaries =====", flush=True)
PAD = 4096
newbufs = []
for b in oracle.BUFS:
    shape = (8,) + tuple(b.shape[1:]); n = 1
    for s in shape: n *= s
    flat = torch.full((n + 2 * PAD,), 12345.0, dtype=b.dtype, device=b.device)
    CANARY[len(newbufs)] = (flat[:PAD], flat[-PAD:])
    newbufs.append(flat[PAD:PAD + n].view(shape))
oracle.BUFS = tuple(newbufs); oracle.SEQ_THRESHOLD = 8; oracle.CAPACITY = 8
sweep("old8", [(1,), (8,), (4, 4), (2, 2, 2, 2), (1, 1, 1, 1)], LS)
print(f"SWEEP DONE in {round(time.time()-t0)}s", flush=True)
