"""Operator oracle: forced-2D vs forced-3D int4 packed attention on identical
inputs. Sol's load-bearing test (chillacks #1448): per-row/head agreement,
NaN-prefilled scratch so any unwritten-slot read becomes visible, q sweep
crossing the verify shapes, seq lengths crossing tile/segment boundaries, and
a zero-length padded row exercising the new reducer guard."""
import os, sys, math
os.environ["VLLM_INT4_MQ_3D"] = "1"
import torch
from vllm.v1.attention.ops.int4_per_token_head import (
    reshape_and_cache_int4, _launch_packed_attn, _INT4_PACKING_FACTOR)

torch.manual_seed(1234)
dev = "cuda"
HKV, HQ, D, BS = 4, 24, 128, 128   # GQA 24/4, head 128, block 128 (profile's shapes)
SEGMENTS = 16

def build(seq_lens, q_lens):
    n_seqs = len(seq_lens)
    total_ctx = sum(seq_lens); total_q = sum(q_lens)
    n_blocks = max(1, sum((l + BS - 1)//BS for l in seq_lens)) + 1
    kc = torch.zeros(n_blocks, BS, HKV, D//2, dtype=torch.uint8, device=dev)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(n_blocks, BS, HKV, dtype=torch.float32, device=dev)
    vs = torch.zeros_like(ks)
    # slot mapping: sequences packed into blocks contiguously
    slots, bt_rows, next_blk = [], [], 0
    for L in seq_lens:
        nb = (L + BS - 1)//BS if L else 0
        row = list(range(next_blk, next_blk + nb))
        for i in range(L): slots.append(row[i//BS]*BS + i % BS)
        bt_rows.append(row); next_blk += nb
    max_nb = max((len(r) for r in bt_rows), default=1) or 1
    bt = torch.zeros(n_seqs, max_nb, dtype=torch.int32, device=dev)
    for i,row in enumerate(bt_rows):
        if row: bt[i,:len(row)] = torch.tensor(row, dtype=torch.int32, device=dev)
    if total_ctx:
        K = torch.randn(total_ctx, HKV, D, dtype=torch.bfloat16, device=dev)
        V = torch.randn(total_ctx, HKV, D, dtype=torch.bfloat16, device=dev)
        reshape_and_cache_int4(K, V, kc, vc,
            torch.tensor(slots, dtype=torch.long, device=dev),
            k_scale_cache=ks, v_scale_cache=vs)
    q = torch.randn(total_q, HQ, D, dtype=torch.bfloat16, device=dev)
    cu = torch.tensor([0]+list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev)
    seqused = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    return q, kc, vc, ks, vs, cu, seqused, bt

def run(q, kc, vc, ks, vs, cu, seqused, bt, max_q, three_d):
    out = torch.full((q.shape[0], HQ, D), float('nan'), dtype=q.dtype, device=dev)
    kw = dict(q=q, k_cache=kc, v_cache=vc, out=out, cu_seqlens_q=cu,
        max_seqlen_q=max_q, seqused_k=seqused, softmax_scale=1.0/math.sqrt(D),
        window_size=(-1,-1), block_table=bt, softcap=0.0, sinks=None,
        alibi_slopes=None, use_alibi_sqrt=False, qq_bias=None, output_scale=None,
        mm_prefix_range=None, k_scale_cache=ks, v_scale_cache=vs,
        packing_factor=_INT4_PACKING_FACTOR, use_causal=True, per_seq_causal_ptr=None)
    if three_d:
        T = q.shape[0]
        kw.update(seq_threshold_3D=4096, num_par_softmax_segments=SEGMENTS,
            softmax_segm_output=torch.full((T,HQ,SEGMENTS,D), float('nan'), dtype=torch.float32, device=dev),
            softmax_segm_max=torch.full((T,HQ,SEGMENTS), float('nan'), dtype=torch.float32, device=dev),
            softmax_segm_expsum=torch.full((T,HQ,SEGMENTS), float('nan'), dtype=torch.float32, device=dev))
    else:
        kw.update(seq_threshold_3D=None, num_par_softmax_segments=None,
            softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None)
    _launch_packed_attn(**kw)
    return out

def torch_ref(q, kc, vc, ks, vs, cu, seqused, bt):
    # dequant reference in fp32 (mirrors write-side asymmetric int4 + zp-in-mantissa)
    outs = []
    for s in range(len(seqused)):
        L = int(seqused[s]); q0, q1 = int(cu[s]), int(cu[s+1]); nq = q1 - q0
        if nq == 0: continue
        Ks, Vs = [], []
        for i in range(L):
            blk = int(bt[s, i//BS]); slot = i % BS
            kp = kc[blk, slot]; vp = vc[blk, slot]           # [HKV, D/2] uint8
            ksc = ks[blk, slot]; vsc = vs[blk, slot]          # [HKV] f32 (zp in mantissa)
            kb = ksc.view(torch.int32); kzp = (kb & 0xF).float(); ksc_c = (kb & -16).view(torch.float32)
            vb = vsc.view(torch.int32); vzp = (vb & 0xF).float(); vsc_c = (vb & -16).view(torch.float32)
            lo = (kp & 0xF).float(); hi = ((kp >> 4) & 0xF).float()
            k = torch.stack([lo, hi], dim=-1).reshape(HKV, D)
            k = (k - kzp[:,None]) * ksc_c[:,None]
            lo = (vp & 0xF).float(); hi = ((vp >> 4) & 0xF).float()
            v = torch.stack([lo, hi], dim=-1).reshape(HKV, D)
            v = (v - vzp[:,None]) * vsc_c[:,None]
            Ks.append(k); Vs.append(v)
        K = torch.stack(Ks); V = torch.stack(Vs)              # [L, HKV, D]
        Q = q[q0:q1].float()                                  # [nq, HQ, D]
        o = torch.empty(nq, HQ, D, device=dev)
        ctx = L - nq
        for h in range(HQ):
            kh = K[:, h % HKV if HKV==HQ else h // (HQ//HKV)] # GQA map
            vh = V[:, h // (HQ//HKV)]
            kh = K[:, h // (HQ//HKV)]
            S = (Q[:, h] @ kh.T) / math.sqrt(D)               # [nq, L]
            pos = torch.arange(nq, device=dev)[:,None] + ctx
            mask = torch.arange(L, device=dev)[None,:] <= pos
            S = S.masked_fill(~mask, float('-inf'))
            o[:, h] = torch.softmax(S, dim=-1) @ vh
        outs.append(o)
    return torch.cat(outs) if outs else torch.zeros(0, HQ, D, device=dev)

CASES = [
    ("q1-4096",   [4096], [1]),
    ("q2-4096",   [4096], [2]),
    ("q7-4096",   [4096], [7]),
    ("q8-4096",   [4096], [8]),
    ("q9-4096",   [4096], [9]),
    ("q16-4096",  [4096], [16]),
    ("q8-127",    [127],  [8]),
    ("q8-128",    [128],  [8]),
    ("q8-129",    [129],  [8]),
    ("q8-2048",   [2048], [8]),
    ("ragged+pad",[4096, 9, 0], [8, 2, 0]),   # zero-len padded row exercises the guard
]
fails = 0
control = None  # q=1 row: 2d-vs-3d divergence between SHIPPED modes = the yardstick
for name, sl, ql in CASES:
    q, kc, vc, ks, vs, cu, seqused, bt = build(sl, ql)
    mq = max(ql) if ql else 1
    o2 = run(q, kc, vc, ks, vs, cu, seqused, bt, mq, False)
    o3 = run(q, kc, vc, ks, vs, cu, seqused, bt, mq, True)
    n2, n3 = torch.isnan(o2).any().item(), torch.isnan(o3).any().item()
    a, b = o2.float(), o3.float()
    d = (a-b).abs()
    # ULP-relative: bf16 has 7 fraction bits; ulp(x) ~ 2^(floor(log2|x|)-7).
    ulp = a.abs().clamp(min=2**-14).log2().floor().exp2() * 2**-7
    umax = (d/ulp).max().item(); mx = d.max().item(); mean = d.mean().item()
    ref = torch_ref(q, kc, vc, ks, vs, cu, seqused, bt)
    if ref.shape[0]:
        dr2 = (a[:ref.shape[0]]-ref).abs().max().item()
        dr3 = (b[:ref.shape[0]]-ref).abs().max().item()
    else: dr2 = dr3 = 0.0
    # PASS = no NaN, both arms equally close to the dequant reference
    # (neither is 'more wrong'), and the 2d-vs-3d divergence at q>1 is
    # indistinguishable from the q=1 CONTROL — the divergence already accepted
    # between the stack's two shipped dispatch modes in production.
    if control is None:
        control = (mean, mx)
    c_mean, c_max = control
    ok = ((not n2) and (not n3) and abs(dr2-dr3) < 0.005
          and mean <= 3*c_mean + 1e-6 and mx <= 2*c_max + 1e-6)
    if not ok: fails += 1
    print(f"{name:>12}: max|d|={mx:.3e} mean={mean:.2e} ULPmax={umax:.2f}  |2d-ref|={dr2:.3f} |3d-ref|={dr3:.3f}  nan={n2}/{n3}  {'OK' if ok else 'FAIL'}")
print("ORACLE", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
