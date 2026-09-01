"""Time _attn_packed 2D vs 3D in isolation at the REAL depth (72k, q=8, GQA 24/4).
Answers: does the split win at the kernel level, and by how much — separating
'verify attention was the cliff' from 'the cliff lives elsewhere (drafter)'."""
import os, math, torch
os.environ["VLLM_INT4_MQ_3D"] = "1"
from vllm.v1.attention.ops.int4_per_token_head import (
    reshape_and_cache_int4, _launch_packed_attn, _INT4_PACKING_FACTOR)
torch.manual_seed(7)
dev="cuda"; HKV,HQ,D,BS,SEG=4,24,128,128,16
L=72576; Q=8
nb=(L+BS-1)//BS
kc=torch.zeros(nb,BS,HKV,D//2,dtype=torch.uint8,device=dev); vc=torch.zeros_like(kc)
ks=torch.zeros(nb,BS,HKV,dtype=torch.float32,device=dev); vs=torch.zeros_like(ks)
slots=torch.arange(L,dtype=torch.long,device=dev)
CH=16384
for i in range(0,L,CH):
    j=min(i+CH,L)
    reshape_and_cache_int4(torch.randn(j-i,HKV,D,dtype=torch.bfloat16,device=dev),
        torch.randn(j-i,HKV,D,dtype=torch.bfloat16,device=dev),
        kc,vc,slots[i:j],k_scale_cache=ks,v_scale_cache=vs)
q=torch.randn(Q,HQ,D,dtype=torch.bfloat16,device=dev)
cu=torch.tensor([0,Q],dtype=torch.int32,device=dev)
seqused=torch.tensor([L],dtype=torch.int32,device=dev)
bt=torch.arange(nb,dtype=torch.int32,device=dev).unsqueeze(0)
out=torch.zeros(Q,HQ,D,dtype=torch.bfloat16,device=dev)
def arm(three_d):
    kw=dict(q=q,k_cache=kc,v_cache=vc,out=out,cu_seqlens_q=cu,max_seqlen_q=Q,
        seqused_k=seqused,softmax_scale=1.0/math.sqrt(D),window_size=(-1,-1),
        block_table=bt,softcap=0.0,sinks=None,alibi_slopes=None,use_alibi_sqrt=False,
        qq_bias=None,output_scale=None,mm_prefix_range=None,k_scale_cache=ks,
        v_scale_cache=vs,packing_factor=_INT4_PACKING_FACTOR,use_causal=True,per_seq_causal_ptr=None)
    if three_d:
        kw.update(seq_threshold_3D=4096,num_par_softmax_segments=SEG,
            softmax_segm_output=torch.zeros(Q,HQ,SEG,D,dtype=torch.float32,device=dev),
            softmax_segm_max=torch.zeros(Q,HQ,SEG,dtype=torch.float32,device=dev),
            softmax_segm_expsum=torch.zeros(Q,HQ,SEG,dtype=torch.float32,device=dev))
    else:
        kw.update(seq_threshold_3D=None,num_par_softmax_segments=None,
            softmax_segm_output=None,softmax_segm_max=None,softmax_segm_expsum=None)
    return kw
for name,td in [("2D",False),("3D",True)]:
    kw=arm(td)
    for _ in range(3): _launch_packed_attn(**kw)
    torch.cuda.synchronize()
    t0=torch.cuda.Event(True); t1=torch.cuda.Event(True); t0.record()
    N=50
    for _ in range(N): _launch_packed_attn(**kw)
    t1.record(); torch.cuda.synchronize()
    ms=t0.elapsed_time(t1)/N
    print(f"{name}: {ms:.3f} ms per call at L={L} q={Q}  -> x17 layers = {ms*17:.1f} ms/step")
