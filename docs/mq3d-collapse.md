# The int4 multi-query 3D verify path collapses in long tool-calling conversations (2026-09-02)

**Status: reproduced, not root-caused.** A defect in the opt-in 3D verify path of the int4
per-token-head KV cache (`VLLM_INT4_MQ_3D=1`, issue #46's extension; the scratch buffers PR #57
sized). The base int4 kernel with the 3D path off does not show it. Nothing is filed upstream yet.

## What happens

On a long tool-calling conversation (about forty tool calls, twelve thousand tokens of context), the
model's output degenerates and never recovers:

- thinking off: after a run of near-identical tool calls (a pending-evaluation tool polled sixteen
  times a turn for three turns), the reply becomes one repeated word ("duct Register Register…")
  for the rest of the run, every turn, `finish_reason=length`;
- thinking on, with a 6,000-token reply cap: the reasoning runs away instead (the same 53,995
  characters every turn, no tool call).

Two runs per cell, identical call for call. The conversation is the flightbench mission
(`benches/mission.py`, v6.1, controller seat), and the trigger every time has been a long run of
near-identical multi-query verify batches.

## What isolates it

Same weights (Qwen3.8-27B-W4A16-AutoRound), same DFlash2 draft (7 drafts), same image, same
launcher and flags, one variable per row, mission score out of 18, two runs per cell:

| KV cache | 3D path | thinking off | thinking on |
|---|---|---|---|
| bf16 (`CTX=fast`) | n/a | 16, 16 | 15, 15 |
| int8 per-token-head (`CTX=long`) | n/a | 15, 15 | 17, 17 |
| int4 per-token-head (`alternative.sh`), `VLLM_INT4_MQ_3D=1` | on | 10, 10 (collapse) | 6, 6 (runaway) |
| int4 per-token-head (`alternative.sh`), `VLLM_INT4_MQ_3D=0` | off | 14, 14 | 18, 18 |
| KVarN k4v2 (`CTX=huge`) | n/a | 16, 18 | 16, 14 |

The same model as a Q4_K_M GGUF on llama.cpp: 15, 15 and 14, 14, no collapse. So: not the
weights, not the draft or speculation (on in every vLLM row), not the int4 cache itself; the 3D
verify path under int4.

Caveat that orders the next step: every int4 row above ran on the image built from ff41191, which
predates the #57 merge. Whether the collapse survives on the current image is the first question.

## Reproduction

```
# the production tier, 3D path on (the container env sets it; the launcher's default is off)
docker run -d --name int4-3d --gpus '"device=<uuid>"' --ipc host -p 127.0.0.1:18025:18025 \
  -v <cache-volume>:/cache -v <models>:/app/models -e HOME=/cache -e PORT=18025 -e VLLM_API_KEY=<key> \
  -e CTX=long -e SPEC=dflash2 -e DFLASH_TOKENS=7 -e PREFIX_CACHE=1 -e VLLM_INT4_MQ_3D=1 -e MAX_SEQS=4 \
  -e GPU_UTIL=0.93 -e MAX_LEN=244224 --entrypoint bash <image> \
  -c 'cd /app && echo "$VLLM_API_KEY" > api_key.txt && exec bash single-user/alternative.sh'
# the conversation (flightbench, https://github.com/cpuchip/flightbench)
KEY=<key> BASE=http://127.0.0.1:18025/v1 MODEL=qwen3.8-27b THINK=off python benches/mission.py
```

Expected on ff41191: the replies for transmissions 6 through 11 are the repeated word; the trace
(`v6-*.trace.jsonl` beside the results file) shows the polling loop in the LOI station first.
With `VLLM_INT4_MQ_3D=0`: 14/18, coherent to the end.

## Ruled out, and not yet

Ruled out: the weights, the draft, speculation as such, the int4 cache as such, the runtime as a
whole (bf16, int8, KVarN fly it), context length alone (the 2D arm is clean at the same length).
Not yet: whether the current image (post-#57) still shows it; prefix caching (on in every row);
the draft width (7 in every row; the 3D path exists for the multi-query verify batch, so the
batch width is the regime); whether onset tracks the repetition or the token count (a padded
prefix moves the count without the repetition).

## Where to look

The trigger is a long sequence of near-identical multi-query verify batches over near-identical
cached content. The first places: the 3D kernel's segment reduce (`softmax_segm_*`) on repeated
content, the per-token-head scales on identical rows, and the dispatch reason bits under
`VLLM_INT4_MQ_3D_DEBUG=1` in the turns before the collapse (a capacity fallback, a buffer
disagreement, or a `DEGRADED` line would each say something different).

## Narrowed (22:20Z): the current image does not collapse, and the eager kernel is clean

Two runs on the current image (2bbd292, the post-#57 tree), same tier, 3D on: 15/18 and 17/18,
every reply finish=stop, no repetition. The collapse reproduces on the production image (ff41191)
and not on the image four commits later. Five patches sit between them. Two are no-ops in this
launch (marlin-tune-table needs an extension that is not installed; triton-prefill-attn-int8 is
opt-in and unset), so three candidates remain:

- spec-decode-scratch-token-units (#57, Patch A): scratch sized in tokens, 8 rows to 32 on this
  config; the old check was memory-safe (total tokens against rows), so the old image routed only
  batches of 8 tokens or fewer to 3D, a subset of what the current image routes there.
- spec-sampler-prewarm (#48): the rejection sampler's Triton kernels compile at boot instead of
  inside the first request that verifies drafts.
- mamba-align-checkpoint-order (#52): keeps the conversation's mamba state snapshots alive, which
  changes what a prefix-cache hit resumes from. The one candidate that touches prefix caching.

The kernel sweep (`bench/mq3d_sweep.py`, built on `bench/mq3d_layer2_oracle.py`): the 3D leg, the
2D leg, and an fp32 dequantized reference, on both images, real geometry (24 query heads, 4 KV
heads, head size 256), history 48 to 16,384 tokens, batches [1], [8], [8,8], [8,8,8,8],
[1,1,1,1], two fills (random; a 24-token pattern repeated down the sequence with 2% noise, the
shape of a polling loop's cached content), plus a phase at the production image's real scratch
sizing (8 rows, the capture-size snap) with 4,096-element canaries on both sides of every scratch
buffer. 160 rows per image:

| | 3D vs reference | 2D vs reference | max abs(3D - 2D) | NaN | canary touched |
|---|---|---|---|---|---|
| production image (ff41191) | 0.041 | 0.041 | 0.016 | none | none |
| current image (2bbd292) | 0.041 | 0.041 | 0.016 | none | none |

Identical numbers on both images, at fp16 rounding. The eager kernel does not produce the
collapse. First control row on the production image with `PREFIX_CACHE=0`, 3D on: 16/18, no
repetition (one run so far).

Running now, one container at a time, then the production image with each candidate applied at
boot (`patch -p1` on the installed package): prefix cache off (2 runs), draft width 3 (2), padded
prefixes 4k and 8k (1 each), Patch A, mamba-align, sampler-prewarm. Whichever arm stops the
collapse names the fix. The dimension not yet toggled is CUDA graph replay: the eager sweep cannot
see a captured graph's baked dispatch, and the old sizing was itself shaped by the capture snap.

## Correction (22:45Z): the 3D split was confounded; the collapse has only ever been observed in one container

Every collapse row so far came from one long-running production container: the one that also
serves the KV-tier comparisons' "int4" column. That container runs the CPU offload KV connector
(`--kv-transfer-config OffloadingConnector`, 20 GB of pinned CPU memory) and had been serving for
20 hours. No control arm had the connector: the 2D arm, the KV-tier arms, and every matrix arm
were fresh containers without it. So the earlier sentence "the defect is the MQ-3D verify path"
rested on a comparison that moved two things at once (the 3D switch, and the container).

Struck, and replaced by what the new arms say:

- Production image, 3D on, prefix cache on, fresh container, padded prefix 4k: 15/18, no
  repetition (the plain condition the collapse was attributed to, without the connector).
- Production image, 3D on, prefix cache off: 16/18 and 16/18.
- The long-running container, same hour, 3D on: 10/18 with the eight LOI-and-after misses (the
  collapse). Its connector counters moved during that run: CPU-to-GPU load batches 33 to 37.

The connector's load path is the one condition present in every collapse and absent from every
clean row.

~~A relevant number: for the int4 per-token-head tier the KV spec's page size is 32,768 bytes per
16-token block while the kernel's block is 16,896 bytes.~~ Struck (23:05Z): that number came from
a hand-built spec at a 16-token block. A native boot with the connector and a print in the offload
worker's cache registration, at the shipped 1,696-token scheduler block, shows the spec's page
(1,790,976 bytes = 4 heads x 1,696 slots x 264 bytes) equal to the kernel view's block stride and
to the copy stride on all 21 attention layers, non-packed (block_stride 0). The tier's layout is
not what the connector's copy breaks. The same boot recorded two facts about the configuration:
the connector requires prefix caching on this hybrid model (its config asserts it, to align block
sizes across the Mamba and attention groups), and the engine logs that prefix caching in Mamba
cache "align" mode is experimental. So the region is align-mode Mamba prefix caching plus the
connector's CPU-to-GPU reloads, on a fresh container as much as on the long-running one.

Running, fresh containers on card 1 with the connector and a churn side-load that forces reloads
between turns: int4 3D on, bf16, int4 3D off, int8, and the current image. If bf16 collapses too,
the KV tier is out of it entirely and the defect is the connector's reload against this hybrid
model's Mamba state; if only the per-token-head tiers collapse, the tier is still in the chain.

## The engine died where the collapse was expected (23:03Z)

First connector arm, fresh container, production image, int4, 3D on, prefix cache on, pool pinned to
35,313 tokens, CPU region 1.5 GB, a side-load of 21,000-token prompts every few seconds so the
conversation's blocks leave the GPU between turns and come back from CPU:

- Run 1 (side-load mis-sized, only three reloads): 15/18, no repetition.
- Run 2 (side-load live, one 21k prompt every ~14 s): the engine core died in the scheduler, in
  `kv_cache_manager.truncate_computed_blocks`, at `assert num_blocks <= len(group_blocks)`.
  Server log: flightbench `results/raw/dave/server-off_int4_3d.log`.

The assert sits in stock vLLM 0.27.1 (no fork patch touches it): when a request's local prefix hit
ends in a partial block and the connector reports a strictly longer external hit, the scheduler
truncates the local hit to the block boundary across every KV group. On this hybrid model the
attention group and the Mamba group do not hit to the same depth (the Mamba group can only resume
at a retained state snapshot; the scheduler calls this a diverged hit), so the Mamba group's
block list is shorter than the truncation point, and the assert fires. The neighbouring branch is
the one to watch: a diverged hit with external tokens present is accepted as is, on the assumption
that the connector's load carries a valid Mamba state for the deeper boundary. If it does not, the
request resumes with attention KV for a span whose recurrent state is wrong, silently. That is a
candidate mechanism for the collapse: it needs prefix caching, a hybrid model, the connector's
loads, and a diverged hit at the same time, which is why a fresh container rarely shows it and a
server that has served for hours does. It does not depend on the KV tier or the 3D path.

Reproduced (23:28Z, n=2): a second fresh int4 container, same connector and pin, the side-load correctly
sized from the start (2.4 GB CPU region, one 21k-token prompt every ~14 s), died on the same assert at
the same line during its first mission run, eight minutes after serving. Server log:
`results/raw/dave/server-off_int4_3d_b.log` in flightbench.

n=3 (23:36Z): int4 with the 3D path off, same connector and pin, same assert, first run, four minutes
after serving. The 3D path is out of the crash. n=4 (23:44Z): the int8 per-token-head tier (the
other launcher, `start_qwen.sh CTX=long`, no pad, 12k-token side-load) died in the same family but
at a second site: the connector's own scheduler, `offloading/scheduler.py` `update_state_after_alloc`,
`assert num_locally_computed_tokens <= num_locally_computed_gpu_blocks * tokens_per_block`, walking
the KV groups with one local token count. Same shape as the first: a single count aligned to the
scheduler's block, asserted per group on a model whose groups do not agree.

n=5 (23:52Z): the current image (2bbd292, with the #52 snapshot-retention patch, Patch A and the sampler prewarm), int4, 3D on: same assert, same line, first run. None of the patches between the images touches this.

Second machine (00:46Z): the same recipe on a native RTX 3090 under Ubuntu, no container, no WSL2, died on the same assert 13 requests and about seven minutes in. Container, WSL2 and card are out. The stack itself proves the connector reported a strictly longer external hit: the scheduler calls `truncate_computed_blocks` only inside the branch guarded by `partial_tail and ext_tokens > partial_tail`.

Two additions from an independent read of the same stock files on a second machine (code read,
not a reproduction): the line above the failing assert, `num_computed_tokens % manager.block_size
== 0`, is the same shape (one token count aligned to the scheduler's single block size, asserted
against every group's own block size), so a reproduction may land on either; and the connector's
only guard is at construction (its config asserts that hybrid models enable prefix caching "to
align block sizes"), which cannot see a hit that diverges forty turns in. Nothing in this path is
the fork's: the #46 and #57 lineage is not in it. The report, if one is written, is against
vLLM, not this repository.

Still running: the connector with the whale's own pool (no pin), then bf16, int4 with 3D off, int8,
and the current image (whose #52 patch changes snapshot retention, not this reconciliation).
