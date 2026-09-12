# The vLLM 0.29.0 pin

What the move from 0.28.0 to 0.29.0 changed in this repo, and what was re-measured.

[← back to the main README](../README.md)

## Dependencies

`vllm==0.29.0` and `huggingface_hub==1.28.0` (0.29 requires it; 1.27 conflicts). torch 2.13.0 and
compressed-tensors 0.17.0 are unchanged. `verify.sh` checks the 0.29.0 pin.

## Patch series

Retired, because upstream carries the change:

- `vllm-pr54282-draft-gumbel-salt.patch` (in 0.29.0)
- `xgrammar-spec-terminated.patch` (in 0.29.0)
- the graph-memory reserve hunk of `hybrid-kv-groups-v2-cudagraph.patch` (0.29 profiles graph memory
  natively; the kv_cache_utils hunks stay)
- the padded-page view hunk of `int4-kv-per-token-head.patch` (0.29's layout strides cover it; the
  triton kernel hunks stay)

Regenerated against the 0.29 tree, same behaviour:

- `dflash2-prewarm.patch` (the launch path gained the context-parallel arguments)
- `dflash2-lookup-drafting.patch` (32 hunks; import placement and the selector-walk anchor moved)

Adjusted for 0.29 API changes:

- `ngram-chains`: the `propose` override takes and forwards `dp_sync` (new runner signature).
- `hybrid-sw-block-promote`: `AttentionSpec.indexes_kv_by_block_stride` is gone; the "can this
  layer pad" check now mirrors upstream's own pad branch, any non-MLA attention layer pads. Reading
  the removed flag with a default of False refused every promotion, so the int4 profile padded the
  five drafter layers at block 16 and could not fit 120k tokens; that is the failure to look for if
  the promotion lines stop appearing at boot.

KVarN (`kvarn/`, `CTX=huge`) is ported: `kvarn-0.29.0.patch` and `kvarn-v2-runner-0.29.0.patch`. The
layout refactor removed the per-backend shape and stride hooks, so the backend now declares its layout
(`LBHNC`: heads outside tokens within a block) and folds the runner's 4D per-layer view back into one
tile per block and head with a `view`, so a wrong layout fails at the first KV update instead of
returning wrong numbers (the first port declared `LBNHC` and did exactly that; the guard caught it).
The old strided-view hunk and the four block-size hunks are retired; their reasons are in the patch
preambles and in `kvarn/README.md`.

## Acceptance (WSL2 4090, card 1, 2026-09-12)

Same script on the 0.28.0 image and the 0.29.0 image, fresh cache volume per run.

| profile | 0.28.0 | 0.29.0 |
|---|---|---|
| fast (dflash2 k=7, prefix caching): ppl / GSM8K n=100 | 10.8437 / 0.95 | 10.8437 / 0.95 |
| fast depth 25k: prefill / decode tok/s | 2678 / 105.0 | 2685 / 103.2 |
| fast depth 47k: prefill / decode tok/s | 2324 / 89.7 | 2323 / 97.9 |
| long (mtp, align, prefix caching): prefix ladder and 60k/160k peaks | pass | pass, same numbers |
| int4 (`kv_cache_dtype=int4_per_token_head`, 120k): KV tokens, mq3d oracle | 179,701, 8/8 | 173,134, 8/8 |
| int4 depth 25k / 90k decode tok/s | 42.1 / 19.5 | 43.2 / 19.1 |
| offload (12 GiB tier, mtp): served after eviction, tier guard | pass | pass |
| huge (KVarN k4v2_g128, dflash2, 262k): block / KV tokens | 2176 / 268,169 | 2176 / 268,169 |
| huge: ppl en / da, GSM8K n=100 | 10.7674 / 10.9097, 0.89 | 10.7691 / 10.9085, 0.93 |
| huge: needle at 32k / 90k / 200k (thinking off) | see note | retrieved at all three |
| huge: request time at 25k / 90k (256 output tokens) | 30.2 s / 125.6 s | 22.4 s / 67.8 s |

The 47k decode column is bimodal per prompt slice on both images (rows land near 89 or near 104), so
the medians differ by draw, not by version; with prefix caching off both images read 102 to 104.

The huge-context rows: the two images compute the same KV geometry and the same drafter acceptance, and
0.29 finishes both requests sooner; the split of that time between first token and streaming differs
between the two boots and is being measured with a stream probe (the 0.28 boot's needle answers came
back empty with thinking left on, which is the probe's default, not the cache's fault). The 0.28 needle
column is the fixed probe's rerun, pending at the time of writing.

One knob worth knowing: 0.29 defaults `prefix_cache_retention_interval` to dense checkpointing for
hybrid models with a draft model (the same behaviour 0.28 had). Setting it to 0 on this model halves
the 47k prefill (1303 vs 2323 tok/s). Leave it at the default.
