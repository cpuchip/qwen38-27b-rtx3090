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

Not ported: KVarN (`kvarn/`, `CTX=huge`). Its 0.28.0 patches do not apply on 0.29 and the Dockerfile
skips `kvarn/install.sh` with a note. Separate port.

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

The 47k decode column is bimodal per prompt slice on both images (rows land near 89 or near 104), so
the medians differ by draw, not by version; with prefix caching off both images read 102 to 104.

One knob worth knowing: 0.29 defaults `prefix_cache_retention_interval` to dense checkpointing for
hybrid models with a draft model (the same behaviour 0.28 had). Setting it to 0 on this model halves
the 47k prefill (1303 vs 2323 tok/s). Leave it at the default.
