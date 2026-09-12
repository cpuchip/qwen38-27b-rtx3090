# The patch series, one line each

What every file in `patches/` (and `kvarn/`) is, where it came from, and what retires it. The Dockerfile applies
them in filename order onto the installed vLLM wheel; `verify.sh` checks each one is in place. Kinds:

- **backport**: a merged or open upstream change carried early. Retires when the pin carries it.
- **fix**: a defect in upstream or in this stack, fixable upstream. Retires when upstream takes it.
- **feature**: something upstream does not have. Stays until upstreamed as a feature.
- **local**: this hardware or environment (WSL2, sm80, a tuned build, env knobs). Stays.
- **own**: a fix to a feature this repo introduced. Rides with that feature.

Cut against: the pin the current hunks were generated on (0.29.0 unless noted). "Retired hunks" lists what
the 0.29 port dropped from the file and why (details in `docs/vllm-0.29.md`).

| patch | kind | what | upstream | cut against | retires when |
|---|---|---|---|---|---|
| dflash2-backport | backport, RETIRED | DFlash2 speculator on 0.27.1 | vllm #52816 (in 0.28.0) | 0.27.1 | done; kept for history, skipped by the Dockerfile |
| dflash2-lookup-drafting | feature | lookup-augmented drafting for DFlash2 (n-gram search over the context) | none | 0.29.0, regenerated (32 hunks) | upstreamed |
| dflash2-ngram-chains | feature | quantized candidate chains for the drafter; `propose` override | none | 0.29.0 (`dp_sync` signature) | upstreamed |
| dflash2-prewarm | fix | compile every DFlash2 rung at boot instead of at first request | none yet | 0.29.0, regenerated (CP args on the launch path) | upstream PR |
| dflash2-z-adaptive-emitted | fix | adaptive z counts emitted tokens, not sampling slots | none yet | 0.28.0 | upstream PR |
| dspark-draft-quant-config | fix | bf16 DSpark drafter beside a quantized target (callable `hf_overrides`) | none yet | 0.28.0 | upstream PR |
| hybrid-kv-groups-v2-cudagraph | fix | KV group sizing when the smallest bucket is the drafter's sliding-window layers | none yet | 0.29.0; graph-reserve hunk retired (vLLM profiles it) | upstream PR |
| hybrid-sw-block-promote | fix | promote a draft SW layer's block to a divisor of the primary block instead of padding its page | none yet (upstream pads) | 0.29.0 (pad check mirrors upstream's non-MLA rule) | upstream PR |
| int4-kv-per-token-head | feature | int4 per-token-head KV cache with the DFlash2 drafter | none | 0.29.0; padded-page view hunk retired (layout strides) | upstreamed |
| mamba-align-checkpoint-order | fix | keep reachable Mamba state snapshots alive until request end (fork #52) | vllm #45238 (not merged) | 0.28.0 | check against upstream #52789 (internal prefill checkpoints, in 0.29) at each pin |
| mamba-chunked-prefill-align | fix | state loss and NaN during chunked prefill on Mamba/GDN | none yet | 0.28.0 | upstream PR |
| marlin-int8-layer-select | local | env vars to pick which layers run W4A8 with the Marlin kernel | none | 0.28.0 | stays |
| marlin-int8-negative-scales | fix | Marlin W4A8 reads group scales as unsigned; AutoRound exports negative ones | none yet | 0.28.0 | upstream PR |
| marlin-repack-staged-sm80 | local | one grow-only staging buffer for the sm80 Marlin repack (fork #27) | none | 0.28.0 | stays |
| marlin-tune-table | local | wiring for a locally built tunable Marlin extension, off by default | none | 0.27.1 source | stays |
| offload-dflash-eagle-groups | fix | OffloadingConnector under dflash flagged every KV group as draft attention (fork #33) | none yet | 0.28.0 | upstream PR |
| offload-wsl2-devptr | local | CPU offload tier device pointers on WSL2 | none | 0.28.0 | stays |
| qwen3_5-embed-quant | fix | pass `quant_config` to the token embedding (main model and MTP module) | none yet | 0.28.0 | upstream PR |
| qwen3_5-mtp-draft-vocab | feature | vocab-truncated draft head for MTP | none | 0.28.0 | upstreamed |
| sampler-small-topk-fast-softmax | feature | sort-free top-k/top-p for small k, multi-block row softmax | none | 0.28.0 | upstreamed or superseded |
| spec-decode-attn | feature | split-KV verify attention on FLASH_ATTN with query-row tiling | none | 0.28.0 | upstreamed |
| spec-decode-int4-kv-mq3d | feature | multi-query 3D int4 verify path | none | 0.28.0 | rides with int4-kv-per-token-head |
| spec-decode-int8-kv | feature | split-KV verify attention over an int8 per-token-head cache | none | 0.28.0 | rides with spec-decode-attn |
| spec-decode-scratch-token-units | own | mq3d scratch sized in tokens, not sequences (fork #46, #57) | none | 0.28.0 | rides with mq3d |
| spec-decode-scratch-within-budget | own | mq3d scratch allocated inside the memory budget (fork #57) | none | 0.28.0 | rides with mq3d |
| spec-sampler-prewarm | fix | compile the rejection sampler's Triton kernels at boot (fork #48) | none yet | 0.28.0 | upstream PR |
| speed-knobs-envs | local | register this repo's env knobs in `envs.py` | none | 0.28.0 | stays while the knobs exist |
| triton-prefill-attn-int8 | feature | int8-QK Triton prefill attention for head_dim 256 | none | 0.28.0 | upstreamed |
| vision-tower-cpu-offload | local | Qwen3 vision tower bulk weights in host RAM | none | 0.28.0 | stays |
| vllm-pr50021-gdn-spec-bounds | backport | bounds checks in GDN/KDA spec-decode state lookups | vllm #50021 (open) | 0.28.0 | the pin that carries #50021 |
| kvarn/kvarn-0.29.0 | feature | KVarN cache dtypes, quant mode, backend registration, page size | none (KVarN is Huawei CSL's, Apache-2.0) | 0.29.0; attn_utils view hunk retired | upstreamed |
| kvarn/kvarn-v2-runner-0.29.0 | own | KVarN with the V2 runner and DFlash2 (SW groups, Mamba block index, selector guards) | none | 0.29.0; kv_cache_utils hunks retired | rides with KVarN |

Retired at 0.29.0 and removed from the tree: `vllm-pr54282-draft-gumbel-salt` (vllm #54282, in 0.29.0) and
`xgrammar-spec-terminated` (in 0.29.0).

Two files still carry raw `diff -ruN` headers with timestamps instead of a preamble (`dflash2-z-adaptive-emitted`,
`offload-wsl2-devptr`); their descriptions live in `docs/gotchas.md` and `docs/MR-DRAFT.md` until they get one.
