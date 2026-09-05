# Speculative decoding: state of the art and where our headroom is, September 2026

Research brief for the qwen38-27b-rtx3090 fork (Qwen3.8-27B W4A16 on RTX 3090/4090, batch 1,
DFlash2 drafter + suffix lookup). Compiled 2026-09-04, finished 2026-09-05. Every claim carries a
URL and the date it was read. Anything I could not confirm against a primary source is marked
UNVERIFIED. The upstream picture moves fast enough that one of the fixes below (#54374) merged the
same day this was written, so re-run the `v0.28.0...main` filter before acting on the Q4 tables.

Our measured baseline, as given in the brief: acceptance per drafted token 30 to 38 percent,
about 2.5 accepted of 7 drafted per round, 3.2 to 3.7 tokens per step.

## Executive summary

1. Our 2.5-of-7 is not a broken drafter. It is what a healthy 7-deep block drafter with roughly
   flat per-position acceptance of about 0.74 produces, and EAGLE's own table (Vicuna 7B, batch 1,
   a single RTX 3090) shows the same shape. Arithmetic in Q2.
2. So "acceptance per drafted token 30 to 38 percent" is a depth artifact, not a defect. Treating
   it as the per-position rate will mislead the next two days.
3. **Three DFlash bugfixes merged upstream after v0.28.0 and we have none of them.** #54374 leaves
   FlashAttention's AOT schedule on for a sliding-window DFlash drafter and collapses acceptance to
   1.0 under full CUDA graphs whenever the target is not on FlashAttention, which is both
   `CTX=long` (FLASHINFER) and `CTX=huge` (TRITON_ATTN).
4. #54282 is worse in kind: on v0.28.0 the fix we adopted for issue #73
   (`draft_sample_method: "probabilistic"`) makes the output distribution no longer the target's.
   The README's losslessness claim does not hold in that configuration.
5. vLLM v0.28.0 already ships block verification (Sun et al., arXiv 2403.10444) behind
   `rejection_sample_method: "block"`, in the exact V2 GPU kernel our DFlash2 path uses. We run the
   default. Lossless, provably never worse, "5%-8%" reported, one config key.
6. The largest structural waste: the selector computes `selector_top_k = 16` candidates at each of
   7 positions and discards 105 of 112. DDTree verifies a tree over exactly those and reports tau
   7.79 to 10.73 on Qwen3-8B MATH-500, same drafter, no retraining.
7. A block-16 DFlash2 checkpoint exists and is architecturally identical to ours apart from block
   size, but the DFlash paper's Table 8 shows a block-8 drafter run at block 16 is *worse* than at
   block 8. Our clamp is correct; a longer block is a training job.
8. Adaptive draft length is a batch-size-larger-than-one win. DSpark's own paper says trimming
   saves little at batch 1. At batch 1 the win is a better static k, not adaptation.
9. vLLM 0.29.0 is not released; v0.28.0 shipped 2026-08-26 and cadence suggests mid-September. No
   release tracker exists. Determinism: `VLLM_BATCH_INVARIANT` hard-fails on Gated DeltaNet
   (open issue #42960), so batch invariance is closed to this model family.
10. Cheapest high-information move we are not making: vLLM already logs a per-position acceptance
    vector, and `rejection_sample_method: "synthetic"` prices any acceptance curve on our own
    silicon without training anything.

---

## Q1. State of the art, September 2026

### What actually moved in 2026

Three things changed the field in 2026, and only one of them is "a better drafter":

* **Block/parallel drafting replaced chained drafting.** DFlash and DFlash2 propose a whole
  block in one forward pass instead of k chained passes. Acceptance length is roughly the same
  as EAGLE-3; the speedup is higher because the drafter costs one pass, not k.
* **The verification algorithm became a first-class knob.** Block verification, tree
  verification, and confidence-scheduled verification length are all 2024 to 2026 work that
  raises accepted tokens per step *without touching the drafter*.
* **Acceptance decay in parallel drafters became the named problem.** DSpark's abstract states
  it directly: parallel drafters "suffer from rapid acceptance decay due to a lack of
  inter-token dependencies."

### Model-based drafters

| Method | What it changes | Reported acceptance / speedup | Batch / GPU of that number | Training | vLLM support | Source (read 2026-09-04) |
|---|---|---|---|---|---|---|
| EAGLE-3 | Autoregressive feature-level drafter, training-time test, multi-layer feature fusion | Baseline that DFlash is measured against; on Qwen3-4B 5-layer: GSM8K tau 4.2 at 2.1x, HumanEval 4.3 at 2.2x, MT-Bench 3.1 at 1.4x | Not stated in the LMSYS table; DFlash comparison run | Yes, draft head | Yes, `method: "eagle3"` | [arXiv 2503.01840](https://arxiv.org/pdf/2503.01840); [LMSYS 2026-06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/) |
| DFlash (block diffusion) | Whole block in one pass, KV injection of target representations into the draft KV cache | Paper abstract: "over 6x lossless acceleration" and "up to 2.5x higher speedup than the state-of-the-art speculative decoding method EAGLE-3". LMSYS Qwen3-4B 5-layer: GSM8K tau 4.2 at 3.3x, HumanEval 4.0 at 3.2x, MT-Bench 3.0 at 2.2x. Qwen3.5-397B-A17B: >4.3x over baseline, 1.5x over MTP at concurrency 1, block size 16 | 8x B200 for the 397B row; paper hardware not checked | Yes | Yes, `method: "dflash"` | [arXiv 2602.06036](https://arxiv.org/abs/2602.06036), Chen, Liang and Liu, ICML 2026, v1 2026-02-05, v2 2026-05-28; [LMSYS 2026-06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/) |
| **DFlash2 (ours)** | Adds two-tap dynamic convolution backbone and a low-rank candidate selector over top-k candidates per position | Acceptance length: GSM8K 5.46, MATH-500 5.28, HumanEval 4.39, MBPP 4.79, MT-Bench 4.10. Up to 3.43x on GSM8K | **Concurrency 1, single H200** | Yes (checkpoint published) | Yes, native in v0.28.0 (PR #52816) | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| DSpark | Semi-autoregressive: parallel backbone plus a lightweight *sequential* module to restore intra-block dependency, plus confidence-scheduled verification length | 60 to 85 percent faster per-user generation vs MTP-1 at matched throughput, in DeepSeek-V4 production | 8x B300, TP=8, concurrency swept 1 to 256, T=1.0 | Yes, includes a confidence head | Yes, `method: "dspark"` (PR #47808) | [arXiv 2607.05147](https://arxiv.org/abs/2607.05147); [vLLM blog 2026-08-14](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) |
| Medusa-style multi-head | Independent heads per future position, typical-acceptance verification | See Q3; superseded by EAGLE-3 and block drafters in every 2026 comparison I found | n/a | Yes | Yes, `method: "medusa"` | [vLLM speculative_decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) |

Notes on this table:

* The DFlash2 acceptance-length figures are the closest published reference to our setup: same
  checkpoint family, same drafter, concurrency 1. But they are H200 and bf16 with a bf16 target;
  we run a W4A16 target with an int4-requantized drafter. The tokens-per-step comparison is fair;
  the tokens-per-second comparison is not.
* Our own MT-Bench-like cohort sits at 3.2 to 3.7 against their 4.10. That is a 10 to 25 percent
  acceptance-length gap that is not explained by the drafter weights: `drafter/README.md` records
  that int4 GPTQ keeps greedy acceptance (3.34 to 3.65 vs 3.54 for bf16) and costs about 5 percent
  under sampling (3.2 vs 3.4). The remaining gap is workload mix and, plausibly, the fact that the
  drafter was distilled against a bf16 target and serves an int4 one. That hypothesis is
  UNVERIFIED and is not a two-day experiment.

### Block length: a block-16 DFlash2 exists, and running ours at 16 is worse than at 8

This is the most directly actionable finding about our own configuration, and it validates a
design choice we made for other reasons.

**A block-16 DFlash2 checkpoint is published today.** `z-lab/Muse-Glimmer-30B-DFlash2` (mirrored at
`incoai/Muse-Glimmer-30B-DFlash2`) declares `"block_size": 16` and is otherwise **architecturally
identical** to our block-8 Qwen3.8-27B drafter: same `conv_kernel_size` 2, same `conv_group_size`
16, same `selector_rank` 256, same `selector_top_k` 16, same five hidden layers, same
`sliding_window` 2048 ([config.json](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2/raw/main/config.json),
read 2026-09-04). Block size and the target-layer ids are the only differences. Its published
acceptance at concurrency 1 on one H200, 15 draft tokens per step, SGLang and FA3, T=1.0, top-p
0.95, top-k 64:

| Task | Official DFlash | DSpark | DFlash 2 (block 16) |
|---|---|---|---|
| GSM8K | 5.43 | 5.45 | **6.57** |
| MATH-500 | 5.39 | 5.01 | **6.56** |
| HumanEval | 4.11 | 4.33 | **5.66** |
| MBPP | 3.74 | 4.02 | **5.30** |
| MT-Bench | 3.52 | 3.59 | **4.42** |

Concurrency-1 throughput 4.59x GSM8K, 3.08x MT-Bench
([model card](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2/raw/main/README.md), read
2026-09-04). **Carry the caveat:** the target model is different (Muse Glimmer 30B, not
Qwen3.8-27B), so 4.42 against our reference 4.10 on MT-Bench is not a clean block-16-versus-block-8
A/B. Block 8 and block 16 are the only two DFlash2 block sizes published.

**Running a block-8 drafter at block 16 is worse than running it at block 8.** DFlash paper section
5.5.4, Table 8, 8-layer drafters with 5 target features, tau:

| Train block | Test block | Math500 | HumanEval | MT-Bench |
|---|---|---|---|---|
| 16 | 16 | 6.33 | 5.29 | 3.50 |
| 16 | 8 | 5.09 | 4.44 | 3.18 |
| **8** | **16** | **5.02** | **4.28** | **3.09** |
| 8 | 8 | 5.21 | 4.61 | 3.29 |

The authors state that larger-block models generalize well to smaller inference block sizes and
"the reverse does not hold". **This is exactly why our clamp is correct.** `DFLASH_TOKENS=15` does
not ask the drafter for 15 tokens; the lookup patch clamps the emitted block to the checkpoint's
trained 7 and fills the remainder from context. Table 8 is the citation for why that clamp exists,
and it means the honest description of `DFLASH_TOKENS=15` is "a longer verify block filled by two
sources", not "a longer draft".

The same table also reports that the block-8 model **fully accepts the entire block 35.7 percent of
the time on Math500**, so block 8 is frequently saturated on structured text. That is the strongest
published argument for a longer trained block on math and code workloads specifically.

**What it would cost to train one.** No source publishes GPU-hours, wall-clock, or step counts for
any DFlash or DFlash2 drafter. Mark that **UNVERIFIED** and do not let anyone infer it. What is
stated ([NVIDIA NeMo AutoModel DFlash recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding),
read 2026-09-04, plus the DFlash paper appendix A.1):

* **Data:** about 800K samples (Nemotron Post-Training Dataset V2 plus CodeAlpaca) with responses
  regenerated by the target model. DFlare (arXiv 2606.02091) scaled 800K to 2.4M for roughly
  +11 percent on Qwen3-4B, so 800K is the working floor and 2.4M the diminishing-returns ceiling.
* **Optimization:** 6 epochs, AdamW, lr 6e-4, grad clip 1.0, cosine schedule, warmup ratio 0.04,
  max sequence length 3072, 512 anchor positions sampled per sequence. Only the draft stack, its
  feature projection and its norms train; `embed_tokens` and `lm_head` are frozen and shared.
* **Memory wall:** multi-GPU wraps the draft in DDP and replicates the frozen target on every rank.
  The **offline mode**, which precomputes and caches target hidden states so the target never runs
  during drafter optimization, is the only path that is tractable on consumer cards. Cache size
  grows linearly with the number of extracted feature layers (we extract 5).
* **Warm start is precedented.** The Muse Glimmer DFlash2 card states it "is finetuned from
  `meta-models/Muse-Glimmer-30B-assistant`, the official DFlash drafter Meta ships with the model".
  A shipped block-16 DFlash2 was produced by finetuning an existing DFlash v1 drafter, not trained
  from scratch. The cost of that finetune is UNVERIFIED.
* **If block size changes, `loss_decay_gamma` must change with it:** NeMo specifies 7 for block 16,
  5 for 10, 4 for 8. `mask_token_id` has no default and NVIDIA warns explicitly against reusing
  `pad`, because it is frequently aliased to `eos` and "quietly erodes acceptance".

**One training objective worth a look.** NeMo supports `loss_type: variable_prefix`, the D2SD
VP-Drafter objective (arXiv 2606.04446), where each block draws a visible-prefix length from a
truncated geometric prior, only the masked suffix is supervised, and the decay restarts at the
prefix boundary. NVIDIA's stated rationale is that it "matches the regime a drafter sees when it
re-drafts behind a partially accepted block". At batch 1, partial acceptance is the norm rather
than the exception, so that objective is better aligned with our workload than the fixed-anchor
default. I did not read the paper; treat this as a lead.

Honest summary of the training question: **unknown compute, known data volume, known warm-start
path.** Nothing in the sources supports "two months" and nothing supports "two weeks".


### Tree and graph verification

This is where 2026 put its gains, and it is directly applicable to us.

| Method | What it changes | Reported numbers | Batch / GPU | Training | vLLM support | Source (read 2026-09-04) |
|---|---|---|---|---|---|---|
| **DDTree** | Builds a draft tree from a block-diffusion drafter's per-position distributions, best-first heap, verified in one pass with ancestor-only attention masking | Qwen3-8B MATH-500 T=0: DFlash 5.56x at tau 7.79, DDTree **7.52x at tau 10.73** (+38 percent tau). Qwen3-4B 2.46x to 7.27x, Qwen3-30B-Coder up to 8.22x | 8x H200, bf16, drafter block size 16, node budgets swept over {16,32,64,128,256,512,1024}. Batch size not stated | **No** (uses the existing DFlash drafter) | No | [arXiv 2604.12989](https://arxiv.org/abs/2604.12989); code [github.com/liranringel/ddtree](https://github.com/liranringel/ddtree) |
| CaDDTree | Chooses the node budget per round from the current per-position distributions and profiled verification cost; proves throughput is unimodal under convex verification cost, so a greedy stopping rule suffices | "matches or surpasses DDTree with oracle budget selection on nearly all tasks", 8 benchmarks | Qwen3-4B and Qwen3-8B; GPU and batch not stated in the abstract | No | No | [arXiv 2606.01813](https://arxiv.org/abs/2606.01813) |
| Graft ("Draft Less, Retrieve More") | Prune low-confidence draft branches at calibrated checkpoints, graft retrieved n-gram candidates into the freed slots, verify the hybrid tree in one pass | Vicuna-13B 5.41x HumanEval at MAT 8.53; LLaMA3.1-8B 4.28x avg; Qwen3-235B 2.09x avg (+21.8 percent over EAGLE-3); T=1 holds at 3.43x on LLaMA3.1-8B. **Preliminary DFlash block drafting: +9.1 percent, 3.40x to 3.71x** | 8x H20; batch sizes 1 to 16 evaluated in serving | **No**, "training-free and lossless" | No | [arXiv 2605.20104](https://arxiv.org/html/2605.20104v1) |
| TreeGraft | Multiple drafters of different cost jointly build one tree; value-guided scheduling decides when to invoke the expensive drafter | 1.60x avg over 10 model pairs and 6 benchmarks, max 2.30x, +15.1 percent over the better single-drafter endpoint; avg accepted length 2.17 | Batch size and GPU not stated; no consumer-GPU results | No | No | [arXiv 2608.26112](https://arxiv.org/html/2608.26112) |
| SpecInfer / Sequoia / EAGLE tree attention | The 2024 lineage that established tree verification: multiple candidate paths verified in one pass with a tree attention mask | Not re-measured here; superseded for our purposes by DDTree, which is the block-diffusion-specific version | n/a | Varies | EAGLE tree attention: partial, via `method: "eagle3"` | UNVERIFIED for 2026-specific numbers |

**Why DDTree matters more to us than any other row in this document.** Read the actual DFlash2
config we serve ([z-lab/Qwen3.8-27B-DFlash2/config.json](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/raw/main/config.json),
read 2026-09-04):

```json
"dflash_config": {
  "block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
  "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5,19,33,47,61]
}
```

`selector_top_k: 16`. The drafter computes 16 candidate tokens at each of the 7 draft positions,
112 in total, and `CandidateSelector` in
`vllm/model_executor/models/qwen3_dflash2.py` (v0.28.0 tag, read 2026-09-04) walks a successor
table to pick a single coherent path through them. We verify 7 tokens and throw away 105 already
computed candidates every step. DDTree is precisely the algorithm for spending some of those.

### Training-free lookup drafting

| Method | What it changes | Reported numbers | Training | vLLM support | Source (read 2026-09-04) |
|---|---|---|---|---|---|
| Prompt Lookup Decoding | Match the generation suffix against the prompt, propose the continuation | Wins hard on quote/edit/RAG workloads, near-zero on free prose. This matches our own measurement: 130 tok/s on chat, up to 381 reproducing context | No | Effectively `method: "ngram"` | Our README; [vLLM docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) |
| SuffixDecoding | Suffix tree over prompt, previous outputs, and a cached corpus; speculation depth adapts to prefix match length | "speedups of up to 5.3x" on agentic benchmarks; "2.8x faster than model-based approaches like EAGLE-2/3 and 1.9x faster than model-free approaches such as Token Recycling" | No | Yes, `method: "suffix"`, but **requires Arctic Inference installed** (`_validate_suffix_decoding` raises ImportError otherwise) | [arXiv 2411.04975](https://arxiv.org/abs/2411.04975); v0.28.0 `vllm/config/speculative.py` |
| vLLM `ngram` / `ngram_gpu` | Built-in n-gram proposer; `prompt_lookup_min`/`prompt_lookup_max` default 5 | No published tau in the docs | No | Yes | [vLLM docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) |

vLLM's `suffix` method exposes exactly the adaptive-depth knob we hand-rolled:
`suffix_decoding_max_spec_factor` ("max_spec_tokens = max_spec_factor * prefix_match_length",
default 1.0) and `suffix_decoding_min_token_prob` (default 0.1). Our `_NMIN`/`_NMAX`/`_NSTRONG`
env vars are a reimplementation of the same idea with more hand-tuning. Worth reading their
policy even though we cannot swap methods (one `method` per engine).

**On combining a model drafter with lookup:** this is now a named research direction and our
architecture is an instance of it. Graft's framing is the useful one: model-based proposals
stabilize the lower bound, retrieval raises the upper bound, and *confidence pruning decides
which positions are unreliable*. We fill positions past the block boundary from context. Graft
fills positions the drafter is *not confident about*, wherever they sit. That is a strictly more
general policy and it is what our `z-adaptive-emitted` argument has been circling.

**The hybrid is measured at batch 1 on a single H100, which is the closest published rig to ours.**
SuffixDecoding's own paper (v2 HTML section 4.2, Llama-3.1-8B, batch 1, one H100) states the
policy plainly: "we speculate with the faster SuffixDecoding method whenever possible and fall
back to EAGLE-3 when the speculation score is too low", where the score estimates how many tokens
would be accepted.

| Workload | Hybrid | Standalone EAGLE-3 | Standalone suffix | Note |
|---|---|---|---|---|
| Spec-Bench (high entropy, non-agentic) | **2.5x** | 2.4x | lower | Token Recycling 2.2x |
| AgenticSQL | 4.1x at **7.5 mean accepted tokens/step** | about 2x lower accepted length | **5.3x at 6.3 accepted tokens/step** | suffix wins on wall time with the *lower* acceptance length |
| SWE-Bench | n/a | EAGLE-2/3 failed outright on several tasks above 8192 context | 7.8 accepted tok/step vs PLD's 3.2 | |

**The line to take from that table: accepted tokens per step is the wrong objective.** The hybrid
had the higher acceptance length on AgenticSQL and lost on wall time, "because its much lower
speculation cost and higher acceptance rate make it the winning solution". The quantity to
maximize is accepted tokens per unit of *drafting cost*. That is worth remembering before we spend
two days buying acceptance length with a bigger verify block.

Arctic Inference productionized the same fallback rule and published the ablation (Llama-3.1-70B,
8x H100 TP=8 FP8, 0.5 req/s, output tok/s, [Snowflake engineering blog, 2025-05-01](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/),
read 2026-09-04):

| Workload | No spec | N-gram | EAGLE | LSTM only | Suffix only | LSTM + Suffix |
|---|---|---|---|---|---|---|
| ShareGPT | 76.0 | 91.2 | 102 | 172 | 113 | **179** |
| HumanEval | 77.2 | 100 | 112 | 204 | 148 | **217** |
| SWE-Bench | 75.8 | 175 | n/a | 123 | 286 | **302** |
| Mixed | 82.9 | 112 | n/a | 154 | 155 | **209** |

The hybrid matches or beats both endpoints on every workload. That is the strongest evidence in
this document that our drafter-plus-lookup architecture is the right shape, and it suggests the
policy question (which source fills which position) is where the remaining value sits, not the
choice of drafter.


### Self-speculative and layer-skip

Medium-high confidence that this family does not compete at batch 1. The 2026 work exists and it
all lands in the same band, well below block drafting's 3x-plus at concurrency 1. All read
2026-09-04.

| Method | Reported speedup | Training | vLLM | Source |
|---|---|---|---|---|
| SWIFT (ICLR 2025) | 1.3x to 1.6x | none | no | [arXiv 2410.06916](https://arxiv.org/abs/2410.06916) |
| ConfLayers (2026-04-16) | up to 1.4x | none | no | [arXiv 2604.14612](https://arxiv.org/abs/2604.14612) |
| KnapSpec (ICML 2026) | up to 1.47x | none | no | [arXiv 2602.20217](https://arxiv.org/abs/2602.20217) |
| S3D | 1.4x to 2x versus a **quantized** EAGLE, at half precision and less VRAM | minimal | no | [arXiv 2405.20314](https://arxiv.org/abs/2405.20314) |

The ceiling is about 1.5x and **none of these are in vLLM**. Skip the family. S3D is the only one
written for our constraint, and it makes one claim worth carrying regardless: under limited memory
and quantization, a method that performs well on a high-end GPU "can slow down by up to 7 times".
That is the general warning against reading any number in this document off an H200 and assuming
it survives the trip to a 3090.


### AMD cross-check

The vLLM AMD blog ([2026-08-23](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus),
read 2026-09-04) benchmarked native MTP, Gemma 4 MTP, EAGLE-3, DFlash and DSpark on MI300X and
MI355X. Reported mean accepted lengths ranged 1.8 to 7.6; DFlash reached 2.87x on
Gemma-4-26B-A4B-it MATH500 and 2.68x on Kimi-K2.5. It explicitly did **not** test n-gram or
suffix decoding. Useful mainly as an independent confirmation that DFlash-family drafters lead
the learned-drafter field in mid-2026.

---

## Q2. Per-position acceptance and draft length at batch 1

### The theory, from the primary source

Leviathan, Kalman and Matias, "Fast Inference from Transformers via Speculative Decoding",
[arXiv 2211.17192](https://ar5iv.labs.arxiv.org/html/2211.17192) (read 2026-09-04):

* Definition 3.1: the acceptance rate is "the probability of accepting x_t ~ q(x_t | x_<t)".
* Definition 3.7: c, the cost coefficient, is "the ratio between the time for a single run of
  M_q and the time for a single run of M_p".
* Expected tokens per iteration: `E = (1 - alpha^(gamma+1)) / (1 - alpha)`, described as "a
  capped geometric distribution with success probability (1-alpha) and cap (gamma+1)".
* Expected walltime improvement factor: `(1 - alpha^(gamma+1)) / ((1 - alpha)(gamma*c + 1))`.

### Applying it to our numbers

This section is my derivation from the brief's measurements, not a literature result. Take it as
a model to falsify, not as a finding.

Solve `(1 - a^8)/(1 - a) = 3.50` for a uniform per-position conditional acceptance a:

| a | a^8 | tokens/step | sum of a^i, i=1..7 (expected accepted drafts) |
|---|---|---|---|
| 0.72 | 0.0722 | 3.31 | 2.31 |
| **0.74** | **0.0899** | **3.50** | **2.50** |
| 0.75 | 0.1001 | 3.60 | 2.61 |
| 0.78 | 0.1370 | 3.92 | 2.92 |

a = 0.74 reproduces **both** reported figures at once: 3.50 tokens per step and 2.50 accepted of
7 drafted. Our README independently records DFlash2 position-0 acceptance at roughly 75 to 78
percent. So within measurement noise, **our drafter's conditional acceptance is close to flat
across the block**, which is exactly what DFlash2's two-tap dynamic convolution is advertised to
achieve. Allowing position 1 to be 0.78 and the rest uniform gives a tail rate of about 0.72,
which is a mild decay, not a collapse.

Two consequences:

1. **30 to 38 percent "acceptance per drafted token" is the geometric-decay artifact.** With flat
   conditional acceptance 0.74, the fraction of *drafted* tokens accepted at depth 7 is
   2.50/7 = 0.357. That number falls automatically as we draft deeper. It is not a quality signal
   and it should not be a KPI.
2. **Extending the block is worth less than raising acceptance.** At a = 0.74, positions 8 through
   15 would add `sum a^i for i=8..15` = 0.315 tokens per step, taking 3.50 to 3.81, about
   +9 percent, and that is the *ceiling* assuming the extension keeps the same acceptance. Raising
   a from 0.74 to 0.80 at the same depth 7 takes us to 4.16, about +19 percent. Raising it to 0.85
   gives 4.85, about +39 percent. Acceptance quality is worth roughly twice what block length is,
   per unit of gain.

That second point is corroborated by DDTree's decomposition-free result: it kept the drafter and
the block and got +38 percent tau by widening the *verified set* rather than lengthening it.

### What the literature reports for the decay curve

**The only complete published per-position table, and its 7B rows were measured on a single
RTX 3090 at batch 1.** EAGLE Table 2 ([ar5iv 2401.15077](https://ar5iv.labs.arxiv.org/html/2401.15077),
read 2026-09-04), caption "Average acceptance length tau and acceptance rate alpha on MT-bench.
T denotes temperature." The paper states: "For Vicuna 7B as the target LLM, operating under a memory
constraint of a single RTX 3090 with 24G of CUDA memory", "In the case of LLaMA2-Chat 70B,
constrained by 4 A100 (40G) GPUs", and on batch size, "the majority of our experiments also adopted
this setting", meaning batch size 1. Definition: 0-alpha is the acceptance rate with entirely
accurate feature predictions and n-alpha the rate with n erroneous features, so the row reads as
per-position conditional acceptance, the same quantity as my `a`. Note the 7B target is fp16, not
int4, and the drafter is autoregressive rather than a block drafter, so this is the right *shape*
comparison and not a like-for-like one.

| Model | T | tau | 0-a | 1-a | 2-a | 3-a | 4-a |
|---|---|---|---|---|---|---|---|
| Vicuna 7B (RTX 3090) | 0 | 3.94 | 0.79 | 0.74 | 0.72 | 0.73 | 0.67 |
| Vicuna 13B | 0 | 3.98 | 0.79 | 0.74 | 0.72 | 0.74 | 0.70 |
| Vicuna 33B | 0 | 3.68 | 0.74 | 0.69 | 0.67 | 0.67 | 0.66 |
| LLaMA2-Chat 7B | 0 | 3.62 | 0.76 | 0.69 | 0.67 | 0.68 | 0.68 |
| LLaMA2-Chat 13B | 0 | 3.90 | 0.77 | 0.69 | 0.69 | 0.70 | 0.71 |
| LLaMA2-Chat 70B | 0 | 3.81 | 0.75 | 0.69 | 0.65 | 0.64 | 0.64 |
| Vicuna 7B (RTX 3090) | 1 | 3.17 | 0.71 | 0.68 | 0.66 | 0.66 | 0.65 |
| LLaMA2-Chat 13B | 1 | 3.45 | 0.73 | 0.69 | 0.66 | 0.67 | 0.67 |

**This supports the flat fit and improves the model.** Almost the entire drop is between position 0
and position 1 (about 5 points), after which the curve flattens to 1 or 2 points per step, and
several rows are non-monotonic (Vicuna 7B goes 0.72 then 0.73; LLaMA2-13B goes 0.69, 0.70, 0.71).
**"One step down, then a floor" fits better than alpha^i.** Our flat 0.74 sits in the middle of
every T=0 row.

Other curves, and the one that should worry us:

| Source | Curve | Setting |
|---|---|---|
| [SpecBlock, arXiv 2605.07243](https://arxiv.org/html/2605.07243v2) | alpha_1 above 0.80 falling to **alpha_8 about 0.544 on Llama-3.1-8B and about 0.369 on Qwen3-8B** | Batch 1, A100-80GB, T=0 and T=1 |
| [vLLM DSpark blog, 2026-08-14](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) | "the last drafted token of a 7-token block survives less than 10% of the time, against more than 70% for the first" (cumulative survival) | DeepSeek-V4-Pro, production |
| [vLLM AMD blog, 2026-08-23](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus) | Gemma 4 MTP GSM8K: "94% acceptance at position 1, declining to 66% at position 5", N=5 | MI300X/MI355X |
| [arXiv 2604.14682](https://arxiv.org/html/2604.14682v1) | TinyLlama-1.1B drafting **Llama-2-7B-Chat-GPTQ 4-bit**, greedy: depth 1/2/3 acceptance 0.567/0.553/0.588 on chat, essentially flat to slightly increasing | 2x Tesla T4, the closest published int4-target setup |
| [DSpark, arXiv 2607.05147](https://arxiv.org/abs/2607.05147) | Names "rapid acceptance decay due to a lack of inter-token dependencies" as the defining failure of parallel drafters | DeepSeek-V4 |

The SpecBlock row is the counter-evidence to take seriously: **Qwen models decay much harder than
Llama models** at depth (0.369 vs 0.544 at position 8). If that holds for Qwen3.8-27B, our positions
6 and 7 are worth less than a flat model predicts, and the payoff from lengthening the block is
smaller than my +9 percent estimate. Reading our own per-position vector settles it.

Note also the DSpark curve (>70 percent first, <10 percent last of 7) is cumulative survival and
tracks a flat-a curve at a ~ 0.74 almost exactly (0.74^7 = 0.122). DeepSeek's production drafter
and ours decay at the same rate. We are not an outlier.

### Adaptive draft length

| Approach | Mechanism | Reported effect | Batch / GPU | Available to us? |
|---|---|---|---|---|
| **DSpark confidence-scheduled verification** | Learned confidence head per drafted token; scheduler converts to cumulative survival, picks budget B by maximizing expected tokens per unit step time against a profiled cost table | Part of the 60 to 85 percent per-user speedup | 8x B300, concurrency 1 to 256 | **No.** `enable_adaptive_verification` raises `ValueError("Adaptive verification only supported with DSpark")` in v0.28.0, and additionally requires every attention builder to report `AttentionCGSupport.ALWAYS`, which only SM100 sparse-MLA backends do. Verified in `config/speculative.py` and `v1/worker/gpu/spec_decode/adaptive_verification.py` at the v0.28.0 tag |
| **DISCO** (Intel Labs, Weizmann, HUJI) | A *classifier*, not a bare threshold. Inputs are the top-10 draft logits, the draft entropy, **and the token's position i**. Halts when confidence falls below tau, capped at SL_max so tensor shapes stay fixed. Output text is identical | **About 10 percent average over the best static lookahead.** MBPP 1.84x vs 1.64x static-optimal (1.46x at SL=5); HumanEval 1.84x vs 1.63x; CNN-DM 2.15x vs 1.85x; Alpaca 2.12x vs 2.02x | A100 80GB; batch size not reported | Portable. The classifier is small and its inputs are things we already compute |
| **SpecDec++** | Trained acceptance-prediction head on the drafter; stop when P(at least one token rejected) crosses a threshold | 2.04x Alpaca (+7.2 percent over baseline speculative decoding), 2.26x GSM8K (+9.4), 2.23x HumanEval (+11.1) | Not reported; Llama-2-chat 7B drafting 70B | Needs a trained head |
| **SGLang adaptive speculative decoding** | EMA of *accepted length*, switching between pre-captured CUDA-graph tiers: `target_steps ~ clamp(round(ema_accept_len)+1, min, max)`. A tier switch is a reference swap, never a recapture | No speedup published. Config: ema_alpha 0.2, update every 5 verify batches, 10 warmup batches. **Tiers for batch 1 to 7 are [1, 3, 7]** | Explicitly tiered by batch size | The cheapest item on this list. Roughly fifteen lines, no training |
| Learning To Draft (LTD), 2026-03-02 | RL depth policy over frontier log-probs plus depth and sequence length; reward is accept_len / (draft_time + verify_time). Explicitly does *not* use per-position acceptance statistics | vs static EAGLE-3: Qwen3-32B **+36.4 percent**, DeepSeek-8B +9.5, Llama-3-8B +6.5. Beats SpecDec++ and DISCO in its Table 1 | Not specified | Research only |
| CaDDTree | Per-round *node budget* from the current per-position distributions plus a profiled verification cost; greedy stopping rule justified by unimodality | "matches or surpasses DDTree with oracle budget selection" | Qwen3-4B/8B, GPU not stated | Not implemented anywhere; the rule is simple enough to port |
| SuffixDecoding spec factor | `max_spec_tokens = max_spec_factor * prefix_match_length` | Not isolated | n/a | Yes, if we switch to `method: "suffix"` (needs Arctic Inference) |
| AdaEDL | Training-free and parameter-free; draft-logit entropy lower-bounds expected acceptance, stop when it falls | **UNVERIFIED.** The PDF would not decode on repeated attempts and no secondary source carries a number | unknown | Lead, not evidence |
| vLLM `num_speculative_tokens_per_batch_size` | Batch-size schedule for k, present in v0.28.0 | n/a | n/a | Present, irrelevant at batch 1 |
| **Ours** | `VLLM_DFLASH2_LOOKUP_ADAPTIVE`: schedule the long verify block only while the lookup is firing, `_STICKY=3` hold | The long block is worth +50 percent where the model reproduces its context and about +9 percent on the short-prompt C1 set | RTX 3090/4090, batch 1 | Shipping |

Sources, all read 2026-09-04: [SpecDec++ arXiv 2405.19715](https://arxiv.org/abs/2405.19715) (COLM 2025);
[DISCO arXiv 2405.04304v5](https://arxiv.org/html/2405.04304v5);
[SGLang adaptive speculative decoding](https://docs.sglang.io/advanced_features/adaptive_speculative_decoding.html);
[LTD arXiv 2603.01639](https://arxiv.org/html/2603.01639v1);
[AdaEDL arXiv 2410.18351](https://arxiv.org/abs/2410.18351). vLLM's own dynamic-speculation-length
RFC ([issue #36657](https://github.com/vllm-project/vllm/issues/36657), PR #35301) proposed a
`draft_confidence_threshold` with a batch-mean exit and reported 1.77x on Llama-3.1-8B/3.2-1B; it
was **closed as not planned**. PACER (arXiv 2602.01274) and GammaTune (arXiv 2504.00030) are
described as using online acceptance statistics, but neither PDF would parse and **zero numbers
from either are verified**. Leads only.

**The batch-1 caveat, which is the most important sentence in this section.** DSpark's own paper
concedes that at batch size 1 the target's verify step barely slows as tokens are added, so
trimming the draft saves almost nothing; their gains come from reclaiming *batch capacity* at
concurrency. Every adaptive result above except SGLang's is measured at server batch sizes on
A100-class hardware. **At batch 1 the win is choosing a better static k, not adapting it.** That
is a concrete reason to rank adaptive-k work below verification-algorithm work on this stack, and
it argues against porting DISCO or SpecDec++ on the strength of their headline numbers.

**The honest gap in what we ship.** Our adaptive policy is binary (short block or long block) and
it is triggered by *lookup match length*, not by *estimated acceptance*. We already have the
estimate: vLLM 0.28.0's `vllm/v1/spec_decode/metrics.py` computes and logs
`acceptance_rates = sum(pos_matrix, axis=0) / num_drafts`, a per-position unconditional acceptance
vector, logged as "Per-position acceptance rate: %s". We are not feeding it back into anything.
If we do want adaptivity, SGLang's EMA rule is the best effort-to-value option on this list.

I found **no** paper studying the specific case of a fixed trained block length with cheaper fill
beyond the boundary. Our configuration appears to be novel; Graft's prune-then-graft is the
closest framing. UNVERIFIED whether anyone has measured the acceptance discontinuity at the
boundary. Our own instrumentation could answer it in an afternoon.

---

## Q3. Drafting under sampling rather than greedy

### The thing we already found, restated

`docs/drafts/issue-73-report.draft.md` is the best evidence in this document because it is ours
and it is paired. Setting `draft_sample_method: "probabilistic"` on the DFlash2 config, RTX 4090,
4 seeds by 8 prompts, paired:

| stratum | cells | shipped | field set | t |
|---|---|---|---|---|
| overall | 32 | 3.440 | 3.730 | 2.77 |
| drafted tokens per round = 7 | 18 | 2.886 | **3.425** | 4.22 |
| drafted tokens per round > 7 | 14 | 4.152 | 4.123 | -0.22 |

Verified against the v0.28.0 source: `draft_sample_method` defaults to `"greedy"`, whose docstring
says "the draft probabilities are treated as one-hot during rejection sampling"
(`vllm/config/speculative.py`, v0.28.0 tag, read 2026-09-04). With one-hot q the ratio test
`p(x)/q(x) > u` degenerates to `p(x) > u`, which is strictly stricter. That is the mechanism.

**Action item before any other sampling experiment:** `single-user/start_qwen.sh:231` in this
checkout builds `SPEC_CFG` as
`{"method":"dflash","model":"$DRAFT","num_speculative_tokens":$DRAFT_TOKENS}` with no
`draft_sample_method`. The upstream maintainer fixed this himself in `0e95195`
(`docs/upstream-tracker.md`). Confirm the fix is in the tree being benchmarked, or every sampling
number below is measured on the wrong baseline.

### Verification algorithms that raise acceptance under sampling

| Method | What it does | Reported effect | In vLLM 0.28.0? | Source (read 2026-09-04) |
|---|---|---|---|---|
| **Block verification** | Verifies the whole draft block jointly using cumulative joint ratios and residual mass, instead of one token at a time. Proven optimal in expected accepted length over algorithms using on-path probabilities, and never worse than token-level | "modest but consistent wall-clock speedups over token verification of 5%-8% across various tasks and datasets" | **Yes.** `rejection_sample_method: "block"` | [arXiv 2403.10444](https://arxiv.org/html/2403.10444v2); implementation verified below |
| Typical acceptance (Medusa) | Accepts a draft token when the *target* probability clears `min(epsilon, delta * exp(-H(p_target)))`. Lossy: output no longer matches the target distribution. At T=0 it reverts to greedy | Reported as "**about 10 percent speedup over greedy decoding**", with the authors noting that speculative decoding with random sampling actually *slowed down* relative to greedy in the same comparison. So the honest framing is that typical acceptance turns a regression into a modest win at temperature, not that it beats rejection sampling by a margin. Reference defaults are `posterior_threshold = 0.09`, `posterior_alpha = 0.3` | Not in the CUDA path I read. Present in vLLM **Ascend** as `enable_entropy_verify` under `additional_config.rejection_sampler_config`, with different defaults (`posterior_threshold` 0.95, `posterior_alpha` 0.4) | [arXiv 2401.10774](https://arxiv.org/html/2401.10774v1); [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa); [vLLM Ascend guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html) |
| Multi-draft / optimal transport | SpecTr and successors; block-level optimal transport formulation of verification | SpecTr-GBV, arXiv 2604.25925; Greedy Multi-Path Block Verification, arXiv 2602.16961 | No | UNVERIFIED numbers, abstracts only |
| Truncated-vocabulary drafting (SlimSpec) | Restricts the drafter to a truncated vocabulary and proves `alpha >= 1 - (1 - p_T(S_T)) / p_M(S_T)`: acceptance is capped by the target probability mass inside the truncated set, regardless of drafter quality or size | Coverage on Llama-3.1-8B/MT-Bench, H100 batch 1 greedy: 4K vocab about 94 percent, 8K about 97 percent, 16K about 99 percent | No | [arXiv 2605.10453](https://arxiv.org/pdf/2605.10453) |

**Block verification is in our exact code path, and it is off.** Verified by reading the v0.28.0
tag source, not the docs:

* `vllm/config/speculative.py:80` at the v0.28.0 tag:
  `RejectionSampleMethod = Literal["standard", "synthetic", "block"]`, and
  `rejection_sample_method: RejectionSampleMethod = "standard"` at line 219. The docstring on
  `main` reads: "'block' uses block verification (Sun et al.), which jointly verifies the draft
  tokens as a block instead of one at a time."
* `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` at the v0.28.0 tag contains the
  kernel branch, commented
  `# Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444`, computing
  `prefix_joint_ratio`, a `residual_mass` over the next draft token, and accepting via
  `h = residual_mass / (residual_mass + 1 - prefix_joint_ratio)`.
* That file is the **V2 GPU runner** sampler. Our README states DFlash2 forces vLLM's V2 model
  runner (`_is_dflash2_draft()`), so this is the sampler our stack actually runs.
* The branch is guarded by `if is_greedy: ... elif USE_BLOCK_VERIFICATION:`, so **it acts only on
  non-greedy requests**. Our default-sampling cohort qualifies; our greedy benchmarks will show
  exactly zero change, which is a useful negative control.
* It handles `HAS_DRAFT_LOGITS` both ways, so it does not require
  `draft_sample_method: "probabilistic"`, though the two should be tested together.

**Why I expect this to pay more for us than the paper's 5 to 8 percent.** Our lookup-filled
positions get point-mass draft logits (`_point_mass_draft_logits_kernel` in
`patches/dflash2-lookup-drafting.patch`). Under standard token-level rejection sampling, a
point-mass q means the acceptance test collapses to `p(x) > u`, which at temperature 1 over a
broad target distribution is a low bar to clear only rarely. Block verification's joint-ratio and
residual-mass formulation is precisely the construction that recovers expected length where
token-level verification is loose. That is a mechanism argument, not a measurement, and it is the
single most falsifiable prediction in this document.

### Temperature effects

Measured greedy-versus-temperature deltas, all read 2026-09-04:

| Pair | T=0 | T=1 | Setting |
|---|---|---|---|
| T5-XXL EnDe + T5-small | alpha 0.75 | alpha 0.62 | Leviathan Table 3, [arXiv 2211.17192](https://ar5iv.labs.arxiv.org/html/2211.17192) |
| T5-XXL CNNDM + T5-small | alpha 0.65 | alpha 0.53 | same |
| LaMDA 137B + LaMDA 8B | alpha 0.75 | alpha 0.74 | same |
| EAGLE Vicuna 7B | tau 3.94, 2.90x | tau 3.17, 2.21x | **single RTX 3090, batch 1, fp16**, [ar5iv 2401.15077](https://ar5iv.labs.arxiv.org/html/2401.15077) |
| EAGLE LLaMA2-Chat 70B | tau 3.81, 3.01x | tau 3.46, 2.92x | 4x A100-40G, batch 1 |
| EAGLE-2 Vicuna 13B | tau 4.83, 4.26x | tau 4.40, 3.80x | batch 1, hardware not stated, [ar5iv 2406.16858](https://ar5iv.labs.arxiv.org/html/2406.16858) |
| Llama-2-13B-chat + Llama-68M | reference | about 30 percent relative speedup drop across T=0 to T=1.0 | A100-40G, batch 1, [arXiv 2410.10141](https://arxiv.org/html/2410.10141v1) |
| **Ours, DFlash2** | 3.34 tokens/step | 3.14 at model-default sampling | RTX 3090, batch 1, README single-stream table |

Three things worth carrying:

* **The penalty is pair-specific and large.** LaMDA loses one point of alpha going to T=1; the T5
  pairs lose twelve or thirteen. There is no universal number.
* **Speedup falls harder than alpha does**, because tau compounds per-position losses. EAGLE's
  Vicuna 7B loses about 8 points of 0-alpha and 24 percent of its speedup.
* **Nobody publishes T=0.7.** Every primary source reports T=0 and T=1 only. The closest nearby
  measurement is a PayPal EAGLE-3 deployment reporting acceptance essentially unchanged between
  T=0 and T=0.5, within plus or minus one point across concurrency 1 to 32
  ([arXiv 2604.19767](https://arxiv.org/pdf/2604.19767)), which suggests the penalty out to 0.7 is
  much milder than the T=1 figures imply. Our own delta (3.34 to 3.14) is consistent with that.

**Temperature-matched drafter distillation is the one training-side fix with a number on it.**
Aligning the knowledge-distillation temperature to the decoding temperature gave 2.34x at 83.6
percent acceptance on Alpaca and 5.62x at 89.5 percent on GSM8K with KD temperatures in
{0.9, 0.8, 0.7}, A100-40G, batch 1 ([arXiv 2410.10141](https://arxiv.org/html/2410.10141v1)). This
matters to us because our drafter was distilled at whatever temperature z-lab used and we serve at
the model default. It is not a two-day experiment, but it is the strongest argument in this
document for eventually re-distilling rather than re-quantizing.

**Truncation-aware drafting is a confirmed gap, with one useful bound.** I found no work at all on
matching a drafter to a target that has had top-k or top-p applied before verification, which is
exactly what vLLM does to our target while the drafter was trained against the untruncated
distribution. The adjacent case is studied: SlimSpec proves that with a drafter restricted to a
truncated *vocabulary*, acceptance is bounded by
`alpha >= 1 - (1 - p_T(S_T)) / p_M(S_T)`, that is, capped by the target mass inside the truncated
set no matter how good the drafter is, and it flags the same objective-versus-acceptance mismatch
we would expect on the target side. Carrying that bound to target-side top-p truncation is an
inference, not a cited result.

**One caution before anyone reaches for typical acceptance.** A vLLM user measured typical
acceptance at default hyperparameters producing a *shorter* accepted length than plain rejection
sampling (T=0.9, MT-Bench, Llama-3.1-8B with a Qwama-0.5B draft, 2 speculative tokens); the issue
was closed with no maintainer answer
([vLLM issue #8639](https://github.com/vllm-project/vllm/issues/8639)). Typical acceptance is also
lossy by construction. Block verification is lossless and provably never worse, which is why it
ranks above typical acceptance in the proposals below.


---

## Q4. vLLM 0.28 to 0.29

### Is 0.29 out?

**No.** Verified 2026-09-04:

| Source | Finding |
|---|---|
| [PyPI release history](https://pypi.org/project/vllm/#history) | Newest is **0.28.0, 2026-08-26**. Preceding: 0.27.1 (2026-08-11), 0.27.0 (2026-08-10), 0.26.0 (2026-07-25), 0.25.1 (2026-07-14), 0.25.0 (2026-07-11). No 0.29 or 0.29 pre-release listed |
| [GitHub releases](https://github.com/vllm-project/vllm/releases) | Newest tag is v0.28.0. No v0.29.0 |
| Search for a 0.29 release-tracker issue | None found |

**Expected date: mid-September 2026, UNVERIFIED.** The inference is from cadence alone:
0.26.0 to 0.27.0 was 16 days, 0.27.0 to 0.28.0 was 16 days. Sixteen days after 2026-08-26 is
2026-09-11. There is no maintainer statement backing this, so plan for "0.28.0 is what we ship
on" and treat 0.29 as a thing to re-check, not a thing to wait for.

### What v0.28.0 itself changed

Read from the [v0.28.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)
on 2026-09-04. The release states "584 commits from 270 contributors (76 new)". These bullets are
as they appear in the release notes; I did not open each PR individually, so treat the PR numbers
as pointers rather than as verified descriptions.

| Area | Change | PR |
|---|---|---|
| Spec decode | DSpark confidence-scheduled verification | #47808 |
| Spec decode | top-k DSpark Markov projection | #49969 |
| Spec decode | **DFlash2 with local convolution and a candidate selector** | #52816 |
| Spec decode | async scheduling auto-enabled for draft models | #48341 |
| Spec decode | fused MTP trailing all-reduce with local-argmax draft tokens | #49793 |
| Spec decode | adaptive budget for speculative scheduled input tokens | #51725 |
| Spec decode | EAGLE3 support declared on KimiLinear | #52171 |
| Spec decode | Qwen3.6 dSpark acceptance coverage | #51310 |
| Linear attention / Mamba | **prefix caching enabled by default for Mamba models** | #50991 |
| Linear attention / Mamba | final part of the Mamba attention module refactor | #44857 |
| Linear attention / Mamba | 3D-grid tiling of state-copy Triton kernels | #49436 |
| CUDA graphs | Blackwell CUDA graph capture default raised to 1024 | #49390 |
| CUDA graphs | encoder CUDA graphs | #49852 |
| Determinism | batch-invariant NVFP4 MoE via CUTLASS | #40372 |
| Triton autotuning | **No matching bullet found in the release notes** | n/a |
| Compile caching | **No matching bullet found in the release notes** | n/a |

### What is on main since v0.28.0, and three of them are ours

`main` is 858 commits ahead of `v0.28.0` as of 2026-09-04
(`api.github.com/repos/vllm-project/vllm/compare/v0.28.0...main`, read 2026-09-04). I pulled the
492 commits dated on or after 2026-08-26 and filtered them. **Three merged bugfixes land directly
on this stack, and all three are post-0.28.0, so we do not have them.** PR bodies quoted below
were read from the GitHub API on 2026-09-04.

#### The three that matter to us

**#54374, "Drop FlashAttention's AOT schedule for a sliding-window DFlash drafter", merged
2026-09-04.** From the PR body:

> A DFlash drafter runs sliding-window attention. FlashAttention's AOT split schedule is computed
> for one window configuration shared by every layer, and whether the drafter gets a correct one
> today is decided by accident. [...] target on any other backend (MLA, FlashInfer) -> the drafter
> is the only FlashAttention consumer -> 1 config -> AOT stays on, and the drafter runs a schedule
> built for the wrong geometry. [...] In the second case acceptance collapses to 1.0 under full
> CUDA graphs, and some configurations hit an illegal memory access in `flash_fwd_combine`.

Our drafter is sliding-window: the [z-lab config.json](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/raw/main/config.json)
declares `"layer_types"` as five `sliding_attention` layers with `"sliding_window": 2048`. And
`CTX=long` and `CTX=huge` put the **target** on FlashInfer for fp8 KV, which is precisely the
"drafter is the only FlashAttention consumer" case. `CTX=fast` keeps the target on FlashAttention
and is, in the PR author's words, "correct by coincidence".

Two of our own README entries describe this symptom. "On WSL2 that showed up as acceptance
collapsing to about one token per step; on bare metal it also corrupted the output." And, for the
FlashInfer backend, "four drafts crash the engine with an illegal memory access as soon as one
request finishes while another is mid-generation." We attributed the first to a graph residue
(fixed in `a75ee4b`) and the second to draft depth. Upstream has now landed a different root cause
for the same two symptoms. Whether `a75ee4b` fixed our instance or merely moved it is **UNVERIFIED**
and is a one-afternoon question.

**#54282, "Decouple the draft's gumbel noise stream from the target's", merged 2026-08-29.** This
one is a correctness bug in the exact configuration we just switched on. From the PR body:

> During target verification, the Gumbel noise used to re-sample a rejected draft token is drawn
> from the same Philox offset as the noise that produced that draft token (the same seed, pos), so
> byte-identical noise. [...] the residual under-weights exactly those tokens and the output
> distribution is no longer the target's.
>
> This affects `draft_sample_method="probabilistic"` only. Under the default greedy, `draft_logits`
> is None, the draft never calls `gumbel_sample`, and there is no shared noise vector.

Issue #73's fix was to set `draft_sample_method: "probabilistic"`. On v0.28.0 that fix buys about
+0.54 tokens per step in the drafter-carried stratum **and silently breaks losslessness**, because
this decoupling is not in v0.28.0. The README's "all of it is lossless" paragraph does not hold on
0.28.0 with the field set. This needs saying in the repo whether or not we act on it.

**#54373, "Take the DFlash draft's RoPE layout from its own config", merged 2026-08-31.** The 0.28.0
loader reads `is_neox_style` off the target and stamps it onto the draft config. The PR reports
that on GLM-5.3 this gives "acceptance 1.03 against 7.78 forced to neox", and a second case at
"acceptance 1.0 against 5.5". Our draft config.json does **not** declare `is_neox_style`, so on
0.28.0 it inherits the target's. If the two agree, nothing happens. Worth a five-minute check of
the target's value, given that the failure mode is a total acceptance collapse rather than a
gradual one.

#### Everything else, by area

| Area | Merged since v0.28.0 (selected) |
|---|---|
| DFlash / spec decode | #54374 AOT schedule (above), #54373 RoPE layout (above), #54282 gumbel decoupling (above), #54826 honour the draft's `attention_backend` on Model Runner V2, #53797 load dflash2 in speculators format, #54418 keep default CUDA graph sizes memory-safe, #50488 capture the widest uniform decode batch by default, #53962 do not pad spec decode up to `max_model_len`, #55245 fix spec decode warmup device selection, #54856 skip DP sync for speculator uniform decodes, #53388 support disabling trailing prefix-cache block dropping, #50514 eagle3 with pipeline parallel |
| Gated DeltaNet / Mamba | #55404 build cudagraph-capture metadata without a device sync, #54251 warm up Qwen GDN gated RMSNorm, #52506 FlashInfer ReplaySSM backend, #53877 keep packed GDN decode beta in FP32, #55178 preserve Mamba state for padded prompt tails, #54044 reset cached Mamba align metadata on profiling teardown, #55041 deprecate the "all" mamba cache mode, **#52743 fix the preprocessor guard so `fused_gdn_decode_post_conv_mtp` compiles on Ampere SM 8.6** |
| Triton / FlashInfer autotuning | **#54794 "Avoid flashinfer autotune each time when vllm source change"**, merged 2026-09-01. The PR shows a tuning run costing minutes per boot |
| torch.compile / CUDA graphs | **#53955 release CUDA graph profiling memory before KV cache allocation**, merged 2026-08-27, #55341 warm up kernels before capturing CUDA graphs, #54782 raise for unavailable piecewise CUDA graphs, #52358 `CUDAGraphStat` in MRV2 |
| Prefix caching | #53388 (above), #53598 serve prefix cache hits under DCP, #53920 warn on warm prefix cache for random serve runs |
| Determinism | **#49209 register matmul and linear batch-invariant kernels for XPU** and nothing else. No GDN work |

Two of these speak to our issue #75 directly. #54794 removes the FlashInfer autotune re-run that
our eight-cold-boot experiment was measuring. #53955 fixes CUDA-graph profiling memory not being
freed before `allocate_kv_cache`, which is a plausible mechanism for the 0.92 GiB KV-pool swing we
attributed to compile-cache warmth. Neither is in v0.28.0.

#### Config surface on main versus the tag

* `vllm/config/speculative.py` on `main` carries the same `RejectionSampleMethod` and
  `DraftSampleMethod` literals as the v0.28.0 tag, plus `dspark_draft_topk`.
* The [speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
  are marked "Developer preview (dated August 20, 2026)" and additionally list `method: "pard"`,
  `method: "dynamic"` ("Useful for RL or workload with fluctuating QPS"), and
  `method: "extract_hidden_states"`. Whether those are post-0.28 additions or documentation drift
  is **UNVERIFIED**.

### Config surface relevant to us, verified against the v0.28.0 tag

Read from `https://raw.githubusercontent.com/vllm-project/vllm/v0.28.0/vllm/config/speculative.py`
on 2026-09-04:

| Key | Default | Note |
|---|---|---|
| `rejection_sample_method` | `"standard"` | `standard` / `synthetic` / `block`. **We are on the default** |
| `draft_sample_method` | `"greedy"` | greedy treats draft probs as one-hot. See Q3 |
| `synthetic_acceptance_rates` | `None` | "Per-position *unconditional* acceptance rates ... must have length num_speculative_tokens, each entry in [0,1], and be monotonically non-increasing" |
| `synthetic_acceptance_length` | `None` | Target mean acceptance length, resolved internally to rates |
| `enable_adaptive_verification` | `False` | Raises `ValueError("Adaptive verification only supported with DSpark")` for any other method |
| `num_speculative_tokens_per_batch_size` | `None` | Batch-size schedule for k. Irrelevant at batch 1 |
| `parallel_drafting` | `False` | "Only compatible with EAGLE and draft model methods" |
| `suffix_decoding_max_spec_factor` | `1.0` | `max_spec_tokens = max_spec_factor * prefix_match_length` |
| `disable_padded_drafter_batch` | `False` | Effective for eagle, eagle3, mtp, dflash, draft_model |

---

## Q5. Numerical determinism

### What vLLM offers

`VLLM_BATCH_INVARIANT=1` selects batch-invariant kernels (single universal reduction strategy per
kernel, so results do not depend on how a token was batched). Coverage as of 2026:

| Path | Batch-invariant? | Evidence (read 2026-09-04) |
|---|---|---|
| Standard attention on SM80 | Yes | PR #42456, referenced in issue #42960 |
| NVFP4 MoE via CUTLASS | Yes, added in 0.28.0 | v0.28.0 release notes, PR #40372 |
| **Gated DeltaNet / linear attention (GDN_ATTN)** | **No. Hard failure at startup** | [vLLM issue #42960](https://github.com/vllm-project/vllm/issues/42960) |
| VLMs | No | [vLLM issue #27059](https://github.com/vllm-project/vllm/issues/27059) |

Issue #42960, "Batch-invariant support for GDN_ATTN (Qwen3-Next / Qwen3.6 hybrid Mamba+GDN MoE
models)", opened 2026-05-18, still **open**. The error is verbatim:

```
RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN.
```

It reports "a hard incompatibility with no fallback". PR #45819 is referenced as open and
possibly a fix; I could not confirm its state or scope, so mark that **UNVERIFIED**.

**Bottom line for us:** Qwen3.8-27B is a hybrid GDN model, so `VLLM_BATCH_INVARIANT=1` is closed
to this stack in 0.28.0 and, absent evidence of #45819 merging, in 0.29 as well. Do not plan
benchmark reproducibility around it.

### The determinism we actually have, and its three leaks

Our own `docs/upstream-tracker.md` and issue #75 already separate these correctly, and the
separation is better than anything I found upstream:

1. **Triton autotune draw affects the trajectory.** Eight cold boots produced eight distinct
   autotune winner sets, with trajectories spanning 58 to 189 tool calls, and the block-inverse
   merge kernel identical in all eight. That is a long tail, not a coin flip. A benchmark row is
   reproducible only with its autotune cache pinned.
2. **Compile-cache warmth affects the profiled memory peak** and therefore the KV pool by
   0.92 GiB.
3. **Position in the run** affects decode rate and time to first token.

I found no upstream vLLM feature that addresses (1). The v0.28.0 release notes contain no Triton
autotuning bullet. Pinning the cache directory ourselves remains the only lever.

Also worth carrying forward from `issue-73-report.draft.md`, because it is a determinism trap
that looks like a measurement: **"Two passes with a fixed seed are two replays, not two samples."**
Repeat runs came back bit-identical on every counter. Reproducibility to three significant figures
measures a deterministic harness, it does not bound an effect. Any A/B below must vary seeds.

---

## What is worth trying on this stack in two days

Ranked by expected value per engineer-hour, given a two-day window and the measurement discipline
this repo already has. All three are testable with the issue-73 protocol we already own.

### Proposal 1: cherry-pick the three post-0.28 DFlash fixes

**Change.** Backport #54374, #54373 and #54282 from vLLM `main` onto our pinned v0.28.0 tree, in
the same style as everything already in `patches/`. All three merged after the v0.28.0 tag, so we
have none of them.

**Why this is first.** Two of the three have a failure mode of *total acceptance collapse*, not a
gradual loss, and one of those matches a symptom this repo already chased and attributed elsewhere.
The third breaks losslessness in the configuration we just adopted. These are the highest expected
value hours in the two days because the downside is a null result and the upside is recovering
acceptance we did not know we were losing.

| PR | What it fixes | Why it applies to us | Test |
|---|---|---|---|
| **#54374** (merged 2026-09-04) | FlashAttention's AOT split schedule is left on for a sliding-window DFlash drafter when the drafter is the only FlashAttention consumer. "acceptance collapses to 1.0 under full CUDA graphs, and some configurations hit an illegal memory access in `flash_fwd_combine`" | Our drafter is sliding-window (`sliding_window: 2048`, five `sliding_attention` layers). `single-user/start_qwen.sh` sets the target backend per profile: `CTX=fast` is `FLASH_ATTN` (line 156), `CTX=long` is `FLASHINFER` (line 166), `CTX=huge` is `TRITON_ATTN` (line 182). **In both `CTX=long` and `CTX=huge` the drafter is the only FlashAttention consumer, which is exactly the triggering case.** Only `CTX=fast` is "correct by coincidence" | A/B `SPEC=dflash2 CTX=long` and `CTX=huge` before and after. Predicted: both improve, `CTX=fast` does not move. If `CTX=fast` moves, our model of the bug is wrong |
| **#54282** (merged 2026-08-29) | The draft's Gumbel noise shares a Philox offset with the target's resampling noise, so "the residual under-weights exactly those tokens and the output distribution is no longer the target's". Affects `draft_sample_method="probabilistic"` only | Issue #73's fix set exactly that field. On 0.28.0 it buys about +0.54 tokens per step in the drafter-carried stratum **and silently breaks losslessness** | A correctness fix, not a throughput fix. Test with the `docs/quality.md` gate (PPL, GSM8K); expect tokens per step to move slightly in either direction |
| **#54373** (merged 2026-08-31) | The 0.28.0 loader reads `is_neox_style` off the target and stamps it onto the draft. The PR reports "acceptance 1.03 against 7.78 forced to neox" on one model and "acceptance 1.0 against 5.5" on another | Our draft `config.json` does not declare `is_neox_style`, so it inherits the target's. If the two agree, nothing happens | Five-minute check of the target's `is_neox_style` before writing any patch. Only worth backporting if they disagree |

**What would make this fail.** #54374 may already be masked by `a75ee4b`, the graph-residue fix we
landed for the same symptom. If so the A/B returns a null and we have learned that our fix was the
real one, which is worth knowing and worth telling upstream. #54282 is a distribution fix, so a
throughput bench cannot validate it; only the quality gate can.

**Also worth pulling while the patch queue is open, both aimed at issue #75:** #54794 ("Avoid
flashinfer autotune each time when vllm source change", merged 2026-09-01) removes the autotune
re-run our eight-cold-boot experiment was measuring, and #53955 ("Release CUDA graph profiling
memory before KV cache allocation", merged 2026-08-27) is a plausible mechanism for the 0.92 GiB
KV-pool swing we attributed to compile-cache warmth.

**Cost.** One day, most of it benchmarking rather than patching.

### Proposal 2: turn on block verification. One config key.

**Change.** Add `"rejection_sample_method": "block"` to the DFlash2 `SPEC_CFG` in
`single-user/start_qwen.sh`. Confirm first that `"draft_sample_method": "probabilistic"` is present
in the tree being benchmarked: line 231 of this checkout does not set it, the v0.28.0 default is
`"greedy"`, and the upstream maintainer fixed it in `0e95195`.

**Evidence.**

* The Triton kernel exists at the v0.28.0 tag in
  `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`, in the V2 GPU sampler our DFlash2
  path forces, commented with the paper reference. Verified by reading the file, not the docs.
* Sun et al. ([arXiv 2403.10444](https://arxiv.org/html/2403.10444v2)) prove block verification is
  optimal in expected accepted length among algorithms using on-path probabilities and "never worse
  than standard token-level verification", and report "5%-8%" wall-clock across tasks.
* The gain should concentrate where our draft distribution is degenerate. Our lookup-filled
  positions carry point-mass draft logits, which makes the token-level test maximally loose; block
  verification is the construction that recovers that slack. **This is the most falsifiable
  prediction in this document.**
* It is lossless, unlike typical acceptance, which a vLLM user measured as producing a *shorter*
  accepted length than plain rejection sampling ([issue #8639](https://github.com/vllm-project/vllm/issues/8639)).

**Test.** The issue-73 protocol unchanged: cohort file at 1024 output tokens, 4 seeds by 8 prompts,
paired by seed and prompt, tokens per step as the metric, stratified on drafted tokens per round
(=7 versus >7), threshold registered before the data. Vary seeds; two fixed-seed passes are two
replays, not two samples. Run greedy as the null control: the kernel branch sits under
`if is_greedy: ... elif USE_BLOCK_VERIFICATION:`, so **greedy must show zero change**. If greedy
moves, something else moved and the experiment is void.

**What would make this fail.** vLLM Ascend's docs warn that Block Verify "may cause minor precision
degradation", which contradicts Sun et al.'s losslessness proof. That warning probably describes an
Ascend approximation rather than this kernel, but it means the `docs/quality.md` gate must run, not
just the throughput bench. Second risk: our patches modify the DFlash2 speculator rather than the
sampler, so the flag should plumb through untouched, but confirm it reaches the kernel on the
patched tree before believing a null.

**Expected outcome.** Between 0 and +8 percent tokens per step on the sampling cohort, zero on
greedy. Cost: under an hour of engineering, half a day of benchmarking.

### Proposal 3: read the per-position vector, then price every curve with synthetic mode

**Change.** Two steps, no kernel work.

1. Capture the per-position acceptance vector we are already logging.
   `vllm/v1/spec_decode/metrics.py` at the v0.28.0 tag computes
   `acceptance_rates = sum(pos_matrix, axis=0) / num_drafts` and logs
   "Per-position acceptance rate: %s". Capture it stratified by whether the lookup fired, so the
   drafter's curve and the lookup tail's curve come out separately.
2. Then run `rejection_sample_method: "synthetic"` with hand-written `synthetic_acceptance_rates`:
   our measured curve as the control, and counterfactual curves as treatments (flat plus 0.05 per
   position; a longer block; a tree-shaped curve).

**Evidence.** The `synthetic_acceptance_rates` docstring at the v0.28.0 tag is explicit:
"Per-position *unconditional* acceptance rates ... Position i's entry is the marginal probability
that the first i+1 draft tokens are all accepted", monotonically non-increasing. That is exactly
the format the metrics logger emits. The two features compose by design.

**Why this earns a slot.** Every ranking in this document rests on numbers measured on H200, B200,
B300, H20 or MI300X hardware, and S3D's warning is that a method which performs well on a high-end
GPU "can slow down by up to 7 times" under limited memory and quantization. Synthetic mode converts
published acceptance curves into a **local exchange rate**: how many tokens per second, on our
silicon, at our context length, with our CUDA graphs, a given acceptance curve is actually worth.
It is the only experiment here that decides what to fund next rather than moving a number.

Step 1 also settles the open question in Q2. If our per-position curve is flat near 0.74, the
EAGLE-table model holds and the block is not the bottleneck. If it looks like SpecBlock's Qwen
curve (alpha_8 about 0.369), positions 6 and 7 are nearly worthless and the block should be
*shortened*, not lengthened.

**Caveat.** Synthetic mode accepts against a rate by construction, so it changes output quality.
This is a throughput-model experiment only. Never report a quality number from it and never leave
the flag on.

**Cost.** Half a day. Almost no risk.

### What to fund after the two days, in order

1. **Tree verification over the candidates we already compute (DDTree).** The drafter emits
   `selector_top_k = 16` candidates at each of 7 positions and discards 105 of the 112. DDTree
   ([arXiv 2604.12989](https://arxiv.org/abs/2604.12989)) reports Qwen3-8B MATH-500 going from
   DFlash 5.56x at tau 7.79 to 7.52x at tau 10.73, same drafter, no retraining; Graft's more
   conservative preliminary DFlash figure is +9.1 percent. Reference code at
   [github.com/liranringel/ddtree](https://github.com/liranringel/ddtree). Scope it at a node budget
   of 16 to 24 rather than DDTree's 64 to 1024, because at batch 1 the int4 `lm_head` over a
   248,320-token vocabulary starts to dominate. The blocker is the verify attention mask:
   `patches/spec-decode-attn.patch` sizes split-KV partial buffers once for the longest query block
   and a captured graph holds their addresses (`VLLM_SPEC_DECODE_ATTN_QMAX`), so buffer sizing has
   to move to the node budget first. `docs/spec-decode-scratch-token-units.md` is the precedent for
   getting that sizing right.
2. **A block-16 drafter, warm-started.** Table 8 of the DFlash paper says a block-8 drafter run at
   block 16 is *worse* than at block 8, so this is a training job and not a config change. But the
   warm-start path is precedented (Muse Glimmer's block-16 DFlash2 was finetuned from a DFlash v1
   drafter), the data volume is known (about 800K regenerated samples), and NeMo's offline mode
   makes it tractable on our cards. Consider `loss_type: variable_prefix` (D2SD VP-Drafter) rather
   than the fixed-anchor default, since it targets exactly the partial-acceptance regime batch 1
   lives in.
3. **Confidence-gated grafting instead of boundary-gated grafting.** Graft's finding is that
   retrieval should fill the positions the drafter is *unsure about*, wherever they sit, not only
   the positions past the trained block. That is the general form of the `z-adaptive-emitted`
   argument, and it needs the per-position vector from Proposal 3 before it can be tuned at all.

### Explicitly not worth the two days

* **Adaptive draft length.** DSpark's own paper concedes that at batch 1 the verify step barely
  slows as tokens are added, so trimming saves little; the published gains are batch-capacity gains
  at concurrency. If we want it anyway, SGLang's EMA-of-accepted-length rule (tiers `[1, 3, 7]` for
  batch 1 to 7) is roughly fifteen lines and needs no training, and is the only item on that list
  worth the effort at our batch size.
* **DSpark's adaptive verification specifically.** Hard-blocked: `ValueError` for any method other
  than `"dspark"`, plus an `AttentionCGSupport.ALWAYS` requirement that only SM100 sparse-MLA
  backends satisfy.
* **Lengthening the *drafted* block without retraining.** Table 8: train-8 / test-16 scores below
  train-8 / test-8 on every task. Our clamp is already the right answer.
* **Re-quantizing or un-quantizing the drafter.** `drafter/README.md` already measured it: int4
  keeps greedy acceptance and costs about 5 percent under sampling.
* **Fine-tuning the drafter on our own outputs.** Already tried for the MTP head: "KL halves, greedy
  top-1 on response tokens unchanged".
* **Typical acceptance.** Lossy, and the one independent measurement in vLLM's own tracker found it
  *shorter* than plain rejection sampling. Block verification is the lossless version of the same
  ambition.
* **Self-speculative and layer-skip.** Ceiling about 1.5x, and none of it is in vLLM.
* **Batch-invariant determinism.** Hard-blocked on GDN. See Q5.
* **Waiting for vLLM 0.29.** Not released, no tracker, no committed date.


## Gaps in this brief

Stated plainly, because the ranking above depends on them.

1. **Almost every headline number in Q1 is from H100, H200, B200, B300, H20 or MI300X.** The two
   exceptions are EAGLE's Table 2 (RTX 3090, batch 1, but fp16 targets and an autoregressive
   drafter) and the TinyLlama-into-4-bit-Llama-2 study on Tesla T4s. **Not one published
   accepted-tokens-per-step figure for a block drafter was measured on a 24 GB consumer card at
   batch 1 with int4 weights.** Tokens-per-step ratios should transfer; wall-clock speedups should
   not, because our cost coefficient c and our bandwidth regime differ. S3D's warning stands as the
   general caution: on limited memory with quantization, a high-end-GPU method "can slow down by up
   to 7 times".
2. **The a = 0.74 fit is mine, derived from two of our own summary statistics.** EAGLE's table and
   DeepSeek's published curve support the shape, but SpecBlock's Qwen3-8B row (alpha_8 about 0.369
   against Llama's 0.544) is direct counter-evidence that Qwen models decay harder at depth.
   Reading our own per-position vector (Proposal 3, step 1) settles it in an afternoon and could
   invalidate the flat-acceptance conclusion and with it the +9 percent block-extension estimate.
3. **The PR numbers in the v0.28.0 release-notes table were not individually opened.** They are as
   the release notes list them. The six post-0.28 PRs I quote from *were* opened and read.
4. **`v0.28.0...main` was filtered by keyword, not read exhaustively.** I pulled the 492 commits
   dated on or after 2026-08-26 out of 858 total and grepped their subject lines. A change with an
   unrevealing subject line would have been missed.
5. **The EAGLE-3 successor landscape is thin here.** I did not get a systematic 2026 sweep of
   EAGLE-4 / HASS / Falcon / FastEagle-class work, and neither EAGLE-2 nor EAGLE-3 tabulates a
   per-position acceptance vector (both put it in figures). Medusa never breaks accuracy out by head.
6. **Truncation-aware drafting is a confirmed gap in the literature.** No work exists on matching a
   drafter to a target that has had top-k or top-p applied before verification, which is exactly
   what we serve. SlimSpec's truncated-*vocabulary* bound is the nearest thing and carrying it to
   the target side is my inference, not a cited result.
7. **No published acceptance measurement at T=0.7.** Everyone reports T=0 and T=1. The PayPal
   EAGLE-3 finding of no change from T=0 to T=0.5 is the only nearby data point.
8. **No source anywhere publishes GPU-hours, wall-clock or step counts for training a DFlash or
   DFlash2 drafter.** Data volume and optimizer settings are known; cost is not. Do not let anyone
   infer a number for it.
9. **AdaEDL, PACER and GammaTune contributed no verified numbers.** Their PDFs would not parse on
   repeated attempts. They are leads, not evidence, and are labelled as such above.
10. **The claim that block verification helps most on point-mass lookup positions is a mechanism
    argument, not a measurement.** It is the most falsifiable prediction in this document, and
    Proposal 2's stratification tests it directly.
11. **Whether #54374 is already masked by our own `a75ee4b` is unknown.** The symptom matches; the
    root causes differ. Proposal 1's A/B is the only way to tell, and a null there is informative
    rather than a wasted day.
