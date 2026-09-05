# How DFlash and DFlash2 were made, and what it would cost us to train a drafter

Research file, written 2026-09-05. All web sources fetched 2026-09-05 unless a different
date is given inline. Local files read from this box on 2026-09-05.

**Binding question (the human's, verbatim):** "can you research how dflash and dflash2 was
created? we can rent a few hours of cluster time for not terribly much money something like
2 to 10$ an hour depending on GPUs and train a new drafter. 70% to 80% honestly sounds like
dflash2 is already quite quite good. so really before I commit 10$ to 100$ on it I'd want to
know if we could. or if 2x4090s could be enough?"

**Three attribution buckets, kept separate throughout:**

| Label | Who | What they did |
|---|---|---|
| **[UCSD]** | Jian Chen, Yesheng Liang, Zhijian Liu, UC San Diego | The DFlash paper and method. Published the training recipe. |
| **[Inco]** | Inco AI | DFlash2 (the selector and the convolution) and the two shipped DFlash2 drafters, including ours. Published the architecture, **not** the recipe. |
| **[syv-ai]** | mhenrichsen / syv-ai | Requantized Inco's drafter to W4A16. **Trained nothing.** |
| **[estimate]** | me, this file | Arithmetic from configs and public specs. Every one shows its working. |

Companion file: `docs/research/spec-decode-sota-2026-09.md` already covers block length,
tree verification and lookup drafting from the serving side. This file is the training side.
Where the two overlap I say so; they agree.

---

## 0. The short version

* **DFlash is a paper** with a fully published training recipe. **DFlash2 is a product
  announcement** with a published architecture and no published recipe.
* **Nobody who trained the drafter we run has said what it cost.** Not Inco, not the
  paper, not any model card. That number does not exist in public.
* **But the training code does exist**, from three independent parties, and one of them
  ships a reference topology for a 27B DFlash2 that uses **exactly two GPUs**.
* **Somebody already did this on consumer hardware for our exact target.** A community
  drafter for Qwen3.8-27B was fine-tuned on **two Intel Arc Pro B70s** from 961 samples in
  440 steps. It works. It is also worse than what we run.
* **Somebody already trained a block-16 DFlash drafter for our exact target.** So the
  block-16 question is not a training question for us at all.
* Two 4090s can host the drafter's optimizer state and can host a W4A16 target. They
  cannot host a bf16 target. That single fact shapes everything below.

---

## 1. What DFlash is, what DFlash2 changed, and who trained ours

### 1.1 DFlash: the paper [UCSD]

**Provenance.** *DFlash: Block Diffusion for Flash Speculative Decoding*, Jian Chen,
Yesheng Liang, Zhijian Liu (UC San Diego, correspondence zhijian@ucsd.edu).
arXiv:2602.06036v1, submitted 5 Feb 2026. Accepted at ICML 2026
(https://icml.cc/virtual/2026/poster/64301). Code at https://github.com/z-lab/dflash,
MIT licence. I downloaded https://arxiv.org/html/2602.06036v1 to a local file and read the
text directly; every quotation below is from that file, not from a summary of it.

**The idea.** The drafter is a small block-diffusion model. Instead of proposing tokens one
at a time, it fills a whole block of mask positions in one forward pass. From section 4.1:

> "All masked positions within a block are decoded in parallel in a single forward pass."

The drafter is not a standalone small LM. It is conditioned on the target's own internals.
Section 4.1:

> "We treat the fused target context feature as persistent contextual information and
> directly inject it into the Key and Value projections of *every* draft model layer"

and on where those features come from:

> "we extract hidden representations from a fixed set of layers uniformly sampled from
> shallow to deep. These hidden states are concatenated and passed through a lightweight
> projection layer to fuse cross-layer information into a compact target context feature"

Section 4.1 also states the negative control, which matters for the "is it worth it"
question: a five-layer block-diffusion drafter with **no** target conditioning gets
"speedups ... typically around 2-3x", against 4.9x for the conditioned version.

**How it is trained** (section 4.2 and appendix A.1, both read in full):

* **Objective.** Cross-entropy over the masked positions, weighted so early positions in the
  block count more. Section 4.2: "Errors at early positions within a draft block invalidate
  all subsequent tokens." The weight is `w_k = exp(-(k-1)/gamma)`.
* **Anchors, not uniform blocks.** "We randomly sample anchor tokens from the response, use
  each anchor as the first position of a block, and mask the remaining positions ... This
  directly matches inference-time behavior, where the draft model always conditions on a
  clean token produced by the target model (i.e., the bonus token from the previous
  verification step)."
* **Frozen sharing.** "the draft model shares the token embedding layer and language
  modeling head with the target model and keeps them frozen during training. Only the draft
  Transformer layers are updated."
* **Does it need target hidden states?** Yes, unavoidably. They are the conditioning signal.
* **Must the target be co-resident?** No. Appendix A.1, verbatim:

  > "Training can be performed either online or offline. In online training, target hidden
  > features are computed on the fly during each training step. In offline training, target
  > hidden features are precomputed and cached, then loaded during draft model optimization
  > to reduce computational overhead."

  This sentence is the whole ballgame for the 2x4090 question. See section 4.

**Full published hyperparameters** (appendix A.1, verbatim): "The draft models are optimized
for 6 epochs using AdamW with a learning rate of 6x10^-4, a gradient clipping threshold of
1.0, and a cosine schedule with a warmup ratio of 0.04. We train on our training data
mixture with a maximum sequence length of 3072 tokens (4096 for Qwen3-Coder); for each
sequence, 512 anchor positions are randomly sampled. The hyperparameter gamma for the loss
decay in Equation 4 is set to 7 for block size 16, 5 for block size 10, and 4 for block size
8 models."

**Data** (section 5, verbatim): "we collect a mixture of around 800K samples from NVIDIA
Nemotron Post-Training Dataset V2 and CodeAlpaca ... we construct our training set with the
responses generated by the target model for better target alignment."

**What the paper does NOT say.** Batch size. Number of GPUs used for training. Wall clock.
Total token count. I grepped the full text for `batch size`, `global batch`, `per-device`,
`wall`, `hours`, `days`, `training cost`, `A100`, `H100`, `4090`. The only compute statement
is section 5: "We conduct all experiments on NVIDIA H200 GPUs unless otherwise specified",
and in context "experiments" means the evaluations; several ablations then say "a single H200
GPU" or "a single B200 GPU" for evaluation. **The paper never states training compute.**
Anyone who tells you the DFlash training cost is quoting something other than the paper.

### 1.2 DFlash2: the product post [Inco]

Source: https://inco.ai/blog/dflash2/ (downloaded and read as text, not summarized).
Released **18 August 2026** with two drafters, ours and Meta's Muse Glimmer.

DFlash2 keeps the DFlash backbone and adds exactly two things.

**(a) A candidate path selector.** DFlash picks each position's top-1 independently, so
neighbours can disagree and the block dies at verification. The blog's motivating table
(Table 1, five-layer Qwen3-4B DFlash on GSM8K, conditioned on every earlier position being
right):

| Metric | pos 0 | 1 | 2 | 3 | 4 | 5 | 6 | Acceptance length |
|---|---|---|---|---|---|---|---|---|
| Recall@1 | 85.4% | 80.3% | 79.4% | 78.3% | 77.5% | 75.9% | 72.9% | 4.27 |
| Recall@16 | 99.5% | 97.3% | 94.8% | 92.6% | 90.8% | 89.4% | 87.8% | 6.79 |

> "An oracle that always picks the right candidate from the top 16 would lift the acceptance
> length from 4.27 to 6.79. That gap is pure selection headroom."

The selector keeps the top 16 at each position and scores every adjacent pair with a
low-rank bilinear form, quoted verbatim from the blog:

> "S_t(a,b) = U_t(b) + <A(a) (*) H(h_t), B(b)>"

where `U_t(b)` is the drafter's own logit, `A` and `B` are 256-dimensional per-token
codebooks, and `H(h_t)` is a context gate. Then a greedy or sampled walk over the
precomputed pair scores.

**This is visible in our checkpoint on disk**, which is how I know the description is
accurate rather than marketing. From the safetensors header of
`models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors`:

```
candidate_selector.successor_codebook      [248320, 256]  BF16   (= A)
candidate_selector.predecessor_codebook    [248320, 256]  BF16   (= B)
candidate_selector.hidden_projection       [256, 5120]    BF16   (= H)
```

`selector_rank: 256` and `selector_top_k: 16` in `config.json` match. Confirmed
mechanism, not a claim.

**(b) A two-tap dynamic depthwise convolution**, before and after each attention and MLP
sublayer, to stop suffix decay. Blog: "each position mixes its own representation with its
predecessor's, and the first position reads the last verified token." Also visible on disk
as `layers.N.attention_conv.base_kernel [2,2,5120]` plus
`layers.N.attention_conv.kernel_projection [1280,5120]`, and the same pair for `mlp_conv`.
`conv_kernel_size: 2` and `conv_group_size: 16` match; 5120/16 = 320 groups and
320 x 2 taps x 2 = 1280, which is the projection's output width. Confirmed.

**Claimed overheads** (blog): selector "+2.0M" params and "+0.6%" cycle latency; convolution
"16.5M added parameters (3%)" and "+0.7%" latency; together "only 1.3% to the five-layer
DFlash draft-verify cycle latency."

> **Discrepancy I could not fully resolve, flagged rather than smoothed.** Those parameter
> counts are for **Qwen3-4B**, whose hidden size is 2560 and vocab 151,936, not for our
> 27B drafter. The convolution figure checks out under that reading: per layer
> 2 x (2x2x2560 + (2x2x2560/16)x2560) = 3.29M, x5 layers = **16.4M**, against the blog's
> 16.5M [estimate, arithmetic shown]. The selector figure does not. Our checkpoint's two
> codebooks alone are 2 x 248,320 x 256 = **127.1M** parameters, and at Qwen3-4B's vocab
> they would still be 77.8M, not 2.0M. The blog does not state its accounting. Do not use
> "+2.0M" to size a training run; use the tensors.

**What the blog does NOT say.** Anything at all about how the DFlash2 drafters were
trained. No dataset, no token count, no steps, no GPUs, no hours. The only training
sentence in the whole post is "We trained the DFlash and DSpark drafters ourselves under
matched setups", which is about the **baselines**, not about DFlash2. **The DFlash2 recipe
is unpublished.**

### 1.3 Who trained the drafter we actually run

**Inco AI trained it.** Not mhenrichsen, not the DFlash authors.

Chain of custody, each link read directly today:

1. `incoai/Qwen3.8-27B-DFlash2` model card
   (https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/main/README.md): "This repository
   contains the DFlash 2 draft model for `Qwen/Qwen3.8-27B`." Developer: Inco AI. Apache-2.0.
   Mirrored at `z-lab/Qwen3.8-27B-DFlash2`.
2. Our local `models/Qwen3.8-27B-DFlash2-W4A16/README.md` names its base model as
   `incoai/Qwen3.8-27B-DFlash2` and says the W4A16 build is "requantized to W4A16
   compressed-tensors ... with GPTQ", crediting "the original drafter by Inco".
3. The repo's own `drafter/README.md` describes only the requantization pipeline
   (`capture_dflash2.py`, `quant_dflash2.py`), never a training run.

So [syv-ai]'s contribution is a **post-training compression**, worth about 2.7 GB less read
per decode step by its own measurement, and no acceptance change at greedy. There is no
syv-ai-trained DFlash drafter.

**One consequence worth stating plainly.** Inco trained against the **bf16** Qwen3.8-27B.
We feed the drafter hidden states from a **W4A16 AutoRound** target. Those are not the same
distribution. Inco's published mean acceptance length for this drafter is **4.80** across
five benchmarks and **4.10** on MT-Bench (blog Table 4, block size 8, model-default
sampling, one H200). We measure **3.14 at default sampling / 3.34 greedy** on the repo's
8-prompt chat protocol (`README.md`, the single-stream table). The repo's own ablation says
int4 on the **drafter** costs about 5% at default sampling (3.2 vs 3.4), which accounts for
roughly 0.2 tokens, not the roughly 1.0-token gap against MT-Bench. **[estimate]** The
residual is some mix of prompt-mix differences and the quantized **target's** hidden states
being off-distribution for a drafter trained on bf16 ones. I did not measure this; it is a
hypothesis with an obvious cheap test, and it is the single best argument for training
anything at all (see 3.4).

---

## 2. The training recipe actually used for this drafter

### 2.1 For our checkpoint specifically: unpublished

Searched and came up empty on all of: the Inco blog, the `incoai/Qwen3.8-27B-DFlash2` card,
the `z-lab/Qwen3.8-27B-DFlash2` mirror, and the z-lab GitHub repo. **No dataset, no token
count, no step count, no batch size, no sequence length, no GPU type or count, no wall
clock, for either shipped DFlash2 drafter.**

I also checked whether the reference repo ships training code. It does not. Full file
listing of `z-lab/dflash` from the GitHub trees API today:

```
.github/workflows/package.yml, .github/workflows/publish-pypi.yml, .gitignore,
LICENSE, README.md, dflash/__init__.py, dflash/benchmark.py, dflash/cli.py,
dflash/model.py, dflash/model_mlx.py, pyproject.toml
```

Twelve files, all inference and benchmarking. **The DFlash authors released no trainer.**

### 2.2 One thing about Inco's method that IS published, and it is load-bearing

The Muse Glimmer DFlash2 card
(https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2/raw/main/README.md) says, verbatim:

> "It is finetuned from
> [`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant),
> the official DFlash drafter Meta ships with the model."

So at least one shipped DFlash2 drafter was **warm-started from an existing DFlash v1
drafter**, not trained from scratch. The convolution's base kernel is an identity and the
selector's successor codebook is zero-initialized (SpecForge's `docs/concepts/DFlash2.md`:
"a new convolution starts as an exact no-op" and "The successor codebook is
zero-initialized, so a new selector initially preserves unary DFlash scores"), which is
precisely the design that makes such a warm start work.

**The Qwen3.8-27B card says nothing equivalent**, and there is no `incoai/Qwen3.8-27B-DFlash`
v1 to have warm-started from. I listed the full `incoai` org today: GLM-5.3-DFlash2,
GLM-5.3-Flash-DFlash2, GLM-5.3-NVFP4, Qwen3.8-27B-DFlash2 (+GGUF), Muse-Glimmer-30B-DFlash2
(+GGUF). No DFlash v1 anywhere. **Whether ours was trained from scratch or warm-started from
something unreleased, the sources are silent.**

### 2.3 The drafter's own shape, computed from the files on this box

Parsed the safetensors header of
`models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors` (1,280,633,960 bytes on disk, 1221 MiB)
and unpacked the int4 tensors (each `weight_packed` int32 holds 8 int4 values):

| Quantity | Value | How |
|---|---|---|
| Tensors | 153 | header count |
| **Dense parameters** | **1,924,404,480** | sum of int4-unpacked + bf16 tensors [estimate, arithmetic in text] |
| Layers | 5 | `num_hidden_layers`, all `sliding_attention`, window 2048 |
| Hidden / heads / KV heads / head_dim | 5120 / 32 / 8 / 128 | `config.json` |
| Intermediate | 17408 | `config.json` |
| Vocab | 248320, untied | `config.json` |
| Fed by target layers | 5, 19, 33, 47, 61 of 64 | `dflash_config.target_layer_ids` |
| Block size | 8 (so 7 drafts) | `dflash_config.block_size` |

That 1.9244B matches the model card's "1.92B parameters, 3.85 GB in bf16" exactly, so the
unpacking arithmetic is validated against an independent statement.

Breakdown of where the parameters are, all computed:

* 5 transformer layers, int4: 5 x (q 20.97M + k 5.24M + v 5.24M + o 20.97M + gate 89.13M +
  up 89.13M + down 89.13M) = **1,599.1M**
* `fc`, int4, 5120 x 25600 (25600 = 5 target layers x 5120): **131.07M**
* Convolutions, bf16: 5 layers x 2 sublayers x (base 20,480 + projection 6,553,600) =
  **65.74M**
* Selector, bf16: two 248,320 x 256 codebooks + a 256 x 5120 gate = **128.45M**
* Norms: negligible

Note what that means for a training run: **the selector's codebooks are 128M trainable
parameters keyed by vocab id**, and our vocab is 248,320. That is a large, sparse-gradient
table. It is also zero-initialized by design, so it is the part a short fine-tune can move
fastest.

**Not in the checkpoint:** `embed_tokens` and `lm_head`. Confirmed by absence from the
tensor list, and consistent with the paper's frozen-sharing design and the local card
("The drafter shares the target's embeddings and lm_head; it ships neither").

### 2.4 What IS published: three independent trainers, one with a 27B DFlash2 reference

This is where the "can we" question actually gets answered.

**(a) SpecForge**, the SGLang team's training framework
(https://github.com/sgl-project/SpecForge). It trains EAGLE3, P-EAGLE, EAGLE3.1, DFlash,
**DFlash2**, Domino and DSpark. `docs/concepts/DFlash2.md` (read in full today) documents
DFlash2 as a first-class draft architecture with its own selector objective and schedule
(`dflash2_selector_loss_alpha`, `dflash2_selector_warmup_ratio`,
`dflash2_selector_ramp_ratio`, `dflash2_selector_stop_gradient`).

Critically, SpecForge ships a **checked-in DFlash2 config for a 27B Qwen target**,
`configs/qwen3.6-27b-dflash2.json`. Read today, it is our checkpoint's config almost
verbatim: same `block_size` 8, `conv_group_size` 16, `conv_kernel_size` 2,
`selector_rank` 256, `selector_top_k` 16, same `mask_token_id` 248070, same hidden 5120,
same vocab 248320, same 64 target layers. The only difference is
`target_layer_ids: [1,16,31,46,61]` against our `[5,19,33,47,61]`, and
`sliding_window` 4096 against our 2048. **A DFlash2 training config for a 27B model with our
exact tokenizer and hidden size is public and runnable.**

Its reference run YAML,
`examples/configs/online/disaggregated/managed-local/qwen3.6-27b-dflash2-disaggregated.yaml`,
read in full today:

```yaml
data:   max_length: 8192, chat_template: qwen3.5
training:
  strategy: dflash
  batch_size: 1
  accumulation_steps: 4
  learning_rate: 0.0006
  warmup_ratio: 0.04
  max_grad_norm: 1.0
  attention_backend: flex_attention
  num_anchors: 512
  loss_decay_gamma: 7.0
  loss_type: dpace
  dflash2_selector_loss_alpha: 1.0
deployment:
  mode: disaggregated
  trainer:   { nnodes: 1, nproc_per_node: 1 }
  disaggregated:
    managed_local:
      trainer_cuda_visible_devices: ["5"]
      capture_servers:
        - port: 30000
          cuda_visible_devices: ["2"]
          tp_size: 1
          mem_fraction_static: 0.8
```

Read that deployment block carefully, because it is the direct answer to the human's second
question. **One GPU runs an SGLang server holding the 27B target and captures hidden states.
One GPU trains the drafter. Total: two GPUs.** That is the maintainers' own reference
topology for a 27B DFlash2 run.

Note also `loss_type: dpace`, not the paper's plain decayed cross-entropy. That is D-PACE
(arXiv:2605.18810, surfaced in search, **not read by me**, flagged as a lead only). So even
SpecForge's DFlash2 recipe has moved past the paper's objective.

**SpecForge has already been run against our exact target.** `RadixArk/Qwen3.8-27B-DSpark`
is SpecForge-trained on Qwen3.8-27B (sibling session, 2026-09-05), which settles the
question of whether the framework can ingest a hybrid gated-delta-net target at all. Its
online mode captures from a patched SGLang server, so anything SGLang can serve, SpecForge
can train against. There is no checked-in Qwen3.8 config among SpecForge's 47, so we would
adapt `configs/qwen3.6-27b-dflash2.json`, changing `target_layer_ids` to our checkpoint's
`[5,19,33,47,61]` and `sliding_window` to 2048.

The offline path is documented too (`docs/basic_usage/data_preparation.md`, read today).
`scripts/prepare_hidden_states.py --strategy dflash` writes records containing
`input_ids`, `loss_mask`, `hidden_states`, where "For the DFlash family, `hidden_states`
concatenates the target layers selected by the draft config." Training then reads
`data.hidden_states_path` and the target never runs. That is the paper's offline mode, with
a script.

**(b) Speculators** (Red Hat / vLLM ecosystem),
https://developers.redhat.com/articles/2026/06/04/speculators-v050-dflash-support-and-online-training,
dated **4 June 2026**. v0.5.0 "introduces training support for the DFlash speculative
decoding algorithm". Its worked example is, verbatim:

```
torchrun --standalone --nproc_per_node 2 scripts/train.py \
    --verifier-name-or-path "Qwen/Qwen3-8B" \
    --speculator-type dflash \
    --block-size 8 \
    --max-anchors 3072
```

Two processes again, though for an 8B target. Its online mode "extracts hidden states during
the training process itself": a vLLM server holds the base model, prompts are sent to it,
"Hidden states are extracted and temporarily written to disk." The article states no
hardware requirement, no memory figure, and no duration.

**(c) NVIDIA NeMo AutoModel**,
https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding.
Also `torchrun --nproc_per_node=2`. Two statements from it matter more than anything else on
that page for our purposes, both verbatim:

> "Multi-GPU wraps the draft in DDP and replicates the frozen target."

> "Set `distributed.tp_size` to shard a large target across ranks."

Replicating a 55.6 GB bf16 target on every rank is exactly what a 24 GB card cannot do. It
documents `block_size` ("tokens drafted in parallel per block (paper: 16)"),
`loss_decay_gamma`, `mask_token_id` ("reserved token id filling non-anchor block positions
(required)") and `loss_type` (`dflash` or `variable_prefix`). It states no GPU count
minimum, no memory figure and no wall clock.

### 2.5 Somebody already did this on our exact target, on two consumer GPUs

Searched the Hugging Face model API for `Qwen3.8-27B-DFlash` today. Thirty hits, mostly
requantizations of Inco's checkpoint (ours among them, `syvai/Qwen3.8-27B-DFlash2-W4A16`).
Three are not requantizations. All three cards read in full today.

**(a) `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70`.** The most directly useful artefact I
found in this entire search. Its card states, verbatim:

> "Trained 1.36B DSpark/DFlash draft model for Qwen3.8-27B FP8 speculative decoding on 2x
> Intel Arc Pro B70 (TP=2)."

and its Training section, verbatim:

> "- Warm-start from released RadixArk Qwen3.8-27B DSpark drafter
> - Fine-tuned on hidden states captured from the vLLM FP8 serving stack
> - Data: 761 ShareGPT-derived + 200 C1-domain (bench-protocol) samples
> - SpecForge DSpark strategy, XPU (eager attention), 440 steps, lr 5e-5"

Results claimed: 72.2 median tok/s against 54.67 for MTP2 and 32.4 for no speculation;
"Acceptance: 62-74% pos-0, mean acceptance length 2.5-3.5."

Four things to take from this and one to be careful about. Take: **961 samples** is enough
to move a warm-started drafter; **440 steps** is a tiny run; **two consumer-class GPUs** was
the whole rig; and they **captured hidden states from the quantized serving stack**, which
is the same idea I raised in 1.3 as our best cheap experiment, done by someone else and
shipped. Be careful about: this is a **DSpark** drafter trained with SpecForge's DSpark
strategy, not DFlash2, and its **mean acceptance length 2.5-3.5 is at or below our current
3.14-3.34**. It beat MTP on their stack. It would not beat DFlash2 on ours.

Its card also documents a trap worth more than the drafter: "vLLM with the DFlash kernel
readout fix ... without it, acceptance collapses to ~24% for ANY SpecForge-trained drafter",
because "SpecForge DSpark trains output position j to predict token anchor+j+1 (LM-style).
Stock vLLM dflash sampled outputs at offsets 1..k (BERT-style) - every draft off by one."
**If we train anything in SpecForge and serve it in vLLM, verify this alignment before
believing any acceptance number.** That is a one-line off-by-one that silently looks like a
bad drafter.

**(b) `onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2`.** A **block-16** DFlash drafter
for Qwen3.8-27B, already trained and published. Card, read today: `DFlashDraftModel`,
**6 layers**, hidden 5120, vocab 248320, `block_size` **16**, BF16 3.96 GiB, 69 tensors,
trained against `unsloth/Qwen3.8-27B-NVFP4`. Its measured comparison on a DGX Spark (GB10),
single-stream, fixed code prompt, temperature 0, median of 5:

| Configuration | tok/s |
|---|---|
| this drafter, `--dflash-gamma 15` | 63.9 |
| `incoai/Qwen3.8-27B-DFlash2`, `--dflash-gamma 7` | 43.9 |
| no speculation | 13.9 |

Carry every caveat: GB10 unified memory at about 273 GB/s, not a 4090; the "Atlas" engine,
not vLLM; an NVFP4 target, not ours; and the card itself says "That figure is
workload-specific ... median ~37 across mixed workloads". It also volunteers its own
weakness: "a substantial fraction of the training corpus was not on-policy, and it contained
essentially no reasoning-span text while reasoning spans dominate what the drafter actually
drafts at serve time. Measured serve-time acceptance is therefore below what training loss
suggested." No GPU count or hours for its training are stated.

**(c) `mrchuy/Qwen3.8-27B-DFlash-drafter-bootstrap-GGUF`.** A negative result worth knowing.
The published `Qwen3.6-27B-DFlash` weights transplanted onto Qwen3.8 metadata with, in the
card's words, "no additional training on Qwen3.8". Measured on 2x RTX 4060 Ti 16 GB: 34.04%
overall draft acceptance, 2.02 tokens per verification, against native MTP's 71.65% and 3.15.
**Weight transplant without training does not work.** Half the acceptance of the thing we
already run.

---

## 3. Is a better drafter for THIS target a training question or a data question?

Short answer: **for block 16 it is neither, because one already exists. For raw acceptance
it is mostly a selection-and-alignment question, and only weakly a data question.**

### 3.1 What the DFlash authors report about block size [UCSD]

Paper section 5.4.4, Table 7 (8-layer drafters, 5 target features, acceptance length tau).
Read from the downloaded paper text:

| Train BS | Test BS | Math500 speedup | tau | HumanEval speedup | tau | MT-Bench speedup | tau |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 4.64x | 6.33 | 3.96x | 5.29 | 2.23x | 3.50 |
| 16 | 8 | 3.87x | 5.09 | 3.39x | 4.44 | 2.12x | 3.18 |
| **8** | **16** | **3.78x** | **5.02** | **3.24x** | **4.28** | **2.09x** | **3.09** |
| 8 | 8 | 3.97x | 5.21 | 3.53x | 4.61 | 2.22x | 3.29 |

The authors' own reading, verbatim: "A model trained with a larger block size generalizes
well to smaller inference-time block sizes ... However, the reverse does not hold."

**This is the row that matters for us: train-8, test-16 is the worst row in the table**,
worse than train-8 test-8 on all three benchmarks. Our drafter is trained at block 8. So
running it wider is the bad direction, and `docs/research/spec-decode-sota-2026-09.md`
already reached this conclusion independently and used it to justify the `DFLASH_TOKENS=15`
clamp. The two files agree. (Citation note: I read **v1**, where this is section 5.4.4
/ Table 7. The companion file and the sibling session read **v2**, dated 2026-05-28, where
the same table is section 5.5.4 / Table 8. Same values, renumbered between versions. Cite
whichever version you actually opened.)

Also from that section: "the block-8 model frequently fully accepts entire blocks (35.7%),
suggesting that block size 8 is often underutilized." That is the strongest published
argument that a block-16 drafter for our target would be worth having, **on math and code**.
MT-Bench, which is closest to chat, gains far less (3.50 vs 3.29).

### 3.2 What they report about training tokens

**Almost nothing.** There is no acceptance-versus-training-tokens curve in the DFlash paper.
The ablations use "100K samples randomly drawn from the full data mixture" against 800K for
the main results, but the paper never puts the two on the same axis, so the 100K-to-800K
delta is not recoverable from it. I checked; the ablation tables and the main tables use
different drafter depths and different block sizes, so they are not comparable.

The nearest published thing is **DFlare** (arXiv:2606.02091, Jiebin Zhang et al., v1
1 Jun 2026, v2 2 Jun 2026). I read the abstract only, not the paper. It says, verbatim:

> "We further scale training data from 800K to 2.4M samples to fully exploit the enlarged
> capacity. On six benchmarks ... DFlare attains average wall-clock speedups of 5.52x on
> Qwen3-4B, 5.46x on Qwen3-8B, and 3.91x on GPT-OSS-20B, improving over DFlash by roughly
> 11%, 8%, and 5% respectively."

**Read that carefully.** The abstract attributes the improvement to the **combination** of a
new layer-wise fusion architecture, a deeper draft model, **and** the 3x data. It is not a
data-scaling curve, and citing it as "3x data buys 11%" overstates it. What it does support:
800K samples is the working floor the field uses, 2.4M is where a stronger architecture was
pushed, and nobody publishes the curve between them.

### 3.3 What they report about capacity

Paper Table 5: 3, 5 and 8 draft layers compared; **5 layers had the best average speedup**
(4.71x on Math500, tau 5.99). Paper Table 6: 5 target hidden features beat 3
(4.69x vs 4.49x on Math500), with the noted cost that "in offline training, the storage
required to cache target hidden states increases linearly with the number of extracted
features."

Inco's blog Figure 2 makes the same point harder: a 5-layer DFlash **with the convolution**
recalls within about 0.6 points of a **15-layer** DFlash with 3x the parameters, at "3%
parameters and 0.7% cycle latency". Depth is not our bottleneck. We already have the
configuration the authors chose after ablating it.

### 3.4 So where IS the headroom, for us specifically

Ranked by how well the sources support it.

1. **Target-distribution mismatch. [estimate, hypothesis, untested]** Inco trained against
   bf16 Qwen3.8-27B and reports 4.80 mean / 4.10 MT-Bench. We serve W4A16 and measure
   3.14-3.34. The drafter's int4 requant explains about 0.2 of that by the repo's own
   ablation. The `rwmacy` card is a published precedent for fixing exactly this by
   "fine-tuned on hidden states captured from the vLLM FP8 serving stack". This is a
   **data-generation** question, not a capacity question, and it is the cheapest experiment
   on the list because the capture tooling already exists in `drafter/capture_dflash2.py`.
2. **Selection headroom.** Inco's own Table 1 says an oracle over the existing top-16 would
   reach 6.79 against DFlash's 4.27, and DFlash2's selector recovers only 0.34 of that gap
   at T=0. Inco's own words: "Pairwise scoring is the simplest selector we could think of,
   and we believe there is plenty to explore." A better selector is a **training** question,
   and the codebooks are only 128M of the 1.92B, and they are zero-initialized by design,
   so this is the part most reachable by a short run.
3. **On-policy and reasoning-span data.** The Apathy card's self-diagnosis is that
   off-policy data with no reasoning spans underperforms at serve time where reasoning spans
   dominate. Our own repo already learned the same lesson in a different guise: the MTP
   draft-vocabulary counted over the model's own outputs went from 92.1% to 97.5% coverage
   and moved 98.0 to 108.6 tok/s greedy. This is a **data** question, and it is the one our
   box is already tooled for (`drafter/gen_data.py`, 5.4M output tokens in 2.2 h).
4. **Block 16.** A **training** question in principle, but one somebody else has already
   answered for our target (`onewhosighs/...-v2`). Trying theirs costs a download.

---

## 4. What it would cost to train a new drafter for Qwen3.8-27B

### 4.1 Memory, computed

All figures below are **[estimate]** with the arithmetic shown, except the two marked as
measured on this box.

**The drafter's own optimizer state.** N = 1,924,404,480 dense parameters.

| Component | Bytes/param | GiB |
|---|---|---|
| bf16 weights | 2 | 3.58 |
| bf16 gradients | 2 | 3.58 |
| fp32 master weights | 4 | 7.17 |
| Adam m (fp32) | 4 | 7.17 |
| Adam v (fp32) | 4 | 7.17 |
| **Total** | 16 | **28.68** |

28.68 GiB does not fit one 24 GiB 4090. **Sharded two ways (ZeRO-2 or FSDP) it is 14.34 GiB
per card**, leaving roughly 8 GiB per card for activations and logits. With bf16 optimizer
moments instead of fp32 it drops to 14.34 GiB total, 7.17 GiB per card. So the parameter
state is comfortably within reach of two 4090s. It is not the binding constraint.

**The frozen head is.** The loss needs `lm_head`, which the drafter does not ship:
248,320 x 5120 = 1,271,398,400 params = **2.37 GiB bf16**, resident on every rank.

**The logits are the real wall.** At the paper's 512 anchors and block 8, one sequence has
512 x 7 = 3,584 predicted positions:

| Anchors | Block | Positions | bf16 logits | with fp32 softmax workspace |
|---|---|---|---|---|
| 512 | 8 | 3,584 | 1.66 GiB | 4.97 GiB |
| 512 | 16 | 7,680 | 3.55 GiB | 10.66 GiB |
| 128 | 8 | 896 | 0.41 GiB | 1.24 GiB |
| 64 | 8 | 448 | 0.21 GiB | 0.62 GiB |

A 248k vocab is unusually punishing here. Either use a chunked or fused cross-entropy, or
cut `num_anchors`. SpecForge's reference YAML sets `objective_chunk_blocks: 128`, which
reads like exactly this mitigation, though I did not read the code to confirm what it does.

**The target.** This is the constraint that decides everything.

| Form | Size | Fits a 4090? |
|---|---|---|
| bf16 Qwen3.8-27B | 51.75 GiB (index `total_size` 55,562,855,904 bytes, read from HF today) | **No** |
| W4A16 AutoRound | **14.71 GiB** measured, fast variant; 15.68 GiB, uncensored variant (repo README) | **Yes** |
| Precomputed hidden states | 0 GiB at training time | **Yes** |

NeMo's "Multi-GPU wraps the draft in DDP and replicates the frozen target" is therefore not
a path for us in bf16, and its `tp_size` escape hatch would consume both cards for the
target alone.

**Precomputed hidden states, the storage cost.** 5 layers x 5120 dims x 2 bytes =
**51,200 bytes per token**, that is 50 KiB per token:

| Corpus | bf16 | fp8 |
|---|---|---|
| 50M tokens (about 100K samples x 500 tok) | 2.56 TB | 1.28 TB |
| 200M tokens | 10.24 TB | 5.12 TB |
| 640M tokens (about 800K samples x 800 tok) | 32.8 TB | 16.4 TB |

**This kills the pure-offline path at any interesting scale on a home box.** The paper's own
warning that cache size "increases linearly with the number of extracted features" is an
understatement at 27B scale with 5 features. Offline is viable only for a small corpus, or
in a hybrid where you stream capture and training together, which is precisely what
SpecForge's "disaggregated" mode is.

### 4.2 Compute, computed

Forward+backward at roughly 6ND FLOPs, N taken as 1.80e9 (transformer plus `fc`, excluding
the codebooks, whose gradient is sparse), plus the frozen `lm_head` at about 4 x (V x H) per
predicted position:

| Scenario | Drafter | lm_head | Total |
|---|---|---|---|
| Paper-scale: 800K samples x ~800 tok x 6 epochs = 3.84e9 token-passes | 4.15e19 | 1.71e19 | **5.9e19** |
| Eighth-scale: 100K samples x ~800 tok x 6 epochs | 5.18e18 | 2.14e18 | **7.3e18** |
| Small: 100K samples x ~800 tok x 1.5 epochs | 1.30e18 | 5.34e17 | **1.8e18** |

Plus a **one-time target forward** over the corpus to produce hidden states: 2 x 27.78e9 x
tokens, that is **3.6e19** FLOPs at paper scale and **4.5e18** at eighth-scale. At paper
scale the target capture is comparable to the entire drafter training. And note this ignores
the largest hidden cost of the paper recipe: it **regenerates all 800K responses with the
target**, which is decode-bound, not compute-bound, and on this box `drafter/gen_data.py`
managed 5.4M output tokens in 2.2 h on one 3090. Generating 640M tokens at that rate would
be about 260 days. Rent for that, or reuse existing responses and accept the paper's warning
about target alignment.

Peak throughput figures used, with the sparsity caveat stated: RTX 4090 dense FP16 tensor
**165.2 TFLOPS** (330.3 with 2:4 sparsity, NVIDIA Ada architecture whitepaper,
https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf);
H100 SXM BF16 tensor **1,979 TFLOPS as NVIDIA quotes it**, which is the sparsity number, so
**989.5 dense**. I could not parse NVIDIA's H100 datasheet PDF directly and the search
summaries disagreed about whether 1,979 is dense; I have used 989.5 dense and flag that if
the higher number were right, every H100 hour below halves. MFU assumptions are mine.

| Hardware | Assumed sustained | Paper-scale 5.9e19 | Eighth-scale 7.3e18 | Small 1.8e18 |
|---|---|---|---|---|
| 1x H100/H200 @ 40% MFU | 396 TFLOP/s | 42 h | 5.3 h | 1.3 h |
| 1x A100 80G @ 40% MFU | 125 TFLOP/s | 134 h | 17 h | 4.2 h |
| 1x RTX 4090 @ 30% MFU | 50 TFLOP/s | 336 h | 42 h | 11 h |
| **2x RTX 4090 @ 25% MFU** | 83 TFLOP/s | 202 h | **25 h** | **6.4 h** |

The 2x4090 MFU is deliberately pessimistic: PCIe-only all-reduce on a 1.9B model with a
128M-row embedding-shaped gradient is not a friendly workload, and WSL2 is not a friendly
host.

### 4.2b Reality check: what real drafter runs actually cost

The table above is FLOPs arithmetic. Published GPU-hours for real drafter training runs are
**an order of magnitude higher**, and I am flagging that rather than defending my estimate.
From the sibling session's report (`<scratchpad>/alternatives_2026-09-05.md`, 2026-09-05):

| Run | Hardware | Wall clock | GPU-hours | Source |
|---|---|---|---|---|
| EAGLE3 for Qwen3-4B-Instruct-2507 | 8x MI308X | 56 h | **448** | `taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3` card |
| EAGLE3 for Qwen3-VL-8B | 8x | 74 h | **592** | SpecForge-trained card |
| EAGLE-3 for Qwen3-8B | 8x A100 | | **696** | DEER paper arXiv:2512.15176 Table 8 |
| EAGLE-3 for Qwen3-14B | 8x A100 | | **1,440** | DEER paper arXiv:2512.15176 Table 8 |
| SpecForge paper reference | 8x H200, Open-PerfectBlend 1.4M convs, 2 epochs | not stated | not stated | arXiv:2603.18567, 2026-03-19 |

Two things reconcile this against my 42-hour paper-scale figure, and neither fully closes
the gap:

1. **These are EAGLE-3, not DFlash, and EAGLE-3 is the expensive one by design.** The DFlash
   paper says so explicitly (section on efficient long-context training, verbatim):
   "Training speculative draft models on long contexts is challenging for methods such as
   EAGLE-3 due to their costly training-time test. DFlash achieves efficient long-context
   training by fixing the number of masked blocks per sequence and randomly sampling anchor
   positions." EAGLE-3 unrolls multi-step chains during training; DFlash does one
   bidirectional pass per block. So DFlash should be materially cheaper per token.
2. **My 40% and 30% MFU assumptions are optimistic**, especially for a 5-layer model with a
   248k-row output and a 128M-row codebook gradient, and they ignore dataloading, capture
   stalls and checkpointing.

**Treat 4.2's table as a floor and these as a ceiling.** The honest statement is that a
from-scratch, paper-scale DFlash2 drafter for a 27B target is somewhere between 40 and
several hundred H100-equivalent GPU-hours, and nobody has published the number for the
drafter we run. This is exactly why every costed scenario below that the human can afford is
a **warm-started fine-tune**, where the empirical anchor is not these 448-to-1,440 GPU-hour
numbers but the `rwmacy` card's 440 steps on two consumer cards.

### 4.3 Rental prices, read from the provider pages on 2026-09-05

Prices below were fetched from the providers' own pricing pages on **2026-09-05** by a
sibling session and recorded with URLs in
`<scratch>/pricing_2026-09-05.md`. I did not re-fetch them. Per GPU per hour, on-demand.

| Card | VRAM | RunPod Community | RunPod Secure | Vast min / median | Vast interruptible min | Lambda 1x |
|---|---|---|---|---|---|---|
| RTX 4090 | 24 GB | 0.34 | 0.74 | 0.223 / 0.435 | 0.158 | not offered |
| RTX 5090 | 32 GB | 0.69 | 0.99 | 0.343 / 0.627 | n/q | not offered |
| RTX A6000 | 48 GB | 0.33 | 0.53 | 0.402 / 0.463 | n/q | 1.09 |
| L40S | 48 GB | 0.79 | 0.99 | 0.801 | n/q | not offered |
| RTX PRO 6000 | 96 GB | 1.69 | 2.09 | 1.341 / 1.602 | n/q | not offered |
| A100 PCIe 80 GB | 80 GB | 1.19 | 1.39 | 0.668 (40 GB card) | n/q | 1.99 (40 GB) |
| A100 SXM 80 GB | 80 GB | 1.39 | 1.59 | 0.564 (40 GB card) | 0.401 (40 GB) | 2.79 (8x only) |
| H100 PCIe | 80 GB | 1.99 | 2.89 | 2.935 | n/q | 3.29 |
| H100 SXM | 80 GB | 2.69 | 3.29 | 2.406 / 4.341 | 0.673 | 4.29 |
| H200 | 141 GB | 3.59 | 4.59 | 3.977 / 4.490 | n/q | not listed |
| B200 | 180 GB | 5.98 | 6.79 | 6.010 / 7.752 | n/q | 6.99 |

Sources, all fetched 2026-09-05: RunPod https://www.runpod.io/pricing and
https://docs.runpod.io/pods/pricing; Vast.ai live offers API
https://console.vast.ai/api/v0/bundles/ because https://vast.ai/pricing renders prices
client-side and has no numbers in static HTML; Lambda https://lambda.ai/pricing (the old
lambdalabs.com URL 404s); Together AI https://www.together.ai/pricing.

Practical notes that change the arithmetic:

* **Together AI is out.** Its smallest H100 unit is an 8-GPU cluster at 8 x $3.99 =
  **$31.92/hr**, and H200/B200 start at 256 GPUs. Wrong shape for a single-node drafter run.
* **Lambda has no 24 GB or 48 GB consumer tier and no spot.** Billed per minute, terminated
  means gone.
* **RunPod bills per second with no egress fees.** Storage is separate at $0.10/GB/mo
  container and volume disk. Its docs also say it "is no longer accepting new hosts for
  Community Cloud", so the cheap tier may thin out.
* **Vast is a marketplace**, every price is host-set and moves hourly; today's verified-host
  datacenter inventory was 2 to 9 offers per card, so those medians are noisy. Bandwidth is
  billed per byte in both directions at host-set rates, which matters if you are moving a
  hidden-state cache.

### 4.4 Costed scenarios

Combining 4.2's hours with 4.3's prices. All **[estimate]**.

| Plan | Rig | Hours | Rate | **Compute cost** |
|---|---|---|---|---|
| **A. Warm-start fine-tune, `rwmacy` scale** (about 1k on-policy samples, a few hundred steps, matched to our W4A16 target) | 2x RTX 4090 | 4 to 10 | $0.34 x2 to $0.74 x2 | **$3 to $15** |
| **B. Same, but on one 48 GB card** (no sharding, no PCIe all-reduce, simpler) | 1x A6000 or RTX 6000 Ada | 4 to 12 | $0.33 to $0.84 | **$1.50 to $10** |
| **C. Eighth-scale run from a warm start** (100K samples, ~6 epochs) | 1x H100 SXM | 6 to 12 incl. capture | $2.69 to $3.29 | **$16 to $40** |
| **D. Same on rented 2x4090** | 2x RTX 4090 | 25 to 40 | $0.68 to $1.48/hr for the pair | **$17 to $60** |
| **E. Paper-scale from scratch**, excluding response regeneration | H100 SXM | 42 (my FLOPs floor) to 400+ (EAGLE-3 empirical ceiling, 4.2b) | $2.69 to $3.29 | **$113 to $1,300, and I do not trust the low end** |
| **F. Paper-scale including regenerating 800K responses with the 27B target** | H100 + serving | hundreds | | **$400 to $2,000+, very wide** |

**What dominates the cost.** Not the drafter's gradient step. In order:

1. **Producing target-aligned data.** The paper regenerates all responses with the target.
   That is decode-bound serving time, and at paper scale it dwarfs the training. This is
   also the item most compressible: reuse the 5.4M tokens `gen_data.py` already produced,
   or generate a few million more, and skip 800K entirely.
2. **Capturing hidden states.** One target forward over the corpus, 3.6e19 FLOPs at paper
   scale, comparable to the whole drafter training, plus terabytes of storage if you cache
   it rather than stream it.
3. **The drafter's own optimization.** Genuinely cheap. Scenario A's actual gradient work is
   single-digit dollars.

So the honest framing is: **at the scale the human is contemplating ($10 to $100), the money
does not buy a from-scratch drafter. It buys a targeted fine-tune of an existing one, on
data we mostly already have.** That is scenario A or B, and it lands inside budget with room
to repeat it three or four times.

### 4.5 Can two RTX 4090s do it? Yes, with three conditions

**Yes, for a warm-started fine-tune, and SpecForge's own 27B reference topology is two GPUs.**
Conditions:

1. **The target must be quantized.** bf16 is 51.75 GiB against 24 GiB. W4A16 is 14.71 GiB
   measured on this box and fits card 0 with headroom, exactly as `SPEC=dflash2` already
   runs it. This is not a compromise for us; it is the **matched** condition, since W4A16 is
   what we serve. The `rwmacy` precedent did exactly this on an FP8 stack.
2. **The topology must be split, not replicated.** Card 0 runs the capture server holding
   the W4A16 target. Card 1 trains the drafter. Do **not** use NeMo's DDP path, which
   "replicates the frozen target" on every rank. Use SpecForge's disaggregated managed-local
   mode, which is written for precisely this and whose checked-in 27B example allocates one
   GPU to each role.
3. **The logits must be chunked and the run must be warm-started.** 512 anchors at 248k
   vocab is 4.97 GiB of softmax workspace on its own; cut `num_anchors` or chunk the
   objective. And 28.68 GiB of AdamW state means either sharding across both cards, which
   conflicts with condition 2, or bf16 moments at 14.34 GiB on one card, which fits. From
   scratch on this rig is 200+ hours and is not worth doing; from a warm start it is a
   single-digit-hours run.

**The honest caveat on condition 2:** if card 0 is fully occupied by the target, you are
training on **one** 4090, not two, and the 4.2 table's single-4090 row applies. That is
still fine for scenario A. It is not fine for scenario E.

**The WSL2 caveat:** the repo already documents that every `SPEC=dflash2` profile needs
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` on WSL2, and this box has a history of WSL2-specific
memory-pinning trouble on this exact path. A training run adds a second framework to that
surface. Budget debugging time, not just GPU time.

---

## 5. Alternatives that need no training

Ordered by cost-to-try, cheapest first. **Every one of these is a serving-side change and
therefore competes with `docs/research/spec-decode-sota-2026-09.md`'s proposals; read that
file's "What is worth trying on this stack in two days" before spending anything here.**

**0. Run our existing drafter BELOW its trained block. Cost: one config value, zero
downloads, zero training.** This is the cheapest item on the list and I did not have it until
the sibling session found it. Two independent sources point the same way:

* vLLM 0.28's `qwen3_dflash2.py` line 131 sets the draft block to `1 + num_speculative_tokens`
  **with no assertion against the checkpoint's trained `block_size`**, so `K = 4, 5, 6` is a
  legal no-retrain knob on the drafter we already have (sibling report, source line verified
  by curl+grep).
* The z-lab README says, for the MLX path, "For quantized targets or drafts, use
  `block_size <= 5`". **We run a quantized target and a quantized drafter**, which is exactly
  the condition that recommendation names.

Paper Table 7 supports the direction only weakly (train-8/test-8 beat train-8/test-16, but
the paper never tests below the trained block), so this is a genuine unknown with a cheap
answer. Combine with `num_speculative_tokens_per_batch_size`, vLLM's in-tree dynamic
speculation, which the config docstring says is "Tested with Eagle, Eagle-3, and DFlash". The
repo's own benchmark harness can price the K sweep in an afternoon. **Do this first.**

**0b. `rejection_sample_method: "block"`.** Block verification, no training, one config key.
The companion file `docs/research/spec-decode-sota-2026-09.md` already has this as its
Proposal 2 with the analysis; not repeating it here.

**1. Try the block-16 community drafter for our target. Cost: a 3.96 GiB download.**
`onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2`, 6 layers, `block_size` 16,
`DFlashDraftModel`. This is the highest-value zero-dollar experiment on the list, because it
directly tests the block-16 hypothesis on our exact target without training anything.
Risks, both real: it is DFlash **v1** (no selector, no convolution, so it gives up DFlash2's
published 21% advantage), and it was published for the Atlas engine against an NVFP4 target,
so vLLM loading is unverified. Also it is 3.96 GiB bf16 against our 1.19 GiB int4, which is
a real per-step read cost on a 24 GB card, and the repo has already measured that read cost
as the thing that "turns DFlash2 from a wash into a win on this card".

**2. Raise `selector_top_k` above 16. Cost: a config edit plus a benchmark run, if the
implementation honours it.** Grounded but with a low ceiling. Inco's Table 1 shows recall@16
is already 87.8% to 99.5%, so the missing token is rarely outside the top 16; the gap is
**selection**, not recall. Their own oracle number (6.79 vs the selector's realized 4.61)
says a wider candidate list adds candidates the pairwise scorer must then get right. No
source publishes any k>16 measurement. The codebooks are per-vocab so nothing in the
checkpoint forbids it. **[estimate]** I would expect small gains and a latency cost, and I
would test it only after item 1.

**3. Change `selector_rank`. Cost: nothing, because you cannot.** `selector_rank: 256` is
the width of the trained codebooks. Changing it requires retraining. Listing it here only to
close it off.

**4. Tree or multi-candidate drafting. Not available at any price in vLLM.** The sibling
session settled this: vLLM PR #42121, "[Attention][Cleanup] Remove tree attention", merged
2026-05-09, states "TreeAttention spec-decode is currently not fully supported and there's
no plans to support it in the short to medium term". No `tree_attn.py` at tag v0.28.0 or on
main, no tree field in `SpeculativeConfig`. DFlash in vLLM is chain-only by construction (one
bonus query plus K mask queries) and the DFlash2 selector deliberately collapses the
candidate lattice to a single path, so verification stays linear. Tree verification for
block drafters exists only in papers (CaDDTree arXiv:2606.01813, Bastion arXiv:2605.29727,
neither read by me). DFlash's own argument against EAGLE-3 is that it beats tree drafting
"while incurring substantially lower verification overhead" (paper section 5.1). See the
companion file's "Tree and graph verification" section.

**4b. Suffix decoding. Cost: one pip install plus a benchmark.** vLLM 0.28 ships
`method: "suffix"` (requires `arctic-inference==0.1.1`), documented at
https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/, defaults
`num_speculative_tokens` 24 with the docs recommending 16 to 32, aimed at "code-editing,
agentic loops, RL rollouts". Worth knowing it exists, but note it would **replace** DFlash2
rather than augment it, and our repo's `VLLM_DFLASH2_LOOKUP` already does the
draft-from-context job while keeping the model drafter. Low expected value here; listed for
completeness.

**5. EAGLE-3 or EAGLE-3.1 for Qwen3.8-27B. There is no such thing to download.** I pulled
the full Hugging Face model API listing for the search term `Qwen3.8` today, 1,000 models
returned, and filtered the ids for `eagle`, `spec`, `draft`, `mtp`, `dspark`, `medusa`,
`domino` and `suffix`. **Zero EAGLE, zero EAGLE-3, zero EAGLE-3.1, zero Medusa, zero
Domino.** The only speculative drafters that exist for any Qwen3.8 model are the native MTP,
the DFlash family (Inco's DFlash2 plus the three community efforts in 2.5), and DSpark
(`RadixArk/Qwen3.8-27B-DSpark` and five derivatives). EAGLE-3.1 is real as a *method*,
SpecForge trains it and ships `configs/qwen3-30B-A3B-eagle3.1.json`, so this would be a
training question rather than a download. It is not worth asking: the DFlash paper's Table 1
puts EAGLE-3 on Qwen3-8B at tau 2.96 (tree size 16) to 3.40 (tree size 60) against DFlash's
6.49. **Training an EAGLE-3 to replace a working DFlash2 would be spending money to go
backwards.**

**6. The knobs already in this repo.** `VLLM_DFLASH2_LOOKUP` and its family,
`VLLM_DFLASH2_CHAIN`, `VLLM_DFLASH2_DRAFT_TOPK_TOPP`, `DFLASH_TOKENS`. The repo's own table
already prices these: lookup drafting takes C1 from 126 to 130 tok/s and up to 259 where the
model reproduces its context, and `DFLASH_TOKENS=15` reaches 381 tok/s on
context-reproduction workloads. **These are already on by default and already measured.**
The zero-cost headroom here has largely been spent.

**7. `RadixArk/Qwen3.8-27B-DSpark`.** Exists for our target and is SpecForge-trained. Inco's
Table 4 measures it at mean **3.62** against DFlash2's 4.80; its own card reports mean
acceptance **3.43** (sibling session). Both are below what we already get. Worse. Do not
bother, except as the warm-start ancestor that the `rwmacy` fine-tune used, which is a
different use for it.

---

## 6. What the sources do not say

Stated plainly so nobody infers a number that does not exist.

* **No GPU-hours, wall clock, GPU type or step count for any DFlash or DFlash2 drafter**,
  from the paper, the blog, any Inco model card, or the z-lab repo. This is the single
  biggest gap and it is not closeable from public sources.
* **No batch size in the DFlash paper.** SpecForge's 27B YAML uses `batch_size: 1` with
  `accumulation_steps: 4`, which is a fact about SpecForge, not about how Inco trained ours.
* **No acceptance-versus-training-tokens curve** anywhere. DFlare's 800K to 2.4M is
  confounded with an architecture change.
* **No k>16 selector measurement.**
* **No statement of whether our Qwen3.8-27B drafter was warm-started.** The Muse Glimmer one
  was; ours has no published ancestor.
* **Unverified on the real path:** I have not run SpecForge, not loaded the Apathy drafter
  in vLLM, not measured any training step, and not tested the quantized-target-mismatch
  hypothesis. Everything in sections 4 and 5 is arithmetic and reading, not measurement.
* **The H100 dense-FLOPS figure** is inferred from NVIDIA's sparsity convention rather than
  read off a parsed datasheet; the PDF would not convert. If 1,979 TFLOPS is dense, halve
  every H100 hour in 4.2.

---

## What I would tell the human

DFlash is a real ICML 2026 paper out of UC San Diego with a fully published recipe, and
DFlash2 is Inco AI's product release on top of it, which published the architecture and not
one word of the recipe, so nobody outside Inco knows what our specific drafter cost to make.
The recipe that is public is the paper's: 800K target-regenerated samples, six epochs, AdamW
at 6e-4, block-8 drafters get a loss decay of 4, and the target's hidden states can be
precomputed offline so the target does not have to be resident while the drafter trains.
Yes, we can: three separate projects ship DFlash trainers, and SpecForge has a checked-in
DFlash2 config for a 27B Qwen target with our exact tokenizer and hidden size, plus a
reference run that uses precisely two GPUs, one holding the target and capturing hidden
states, one training the drafter. Somebody has already done a version of this for our target
on two Intel Arc Pro B70s: warm-started from an existing drafter, 961 samples, 440 steps, and
it worked, though its acceptance came out at 2.5 to 3.5 tokens per step, which is no better
than what we already get. Two 4090s are enough for that kind of run, but only if the target
is quantized, because bf16 Qwen3.8-27B is 51.75 GiB against a 4090's 24, whereas the W4A16
target we already serve is 14.71 GiB and fits one card with room; the drafter's own AdamW
state is 28.7 GiB, which needs either sharding or bf16 moments, and the 248k-row vocab means
the logits need chunking. Two 4090s are not enough to train from scratch: that is 200-plus
hours on this rig and $113 to $263 of rented H100 time, and the cost is dominated not by the
gradient step but by regenerating training responses with the 27B target and capturing its
hidden states. So the $10 to $100 budget does not buy a new drafter, it buys three or four
attempts at a targeted fine-tune, roughly $3 to $15 each on two rented 4090s or $16 to $40 on
one H100. Before spending any of it, do the three free things: sweep `num_speculative_tokens`
down to 4, 5 and 6, which vLLM permits on our existing checkpoint with no retrain and which
z-lab explicitly recommends for quantized targets and drafts; try
`onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2`, a block-16 DFlash drafter for our exact
target that someone else already trained; and notice that our measured 3.14 to 3.34 tokens
per step sits about a full token below Inco's published 4.80 on a bf16 target, which points
at our quantized target's hidden states being off-distribution for a drafter trained on bf16
ones. That mismatch, not model capacity, is the thing I would spend the first ten dollars on,
and the capture tooling for it is already sitting in `drafter/capture_dflash2.py`.

---

*Scratch and provenance for this file, all under the session scratchpad:*

| File | What it is | How verified |
|---|---|---|
| `dflash.txt` | arXiv:2602.06036**v1** full text, downloaded and converted locally | Every DFlash quotation in this file is character-exact from it |
| `dflash2blog.txt` | inco.ai/blog/dflash2 full text, downloaded and converted locally | Every Inco quotation is character-exact from it |
| `pricing_2026-09-05.md` | Provider pricing, fetched from provider pages 2026-09-05 by a sibling session | Provider pages read directly except Vast, whose page is JS-only and whose numbers come from its live offers API |
| `alternatives_2026-09-05.md` | vLLM knobs, EAGLE availability, SpecForge costs, from a sibling session | Its own note: "vLLM source lines, PR commit messages, directory listings, and release dates were curl+grep exact; HF ids/dates from the HF JSON API; paper/blog/doc quotes came via the page summariser (asked for verbatim, high confidence but not character-exact)" |

Local files read on this box: `models/Qwen3.8-27B-DFlash2-W4A16/{config.json,README.md,model.safetensors}`
(header parsed, not loaded), `drafter/README.md`, `README.md`.
Remote files read raw rather than summarized: the two HF `config.json`s, the four HF model
cards in 2.5, `Qwen/Qwen3.8-27B/config.json` and its safetensors index, the SpecForge repo
tree and five of its files. Sources reached only through the page summarizer, and therefore
paraphrase-grade rather than quote-grade: the NeMo AutoModel recipe page, the Red Hat
Speculators article, the Baseten post, the DFlare abstract page.

---

## Verification by the presiding seat (12:37Z, shell clock) before this file was committed

Checked on the real path, not from the file: the Hugging Face model API returns all four checkpoints named here (onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2, last modified 2026-08-24, tag dflash; rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70, 2026-08-16, tag dflash; incoai/Qwen3.8-27B-DFlash2 and z-lab/Qwen3.8-27B-DFlash2, both 2026-08-19, tags dflash2 and block-diffusion); the SpecForge repository holds configs/qwen3.6-27b-dflash2.json (1238 bytes) at its default branch; and the rwmacy model card, read raw, says 2x Intel Arc Pro B70 (TP=2), warm start from the released RadixArk Qwen3.8-27B DSpark drafter, SpecForge DSpark strategy, 440 steps, lr 5e-5, and that vLLM needs its DFlash kernel readout fix, without which acceptance collapses. One caveat this seat adds: the sentence comparing our measured 3.14 to 3.34 tokens per step with Inco's published 4.80 compares numbers from different cohorts, sampling settings and possibly different definitions of tokens per step (the arc's default-width figure is 3.8 accepted tokens per step including the bonus token, 2.8 without it, on an eight-prompt cohort at the model's own sampling), so the gap is not established at one token and the off-distribution hypothesis it motivates stands on its own plausibility, not on that subtraction. Everything in sections 4.1 to 4.5 is arithmetic and reading, as the file says; no training step has been timed on this box.
