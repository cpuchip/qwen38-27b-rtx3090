# EAGLE-4 speculative decoding on one 24 GB card (experimental)

This directory serves Qwen3.8-27B W4A16 with the EAGLE-4 draft (`EAGLE-Corp/Qwen3.8-27B-EAGLE4`) on a single RTX 4090,
using eagle-hq's published vLLM image and a few patched files. It works, and on this card it is slower than this
repo's DFlash2 at one user, so it is an experiment and not a recommended mode.

> This project includes or is derived from Eagle 4 technology developed by Eagle AI Corp. It is not an official Eagle
> AI Corp project. The files under `eagle_harness/` are governed by the Eagle 4 Source-Available License (not an
> OSI-approved open-source license), not by this repository's Apache-2.0 license; see `eagle_harness/README.md`.
> Downloading, using or modifying the EAGLE-4 software or draft means accepting that license and the Eagle Trademark
> Policy.

## Results (RTX 4090, one request at a time)

Decode tok/s = (completion tokens - 1) / time from the first to the last streamed token. Thinking off. Five short
prompts plus one copy task (repeat a public-domain passage verbatim: 1,956 prompt tokens, 1,926 out), greedy, then the same at
temperature 1 (top_p 1, top_k off, seed 0). One run per cell (EAGLE-4 was run twice; the table has the first), measured 2026-10-21; `bench/probe.py` is the probe.

| decode tok/s | no speculation | EAGLE-4 (this directory) | DFlash2, k7 (`single-user/start_qwen.sh SPEC=dflash2 CTX=fast`) |
|---|---|---|---|
| greedy, all six | 58.3 | 241.1 | 284.3 |
| greedy, the five prompts only | ~59 | ~140 | ~193 |
| temperature 1, all six | 58.2 | 228.8 | 269.5 |
| greedy copy task | 58.0 | 392.2 | 379.3 |

- The no-speculation column ran on the same EAGLE-4 image with the same target and flags, without a draft. DFlash2 ran
  on a build of this repo's vLLM 0.32 port (port-0.32, not yet merged), so the two speculation columns differ in engine as well
  as in drafter.
- Tokens accepted per step on the prose prompts are about the same for both drafters (EAGLE-4 3.05, 3.20, 3.44, 3.82;
  DFlash2 3.10, 3.00, 3.67, 3.09), but an EAGLE-4 step costs more on this card: about 37 ms against 21 ms for DFlash2
  and 17 ms for plain decoding (from the copy task's tok/s and tok/step). The tree drafts 14 levels every step.
  A shallower tree (depth 6, width 8) cut the step to about 26 ms and still trailed DFlash2 (~147 tok/s on the five
  prompts). Their cost model can stop the tree early; it needs an offline calibration this card could not run at
  its default sweep (verify widths up to 64 need more memory than is left), so it is untested here.
- A second EAGLE-4 run, from `prepare_draft.py` and `start_eagle4.sh` as described below, read 246.3 greedy and 235.3 at temperature 1
  for all six, and the same 392.7 on the copy task.
- Greedy output against no speculation (`bench/score.py`), over those two runs: the copy task and two prompts were
  token-identical both times, and one prompt first differed at token 54 both times. The other two prompts each
  differed in one run only (at token 2 in one, at token 81 in the other) and matched in the other run, so greedy
  output with EAGLE-4 was not repeatable run to run here. That looks like near ties flipping on run-to-run numerics;
  it is not checked.
- EAGLE-4's own published results are for a BF16 target on B300 GPUs. Nothing here speaks to those.

## What it takes to fit

As published, the target, the 5.75 GiB draft and their caches do not fit in 24 GB: with both models loaded, vLLM had
no memory left for the cache at `--gpu-memory-utilization 0.92`. These changes make a 16K context fit for one request:

| change | where | saves |
|---|---|---|
| int8 token embedding for the target (the checkpoint already stores it) | `../patches/qwen3_5-embed-quant.patch`, as the main launchers use | 1.16 GiB |
| the draft's own copy of the embedding dropped; vLLM then shares the target's (they are byte-identical) | `prepare_draft.py` | 2.37 GiB |
| the draft's lm_head in int8 with one fp32 scale per row | `prepare_draft.py` + `eagle_harness/` | 1.19 GiB |
| recurrent state in float16 (the config says float32) | `--mamba-ssm-cache-dtype float16`, as the main launchers use | half of ~2.25 GiB per request |
| one request at a time, `--gpu-memory-utilization 0.93` | `start_eagle4.sh` defaults | |

With all of them the cache holds 21,746 tokens, 1.33 requests at 16K. Speculation keeps 16 recurrent-state slots per
request (one per verify position), which is most of what a request costs here.

The int8 lm_head only re-scores the 128 candidates per draft row that EAGLE-4's low-rank head picks, so the change is
in one kernel (`gather_dot`). On a GPU check against a torch reference (vocab 248,320, hidden 5,120, 16 rows of 128
candidates) the int8 kernel matched the dequantized reference to 1.7e-3 relative error, the same as the original
bf16 kernel against its reference, and agreed with the bf16 head's top candidate on all 16 rows. Dropping the scale
fails that check (error 1.7e+3).

Two other patches:

- `patches/vllm/eagle-share-quantized-head.diff`: vLLM's Model Runner V2 compares the draft's and the target's lm_head
  by reading `.weight`, which a quantized lm_head does not have, and the draft fails to load. Proposed upstream as
  [cpuchip/vllm `up/eagle-share-quantized-head`](https://github.com/cpuchip/vllm/tree/up/eagle-share-quantized-head).
- `eagle_harness/model_rows.diff`: the draft's fused input path reads `embed_tokens.weight` directly; with a shared
  int8 embedding it takes the plugin's existing unfused path instead.

## Run it

1. Read the [Eagle 4 Source-Available License](eagle_harness/LICENSE). Downloading the draft or the image accepts it.
2. Download the draft (`EAGLE-Corp/Qwen3.8-27B-EAGLE4`, 6.2 GB) into your models directory, next to the target
   `Qwen3.8-27B-W4A16-AutoRound` that the rest of this repo uses.
3. `python eagle4/prepare_draft.py <models>/Qwen3.8-27B-EAGLE4 <models>/Qwen3.8-27B-EAGLE4-24g` (numpy only; writes
   a 2.4 GB draft). `--bf16-head` keeps the lm_head as it is; that draft does not fit at 16K on a 24 GB card.
4. `MODELS_DIR=<models> bash eagle4/start_eagle4.sh`. It pulls `ghcr.io/eagle-hq/eagle-4-vllm:0.1.0` by digest
   (49 GB unpacked), copies the six files it patches out of that image, patches the copies, and starts the server on
   `127.0.0.1:8000` with the copies mounted. `VLLM_API_KEY` in the environment sets the server's key. Knobs at the top
   of the script: `PORT`, `GPU`, `MAX_LEN`, `MAX_SEQS`, `GPU_UTIL`, `TREE`, `LM_HEAD_INT8`, `PREPARE_ONLY=1`.
5. `python eagle4/bench/probe.py <label> --port 8000 --passage <text file>` reproduces the table's rows.

Requirements: an NVIDIA GPU from the A100 generation on (sm_80+, the Triton kernels' floor), CUDA 13 drivers for the
image, and Docker with GPU access. Tested on one RTX 4090 under WSL2.

## Limits

- The KV cache must be unquantized: EAGLE-4's tree verification on vLLM does not support a quantized KV cache. This
  repo's int4 KV and KVarN profiles cannot use it; a tree attention that reads those pages would have to be written.
- One request at a time at 16K. Batches that carry a grammar (structured output) verify only the root token per
  round, per EAGLE-4's own docs.
- The image is eagle-hq's build of vLLM (`0.32.0rc1.dev283+eagle.4`), not this repo's. None of this repo's other
  vLLM patches are in it.
