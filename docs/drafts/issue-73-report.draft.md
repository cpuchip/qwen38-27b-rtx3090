**A cause, a one-line change that partly fixes it, and an explanation for why this was so hard to measure.** Two boxes, and the short version is that the fix helps exactly where the drafter is carrying the whole verify block, which may or may not be most of your cohort.

## The cause: a launcher field that is set for MTP and omitted for DFlash2

`single-user/start_qwen.sh:432` builds the MTP config with `"draft_sample_method":"${DRAFT_SAMPLE:-probabilistic}"`. Line 219, the DFlash2 config, does not set the field at all, so it takes vLLM's default of `"greedy"`, documented as treating the draft probabilities as one-hot.

On 0.27.1 that omission was harmless, because the fork allocated the buffer unconditionally in its own speculator with a comment stating why: the selector samples a probabilistic path for non-greedy requests, so rejection sampling always needs the realized proposal distribution. The 0.28 port dropped that unconditional allocation and inherited the upstream base class, which allocates only when the config asks for it. Both boot logs say so outright: `draft_logits=True` on 0.27.1, `draft_logits=False` on merged, same profile, same width.

The consequence is in the rejection kernel. With no proposal distribution the draft probability is pinned to one, so the acceptance test collapses from the ratio of target to draft down to the bare target probability. Since the draft probability is never above one, the ratio test accepts at least as often, and the merged path is strictly stricter. That predicts the direction of your regression, and it is invisible at temperature zero, which is why greedy probes come back clean.

## The fix, and the stratification that limits it

Setting `draft_sample_method: "probabilistic"` on the DFlash2 config. Measured on an RTX 4090 under WSL2, your cohort file at 1024 output tokens, four seeds by eight prompts, paired by seed and prompt, tokens per step as the metric, split on the pre-treatment block size:

| stratum | cells | shipped | field set | t |
|---|---|---|---|---|
| overall | 32 | 3.440 | 3.730 | 2.77 |
| drafted tokens per round = 7 | 18 | 2.886 | 3.425 | 4.22 |
| drafted tokens per round > 7 | 14 | 4.152 | 4.123 | -0.22 |

The paired difference is the claim; the absolute levels are single-cohort figures from a box whose throughput swings run to run, so they are not reference numbers for that card. The threshold was registered before the data: three or above established, two to three suggestive. So the overall figure is suggestive and the stratification is the finding. **The field only matters where the drafter supplies the whole block.** Positions filled from context by the lookup never came from the drafter, so they carry no proposal distribution to normalise against whether the field is set or not, and the effect dilutes to nothing where the lookup is engaged. That split was registered in advance as the interesting outcome, not found afterwards. The same direction appears on an RTX 3090 at a quarter of the sample size, positive in the first stratum and not in the second, with no significance claimed.

**The overall number is a mixture, not a constant of the fix.** It depends on what fraction of a workload sits in each stratum, so a different prompt mix gives a different overall figure from the same mechanism. Ours is roughly eight percent; yours will differ.

## Your regression reproduces, at your workload

Raw speculative-decode counters around a fixed workload, no teacher forcing:

| box | 0.27.1 | merged | change |
|---|---|---|---|
| yours, as filed | 3.19 | 2.66 | -16.6% |
| RTX 4090 | 3.176 | 2.621 | -17.5% |
| RTX 3090, via your `prefill_ab.sh` | 3.10 | 2.72 | -12.3% |

Drafted tokens per round is unchanged across versions on both boxes, so this is proposal survival rather than block length. Decode rate tracks tokens per step to within about one point on both, independently confirming your structural claim.

## Why this is hard to measure, which cost us most of the day

- **Seed spread swamps the effect at short generation lengths.** Six seeds on one build, one prompt, 256 tokens: acceptance ranged over ten points, standard deviation around four. Thirty seeds still could not see the version effect at that length. Every short-prompt bisect cell we ran was underpowered, for the version effect, for the patch reversal, and for degeneracy rate alike. The same cells at cohort scale saw it immediately.
- **Two passes with a fixed seed are two replays, not two samples.** Repeat runs came back bit-identical on every counter on the quiet box, and two of three did on the noisy one. Reproducibility to three significant figures measures a deterministic harness rather than bounding the effect, so two passes are one draw per arm. Ours were too, and we did not notice either until a seed sweep forced it.
- **Prompt choice moves acceptance by more than the regression does.** On one build, 36.8% on a short prompt against 22.7% on the full cohort.
- **Acceptance is heavy-tailed, because a collapsed generation scores near-perfect acceptance.** One repetition loop hit 79.7%, which is the drafter predicting degenerate text rather than performing well. A high acceptance reading is ambiguous between the two and the counter cannot separate them. Guard with drafted tokens per round above about 1.2 times the drafter width, calibrated on 96 rows, and note the guard must run **before** any stratification on block size, since degenerate rows sort into the engaged stratum by construction.

## Two traps for anyone bisecting this

**At width 15 on ordinary text the engine drafts 7 and queries 8, by design.** The clamp sets the drafted block to the checkpoint's trained block and `num_query_per_req` follows it, and the adaptive policy asks for the long block only while a request is reproducing its context. So width 15 and width 7 are close to the same configuration unless the text repeats, and your matching results at both widths are weaker evidence for "not lookup-specific" than they look. Your own counters answer it: how much of your cohort actually exceeds 7 drafted tokens per round?

**Request order changes measured block length.** The long block is sticky and coasts on prior state without consulting the emitted-token count, so per-request drafted-token counts are not independent within a boot and per-request acceptance figures are only comparable within a fixed running order. Anything that scores per-request acceptance picks this up silently.

## The residual, and why one bisect cell saw it and another did not

`dflash2-z-adaptive-emitted.patch` is real and its effect is workload-dependent, which reconciles your 2.66 to 2.79 with a null on another box.

Instrumenting the predicate per step rather than per round shows why. The patch changes the emitted-token count that the block-length policy reads, and the policy compares it against one plus the drafted block. The two definitions can only disagree in a narrow band: the step has to sample at least eight tokens and reject enough of them to fall back under eight. That needs a long block and moderate rejection at the same time. Over 99 logged steps on a short single-prompt cell, the predicate differed on exactly one step, and block entry requires two qualifying steps in succession, so a single flip changes nothing and every counter matches.

A cohort at 1024 output tokens spends far more steps in the long block, so it lands in that band far more often. That is the difference between a box that measures a quarter of the gap from this patch and one that measures nothing, and it is exposure rather than disagreement. What would break this account: if your cohort turns out to spend few steps in the long block and reversing the patch still moves your number by that much, the band explanation is wrong.

Two notes for anyone reversing it. The patch touches two files and only one is live in this configuration, since the other sits in a chain path that is off by default. And the block-length policy is sticky: once entered it coasts on prior state without consulting the emitted count at all, so those steps are patch-independent by construction.

The general lesson cost us several hours and is worth stating: **when a mechanism turns on a threshold, instrument the distribution and not the mean.** Every wrong reading we produced about this patch came from reasoning about per-round averages toward a per-step predicate.

## The one thing that would help most

You already have the number. Report drafted tokens per round from your cohort. If it sits at 7, the field should recover most of your gap; if much of it is above 7, it should recover little; and any rows well above 7 should be checked for repetition before being read at all. One counter you are already scraping, three distinguishable predictions.
