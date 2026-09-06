# Two independent surveys of the speculative-decoding frontier, compared (2026-09-06)

Michael's question (2026-09-06): "could you and astra websearch for the latest in speculative decoding research? see if theres something we could implement maybe?"

Two surveys ran in parallel without reading each other: [the agent's](spec-decode-frontier-2026-09-06.md) (a research subagent of the head seat) and [Astra's](spec-decode-frontier-2026-09-06-astra.md) (a Codex seat; its report states the parallel report was not read). This file compares them and adds three readings taken tonight from the overnight soak's own counters (the shipped default on card 0, `curl /metrics` run inside the container at 04:33:37Z, 37,965 drafts into pass 5). Convergence between two independent searches is evidence about the field, not proof of a local gain; every projected local gain in both surveys is unverified until it is run here.

## Three readings from the soak's counters (measured tonight, this box)

**1. The preflight both surveys asked for: the ceiling bin is 14.1 percent of steps.** Both surveys cite DBloom (arXiv 2608.30427): measure how often the whole block is accepted before deciding to widen it. The engine's `spec_decode_num_accepted_tokens_per_pos` counters at 04:33:37Z, over 37,965 drafts of width 7 (raw: 27,239 / 19,891 / 14,804 / 11,274 / 8,823 / 6,913 / 5,368; they sum to the 94,312 accepted total):

| position | accepted in this share of steps | continuation given the previous position |
|---|---|---|
| 1 | 71.7% | |
| 2 | 52.4% | 0.730 |
| 3 | 39.0% | 0.744 |
| 4 | 29.7% | 0.762 |
| 5 | 23.2% | 0.783 |
| 6 | 18.2% | 0.784 |
| 7 | 14.1% | 0.777 |

As a histogram of exactly-k accepted: 0: 28.3%, 1: 19.4%, 2: 13.4%, 3: 9.3%, 4: 6.5%, 5: 5.0%, 6: 4.1%, 7 (the ceiling): 14.1%. The ceiling bin is three times the bin beside it, so one step in seven wanted more than the block offers. The record already has what happens when the block is widened without training: on this box width 15 read 3.739, 3.727 and 3.727 tokens per step against 3.757 at width 7 (minus 0.5 percent, three boots), on the native 3090 4.116 against 3.900 (plus 5.5 percent), with the long block "entered on both boxes and about a third as often here" (v0.28-validation.md, the open box difference of 2026-09-05). Reading: the demand is there, and the block-8 head does not carry its 0.73 to 0.78 in-block continuation past position 7 on this box. Widening pays only with training (DBloom's continuation curriculum: Astra's survey reports committed length 6.54 to 7.02 on Qwen3-8B, the agent's reports plus 0.8 tokens median, both from the same paper). That folds into the fine-tune plan as a possible second phase and closes widening as an inference-time change.

**2. Zero prefix-cache hits on this cohort are structural, not vLLM #54360.** `prefix_cache_hits_total` 0 against 12,463 queried tokens, after each of the eight prompts had been sent at least 16 times (four seeds, four complete passes) with `enable_prefix_caching=True`. The engine log says why: "Setting attention block size to 448 tokens to ensure that attention page size is >= mamba page size" (the mamba page is padded 3.23 percent to match). The cohort's prompts are 180 to 1,211 characters (bench/prompts_real.jsonl, recorded in v0.28-validation.md), shorter than one 448-token block, so no block is ever cacheable. The check that would answer #54360 on the fork: a prompt over 448 tokens, better several thousand, sent twice, reading hits and time to first token. One boot, under 15 minutes, on a free card.

**3. Two statistics both called "tokens per step" differ by 9 percent on this cohort.** The record's figure is the mean over the 32 requests of (accepted / drafts + 1) (analyze_drafters.py line 3; soak_summary.py the same). The drafts-weighted figure, sum of accepted over sum of drafts plus 1, over the same rows is 3.476 on each of passes 1 to 4 (row mean 3.789 on each), and the engine's counters give 94,312 / 37,965 + 1 = 3.484. Long completions accept less per step than short ones here. The row mean is the fair statistic for an A/B between drafters on the same rows, which is how every comparison in the record uses it; the drafts-weighted figure is the one that predicts throughput over a run, and the one an outside aggregate is most likely to be. The agent's survey sets our 3.83 beside the official card's 4.80 without knowing which statistic the card reports; a note is added there. From here, both statistics, labelled.

## Where the two surveys converge

1. **The gain lives in training the head, not in a runtime trick.** The agent ranks the warm-start fine-tune first (Verification-Aware Training, arXiv 2608.30135, as an objective; SpecForge PR 831 `target_head_path`; the DFlash2 training metrics merged 2026-09-02 and 03). Astra puts the merged SpecForge diagnostics (PR 810, 2026-09-03; PR 793, 2026-08-31) above any new recipe and says to use them to locate loss before the continuation work. Tonight's round-trip pipeline on card 1 is the first step of exactly this.
2. **Measure the ceiling before widening the block.** Both asked; measured above.
3. **Tree verification and tree drafters are nowhere a single-user win.** Astra found concrete hybrid code (SGLang PR 36196, a DFlash2 tree prototype) but its numbers are at four concurrent requests (width 1: 132 to 142 tok/s; width 4: 172; recurrent workspace 2.81 to 10.20 GB; sampling rejected; decode graphs disabled). The agent found TreeWY (vLLM RFC #54080) with "throughput falls at every width". Bole, DARTree and JetSpec: both say defer.
4. **LiLiCorr: low value here.** No compatible weights; DFlash2's selector already scores adjacent candidates; the paper does not compare against it.
5. **ReplaySSM: nothing at batch 1** (agent: 0.98x to 0.99x; Astra: the SGLang PR has no filled benchmark).
6. **No Qwen3.8 EAGLE, Domino, DFly or JetSpec heads exist on the Hub**; the Apathy v3 card lacks a matched DFlash2 control; the FP8 head card compares FP8 against bf16, which is not our question.

## Where they diverge

1. **Vocabulary trimming for the draft (Astra's first pick; the agent did not find it).** The memra card (tiyuvta/Qwen3.8-27B-DFlash2-memra, modified 2026-08-30) reports that limiting draft scoring to 32,768 vocabulary rows raises chat throughput 136.3 to 142.9 tok/s (plus 4.8 percent) and agentic 154.5 to 157.2 (plus 1.7 percent), RTX PRO 6000, q4 drafter, greedy, another engine. The fork hook is real: patches/dflash2-backport.patch says of the selector's top-k that it "spans the vocabulary and is the selector's largest single cost" (line 401), and the draft scores the full shared Marlin lm_head (lines 702 to 709). Astra's estimate: 1 to 2 days (repack the kept head rows, keep token ids through the selector, check rejection semantics); risk, acceptance loss outside the kept vocabulary (multilingual, rare code tokens). Head seat's read: profile first. One hour with the torch profiler says how many of the 23.4 ms per step are the draft's lm_head projection and top-k; under 1 ms the ceiling is about 4 percent and the acceptance risk is not worth it, at 2 ms or more it is the best runtime candidate on either list.
2. **The head-precision test (agent's item 3).** Astra reads the FP8 card as inapplicable, which is right for the card. The agent's item is a local question the card does not answer: does the W4A16 head lose acceptance against the bf16 head at the fitting profile? The round-trip pipeline's bf16 control arm answers it tonight if the pipeline reaches its serve stage; no separate work.
3. **DaoCloud's seven-query checkpoint (Astra's item 2).** Needs vLLM PR 54154 (Model Runner V2 only); no matched timing win reported. Skip until that PR merges; watch it.
4. **vLLM PR 54485, the host metadata skip (Astra's item 3).** 0.125 to 0.007 ms per cycle on GB200 with Model Runner V2; on a 23.4 ms step that is half a percent at most. Only if we are profiling anyway for item 1.
5. **Tensor parallel across both 4090s (agent's item 4).** One flag, one boot when both cards are free; a community pair of 3090s reports 218 tok/s single request with DFlash2 at TP 2. The WSL2 no-P2P risk is real and Astra did not raise it. It takes card 1 from training while it runs.
6. **Block verification under sampling (agent's item 6).** Hours, zero code, moves only under sampling.

## Recommended order, for Michael's word

Tomorrow, hours each, no training, in this order:

- (a) Read the round-trip pipeline (card 1, `specforge-roundtrip/WATCH.md`): did the fine-tuned head export, load in the fork's vLLM, and what did acceptance do against the shipped head and the bf16 control on the same 32 requests. This decides everything under "days".
- (b) The long-prompt prefix-cache check (answers vLLM #54360 on the fork; 15 minutes).
- (c) Profile the draft's lm_head projection and top-k share of the step (decides vocabulary trimming; one hour).
- (d) Tensor parallel across both cards, one boot, when both are free.
- (e) Block verification under sampling, one boot.

Days, on his word:

- (f) If the round trip survives: the fine-tune proper (both surveys' first choice), with DBloom's widening as a second phase only if the fine-tuned head's ceiling bin stays above about 10 percent.
- (g) Vocabulary trimming, 1 to 2 days, only if (c) shows 2 ms or more.

Not now: trees (all of them), LiLiCorr, ReplaySSM, the DaoCloud checkpoint (watch PR 54154), an engine swap.

## What was not done tonight

No new boots (card 0 carries the soak, card 1 the round-trip pipeline), no posts, no upstream traffic. The counters were read with one `docker exec` curl inside the soak container, which does not enter the engine's request stream. Written 04:42:43Z by the shell clock.
