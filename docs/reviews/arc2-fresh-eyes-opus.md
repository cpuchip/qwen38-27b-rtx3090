# Arc two, fresh-eyes review (opus seat, 2026-09-05)

No prior context on this project. Read `docs/v0.28-validation.md` from the "Arc two: throughput.
Registrations before numbers" heading to the end of file (961 lines at time of reading; the file grew
from 945 to 961 while I worked, and I re-read the tail). Ran `analyze_arc2.py` and `analyze_block.py`
and recomputed every number in the arc-two sections from the raw ROW lines. Read the fork's launcher
and patch files for every engine claim. I did not modify anything except this file.

Engine source under `/app` is not in this checkout. Where the doc cites an installed-tree line, I say so
rather than pretending to have read it.

---

## 1. Arm 1, draft width 3 to 7

**Recomputed against the doc, all five widths, 31 paired cells, guard removed 0:**

| width | doc tok/step | mine | doc tok/s | mine | doc ms/step | mine |
|---|---|---|---|---|---|---|
| 7 | 3.757 | 3.757 | 159.4 | 159.4 | 23.6 | 23.62 |
| 6 | 3.556 | 3.556 | 148.0 | 148.0 | 24.0 | 24.03 |
| 5 | 3.375 | 3.375 | 141.8 | 141.8 | 23.8 | 23.87 |
| 4 | 3.140 | 3.140 | 133.6 | 133.6 | 23.5 | 23.51 |
| 3 | 2.814 | 2.814 | 115.4 | 115.4 | 24.4 | 24.37 |

Every published figure reproduces exactly. The table is honest arithmetic.

**Is ms/step from wall/drafts a valid step-cost measure?** No, it is a step cost plus a per-row prefill
term, and the term is not constant across arms. `perpos_client.py` starts its timer before the HTTP POST
and stops after the response, so `wall` carries prompt tokenisation, prefill of the 1024-token prompt,
queueing and client overhead. Measured `ms/step` is therefore `s + P/drafts`, and `drafts` is *larger* at
narrow widths (287 at w7, 371 at w3), so the prefill inflation is *smaller* at w3. The bias runs against
the registered hypothesis, not for it. Subtracting a constant prefill widens the spread instead of
closing it:

| width | raw ms/step | minus P=0.3s | minus P=0.5s | minus P=1.0s |
|---|---|---|---|---|
| 7 | 23.62 | 22.45 | 21.66 | 19.70 |
| 6 | 24.03 | 22.86 | 22.08 | 20.12 |
| 5 | 23.87 | 22.46 | 21.51 | 19.15 |
| 4 | 23.51 | 22.49 | 21.82 | 20.12 |
| 3 | 24.37 | 23.49 | 22.91 | 21.45 |

So "flat within 4 percent" is true of the raw statistic (24.37 / 23.51 = 3.7 percent) and false of the
step cost it is standing in for (6.5 percent at P = 0.5s, with w3 the most expensive). The conclusion the
number is used for survives and in fact strengthens, because the correction makes the narrow arms *more*
expensive per step, never less. **Fix the sentence, keep the finding.** The honest form is: step cost does
not fall as the block narrows, and the residual ordering after removing prefill runs the wrong way for the
registered idea.

Note also the raw column is not monotone. w4 (23.51) is below w7 (23.62). The step cost is flat and noisy,
not ordered, and the doc should not let "flat within 4 percent" imply a measured trend.

**A confound other than acceptance that would make tok/s monotone in width.**

Three, in order of how much they matter.

(a) **The three metrics are two.** `out` is the token count and a spec-decode round emits `1 + accepted`
tokens, so `out` is `drafts * tok_per_step` up to the max_tokens truncation, and therefore
`tok_s = 1000 * tok_per_step / ms_per_step` identically. I checked it numerically: the counter-derived rate
matches the reported `tok_s` to 0.2 across all five arms (159.4 / 148.0 / 141.7 / 133.7 / 115.5 against
159.4 / 148.0 / 141.8 / 133.6 / 115.4). So "monotone on tok/step **and** tok/s" is one measurement and an
identity, not two independent confirmations. The arm has exactly two facts: tok/step is monotone, and
ms/step is flat. Say that.

(b) **The drafter may not be narrowing at all.** `single-user/start_qwen.sh:220-222` says in the fork's own
words: "DFLASH_TOKENS is the *verify* block, which no longer has to equal the drafter's: the DFlash2
checkpoint always proposes the 7 tokens it was trained for". If the drafter forward pass is fixed at its
trained block regardless of the setting, then widths 3 to 6 remove verify positions only, and flat ms/step
is expected by construction rather than discovered. The rows cannot exclude this: `round` is `dtok/drafts`,
which is what was put up for verification, not what the drafter computed. The claim "every drafted position
pays for itself" then needs restating as "every *verified* position pays for itself, and the drafter's cost
is paid whether or not you verify them" which is a materially different recommendation. **This is the one
confound I could not close from the artefacts and it is the one that most changes the reading.**

(c) **Early termination changing prefill amortisation.** Not all rows reach 1024 output tokens: 2 short rows
in w7, 4 in w6, 5 in w5 (one at `out=93`), 3 in w4, 2 in w3. Short rows have prefill as a larger share of
wall and depress `tok_s`. I tested it by restricting to the 26 cells that hit 1024 in *every* arm. The
result is unchanged: 3.746 / 3.556 / 3.354 / 3.122 / 2.762 on tok/step and 158.7 / 147.9 / 141.3 / 132.8 /
113.2 on tok/s, with the same signs and t values of -2.94, -5.23, -6.15, -8.19. **Excluded, and I looked.**

Time-ordered drift is also largely excluded by design, though the doc does not claim the credit: the run
order in `.width_arms.sh` is `for W in 7 5 4 3 6`, so w6 ran last and still lands second best. Card 1 was
running the cheap-context cells throughout, so contention was roughly constant across the arm.

**A number that is wrong.** `docs/v0.28-validation.md` line 880: "the late positions cost nothing to verify
and still accept 6 to 9 percent of the time." That 6 to 9 percent is the *share of all accepted tokens*
printed in the analyzer's brackets (w7 p5 = 7.5 percent, p6 = 6.0 percent), not an acceptance rate. The
marginal acceptance rates are 18.4 and 14.6 percent of rounds, and the rate conditional on the round
reaching that position is 80.7 and 79.6 percent. The error is conservative, so the conclusion is safe, but
the sentence states a wrong quantity.

**Verdict on 1: stands, with two corrections.** Restate ms/step as prefill-inflated and drop the
"within 4 percent" framing; restate the 6 to 9 percent as a share. Flag (b) as unresolved.

---

## 2. Arm 2, the cheap-context threshold

**Recomputed, 32 paired cells, card 1:**

| cell | doc tok/step | mine | doc d | mine d | doc t | mine t | doc p7-14 share | mine |
|---|---|---|---|---|---|---|---|---|
| cc0 | 3.739 | 3.739 | | | | | 0.8% | 0.70% |
| cc4096 | 3.769 | 3.769 | +0.029 | +0.029 | 0.58 | +0.58 | 1.7% | 1.62% |
| cc8192 | 3.766 | 3.766 | +0.026 | +0.026 | 0.51 | +0.51 | 1.6% | 1.59% |
| cc16384 | 3.758 | 3.758 | +0.019 | +0.019 | 0.36 | +0.36 | 1.6% | 1.61% |

The tok/step column reproduces exactly. The share column is a shade high in the doc (0.8 against 0.70, 1.7
against 1.62), consistent with summing the analyzer's rounded per-mille brackets rather than the raw
counters. Cosmetic.

The source claim is verified in the tree, not recalled: `patches/dflash2-lookup-drafting.patch:406` reads
"cheap enough that taking the long block unconditionally wins (+8% at C1)", with the cost model at line 396,
"+6% per step at 1.5k against +27% at 25k". The env read is at line 317 of the same patch. The forcing is
real: `round` is exactly 15.000 on all 32 rows of every non-zero cell against 7.172 at cc0.

**Is a wall-clock null on card 1 informative at all?** I tried to break the doc's caution here and ended up
agreeing with it, then found a stronger argument the doc does not make.

First the attempt. Paired ms/step, cc4096 minus cc0, is +1.60 ms with sd 2.17 and t = +4.19, which looks
like a solid measured cost. It is not trustworthy. The four cc cells are four separate boots run at four
different times against four different card-0 neighbours, so pairing by (seed, prompt) removes prompt
variance and leaves boot variance entirely inside the point estimate. The three same-behaviour forced boots
give 94.0, 92.7 and 93.9 ms/step, a 1.3 ms spread, and cc0 at 92.4 sits essentially on cc8192. So the
+1.6 ms is inside the boot noise. **The doc's refusal to read card-1 wall clock is correct and I could not
overturn it.** For scale, the same card at DFLASH_TOKENS=15 ran at 97.7 ms/step during the cc window and
77.4 ms/step in the newer `ffc-ff1` boot on the same eight rows, an 18 percent swing between boots on one
card.

Now the stronger argument. Because `tok_s = 1000 * tok_per_step / ms_per_step` identically (see 1a), an
8 percent throughput gain with tok/step flat at +0.8 percent requires ms/step to *fall* by about 7 percent.
Adding eight verify positions to every step cannot make the step 7 percent cheaper. The only channel by
which forcing could reduce step cost is removing the 8-to-16 block alternation, and the measured change on
card 1 is between -0.3 and +1.7 percent, nowhere near -7. **So H1's +8 percent is falsified on this cohort
by counters plus monotonicity, not only on tok/step, and not contingent on the card.** The doc under-claims
here. The card-0 rerun is a robustness check, not the missing half of the answer.

**Is the per-position discriminator enough to close the cliff question?** It closes the question it was
built for and it is read against the wrong statistic.

The question it closes: the earlier cell could not separate "the long block is rarely entered" from "the
positions exist and accept near zero". Under forcing, `round` is 15.000 on every row, so the positions exist
unconditionally, and they still return 1.6 percent of accepted tokens. That separation is clean, is
counter-based, and is card-independent. Good instrument, right answer to that question.

The statistic problem. The doc's own registration at line 790 names an **accept rate**: "The discriminating
number is the per-position accept rate at seven to fourteen under the highest threshold... Under two percent
is H2... tens of percent is H1 back in play." The reported number is a **share of accepted tokens**. Those
are different quantities and they disagree. Computing the rate conditional on the round having reached each
position (a position can only be accepted when every earlier one was):

| cell | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cc0 | 70.2 | 71.9 | 74.9 | 76.2 | 78.2 | 78.7 | 76.7 | **1.8** | 95.5 | 90.5 | 94.7 |
| cc4096 | 70.0 | 72.0 | 75.1 | 76.4 | 77.6 | 79.0 | 78.3 | **6.4** | 74.4 | 82.0 | 88.0 |
| cc16384 | 69.4 | 71.8 | 74.9 | 76.3 | 77.5 | 78.6 | 78.0 | **6.4** | 74.1 | 83.3 | 88.0 |
| w7 | 70.7 | 72.6 | 75.4 | 77.4 | 78.2 | 80.7 | 79.6 | n/a | | | |

Read against the registration as written, 6.4 percent at position 7 is above the 2 percent line, and
positions 8 to 14 accept at 74 to 93 percent, which is "tens of percent" by any reading. **The registered
discriminator is not met; the goalpost moved from a rate to a share between registration and result.**

This is not a quibble, because the conditional table says something the share does not. The drafter's own
curve is flat at 69 to 81 percent and *rises* with depth, which is the opposite of the "steeply decreasing"
premise the arm was opened on (line 786). There is exactly one cliff, at position 7, the first lookup
position, and the lookup **recovers to near the drafter's own rate at position 8 and stays there**. That is
the fingerprint of a predictor that cannot *start* correctly but is nearly perfect once started, which is
what a verbatim-copy suffix match should look like. Forcing raises position 7 from 1.8 to 6.4 percent, a
3.5x improvement that the share statistic hides entirely. "The lookup is not starved, it is wrong" is too
strong; the supported statement is "the lookup almost never gets its first position right, and the tail is
worthless because it is gated behind that." A boundary or off-by-one at the drafter-to-lookup seam is a live
candidate the arm did not consider, and it is cheap to probe.

The aggregate conclusion (forcing buys nothing worth having: +0.029 tok/step, 0.8 percent) is unaffected and
stands.

**What the card-0 cells must show.** c0cc0 and c0cc16384 are queued and `cheapctx-c0cc0.txt` is 0 bytes, so
nothing has run. They must show: (i) `round` 15.000 on every c0cc16384 row and near 7.17 on c0cc0, or the
cell is void; (ii) tok/step within about 0.03 of the card-1 values, or the cohort is not the same cohort;
(iii) a **positive** ms/step delta on card 0 larger than card 1's, because the four-lane card hides GPU-side
attention cost, so the extra positions should look *more* expensive where the card is compute-bound. A
negative ms/step delta on card 0 would be the surprise and would put H1's wall-clock half back in play. Run
at least two boots per cell; one boot per cell cannot separate the treatment from the 1.3 ms boot spread
already measured.

**One more thing the doc should say.** cc4096 and cc16384 are **31 of 32 rows identical**, cc4096 and cc8192
are 24 of 32, cc8192 and cc16384 are 23 of 32. On a cohort whose context never exceeds about 2k, all three
thresholds force unconditionally and are the same configuration. So arm 2 is one control boot against one
forced condition sampled three times, not four cells at four thresholds. The three t values are one test
reported three times, and "falsified at every threshold" (line 917) reads as three confirmations when it is
one. The upside: those three boots are the best same-behaviour replay controls on the box and belong in the
boot-replay table in section 5.

**Verdict on 2: conclusion stands, discriminator falls.** The +8 percent is falsified, more strongly than
the doc claims. The registered per-position accept rate was not met and was answered with a different
statistic. The cliff is real but is a cliff at one position followed by a full recovery, not a dead tail.

---

## 3. Block verification and the row-identity decision rule

**Recomputed, std1 against blk1, 32 rows each:**

| quantity | doc | mine |
|---|---|---|
| tok/step std1 | 3.739 | 3.739 |
| tok/step blk1 | 3.745 | 3.745 |
| difference | +0.006 | +0.0056 |
| rows positive | 16 of 32 | 16 of 32 |
| row sd | 0.287 | 0.2868 |
| row SE | about 0.05 | 0.0507 |
| tok/s | 64.9 vs 65.0 | 64.9 vs 65.0 |
| round | 7.093 vs 7.114 | 7.093 vs 7.114 |
| rows identical | 0 of 32 | 0 of 32 |

Every figure reproduces. Per-position rates are within a point at every position, confirmed.

**Is the row-identity rule sound?** It is sound in one direction and close to vacuous in the other.

Sound direction: identity *at* the control rate means the flag did nothing. That inference is good, and the
doc is right that a no-op would itself have been a real finding.

Vacuous direction: 0 of 32 is read as "block verification changes acceptance decisions", and therefore as a
licence for the aggregates. But block verification is claimed **lossless**, and any lossless resampling
scheme that consumes the random stream in a different order decorrelates trajectories completely while
changing the output distribution not at all. The comparison key is `(drafts, acc, dtok, out)` aggregated over
roughly 300 rounds per row, so a single flipped decision anywhere in a row destroys identity for that row.
0 of 32 is therefore the *expected* outcome of any implementation that touches the sampler, including one
that is exactly distribution-preserving and exactly worthless. The gate passes on evidence that cannot fail.

**What would fool it.** (i) Exactly the above: a lossless method scores 0 of 32 and buys nothing, which is in
fact what the aggregates then say. (ii) An unrelated per-boot perturbation. The blk boots differ from the std
boots by a `sed` on the launcher (`.block_arms.sh`, the `BLOCK` variable) that appends
`"rejection_sample_method":"block"` to `SPEC_CFG`. If the engine parsed that key and changed *any* code path,
even kernel selection, you get 0 of 32 with no change to verification semantics. Nothing in this checkout
confirms the engine honoured it. (iii) The control rate is not a constant: same-behaviour boot pairs on this
box give 31 of 32, 28 of 32, 24 of 32, 23 of 32 and 20 of 32, so "the control replay rate" spans 63 to 97
percent. 0 of 32 is far below all of them, so the verdict survives, but the rule needs a distribution, not
the single number 28 of 32.

**The instrument gap the doc's own rule forbids.** Line 782 of the doc sets the standing rule: "print the
engine's resolved configuration, then the number." The block cells do not. `.block_arms.sh` greps the server
log for `rejection_sample_method` or `use_block_verification` and both come back empty; the RESOLVED lines in
`block-std1.txt` and `block-blk1.txt` read `RESOLVED draft_logits=True  drafting 7 tokens per step`, where
"7" is the drafter's block, not `num_speculative_tokens` (which is 15 in both, per `-e DFLASH_TOKENS=15` in
`.block_arms.sh` and confirmed by the p7 to p14 counters existing at all). The only evidence the treatment
took is `SPEC_CFG_LINE`, which proves the `sed` edited the launcher, not that the engine accepted the key.
The arm violates the document's own rule and should say so.

The same gap exists in arm 1: `.width_arms.sh`'s RESOLVED grep returns empty for the width, so
`width_arms.out` records `RESOLVED draft_logits=True  enable_prefix_caching=True` with nothing in between.
Arm 1 is rescued by geometry (`round` is exactly 3.000, 4.000, 5.000, 6.000, 7.000) which is a genuine proof
of the setting; the block flag has no such geometric proof.

**The right way to state the comparison.** One boot pair, 32 paired prompts, difference +0.006 tokens per
step, row SE 0.051, 95 percent within-pair interval [-0.094, +0.105], 16 of 32 rows positive, step cost
identical (paired ms/step delta +0.03 ms). Boot-level variance is not in that interval, but it is small and I
can bound it: the three same-behaviour forced cells in arm 2 have boot-mean tok/step of 3.7686, 3.7656 and
3.7580, an sd of 0.0055, an order of magnitude below the row term. So adding boot variance does not change
the interval materially.

**Is "5 to 8 percent excluded" justified? Yes.** At a base of 3.739, 5 percent is 0.187 and 8 percent is
0.299 tokens per step, which sit 3.58 and 5.79 standard errors above the estimate. Power arithmetic agrees:
resolving 0.19 tokens per step at 80 percent needs 18 rows and there are 32. **The claim is justified as
written, and it is the one strong claim in this section.**

**What n does a real answer need.** It depends on the question, and the doc should say which it is asking.

| effect to resolve | tokens per step | paired rows at 80 percent power |
|---|---|---|
| 5 percent | 0.19 | 18 |
| 2.7 percent | 0.10 | 64 |
| 1.3 percent | 0.05 | 258 |
| 0.8 percent | 0.03 | 717 |

The current pair already answers "is it 5 to 8 percent" with no. It cannot answer "is it 1 to 2 percent",
which needs 8 to 20 times the cohort. Since the step cost is method-independent, a 1 to 2 percent tok/step
gain would still be free throughput and worth having, so that is probably the question that matters. Say
which one you are answering.

Finally: `block-blk2.txt` holds 2 rows and `block-std2.txt` does not exist, so the second pair and the
control pair are not in yet. Every statement about block verification is currently one pair, in reproduction
mode, on one card.

**Verdict on 3: rule is half-sound, exclusion stands, framing needs replacing.** The rule's "live" branch
proves nothing that matters for a lossless method. The +0.006 with its interval, and the exclusion of 5 to 8
percent, are correct and well-founded. The engine never confirmed the flag.

---

## 4. The 15-token step cost

**Recomputed, card 0, per prompt, seed 1, `wall/drafts`:**

| cell | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|---|
| w7 | 29.7 | 25.5 | 23.1 | 23.3 | 23.3 | 23.4 | 23.3 | 23.6 |
| std1 | 86.8 | 77.9 | 56.2 | 56.5 | 56.8 | 56.3 | 56.3 | 56.6 |
| blk1 | 86.9 | 78.8 | 56.3 | 56.8 | 56.4 | 56.5 | 56.5 | 57.5 |

The doc's three ranges (23.1 to 23.6, 56.2 to 56.8, 56.3 to 57.5) reproduce exactly. Pooling all four seeds
over prompts 2 to 7 gives 23.4 (range 23.1 to 23.9), 56.5 (56.1 to 57.4) and 56.5 (56.2 to 57.5). The ratio
is 2.41. **The measurement is solid and reproducible, and the two boots agree to well under 1 percent.**

**What the rows and logs can already discriminate.**

They can rule out prefill. At steady state std1 spends 30.8 s on prompt 2 against w7's 13.0 s for a nearly
identical round count (548 against 562). No prefill term of any plausible size accounts for 17.8 s.

They can rule out the round count. `round` is 7.093 in std1 against 7.000 in w7, so the two configurations
verify essentially the same number of positions per step. The cost is not more verify work per step; it is
the same work costing 2.4 times as much.

They can locate the warm-up, and the shape is informative. Pooled over seeds, std1's p0 is 64.0 and p1 is
61.7 against a steady 56.4, and seed 1 alone shows 86.8 and 77.9. Working it through, seed 1's first request
carries about 7.8 s of extra time and its second request carries about 7.8 s again, two warm-ups of the same
size, where w7 has one of about 1.8 s. Two equal warm-ups is the fingerprint of two graph shapes being
captured lazily, which is consistent with the launcher's own note at `start_qwen.sh:405-407` that above 7
drafts "decode graphs are captured for both block lengths". Seeds 2 to 4 show no elevation, so the doc's
"the first two prompts of each boot carry warm-up" is loosely stated: it is the first two *requests of the
boot*, not prompts 0 and 1 of every seed. Harmless to the result, wrong as a description.

They cannot discriminate the cause. Nothing else in the rows separates the candidates, and the server log
is not saved: `.block_arms.sh` writes it to `/tmp/server.log` inside a `--rm` container and only greps two
lines out. **Save the server log for the attribution cells.** The graph-capture question is answerable
directly from the "Capturing CUDA graphs" lines and currently is not.

**Is the registration sharp enough to be falsified? No, and the reason is a confound in the launcher.**

`single-user/start_qwen.sh:291-296`:

```
if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
  ASYNC_SCHED=${ASYNC_SCHED:-0}
fi
```

and `single-user/start_qwen.sh:658`:

```
ASYNC_ARGS=$([ "${ASYNC_SCHED:-1}" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)
```

So **w7 runs with async scheduling on and every 15-token cell runs with it off.** That is a second variable
changing between the 23 ms and 56 ms cells, and it is exactly the shape of the observed cost: without async
scheduling every decode step pays a full worker-to-scheduler host round trip, and this box is WSL2, where a
host round trip is milliseconds rather than microseconds. The launcher's own comment at line 295 claims
"Measured cost of losing async scheduling at batch 1: under 1%", but that measurement is from elsewhere and
the doc itself observes that the native Linux box pays 2.5 percent for reproduction mode while this box pays
140 percent. A per-step host synchronisation that is cheap on Linux and expensive under WSL2 fits both
numbers with one mechanism.

It is not the only thing that moves at the boundary. At `CTX=fast` with `DRAFT_TOKENS` above 7, the launcher
also changes `MAX_SEQS` from 8 to 4 (line 402), `MAX_LEN` from 65536 to 57344 (line 403),
`VLLM_V2_CUDAGRAPH_MEM_MIB` from 1400 to 1900 (lines 406 to 410) and `VLLM_SPEC_DECODE_ATTN_QMAX` from 8 to
16 (line 236). `CG` happens to be 64 in both cases (line 429, capped). The doc's three registered candidates
(lookup host work, the wide verify path, a step function at the trained-block boundary) do not name any of
these.

**The registration is worse than incomplete, it is confounded.** The `lk15off` cell sets
`VLLM_DFLASH2_LOOKUP=0`, which makes the line-291 condition false and **restores async scheduling at the
same time as it removes the lookup**. So if `lk15off` returns about 23 ms, the registered reading "the cost
is the lookup's host work" is not supported; async scheduling is an equally good explanation and the cell
cannot tell them apart. `t8` and `t11` are both above 7 with the lookup on, so both also lose async
scheduling, and "t8 pays most of it" would be read as a boundary step function when the boundary is exactly
where async scheduling turns off. As written, all three cells fail to separate the doc's own candidates from
one the doc has not named.

**The missing cell, and it is one environment variable.** `ASYNC_SCHED` is honoured by the launcher, so
`DFLASH_TOKENS=15 ASYNC_SCHED=1` with the lookup left on isolates scheduling from everything else. Pair it
with `DFLASH_TOKENS=7 ASYNC_SCHED=0` and the two cells bracket the effect completely. Run these before
`lk15off`, or `lk15off` will be misread.

**Other things that could produce the number.** Client overhead is excluded, since the client is identical
across cells and w7 shows the same client at 23 ms. Prefill is excluded above. First-request warm-up is
excluded by using prompts 2 to 7. Host contention from card 1 is not fully excluded (card 1 was running the
cheap-context cells during w7 and during std1) but is unlikely to produce a 2.4x that is stable to 1 percent
across two boots hours apart.

**Operational.** `lookup_cost.out` reads `bash: results/raw/v028/.lookup_cost.sh: No such file or directory`,
timestamped 00:45, one minute before `.lookup_cost.sh` was written. **The attribution cells did not launch.**
No `lkcost-*.txt` exists. `.block7.sh` waits on pid 731782, which is the launch that failed, so the default-
width block pair may have started early or not at all; no `block-std7.txt` or `block-blk7.txt` exists either.
The doc says these are "queued"; they are not running.

**Verdict on 4: the measurement stands, the attribution plan falls.** 2.4x is real and reproducible. The
registered discriminators are confounded by the launcher's own async-scheduling branch and cannot be read as
registered. Add the two `ASYNC_SCHED` cells and save the server log.

---

## 5. Boot replay

**Recomputed, on `(drafts, acc, dtok, out)`:**

| pair | doc | mine |
|---|---|---|
| cc0 vs coh-flag | 28 of 32 | 28 of 32 |
| cc4096 vs cc8192 | 19 of 25 | **24 of 32** |
| pol-sticky0 vs pol-zrev | 20 of 32 | 20 of 32 |
| cc4096 vs cc16384 | not listed | **31 of 32** |
| cc8192 vs cc16384 | not listed | **23 of 32** |
| std1 vs blk1 | 0 of 32 | 0 of 32 |

Two corrections. The cc4096 vs cc8192 row is stale, presumably counted while cc8192 was still being written;
it is 24 of 32 now. And its description, "a threshold that rarely fires", is wrong: on a cohort whose context
never exceeds about 2k, thresholds of 4096, 8192 and 16384 all force unconditionally, and all three cells
show `round` exactly 15.000. Those are same-behaviour boot pairs, which makes them better controls than the
table claims, and two of them are missing. Adding them, the same-behaviour replay rate on this box ranges
from 23 of 32 to 31 of 32.

**Attacking the autotune hypothesis.** The hypothesis is internally coherent and the discriminating test
(`VLLM_TRITON_FORCE_FIRST_CONFIG`, cells ff1 and ff2) is exactly right, is already running, and was
registered with a numeric prediction before the run. I have nothing better to offer as a *test*. Three
things weaken the hypothesis as stated.

First, it is a hypothesis about kernel numerics but the divergence rate is not stable: 1 in 32 for one pair
and 9 in 32 for another, both same-behaviour. A single rounding-tie mechanism with a fixed flip probability
does not obviously produce that spread over the same eight prompts.

Second, there is a candidate the doc has not named that is visible in the tree: **prefix caching is on in
every arm-two cell** (`-e PREFIX_CACHE=1` in `.width_arms.sh`, `.cheapctx_arms.sh`, `.block_arms.sh`), and
the launcher's own note at `start_qwen.sh:495-496` says prefix caching does not merely reuse KV, it
"resume[s] the recurrent (GDN) state from the last cached block boundary instead of re-running the prompt".
Resuming a recurrent state from a boundary is a numerically different path from recomputing it, so which
rows resume and from where determines the arithmetic, and that depends on pool occupancy and eviction, which
depends on the whole boot's history. This box's own arc-one work already found the block-length policy
coasting between requests (doc line 780). A cheap discriminating cell exists: two boots at `PREFIX_CACHE=0`
in an otherwise identical configuration. If they replay 32 of 32 while the cache-on pair replays 28, the
cache is the source and Triton is not. That cell costs the same as ff1 and ff2 and should run beside them.

Third, `patches/marlin-tune-table.patch:60` shows `use_atomic_add=False` in the tuned Marlin path's fake
registration, and the only `tl.atomic_add` I found in the fork
(`patches/dflash2-lookup-drafting.patch:781`) writes a hit counter, not numerics. So atomic-order
nondeterminism in the quantised GEMM is **not** supported by the patches, which is worth recording as a
candidate actively excluded rather than never considered.

**Is "the replay unit is the card" the right statement? No, on the evidence shown.** The doc gives "same
config across the two cards replays 0 of 32", but the two cards never ran the same configuration in this
arc: card 0's `w7` is DFLASH_TOKENS=7 with async scheduling on, and card 1's `cc0` is DFLASH_TOKENS=15 with
async scheduling off and a different cache volume. `cheapctx-c0cc0.txt` is 0 bytes, so the one cell that
would have made a genuine cross-card same-config pair has not run. Until it does, the cross-card 0 of 32 has
no evidential weight and the correct statement is "the replay unit is the boot, and the flip rate varies by
configuration". I would also drop "the other box replays 32 of 32" as a contrast: the doc's own addendum
already computes that two 8-row boots match everywhere with probability about 0.34 at this box's flip rate,
so it is not evidence of a different regime.

One live inconsistency worth resolving. The doc discounts every card-1 clock as a four-lane artefact, but
`ffc-ff1.txt`'s own probe line reads `card1 pcie_x8`, and on the eight matched rows ff1 runs at 77.4 ms/step
against cc0's 97.7. Either the link renegotiated, or the earlier idle sample was not the loaded state, or
card-0 contention during the cc window is doing work the four-lane story is being credited for. The
four-lane sentence at doc line 796 is load-bearing for a lot of downstream caveats and should be re-checked
under load.

**Verdict on 5: hypothesis survives as a hypothesis, "the replay unit is the card" falls.** The cross-card
claim has no artefact behind it. Prefix caching is an untested and tree-visible alternative.

---

## 6. MTP at 7 is void

**This one falls, and I think the deciding cell is recoverable for the price of one environment variable.**

The doc says: "The launcher passes the knob at line 432 and the engine overrides it." In this checkout, line
432 is a comment inside the resident-request note about state pages. It is not the MTP knob. The MTP
speculative config is built at `single-user/start_qwen.sh:486`:

```
SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
```

`DRAFT_TOKENS` there is whatever the context profile set. At `CTX=fast` that is
`single-user/start_qwen.sh:155`, `DRAFT_TOKENS=${DRAFT_TOKENS:-4}`. The line that maps `DFLASH_TOKENS` onto
`DRAFT_TOKENS` is `single-user/start_qwen.sh:231`, and it lives **inside the `SPEC=dflash2` branch only**. I
grepped every use of `DFLASH_TOKENS` in the launcher (lines 220, 226, 231, 251, 278, 308, 324, 326, 349, 364,
373, 374, 401, 424, 432, 442, 538) and none of them is on the MTP path.

**So `DFLASH_TOKENS` is inert under `SPEC=mtp`.** A cell that sets `DFLASH_TOKENS=7` or `DFLASH_TOKENS=15`
and observes the engine resolve `num_speculative_tokens` to 4 has observed the launcher's own `CTX=fast`
default at line 155, not an engine clamp and not the checkpoint. The void condition fired, but for a reason
that has nothing to do with `mtp_num_hidden_layers`.

**Is there a configuration path that lifts MTP above 4?** Yes, in this tree: set `DRAFT_TOKENS` directly.
`DRAFT_TOKENS=7 SPEC=mtp CTX=fast` puts 7 into `SPEC_CFG` at line 486 with nothing in the launcher to stop
it. Whether the *engine* then clamps to the checkpoint's `mtp_num_hidden_layers` is a real and interesting
question, and it is completely untested, because the experiment never asked the engine for 7.

I cannot see the engine code that would clamp it, and I cannot see the checkpoint config either
(`text_config.mtp_num_hidden_layers = 1` is not in this checkout; the models directory is outside it). I also
cannot see the other box's launcher, which may differ from this one. Both are marked unverified below. But
the fork file the doc cites does not say what the doc says it says.

**Consequence.** "MTP cannot be run wider on this checkpoint" is unsupported. "H-width versus H-drafter stays
unresolved" is correct but for a different reason: not because the cell was void, but because the cell asked
the wrong question. The practical finding the doc says a reader needs, "the DFlash2 lead over MTP is not
closable by tuning MTP", is exactly the claim that depends on this, and it is the one that should not go to
the maintainer yet.

The rest of the decomposition is unaffected. The default-width table (MTP 4 wide at 2.218 accepted per step
and 24.6 ms; DFlash2 at 7 at 2.917 and 25.1) stands on its own numbers, and the doc's own prefill caveat on
those ms figures is correctly stated.

**Verdict on 6: falls.** Re-run the deciding cell with `DRAFT_TOKENS=7 SPEC=mtp` and print the resolved
`num_speculative_tokens` before the rows. If the engine clamps to 4 there, the doc's conclusion is restored
with a correct mechanism and a correct line citation.

---

## 7. The stranger read

Claims in the arc-two sections that a reader cannot trace to a file in this tree or to a named artefact on
the other box. Quoted, with what is missing.

1. "rejection_sampler.py:96-97 wires the method into the V2 sampler" (line 900; also `.block_arms.sh` and
   `.block7.sh` headers). `rejection_sample_method` appears **nowhere** in `patches/`, `single-user/` or
   `README.md`. This is an installed-tree line number in `/app`. Unverifiable here, and unconfirmed by the
   cells themselves, whose RESOLVED grep for the key came back empty.

2. "The rejection-sample method accepts the value block at 0.28.0 ... non-greedy only, lossless, never worse
   than token-by-token, five to eight percent reported" (line 826). The 5 to 8 percent is attributed to
   Sun et al. in the script headers but no citation reaches a file. The "lossless" and "never worse"
   properties are the load-bearing part of the arm's premise and are asserted, not shown.

3. "The launcher passes the knob at line 432 and the engine overrides it" (line 886). Line 432 of
   `single-user/start_qwen.sh` in this checkout is a comment about resident state pages. See section 6.

4. "the checkpoint config carries text_config.mtp_num_hidden_layers = 1" (line 886). Not in this checkout.
   The models directory is outside the repo. No artefact named on the other box either.

5. "Both methods replay bit-identically across boots on that box (m-mtp1b equals m-mtp2; m-dfl1 equals
   m-dfl2, every counter, only the clock moving)" (line 886). Named cells, but on the other box, with no
   path given and nothing in `flightbench results/raw/v028/`.

6. "the other box's bit-identity turned out to come from a warm persisted AOT compile cache (both its boots
   log 'Directly load AOT compilation' for the same two artefacts)" (line 902). Log lines on the other box,
   no path.

7. "Checked here: the card-0 volume holds 21 AOT entries and the card-1 volume 27" (line 902). Docker volume
   contents, no artefact written. A reader cannot re-derive this.

8. "nvidia-smi reports this box's card 1 negotiating PCIe Gen4 at 4 lanes, with 16 available, while card 0
   negotiates 8 at idle" (line 796). No capture file. And `ffc-ff1.txt` records `card1 pcie_x8`, which
   contradicts it. Every card-1 caveat in the document rests on this sentence.

9. "Upstream 54282, merged three days after the tag ... The salt that fixes it is absent from the built
   tree" (line 822) and "Upstream 54374 shows FlashAttention's ahead-of-time schedule stays on ..." (line
   824). Upstream PR numbers with no local diff or grep result shown.

10. "the other box's arm-3 cell supplied the curve on the same cohort at production sampling: 80.1 percent at
    position zero falling to 19.9 at position six" (line 786). No file. This is also the premise arm 1 was
    opened on, and my conditional table on this box's own rows shows a *flat* curve at 69 to 81 percent, so
    the two numbers are almost certainly different statistics. Worth reconciling explicitly before either
    reaches a reader.

11. "MTP, 4 wide, the checkpoint's fixed width" (line 949 table). The width 4 is the launcher's `CTX=fast`
    default at line 155. Calling it the checkpoint's fixed width assumes the conclusion of section 6.

12. "on 0.28.0 as shipped MTP is lossless and probabilistic DFlash2 is not until the upstream noise-stream
    salt is applied" (line 859). Follows from item 9 and inherits its unverified status.

13. "the README calls 15 'reproduction mode' and measures it at +1 percent on chat on the maintainer's
    hardware" (line 921). **This one traces.** `README.md:137` reads "On chat it is worth 1%". Verified, and
    listed here only to note that the launcher gives a different figure for a different cohort at
    `start_qwen.sh:227`, "+9% on the short-prompt C1 set", so the reader should be told which cohort.

---

## Recomputed-number index

Everything in the arc-two result tables reproduces from the raw rows. The exceptions are listed here.

| doc | mine | where |
|---|---|---|
| p7-14 share, cc0 0.8% | 0.70% | rounding, cosmetic |
| p7-14 share, cc4096 1.7% | 1.62% | rounding, cosmetic |
| "late positions accept 6 to 9 percent of the time" | share, not rate; marginal rate 14.6 to 18.4%, conditional 79.6 to 80.7% | doc line 880, wrong quantity |
| cc4096 vs cc8192 replay 19 of 25 | 24 of 32 | doc line 895, stale |
| ms/step flat within 4 percent | 3.7% raw, 6.5% after removing 0.5 s prefill | doc line 880 |
| registered p7-14 accept rate under 2% | 6.4% conditional at cc16384 | doc line 790 against line 917 |

---

VERDICT: Conclusion 1 stands with two corrections (ms/step is prefill-inflated so drop "flat within 4 percent"; "6 to 9 percent" is a share not a rate) and one open confound (the drafter may propose its trained 7 at every width, per start_qwen.sh:220-222, so the arm may be measuring verify cost only). Conclusion 2's finding stands and is stronger than claimed (the +8 percent is falsified by counters plus the tok_s identity, not only on card 1) but its registered discriminator FALLS: the doc registered a per-position accept rate under 2 percent and reported a share of accepted tokens instead, and the conditional rate at position 7 is 6.4 percent with positions 8 to 14 at 74 to 93 percent, so the finding is a one-position cliff with full recovery, not a dead tail; arm 2 is also one forced condition sampled three times, not four thresholds. Conclusion 3's exclusion of 5 to 8 percent STANDS on correct arithmetic (3.6 to 5.8 SE, and boot variance is 0.0055, an order below the row term), but the row-identity rule's "live" branch is near-vacuous for a method claimed lossless, the control rate is a distribution from 20 of 32 to 31 of 32 rather than the single 28 of 32, and the engine never confirmed the flag, in violation of the document's own resolved-configuration rule. Conclusion 4's measurement STANDS (2.4x, reproducible to 1 percent, prefill and round-count excluded) but its attribution plan FALLS: start_qwen.sh:291-296 turns async scheduling off for every cell above 7 drafts and lk15off silently turns it back on, so all three registered cells are confounded, and the fix is two cells setting ASYNC_SCHED explicitly; separately, the attribution script never launched (lookup_cost.out records a missing-file error). Conclusion 5's autotune hypothesis survives as a hypothesis with a good registered test already running, but "the replay unit is the card" FALLS for want of any same-configuration cross-card pair (cheapctx-c0cc0.txt is empty), and prefix caching with GDN state resume is a tree-visible untested alternative. Conclusion 6 FALLS: DFLASH_TOKENS is inert under SPEC=mtp, the width 4 comes from start_qwen.sh:155 and not from the checkpoint, DRAFT_TOKENS=7 SPEC=mtp is an untried path that the launcher does not block, and the doc's line-432 citation does not exist in this tree, so "the DFlash2 lead over MTP is not closable by tuning MTP" must not go to the maintainer yet. Conclusion 7 lists thirteen untraceable claims, of which the load-bearing ones are the block-verification wiring, the MTP checkpoint clamp, and the card-1 four-lane sentence that ffc-ff1.txt now contradicts.
