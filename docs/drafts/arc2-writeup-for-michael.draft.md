# Arc two, what four seats found (draft for Michael, 2026-09-05)

Everything below is in docs/v0.28-validation.md from the heading "Arc two: throughput" with the fresh-eyes review beside it and every correction struck in place. Numbers carry their conditions. Two items are still running and are marked.

## The question you asked

Is DFlash2's lead over MTP real, or a property of these cards, and is 3.3 of 7 leaving something on the table.

**Real, and it is the drafter.** On the native 3090, at the width users run, DFlash2 accepts 31.5 percent more per step than MTP's default and gives 19.6 percent more throughput. At matched width 7 (MTP can be run at 7; the launcher's own default for it is 4), DFlash2 accepts 15.9 percent more per step and is 8.3 percent cheaper per step. The two drafters are identical at position 0 and MTP has lost 40 percent of DFlash2's conditional rate by position 6. Widening MTP nets slightly worse throughput. A block-trained drafter stays useful at depth; chained distilled heads do not.

**3.3 of 7 is the drafter's ceiling, not the width's.** Conditional on reaching a position, the drafter accepts at a flat 70 to 80 percent at every one of its seven positions on this cohort. The steep unconditional curve is that constant hazard multiplied out. Narrowing the verify block only loses accepted tokens (monotone 3 to 7) and does not lower step cost. A wider-trained drafter is the lever, and one exists for a different target (a block-16 DFlash2 for Muse Glimmer), not for this one.

## What was measured and what it is worth

| item | result | conditions |
|---|---|---|
| Block verification (`rejection_sample_method: "block"`, present upstream, off) | live, worth nothing: +0.4 percent at 15, +0.6 percent at 7, the paper's 5 to 8 percent excluded at both | two boot pairs, 32 rows each, card 0 |
| Reproduction mode (DFLASH_TOKENS=15) | on this cohort, no gain in accepted tokens; positions 7 to 14 give 1.2 to 1.6 percent of accepted tokens; step cost 2.4x on this host, 0.3 percent on the native 3090 | cause on this host still running (see below) |
| The lookup's first position | a cliff: 5 to 6 percent accepted at position 7, 70 to 96 percent at 8 to 14 once 7 is accepted | forced block, both cards |
| Async scheduling at width 7 | costs about nothing here, 2.07 percent of throughput natively when lost | direct pairs, engine-confirmed |
| Async scheduling at width 15 | on: a 9.1 percent loss natively, because padding every step to 15 bypasses the adaptive length; the launcher's coupling is a saving, and its "under 1 percent" comment undersells the reason | KV pin raised 10 MiB to boot it |
| VLLM_TRITON_FORCE_FIRST_CONFIG | 2x slower at the default on the 4090, boots diverge more under it, and it flips which seeds terminate immediately (prompt 5, seeds 2 and 3, empty on both cards; clean at greedy on the 3090) | a caution, not a defect |
| Cross-box acceptance levels | inside one box's own sampling spread (row sd 0.98, cross-box gap 0.14); withdrawn | the greedy texts agree for the first 30 to 60 tokens on 7 of 8 prompts, then part at a near-tie |

## Determinism

The launcher default replays across boots on this host to the token (five boots, one trajectory; the native 3090 has not yet run a sampled pair at the default, one is running). Reproduction mode does not: five width-15 boots form exactly two trajectories on card 0, differing on the same ten rows; card 1 gives more than two groups; and the native 3090 at width 15 shows one trajectory across eight boots with prefix caching on and two groups across three with it off. The Triton autotune race is excluded as the source (skipping it makes boots diverge more) and so is prefix caching (turning it off does the same); whatever the binary boot state is, its footprint grows with how much a boot computes from scratch, and it never flips a greedy argmax. Issue 75 narrows to the width-15 path.

## Still running (card 0, card 1 idle)

- The width-15 step cost on this host: three cells (width 8; width 15 with the race skipped; width 15 with the fork's split-KV verify attention off) plus async-on at 15 with the KV pin raised. The lookup cannot be switched off at 15 (the engine crashes; your issue 73 incidental says so). The note to Mads gets the mechanism when these land.
- Greedy under the flag on this card (the inverse case only).
- Two more cache-off boots on card 1 (group count).

## What goes to Mads, with your pen

docs/drafts/note-mads-upstream-fixes.draft.md, items 1 to 9. Items 5 to 8 are final; item 9 waits on the cells above. Nothing has been posted.

## What the night cost and taught

Instruments were corrected three times: the launcher inside the image is the fork's (LOOKUP is the alias it honours), the engine's own non-default-args line is the resolved configuration, and two engines on this host slow each other through the WSL2 GPU path (a fast cell on card 1 doubled card 0's step cost with the counters unchanged), so wall clocks here are clean only one engine at a time. Journal lessons 49 to 68.
