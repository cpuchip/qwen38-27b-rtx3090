# Upstream tracker: syv-ai/qwen38-27b-rtx3090

Generated 2026-09-05 01:09Z by `scripts/track_upstream.py`. Regenerate rather than hand-edit;
the notes section below the table is the part that is written by hand.

## Ball in our court

Nothing open is waiting on a reply from us.

## Every thread we are on

| # | kind | state | ours | unanswered | updated | title |
|---|---|---|---:|---:|---|---|
| 25 | issue | open | 4 | 0 | 2026-09-05T01:08 | Known issues and progress: KVarN / DFlash2 / CTX=huge (tra |
| 80 | pr | open | 0 | 0 | 2026-09-05T01:08 | gotchas: five from the 0.28 acceptance bisect |
| 79 | pr | open | 0 | 0 | 2026-09-05T01:06 | dflash2-z-adaptive-emitted: keep the code, fix the comment |
| 75 | issue | open | 2 | 0 | 2026-09-05T00:19 | Autotune nondeterminism decides the trajectory: a bench ro |
| 73 | issue | closed | 4 | 1 | 2026-09-04T18:20 | DFlash2 acceptance regression on vLLM 0.28.0: 3.19 -> 2.66 |
| 74 | issue | closed | 0 | 1 | 2026-09-04T18:20 | Retraction: the tool-call cap in my 0.28 validation is my  |
| 43 | pr | merged | 9 | 0 | 2026-09-04T14:08 | Update vLLM stack to 0.28.0 |
| 69 | pr | closed | 1 | 1 | 2026-09-04T07:26 | kvarn: restore the sliding-window page padding hunk droppe |
| 67 | pr | merged | 0 | 1 | 2026-09-03T06:56 | int4 3D scratch: allocate inside the memory budget (#57 fo |
| 57 | pr | merged | 1 | 0 | 2026-09-03T04:24 | Fix the int4 KV speculative-attention scratch-buffer sizin |
| 48 | issue | closed | 5 | 1 | 2026-09-02T08:54 | Engine core stalls on the first large chunked-prefill afte |
| 56 | pr | closed | 1 | 1 | 2026-09-02T06:26 | fix(single-user): alternative.sh exits silently under set  |
| 46 | pr | merged | 2 | 1 | 2026-08-31T10:47 | Speculative decoding at depth: offload WSL2 fix, int4 MQ-3 |
<!-- HAND-WRITTEN BELOW -->

## Notes

### Where this landed (2026-09-04)

The maintainer moved his production box to vLLM 0.28 on the strength of two findings from
here. The dropped sliding-window padding hunk unblocked the merge; the unset
`draft_sample_method` closed the acceptance regression that was the last thing holding
production back. He applied the launcher fix himself in `0e95195`, using the same
`${DRAFT_SAMPLE:-probabilistic}` pattern we had prepared and extending it to the int4
launcher we had not checked, so **our prepared one-line PR is superseded and should not be
filed**. His measured result: 2.920 to 3.305 tokens per step with the field set, landing
slightly ahead of 0.27.1, prefill untouched throughout.

### Open work, in the order I would take it

1. **Answer the question on #75.** He asked for one number and we already have it: how many
   distinct autotune winner sets appear over N cold boots, because a coin flip and a long tail
   mean different things for whether a benchmark row can be called reproducible. Our eight-draw
   experiment gives eight distinct winner sets in eight boots, with the trajectory spanning 58
   to 189 tool calls and the block-inverse merge kernel identical in all eight. That is a long
   tail, not a coin flip, and it means a row is reproducible only with its cache pinned.
2. **The z-adaptive patch, PR invited.** He said `dflash2-z-adaptive-emitted.patch` still
   contradicts the comment above it, is not being reverted in this pass now that the real cause
   is fixed, and should be re-justified or dropped on its own merits. He would take a PR. Our
   band analysis is the argument either way: the patch can only act where the lookup qualifies
   twice running and acceptance is low enough to fall under the threshold.
3. **Three cache channels, his framing on #75.** He separated what we found into autotune draw
   affecting trajectory, compile-cache warmth affecting the profiled peak and therefore the KV
   pool by 0.92 GiB, and position in the run affecting decode and time to first token. Only the
   first is ours. The second and third are his and worth carrying into how we bench.

### Flightbench, our side

The mission bench had run greedy since it was written, which saturates it: eighteen of eighteen
three times identically, against 17, 16 and 10 at the model's own recommended sampling, with
failures including a burn started without clearance and the honesty check. Sampling is now
configurable and the default is unchanged so old rows stay comparable. Open question is whether
recommended becomes the default, which is a judgement about what the bench is for.

### Standing cautions earned here

- Compare built trees, not patch files: a hunk that moved its boundaries reads as new behaviour.
- Read the conditions before the diff. Five candidates died to the fact that the measured regime
  runs at temperature 1.0, where a division by temperature is the identity.
- Determinism is an imposter for reproducibility. Two passes on a fixed seed are two replays.
- Match the workload to the claim. A short single-prompt cell was underpowered for three
  different questions.
- Acceptance is ambiguous: degenerate text scores near-perfect acceptance.
- When a mechanism turns on a threshold, instrument the distribution and never the mean.
