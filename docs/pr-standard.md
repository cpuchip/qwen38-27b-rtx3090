# The PR standard, and the gate that checks the mechanical half

Written 2026-09-13 from what the maintainer of syv-ai/qwen38-27b-rtx3090 actually checks before he merges,
read off his review comments on #90, #91, #93, #99, #100, #101 and #54. He verifies on the box, and he reads the
wrapping (title, body, commit messages, comments, docs) as carefully as the diff. Every hold he has placed on one
of our PRs was on the wrapping, not the engineering. This file is the bar; `scripts/pr-gate/pr_gate.py` checks
the part a script can check, and prints the part a person has to.

## The bar

1. **Title, body, lead commit and diff say one thing.** Counts in the body ("three fixes, seven hunks") match
   the files. A body written before the last commit is refreshed after it.
2. **Verified on the box, and the body says how.** What was run, on which card, with which configuration, and
   the numbers with the conditions that produced them. A number without its configuration is not a result.
3. **What is still unproven, stated plainly.** Every PR body carries that section. It is where the reviewer's
   time goes first, and it is what stops a claim from being read wider than its measurement.
4. **Every promised document lands in the same PR.** A gotcha named in the review thread is in the diff, not in
   a follow-up nobody files. A retracted claim is grepped out of comments, docs and launcher headers, not only
   out of the PR body.
5. **Sibling surfaces are swept.** The three launchers (`single-user/start_qwen.sh`, `batch/start_qwen.sh`,
   `single-user/alternative.sh`) carry the same blocks; a fix to one is a fix to all three or the body says why
   not. Docs that describe the changed behaviour (gotchas, README tables, `docs/*.md`) move with it.
6. **A new patch file has a `patches/series` line**, in dependency position, and the series applies to the
   pristine pin with `--fuzz 0` (`patches/check_vllm_series.sh`).
7. **A new environment knob is registered in `envs.py`** (via `patches/speed-knobs-envs.patch` or its own
   patch), never read with `os.environ.get` inside a kernel: an unregistered knob is outside the
   torch.compile cache key, and a warm-cache A/B of it is invalid. And a knob that is registered must be read
   through `envs.<NAME>` everywhere: a raw `os.environ.get` returns the string "0", which is truthy, while the
   registered bool is False, so the same knob has two senses and its off switch is an on switch (found on #90).
8. **Benches guard the hardware they need.** A test that requires sm89 skips on sm86 with a line saying so; it
   does not die with a compiler error. A test half that cannot exercise anything on this repo's kernels is
   dropped, not shipped as coverage.
9. **No claim the measurement does not carry.** "Identical output" is never claimed for a different reduction
   order. A headline from one box names the box. "Rules out" needs a control that could have failed.
10. **Credit and asks are specific.** Who found what, and exactly which line, file or run would unblock the
    merge.

## The gate

```
python scripts/pr-gate/pr_gate.py --pr 100                  # a PR on syv-ai: body from GitHub, diff from the local branch
python scripts/pr-gate/pr_gate.py --branch offload-mtp-serve --body body.md   # before the PR exists
python scripts/pr-gate/pr_gate.py --pr 100 --pristine /path/to/vllm-v0.28.0   # also run the integrity script
```

With `--pr` it refuses to run when the local branch is not at the PR's head (a stale checkout once reported six
hunks for a seven-hunk head, green); `--local-head` is the pre-push case where the local head is meant to
replace the PR's. It reports FAIL, WARN and OK per check and exits non-zero on any FAIL. It checks bar items 1, 4, 5, 6, 7, 8 and
9 as far as text can (counts against hunks, series line, env registration, retracted phrases from
`scripts/pr-gate/retracted.txt`, capability guards, sibling launchers, promised docs, numbers without a
configuration, em-dashes and attribution lines), and prints the manual list for 2, 3 and 10. A green gate is
not a merged PR; it is the wrapping not lagging the code.

## Verification asks, and two rules the second seat paid for

An ask to the verifying seat names the branch head, the exact command, and the expected row. Two rules for
the row (threadchip, 2026-09-14, on #90 and #93):

- **Run the positive control before the negative.** A row that expects silence ("=0 prints nothing") is
  satisfied by a check that never fires; the expected string in the row was wrong once and the =1 run showed
  zero lines, which would have read as a pass on =0 in the other order. The positive run first proves the
  instrument sees the thing at all.
- **A sense test for a knob is valid only on a tree where the knob's new reader is installed.** Against a
  venv that still carries the old reader, "=0 turns it off" passes for the wrong reason. Build the image from
  the branch, or replace the installed file with the branch's, and say which in the row.
- **Prove which module the interpreter loaded before trusting a before-arm.** A stale `.pyc` can run the fixed
  code under an unfixed source file, making a bug look absent; a one-line guard on the loaded module (line
  count, or the presence of the fix's own text) turns "no failure" into a measurement (threadchip, #109).
- **Reset and clean the replay tree between arms**, every time: a tree carrying a partial apply or staged files
  reports the next arm against the wrong base and looks like a result.
- **Peak-memory rows compare only inside one session.** On both boxes a peak KV-usage row moved 59% between
  sessions with the images bit-identical to each other in the same session (pool, block, layout and compile
  cache excluded as causes). A peak claim stands on same-session arms; a cross-session peak is not a
  measurement of the change (threadchip, the 0.29 rebase vet).
- **A launch line in an ask names which values are the launcher's defaults and which are overrides.** A line
  copied with `MAX_LEN=65536` and `DFLASH_TOKENS=15` failed engine init on two boxes and two pins and read as
  a regression for an hour; the launcher's own maximum at k=15 is 57,344 and the override was ours. State the
  defaults, or pass none.
- **Settle between arms, and a dead second arm is suspect before it is a finding.** A boot started seconds
  after a 20+ GiB container exits can fail engine init on the same line that boots after 25 s; the harness
  gap is part of the row.
- **A registration sweep ships with a default-preservation table.** For every knob moved from a raw read to
  the registry, the raw expression's default from the pre-sweep tree beside what the built image reads with
  nothing set; a knob read two ways at two sites (an int default at one, presence at the other) gets both
  sites named. A registration that moves a default is the defect the sweep exists to prevent (threadchip, the
  KVarN sweep).
- **A knob census is a set difference, not a grep.** Three sweeps in one day each declared "no raw read left"
  and each was scoped by a spelling or a prefix nobody had questioned (`os.environ.get(` only; `VLLM_` names
  only). Count knobs by structural extraction over the tree (lodestar's `config_key` nodes) or by AST set
  difference against the stock registry, and remember that vLLM's unknown-variable warning fires only for
  `VLLM_`-prefixed names, so silence proves nothing for any other prefix.
- **Classify a stats-line silence by the line that ends it, and compare timestamps with their date.** The stats
  line before a gap is emitted as a prefill begins and reads zero prompt throughput; the prefill shows only on
  the closing line, so a prefill filter keyed on the pre-gap line passes the very fires it targets. And a
  tally that compares `HH:MM:SS` strings across midnight matches the wrong lines while reporting a verdict that
  can still be right, which is the worse failure.

## Public replies, same bar

A comment on someone's issue or PR is read by the same people with the same care. `scripts/pr-gate/reply_gate.py
<file>` runs on every comment before it is posted: em-dashes, attribution lines, overclaim phrases ("rules out",
"proves", "identical", "always", "never"), numbers without a configuration, missing caveat when numbers are
present, length, undefined shorthand we coined in private (the jargon rule: define it in one plain sentence or
do not use it). The pattern of a good reply here is the maintainer's own: decision first, what was checked and
how, what is still unproven, what would unblock it, credit.

## Why this exists

The week of 2026-09-07 to 09-13 shipped correct engineering with lagging wrapping: a PR body left at "two fixes,
four hunks" after the commit that made it three and seven; a gotcha promised in review that never landed; an
allocator default fixed in one launcher and not its two siblings; a retracted "decode unchanged" claim still
in a launcher header; a bench that dies on sm86 instead of skipping; a knob read from the environment inside a
kernel; a 3.6x headline from one box; and one sentence that said a log ruled out a crash it could not rule out.
Each was caught by the maintainer, each cost a review round, and none needed a second thought once named. The
gate exists so they are caught by us, before the push.
