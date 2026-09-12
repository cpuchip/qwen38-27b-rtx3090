# Acceptance harness

One script boots each serving profile of an image in a container, probes it, and writes the facts to an output
directory. It exists so that a port to a new vLLM pin, or a patch that touches the KV cache, can be judged the same
way on any box with a card and Docker.

```bash
IMAGE=qwen38-27b-rtx3090:port-0.29 TAG=029 DEPTH_CORPUS_GLOB='/data/prose/**/*.txt' bash bench/acceptance/acceptance.sh
```

Profiles (`PROFILES="A B C D H P"`, default `A B C D`), what each boots and what it records:

| profile | boot | records |
|---|---|---|
| A fast | dflash2 k=7, prefix caching | boot facts, quality battery (n=100), depth rows at 25k and 47k tokens, three prompts each |
| B long | mtp, align mode, prefix caching, 64k | prefix ladder (cached tokens vs prompt length), peak pool usage at 60k and 160k chars |
| C int4 | int4 per-token-head KV, 120k | boot facts (the drafter block promotion lines), the mq3d oracle, depth rows |
| D offload | mtp long, 12 GiB CPU tier | the serve oracle: is a stored request served back from the tier after eviction |
| H huge | dflash2 + KVarN, 262k | quality battery, needle at 32k/90k/200k, first stream deltas, depth rows |
| P probe | dflash2 fast | one fixed prompt five times with the spec-decode counters per request |

## Reading two runs against each other

Pair a port against a control image built from the same repo commit with the old pin (the merge base), so the two
images differ by the vLLM pin alone. Then:

- **Geometry compares outright.** Attention block size, KV pool tokens, backend and layout lines, the promotion
  lines: the same model on the same card must compute the same numbers on both pins.
- **Quality compares on en and da.** The code-perplexity lane reads the image's own vLLM source, so it never
  compares across pins; the GSM8K lane is n=100, so one question is inside its error.
- **Prefill compares by paired row.** Depth prompts are deterministic per rep (`depth-{tokens}-{rep}` in the
  salt, no timestamp), so rep i is the same prompt on every image, run and box.
- **Decode does not compare at three rows**, even on identical prompts: the two versions generate different
  continuations, so the accepted-length sequences diverge. Report it as a band. If a decode number is needed, use
  profile P (fixed prompt, counters) or a fixed-output-length probe.
- **Absolutes do not compare across boxes or corpora.** A 3090 with one corpus and a 4090 with another give
  different rates; each box's own deltas and the geometry are the comparable parts.

## Faults the harness has already paid for

- A timestamp in the prompt salt meant no two runs ever shared a prompt. Removed.
- The quality battery mounted outside `/app/bench` had an empty code corpus and reported a lower "all". Mounted
  at `/app/bench/q.py` now; and the `:ro` data mount makes the battery traceback after its last line when it writes
  its result file, which is harmless (the numbers are complete). The harness mounts it writable.
- The needle probe left thinking on, so with the qwen3 reasoning parser its 32 tokens went to reasoning and the
  answer came back empty. Thinking is off by default now (`NEEDLE_THINKING=1` restores it).
- Filtering the oracle's output through `tail` dropped rows. The harness keeps every `[oracle]` line.
- The acceptance probe picked its counters by substring, and `num_drafts` and `num_draft_tokens` both matched, in
  hash order. It selects by exact suffix and reports both now.

Card selection is `GPU` (index or UUID) for Docker and `ACC_GPU_SMI` for the clock lines; the harness does not
pause or restore anything else on the box.
