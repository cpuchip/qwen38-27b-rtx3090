# The int4 multi-query 3D verify path collapses in long tool-calling conversations (2026-09-02)

**Status: reproduced, not root-caused.** A defect in the opt-in 3D verify path of the int4
per-token-head KV cache (`VLLM_INT4_MQ_3D=1`, issue #46's extension; the scratch buffers PR #57
sized). The base int4 kernel with the 3D path off does not show it. Nothing is filed upstream yet.

## What happens

On a long tool-calling conversation (about forty tool calls, twelve thousand tokens of context), the
model's output degenerates and never recovers:

- thinking off: after a run of near-identical tool calls (a pending-evaluation tool polled sixteen
  times a turn for three turns), the reply becomes one repeated word ("duct Register Register…")
  for the rest of the run, every turn, `finish_reason=length`;
- thinking on, with a 6,000-token reply cap: the reasoning runs away instead (the same 53,995
  characters every turn, no tool call).

Two runs per cell, identical call for call. The conversation is the flightbench mission
(`benches/mission.py`, v6.1, controller seat), and the trigger every time has been a long run of
near-identical multi-query verify batches.

## What isolates it

Same weights (Qwen3.8-27B-W4A16-AutoRound), same DFlash2 draft (7 drafts), same image, same
launcher and flags, one variable per row, mission score out of 18, two runs per cell:

| KV cache | 3D path | thinking off | thinking on |
|---|---|---|---|
| bf16 (`CTX=fast`) | n/a | 16, 16 | 15, 15 |
| int8 per-token-head (`CTX=long`) | n/a | 15, 15 | 17, 17 |
| int4 per-token-head (`alternative.sh`), `VLLM_INT4_MQ_3D=1` | on | 10, 10 (collapse) | 6, 6 (runaway) |
| int4 per-token-head (`alternative.sh`), `VLLM_INT4_MQ_3D=0` | off | 14, 14 | 18, 18 |
| KVarN k4v2 (`CTX=huge`) | n/a | 16, 18 | 16, 14 |

The same model as a Q4_K_M GGUF on llama.cpp: 15, 15 and 14, 14, no collapse. So: not the
weights, not the draft or speculation (on in every vLLM row), not the int4 cache itself; the 3D
verify path under int4.

Caveat that orders the next step: every int4 row above ran on the image built from ff41191, which
predates the #57 merge. Whether the collapse survives on the current image is the first question.

## Reproduction

```
# the production tier, 3D path on (the container env sets it; the launcher's default is off)
docker run -d --name int4-3d --gpus '"device=<uuid>"' --ipc host -p 127.0.0.1:18025:18025 \
  -v <cache-volume>:/cache -v <models>:/app/models -e HOME=/cache -e PORT=18025 -e VLLM_API_KEY=<key> \
  -e CTX=long -e SPEC=dflash2 -e DFLASH_TOKENS=7 -e PREFIX_CACHE=1 -e VLLM_INT4_MQ_3D=1 -e MAX_SEQS=4 \
  -e GPU_UTIL=0.93 -e MAX_LEN=244224 --entrypoint bash <image> \
  -c 'cd /app && echo "$VLLM_API_KEY" > api_key.txt && exec bash single-user/alternative.sh'
# the conversation (flightbench, https://github.com/cpuchip/flightbench)
KEY=<key> BASE=http://127.0.0.1:18025/v1 MODEL=qwen3.8-27b THINK=off python benches/mission.py
```

Expected on ff41191: the replies for transmissions 6 through 11 are the repeated word; the trace
(`v6-*.trace.jsonl` beside the results file) shows the polling loop in the LOI station first.
With `VLLM_INT4_MQ_3D=0`: 14/18, coherent to the end.

## Ruled out, and not yet

Ruled out: the weights, the draft, speculation as such, the int4 cache as such, the runtime as a
whole (bf16, int8, KVarN fly it), context length alone (the 2D arm is clean at the same length).
Not yet: whether the current image (post-#57) still shows it; prefix caching (on in every row);
the draft width (7 in every row; the 3D path exists for the multi-query verify batch, so the
batch width is the regime); whether onset tracks the repetition or the token count (a padded
prefix moves the count without the repetition).

## Where to look

The trigger is a long sequence of near-identical multi-query verify batches over near-identical
cached content. The first places: the 3D kernel's segment reduce (`softmax_segm_*`) on repeated
content, the per-token-head scales on identical rows, and the dispatch reason bits under
`VLLM_INT4_MQ_3D_DEBUG=1` in the turns before the collapse (a capacity fallback, a buffer
disagreement, or a `DEGRADED` line would each say something different).
