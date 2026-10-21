# Modification log (Eagle 4 Source-Available License, section 4b)

MODIFICATION NOTICE. This file or component has been materially modified from Eagle 4. The accompanying change log
identifies the modifying person or entity, the date of each material modification, the affected files or components,
and a summary of the material changes. This modified version is not an official Eagle 4 release and is provided
subject to the Eagle 4 Source-Available License.

Base: eagle-harness 0.1.0 as installed in `ghcr.io/eagle-hq/eagle-4-vllm:0.1.0`
(`sha256:93cc820f7ace448804ae4732bbe630b8ced6eac6784286c7c190e01634dd78cc`); the three files are byte-identical to
eagle-hq/EAGLE-4 commit 7b7b5a52.

| date | modified by | file | change |
|---|---|---|---|
| 2026-10-21 | Michael Stufflebeam | `eagle_harness/kernels/gather_dot.py` | An int8 path for the draft's lm_head rows: the kernel takes an optional per-row fp32 scale (`weight.row_scale`), loads int8 rows, and multiplies each candidate's fp32 dot product by its row's scale. The torch reference applies the same scale. bf16 weights take the unchanged path. |
| 2026-10-21 | Michael Stufflebeam | `eagle_harness/vllm/models.py` | With `EAGLE4_LM_HEAD_INT8=1`, the draft replaces `lm_head.weight` with an int8 parameter of the same shape plus `lm_head.weight_scale` (fp32, one per vocab row), loaded from a checkpoint quantized offline (`../prepare_draft.py`). |
| 2026-10-21 | Michael Stufflebeam | `eagle_harness/vllm/model_rows.py` | The fused draft-input path reads `embed_tokens.weight`; when the shared target embedding is quantized (no plain weight tensor), the existing unfused path is taken instead. |
