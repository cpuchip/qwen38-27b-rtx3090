#!/usr/bin/env python3
"""Prepare the EAGLE-4 Qwen3.8-27B draft for one 24 GB card.

    python prepare_draft.py <draft dir> <output dir> [--bf16-head]

Reads the draft's model.safetensors and writes a copy to <output dir> with:

- embed_tokens.weight removed (2.37 GiB). The draft's embedding table is the target's: it is byte-identical to the
  bf16 embedding of Qwen/Qwen3.8-27B. Without it in the checkpoint, vLLM shares the target's embedding with the
  draft instead of loading a second copy.
- lm_head.weight quantized to int8, one fp32 scale per vocab row (scale = absmax / 127), stored as lm_head.weight
  (I8) and lm_head.weight_scale (F32): 2.37 GiB -> 1.19 GiB. The patched draft model reads it when
  EAGLE4_LM_HEAD_INT8=1 (start_eagle4.sh sets it). --bf16-head keeps the head as it is.

Every other tensor is copied byte for byte. Needs numpy only. Prints the int8 reconstruction error.
"""
import argparse
import json
import os
import shutil
import struct

import numpy as np

ROWS = 8192
CHUNK = 64 << 20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def copy_range(fi, fo, start, length):
    fi.seek(start)
    while length:
        buf = fi.read(min(CHUNK, length))
        fo.write(buf)
        length -= len(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bf16-head", action="store_true", help="keep lm_head in bf16 (no int8)")
    args = ap.parse_args()
    int8 = not args.bf16_head

    os.makedirs(args.dst, exist_ok=False)
    for name in os.listdir(args.src):
        if name != "model.safetensors" and os.path.isfile(os.path.join(args.src, name)):
            shutil.copy2(os.path.join(args.src, name), os.path.join(args.dst, name))

    src = os.path.join(args.src, "model.safetensors")
    n, h = header(src)
    meta = h.pop("__metadata__", None)
    if h.pop("embed_tokens.weight", None) is None:
        raise SystemExit("no embed_tokens.weight in the draft; nothing to strip")
    lm = h["lm_head.weight"]
    if int8 and lm["dtype"] != "BF16":
        raise SystemExit(f"lm_head.weight is {lm['dtype']}, expected BF16")
    V, H = lm["shape"]

    out, off = {}, 0
    names = sorted(h, key=lambda k: h[k]["data_offsets"][0])
    for k in names:
        if k == "lm_head.weight" and int8:
            out[k] = {"dtype": "I8", "shape": [V, H], "data_offsets": [off, off + V * H]}
            off += V * H
            out["lm_head.weight_scale"] = {"dtype": "F32", "shape": [V], "data_offsets": [off, off + 4 * V]}
            off += 4 * V
        else:
            a, b = h[k]["data_offsets"]
            out[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + b - a]}
            off += b - a
    if meta is not None:
        out["__metadata__"] = meta
    hb = json.dumps(out, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)

    err_sq = ref_sq = worst = 0.0
    scales = np.empty(V, dtype=np.float32)
    with open(src, "rb") as fi, open(os.path.join(args.dst, "model.safetensors"), "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for k in names:
            a, b = h[k]["data_offsets"]
            if k != "lm_head.weight" or not int8:
                copy_range(fi, fo, 8 + n + a, b - a)
                continue
            fi.seek(8 + n + a)
            for r0 in range(0, V, ROWS):
                r = min(ROWS, V - r0)
                bits = np.frombuffer(fi.read(r * H * 2), dtype=np.uint16).astype(np.uint32) << 16
                w = bits.view(np.float32).reshape(r, H)
                s = np.abs(w).max(axis=1) / 127.0
                s[s == 0] = 1.0
                q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
                scales[r0:r0 + r] = s
                d = q.astype(np.float32) * s[:, None] - w
                err_sq += float((d.astype(np.float64) ** 2).sum())
                ref_sq += float((w.astype(np.float64) ** 2).sum())
                rel = np.sqrt((d ** 2).sum(axis=1)) / np.maximum(np.sqrt((w ** 2).sum(axis=1)), 1e-30)
                worst = max(worst, float(rel.max()))
                fo.write(q.tobytes())
            fo.write(scales.tobytes())

    print(f"wrote {args.dst}/model.safetensors: {os.path.getsize(os.path.join(args.dst, 'model.safetensors'))} bytes "
          f"(source {os.path.getsize(src)})")
    if int8:
        print(f"lm_head int8 relative error: {np.sqrt(err_sq / ref_sq):.5f} overall, worst row {worst:.5f}")


if __name__ == "__main__":
    main()
