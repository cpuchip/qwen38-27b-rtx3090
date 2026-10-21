# eagle4/eagle_harness: Eagle-licensed files

**This directory is NOT covered by the repository's Apache-2.0 license.** Its diffs modify files of eagle-harness
(Eagle 4) and are governed by the Eagle 4 Source-Available License in `LICENSE` here, which is not an OSI-approved
open-source license and incorporates the Eagle Trademark Policy. The policy is `TRADEMARKS.md` in the Eagle 4
repository: https://github.com/eagle-hq/EAGLE-4/blob/7b7b5a522d7a7284b9eeb9889d03c8c2038c112f/TRADEMARKS.md

Downloading, cloning, forking, copying, accessing, modifying, using, hosting, distributing, or otherwise exercising
any licensed right in the files of this directory constitutes acceptance of the Eagle 4 Source-Available License and
the incorporated Eagle Trademark Policy.

DIRECTORY LICENSE NOTICE. Files in this directory that contain, reproduce, modify, implement, port, or derive from
Eagle 4, including EAGLE-Harness, EAGLE Efficient LM Head, model materials, and other core Eagle 4 features, are
Eagle Files governed by the Eagle 4 Source-Available License and the incorporated Eagle Trademark Policy. A file is
governed by Apache-2.0 only if its copyright owner expressly designates that file as Apache-2.0. See the Eagle
LICENSE and NOTICE files for file-level identification. Preserve all applicable copyright, attribution, source-file,
trademark, and modification notices. Commercial Use may require a separate written commercial license from Eagle AI
Corp.

This project includes or is derived from Eagle 4 technology developed by Eagle AI Corp. It is not an official Eagle
AI Corp project unless expressly stated otherwise. Eagle 4 is provided under the Eagle 4 Source-Available License,
which is not an OSI-approved open-source license and incorporates the Eagle Trademark Policy. The applicable LICENSE
file, NOTICE file, source-file notices, and modification notices accompany this distribution. Downloading, cloning,
forking, copying, accessing, modifying, using, hosting, distributing, or otherwise exercising any licensed right
constitutes acceptance of that license and policy.

EAGLE, Eagle 4, EAGLE-Harness, EAGLE Efficient LM Head, and associated logos are trademarks or other protected
designations of Eagle AI Corp. This product is an independent modification or offering. It is not affiliated with,
certified by, sponsored by, or endorsed by Eagle AI Corp. References to Eagle 4 are solely truthful nominative
references identifying origin, compatibility, or technology used.

## Contents

| file | what |
|---|---|
| `LICENSE` | the Eagle 4 Source-Available License, verbatim from eagle-hq/EAGLE-4 at 7b7b5a52 |
| `NOTICE` | the Eagle 4 NOTICE file, verbatim from the same commit |
| `CHANGELOG.md` | the modification log: who, when, which files, what changed |
| `models.diff`, `gather_dot.diff`, `model_rows.diff` | the modifications, as unified diffs against eagle-harness 0.1.0 as installed in `ghcr.io/eagle-hq/eagle-4-vllm:0.1.0` |

The diffs carry only the changed lines and their context. `../start_eagle4.sh` takes the original files from that
image when it runs and applies the diffs to copies; it never ships the whole files.
