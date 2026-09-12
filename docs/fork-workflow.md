# The fork workflow: three repositories, one source of truth

Ground rules for carrying this stack on top of vLLM without drift between the patch files, the vLLM fork branch,
Mads's repository and ours. Written 2026-09-12 after the 0.29 port; every rule here has a check, and the checks
run before an image is built or a pin moves.

[← back to the main README](../README.md)

## Where things live

| repository | holds | never holds |
|---|---|---|
| **cpuchip/vllm** (fork of vllm-project/vllm) | vLLM source only. `qwen38/<pin>` = the release tag plus this repo's series as one commit per topic. `up/<topic>` = a topic cherry-picked onto upstream main for an upstream PR. | launchers, docs of ours, bench, notes, anything that is not vLLM |
| **this repository** (syv-ai/qwen38-27b-rtx3090 and our fork of it) | launchers, Dockerfiles, docs, bench and the acceptance harness, PATCHES.md, the model store notes. The Dockerfile pins a commit of the vLLM fork. | vLLM source changes as their primary form (they are commits on the fork; `patches/` is an exported view) |
| the private workspace | findings, run records, journals, the series builder, proposals | anything meant for either public repo |

## Branches

In cpuchip/vllm:

- `upstream/main` and the release tags: read-only, fetched, never committed to.
- `qwen38/<pin>` (for example `qwen38/0.29`): the integration branch the image is built from. Base = the release
  tag. One commit per topic, authored by whoever wrote it, signed off by the committer, Co-authored-by only where
  an AI produced the content (vLLM's own contributing rule, applied here as well so a commit can go upstream
  unchanged). A new pin is a new branch: `git rebase --onto v<new> v<old>` of the old one, then the acceptance
  harness on two boxes against a control image built from the same repo commit with the old pin.
- `up/<topic>`: cut from `upstream/main`, one topic cherry-picked from the integration branch, conflicts resolved
  against main, PR opened from it with the disclosure vLLM asks for. Deleted when merged; the integration branch
  keeps its copy until the pin that ships the merge, when the commit retires.

In this repository (our fork of syv-ai):

- `syv-main`: a mirror of syv-ai/main. Fast-forward only, never committed to. Refreshed whenever upstream moves.
- `main`: our production line. `syv-main` merged in (merge, not rebase; it is published), plus what we run
  ahead of Mads: the Dockerfile pin on our fork commit, fixes that are still open as PRs to syv-ai, the harness.
  This is what our boxes build from, and it is how we adopt a fix or a pin before it lands in syv-ai.
- `pr/<topic>`: cut from `syv-main`, never from `main`, so a PR to Mads carries only its topic. The vLLM change
  in such a PR is the patch file exported from the fork commit (`scripts/export-patch.sh`), so the patch he merges
  and the commit we run are the same bytes by construction.

## The invariants, and the check that guards each

1. **The image is the branch.** The vLLM tree inside an image built from `qwen38/<pin>` is byte-identical to the
   branch's `vllm/` tree (the version stamp excepted). Check: `scripts/series-check.sh --image <tag>`.
2. **Every patch file is a commit, and the same bytes.** While `patches/` exists, applying the files to the
   release tag yields exactly the integration branch's tree. Check: `scripts/series-check.sh` (applies the files
   into a scratch worktree at the tag and diffs against the branch).
3. **One pin, stated once.** `docker/requirements.txt`'s `vllm==`, `verify.sh`'s pin, and the base tag of the
   fork commit the Dockerfile pins all agree, and that commit exists on origin. Check: `scripts/series-check.sh`.
4. **The mirror is a mirror.** `syv-main` equals syv-ai/main; `main` contains `syv-main`. Check:
   `scripts/series-check.sh --mirror`.
5. **No mixed commits on the integration branch.** One topic per commit; a follow-up is a `git commit --fixup`
   folded into its topic at the next rebase, so every topic stays one cherry-pickable unit. Check: subjects on
   `qwen38/<pin>` are `[qwen38] <topic>` with unique topics, or `fixup!` lines awaiting the next rebase.
6. **PATCHES.md is the ledger.** One row per topic: kind, what, upstream reference and status, cut against, what
   retires it, and the fork commit. A topic without a row, or a row without a commit, is drift. Check:
   `scripts/series-check.sh` cross-checks rows against branch subjects.
7. **A pin moves only after the two-box bar.** Quality (en/da), geometry and the paired prefill rows from
   `bench/acceptance/acceptance.sh` on two boxes, against the merge-base control, recorded in the PR that moves
   the pin. No check can enforce this; the PR template asks for it.

## The three cadences

**When syv-ai/main moves.** Fast-forward `syv-main`, merge it into `main`. If the merge adds or changes a patch
file, apply that change as a commit on `qwen38/<pin>` (or drop the file if the fork already carries it), rerun
`series-check.sh`, and only then rebuild the image. A patch file must never exist without its commit.

**When we fix something.** The commit goes on `qwen38/<pin>` first, with the topic's name, the author who wrote
it, and the reason in the body. If it is also for Mads, export the patch file into a `pr/<topic>` branch off
`syv-main` and open the PR. If it is also for vLLM, cherry-pick onto an `up/<topic>` branch off `upstream/main`
and open that PR with the disclosure and trailer. The row in PATCHES.md is updated in the same change.

**When vLLM releases.** New integration branch by rebase; the conflicts are the port and each gets its cause from
`git log v<old>..v<new> -- <file>`; the two-box acceptance against the control; then the pin moves in `main` and
the export refreshes `patches/`. The procedure in detail: `docs/vllm-0.29.md`, "Porting the next pin".

## What this buys, stated plainly

We can run a fix or a new pin the day we have proven it on two boxes, without waiting for it to be merged
upstream or by Mads; our PRs to both carry exactly the bytes we run; a port is a rebase whose conflicts are the
work, not a regeneration whose failures are silent; and drift between the three places is a script's exit code
rather than a feeling.
