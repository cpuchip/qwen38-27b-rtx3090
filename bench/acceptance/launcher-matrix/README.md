# The launcher-matrix gate

Every launcher setting a user can pick, run through vLLM's own configuration checks in the new image, on CPU, before
a pin moves. Added 2026-11-07 after syv-ai #444: `REQ_METRICS=1` had passed `--per-request-spec-decode-metrics`
with speculation off since our 0.29.0 pin flip (#148), which vLLM refuses at startup, and three ports carried it
because the acceptance runs only booted the profiles we use.

    bash bench/acceptance/launcher-matrix/drive.sh IMAGE TAG MODELS_DIR [OVERLAY_DIR]

- **What runs:** each case in `cases.py` (312 at writing: single-user, alternative and batch, each crossed over the
  knobs that change its argv) runs the real launcher in the image. `venv/bin/vllm` is replaced by
  `build_config.py`, which parses the argv with vLLM's own serve parser and calls `create_engine_config()` in the
  environment the launcher built, so exported env knobs count as well as flags.
- **What it costs:** about 6 minutes for 312 cases at 16 jobs on threadchip. The card is visible only for platform
  detection (vLLM 0.33 cannot build a config with no device); card memory stayed at 1-5 MiB in every run so far,
  and a watchdog kills the container past 2000 MiB.
- **What it does NOT check:** anything after the config is built (weights, kernels, memory, the first request).
  Item 1 of the port runbook still owes the boots. A case that builds here can still fail at load.
- **Reading the score** (`out-TAG/score.txt`): RAISE and PARSE_ERROR are vLLM refusing what the launcher made, so
  a bug in the launcher or a change in the new pin. REFUSED is the launcher's own refusal, fine when its message
  is clear. NO_RESULT means the harness broke and is never a pass. The warning tally is where deprecations show up
  a release before they become PARSE_ERRORs.
- **Falsify it on each new pin:** run once on the image as built and once with a known-bad overlay (or a known
  fix over a known-bad image) and check the verdicts flip on exactly the cases the change touches. The first run
  did this with #444's launchers over the cut3 image.

Results go to `private-workspace/.spec/scratch/`, not here.
