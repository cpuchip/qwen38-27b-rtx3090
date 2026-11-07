"""Stands in for venv/bin/vllm inside the image: builds vLLM's engine config from the launcher's argv, in the
launcher's environment, and writes $MATRIX_OUT/$CASE_ID.json. No model load, no serve.

Verdicts: OK (create_engine_config returned), RAISE (vLLM refused the configuration), PARSE_ERROR (vLLM's
argument parser refused the argv; its message is in the case's launcher.err). vLLM logs to stdout, so its
warnings land in the case's launcher.out, which score.py reads.
"""
import json
import os
import sys
import traceback

case = os.environ.get("CASE_ID", "adhoc")
path = os.path.join(os.environ.get("MATRIX_OUT", "/m/out"), case + ".json")
out = {"case": case, "argv": sys.argv[1:]}


def write(**kw):
    out.update(kw)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


try:
    args = sys.argv[1:]
    if args and args[0] == "serve":
        args = args[1:]
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.engine.arg_utils import AsyncEngineArgs
    parser = make_arg_parser(FlexibleArgumentParser())
    try:
        ns = parser.parse_args(args)
    except SystemExit as e:
        write(verdict="PARSE_ERROR", error=f"argparse exit {e.code} (message in launcher.err)")
        sys.exit(3)
    if getattr(ns, "model_tag", None):
        ns.model = ns.model_tag
    cfg = AsyncEngineArgs.from_cli_args(ns).create_engine_config()
    sc = cfg.speculative_config
    write(verdict="OK", spec=None if sc is None else sc.method, max_model_len=cfg.model_config.max_model_len,
          kv_dtype=str(cfg.cache_config.cache_dtype), prefix=cfg.cache_config.enable_prefix_caching)
except Exception as e:
    write(verdict="RAISE", error_type=type(e).__name__, error=str(e)[:800],
          tb=traceback.format_exc(limit=4)[-1500:])
    sys.exit(4)
