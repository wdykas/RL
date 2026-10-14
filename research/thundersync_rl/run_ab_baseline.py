"""Stock NeMo-RL GRPO (sync ``grpo_train`` or ``async_grpo_train``) with a per-step history file.

Baseline arms of the ThunderSync A/B kit (configs/ab/, thundersync_work/sbatch/submit_ab.sh).
Runs ``examples/run_grpo.py`` unchanged; the only addition is ``--history-json PATH``: every
metric NeMo-RL logs through ``Logger.log_metrics`` is also recorded as ``<prefix>/<name>`` in the
entry of its step, and the list is rewritten after every log call so a run cut short keeps
the steps it finished. The A/B summarizer reads this file next to ThunderSync's history.json.
"""

import argparse
import json
import os
import runpy
import sys


def _install_history(path: str) -> None:
    # Import order as examples/run_grpo.py (algorithms first): importing the logger
    # module first loads the importlib_metadata backport before transformers, which
    # then trips over a dangling dist-info in the container venv.
    import nemo_rl.algorithms.grpo  # noqa: F401
    from nemo_rl.utils import logger as nrl_logger

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    steps: dict[int, dict] = {}
    inner = nrl_logger.Logger.log_metrics

    def log_metrics(self, metrics, step, prefix="", *args, **kwargs):
        out = inner(self, metrics, step, prefix, *args, **kwargs)
        entry = steps.setdefault(int(step), {"step": int(step)})
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                entry[f"{prefix}/{k}" if prefix else k] = float(v)
            elif hasattr(v, "item") and getattr(v, "numel", lambda: 0)() == 1:
                entry[f"{prefix}/{k}" if prefix else k] = float(v.item())
        with open(path + ".tmp", "w") as f:
            json.dump([steps[s] for s in sorted(steps)], f, indent=1)
        os.replace(path + ".tmp", path)
        return out

    nrl_logger.Logger.log_metrics = log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--history-json", type=str, default=None)
    args, rest = parser.parse_known_args()
    if args.history_json:
        _install_history(args.history_json)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(root, "examples", "run_grpo.py")
    sys.argv = [script, *rest]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
