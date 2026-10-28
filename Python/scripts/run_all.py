"""Run the analysis pipeline: CSV tables in outputs/results, LaTeX snippets in outputs/tex, figures in outputs/figures.

    python scripts/run_all.py                      # all steps
    python scripts/run_all.py regression size_effect
    python scripts/run_all.py --list
"""

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glulam.analyses import STEPS  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="steps to run (default: all)")
    ap.add_argument("--list", action="store_true", help="list the available steps")
    args = ap.parse_args()
    if args.list:
        print("\n".join(STEPS))
        return
    for name in args.steps or STEPS:
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        importlib.import_module(f"glulam.analyses.{name}").main()
        print(f"----- {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
