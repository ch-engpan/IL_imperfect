#!/usr/bin/env python3
"""Generate the expert-query-count and target-Pi ablation PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from reporting_plotting import save_separate_reward_figures
from reporting_common import RANGES, ResultsError, expert_check_specs, load_reward_runs, pi_test_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", choices=("expert-check", "target-pi", "all"), default="all")
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    return parser.parse_args()


def load_methods(results_root: Path, specs):
    return [(label, color, load_reward_runs(results_root, spec)) for label, color, spec in specs]


def main() -> int:
    args = parse_args()
    range_keys = RANGES if args.range == "all" else (args.range,)
    for range_key in range_keys:
        config = RANGES[range_key]
        if args.group in ("expert-check", "all"):
            methods = load_methods(args.results_root, expert_check_specs(config))
            save_separate_reward_figures(methods, ("Ant-v4",), args.output_dir, f"{config.label}_ablations_expCheckNum")
        if args.group in ("target-pi", "all"):
            methods = load_methods(args.results_root, pi_test_specs(config))
            save_separate_reward_figures(methods, ("Ant-v4",), args.output_dir, f"{config.label}_ablations_piTest")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultsError as exc:
        raise SystemExit(f"error: {exc}") from exc
