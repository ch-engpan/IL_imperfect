#!/usr/bin/env python3
"""Generate the paper's main reward-curve PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reporting_plotting import add_offline_line, plot_reward_axis
from reporting_common import (
    ENVIRONMENTS,
    PAPER_ENVIRONMENTS,
    PUBLISHED_OFFLINE,
    RANGES,
    ResultsError,
    aggregate_reward_curve,
    load_reward_runs,
    main_reward_specs,
)


def read_offline_csv(path: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    import csv

    values = dict(PUBLISHED_OFFLINE)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") not in (None, "", "best_last5_mean_eval"):
                continue
            values[(row["range"], row["method"], row["environment"])] = (
                float(row["mean"]),
                float(row["std"]),
            )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="Root containing alphaEst_randomPi_* result directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which PDFs are written")
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    parser.add_argument("--include-hopper", action="store_true", help="Also emit Hopper PDFs (generated originally but not included in the paper)")
    parser.add_argument("--offline-summary", type=Path, help="CSV from table_offline_bc_ileed.py; defaults to the notebook's published constants")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environments = ENVIRONMENTS if args.include_hopper else PAPER_ENVIRONMENTS
    offline = read_offline_csv(args.offline_summary) if args.offline_summary else PUBLISHED_OFFLINE
    args.output_dir.mkdir(parents=True, exist_ok=True)
    range_keys = RANGES if args.range == "all" else (args.range,)
    for range_key in range_keys:
        config = RANGES[range_key]
        methods = []
        for label, color, spec in main_reward_specs(config):
            methods.append((label, color, load_reward_runs(args.results_root, spec)))

        for environment in environments:
            fig, axis = plt.subplots(figsize=(8, 6))
            plot_reward_axis(axis, methods, environment, show_legend=False, fontsize=26)
            online = next(data for _, _, data in methods if environment in data)
            steps, _, _ = aggregate_reward_curve(online[environment])
            for method, color, zorder in (("ILEED", "black", 1), ("BC", "C5", 0)):
                key = (range_key, method, environment)
                if key in offline:
                    mean, std = offline[key]
                    add_offline_line(axis, steps, label=method, color=color, mean=mean, std=std, zorder=zorder)
            if environment == "Ant-v4":
                axis.legend(fontsize=20, loc="upper left")
            fig.tight_layout()
            output = args.output_dir / f"reward_{environment}__{config.label}.pdf"
            fig.savefig(output, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultsError as exc:
        raise SystemExit(f"error: {exc}") from exc
