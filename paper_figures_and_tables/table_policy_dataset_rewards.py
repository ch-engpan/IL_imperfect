#!/usr/bin/env python3
"""Compute policy and mixed-dataset reward tables from a simple input CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from reporting_common import ResultsError, print_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewards-csv", type=Path, required=True, help="CSV columns: environment,quality,reward; quality is optimal or suboptimal")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def weighted_stats(optimal: np.ndarray, suboptimal: np.ndarray, optimal_ratio: float) -> tuple[float, float]:
    suboptimal_ratio = 1.0 - optimal_ratio
    optimal_mean, suboptimal_mean = optimal.mean(), suboptimal.mean()
    mean = optimal_ratio * optimal_mean + suboptimal_ratio * suboptimal_mean
    variance = (
        optimal_ratio * (optimal.var() + (optimal_mean - mean) ** 2)
        + suboptimal_ratio * (suboptimal.var() + (suboptimal_mean - mean) ** 2)
    )
    return float(mean), float(np.sqrt(variance))


def main() -> int:
    args = parse_args()
    grouped = defaultdict(lambda: defaultdict(list))
    with args.rewards_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"environment", "quality", "reward"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ResultsError(f"{args.rewards_csv} must contain columns: environment, quality, reward")
        for row in reader:
            quality = row["quality"].strip().lower()
            if quality not in ("optimal", "suboptimal"):
                raise ResultsError(f"Unknown quality {row['quality']!r}; expected optimal or suboptimal")
            grouped[row["environment"]][quality].append(float(row["reward"]))
    rows = []
    for environment, qualities in sorted(grouped.items()):
        if not qualities["optimal"] or not qualities["suboptimal"]:
            raise ResultsError(f"{environment} needs both optimal and suboptimal rewards")
        optimal = np.asarray(qualities["optimal"])
        suboptimal = np.asarray(qualities["suboptimal"])
        for quality, values in (("optimal", optimal), ("suboptimal", suboptimal)):
            rows.append(
                {
                    "environment": environment,
                    "dataset": quality,
                    "optimal_ratio": 1.0 if quality == "optimal" else 0.0,
                    "mean": values.mean(),
                    "std": values.std(),
                    "sample_count": len(values),
                }
            )
        for ratio, label in ((0.5, "mixed_50_50"), (0.1, "mixed_10_90")):
            mean, std = weighted_stats(optimal, suboptimal, ratio)
            rows.append(
                {
                    "environment": environment,
                    "dataset": label,
                    "optimal_ratio": ratio,
                    "mean": mean,
                    "std": std,
                    "sample_count": len(optimal) + len(suboptimal),
                }
            )
    if not rows:
        raise ResultsError(f"No rewards found in {args.rewards_csv}")
    print_rows(rows)
    write_csv(rows, args.output_dir / "table_policy_dataset_rewards.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ResultsError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
