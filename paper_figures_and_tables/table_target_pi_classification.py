#!/usr/bin/env python3
"""Write target-Pi classification accuracy and precision tables."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from reporting_common import (
    RANGES,
    ResultsError,
    load_classification_records,
    print_rows,
    scalar_count,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    return parser.parse_args()


def metrics(values):
    tp = scalar_count(values["label_opt_num"], "label_opt_num")
    fn = scalar_count(values["label_nonopt_num"], "label_nonopt_num")
    fp = scalar_count(values["label_opt_num_"], "label_opt_num_")
    tn = scalar_count(values["label_nonopt_num_"], "label_nonopt_num_")
    total = scalar_count(values["true_opt_num"], "true_opt_num") + scalar_count(values["true_nonopt_num"], "true_nonopt_num")
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / total
    flipped = precision < 0.1 and accuracy < 0.1
    if flipped:
        precision, accuracy = 1.0 - precision, 1.0 - accuracy
    return precision, accuracy, flipped


def main() -> int:
    args = parse_args()
    range_keys = RANGES if args.range == "all" else (args.range,)
    rows = []
    ablation_root = args.results_root / "rebt_ablation"
    for range_key in range_keys:
        prefix = f"{RANGES[range_key].directory}_expCheckNum3_piTest"
        for experiment in sorted(ablation_root.glob(f"{prefix}*")):
            if not experiment.is_dir():
                continue
            match = re.search(r"piTest([0-9.]+)$", experiment.name)
            if not match:
                continue
            target_pi = float(match.group(1))
            grouped = defaultdict(lambda: {"precision": [], "accuracy": [], "flipped": 0})
            records = load_classification_records(experiment / "opt_nonopt_trajs_alphaEst")
            for record in records:
                precision, accuracy, flipped = metrics(record.values)
                grouped[record.environment]["precision"].append(precision)
                grouped[record.environment]["accuracy"].append(accuracy)
                grouped[record.environment]["flipped"] += int(flipped)
            for environment, values in sorted(grouped.items()):
                precision = np.asarray(values["precision"])
                accuracy = np.asarray(values["accuracy"])
                rows.append(
                    {
                        "range": range_key,
                        "target_pi": target_pi,
                        "environment": environment,
                        "seed_count": len(accuracy),
                        "accuracy_mean": accuracy.mean(),
                        "accuracy_std": accuracy.std(),
                        "precision_mean": precision.mean(),
                        "precision_std": precision.std(),
                        "flipped_label_seeds": values["flipped"],
                    }
                )
    if not rows:
        raise ResultsError(f"No target-Pi experiments found under {ablation_root}")
    print_rows(rows)
    write_csv(rows, args.output_dir / "table_target_pi_classification.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ResultsError, KeyError, ZeroDivisionError) as exc:
        raise SystemExit(f"error: {exc}") from exc
