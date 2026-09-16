#!/usr/bin/env python3
"""Write expertise estimates and estimation-error tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from reporting_common import RANGES, ResultsError, load_classification_records, print_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    parser.add_argument("--include-bag-ablation", action="store_true", help="Also process bag counts 2, 4, and 8")
    return parser.parse_args()


def summarize(directory: Path, range_key: str, bag_number: int) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for record in load_classification_records(directory):
        grouped[record.environment].append(record)
    detail_rows, error_rows = [], []
    for environment, records in sorted(grouped.items()):
        estimates = [np.asarray(record.values["est Pi"], dtype=float) for record in records if "est Pi" in record.values]
        truths = [np.asarray(record.values["true Pi"], dtype=float) for record in records if "true Pi" in record.values]
        if not estimates or not truths:
            raise ResultsError(f"Missing est Pi or true Pi in {directory} for {environment}")
        stacked = np.stack(estimates)
        truth = truths[0]
        if stacked.shape[1:] != truth.shape:
            raise ResultsError(f"Estimate/true Pi shape mismatch for {environment} in {directory}")
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        errors = np.abs(mean - truth)
        for index, (true_value, mean_value, std_value, error) in enumerate(zip(truth, mean, std, errors)):
            detail_rows.append(
                {
                    "range": range_key,
                    "bag_number": bag_number,
                    "environment": environment,
                    "pi_index": index,
                    "true_pi": true_value,
                    "estimated_mean": mean_value,
                    "estimated_std": std_value,
                    "absolute_error": error,
                    "seed_count": len(stacked),
                }
            )
        error_rows.append(
            {
                "range": range_key,
                "bag_number": bag_number,
                "environment": environment,
                "mean_absolute_error": errors.mean(),
                "std_absolute_error": errors.std(),
                "seed_count": len(stacked),
            }
        )
    return detail_rows, error_rows


def main() -> int:
    args = parse_args()
    range_keys = RANGES if args.range == "all" else (args.range,)
    details, errors = [], []
    bag_numbers = (6, 2, 4, 8) if args.include_bag_ablation else (6,)
    for range_key in range_keys:
        config = RANGES[range_key]
        for bag_number in bag_numbers:
            directory_name = config.directory if bag_number == 6 else f"{config.directory}_bagNum{bag_number}"
            new_details, new_errors = summarize(
                args.results_root / directory_name / "opt_nonopt_trajs_alphaEst",
                range_key,
                bag_number,
            )
            details.extend(new_details)
            errors.extend(new_errors)
    print_rows(errors)
    write_csv(details, args.output_dir / "table_expertise_estimates.csv")
    write_csv(errors, args.output_dir / "table_expertise_errors.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ResultsError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc
