#!/usr/bin/env python3
"""Write before/after relabel accuracy, precision, and confusion-count tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from reporting_common import (
    RANGES,
    ResultsError,
    discover_relabel_runs,
    print_rows,
    relabel_metrics,
    write_csv,
)


METRICS = ("TP", "TN", "FP", "FN", "recall", "precision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    parser.add_argument(
        "--include-query-variants",
        action="store_true",
        help="Recursively include matching logs_scoring_* directories, including expert-query ablations",
    )
    return parser.parse_args()


def candidate_log_directories(results_root: Path, config, recursive: bool) -> list[Path]:
    baseline = results_root / config.directory / "logs_scoring_sin-train_noReplacing"
    if not recursive:
        return [baseline]
    candidates = {
        path
        for path in results_root.rglob("logs_scoring_sin-train_noReplacing*")
        if path.is_dir() and config.directory in str(path)
    }
    candidates.add(baseline)
    return sorted(candidates)


def main() -> int:
    args = parse_args()
    range_keys = RANGES if args.range == "all" else (args.range,)
    rows = []
    for range_key in range_keys:
        config = RANGES[range_key]
        for log_directory in candidate_log_directories(args.results_root, config, args.include_query_variants):
            runs = discover_relabel_runs(log_directory, ("_scoring_SAC_", config.alpha_token))
            grouped = defaultdict(lambda: {"before": [], "after": []})
            for environment, _seed, artifact in runs:
                before, after = relabel_metrics(artifact)
                grouped[environment]["before"].append(before)
                grouped[environment]["after"].append(after)
            experiment = str(log_directory.relative_to(args.results_root))
            for environment, stages in sorted(grouped.items()):
                for stage, values in stages.items():
                    array = np.stack(values)
                    tp, tn, fp, fn = (array[:, index] for index in range(4))
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    row = {
                        "range": range_key,
                        "experiment": experiment,
                        "environment": environment,
                        "stage": stage,
                        "seed_count": len(array),
                        "accuracy_mean": accuracy.mean(),
                        "accuracy_std": accuracy.std(),
                    }
                    for index, metric in enumerate(METRICS):
                        row[f"{metric}_mean"] = array[:, index].mean()
                        row[f"{metric}_std"] = array[:, index].std()
                    rows.append(row)
    print_rows(rows)
    write_csv(rows, args.output_dir / "table_relabel_metrics.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultsError as exc:
        raise SystemExit(f"error: {exc}") from exc
