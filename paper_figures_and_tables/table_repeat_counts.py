#!/usr/bin/env python3
"""Write the expertise-estimation repeat-count table from Pi-estimation CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from reporting_common import RANGES, ResultsError, print_rows, require_directory, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    return parser.parse_args()


def final_first_column(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or not rows[-1]:
        raise ResultsError(f"CSV has no data rows: {path}")
    return float(rows[-1][0])


def main() -> int:
    args = parse_args()
    range_keys = RANGES if args.range == "all" else (args.range,)
    rows = []
    for range_key in range_keys:
        directory = args.results_root / RANGES[range_key].directory / "Pi_estimation_csv"
        require_directory(directory, "Pi-estimation CSV directory")
        grouped = {}
        for path in sorted(directory.glob("*.csv")):
            match = re.search(r"env_(\d+)_seed_(\d+)", path.name)
            if match:
                grouped.setdefault(int(match.group(1)), []).append(final_first_column(path))
        if not grouped:
            raise ResultsError(f"No env_<n>_seed_<n> CSVs found in {directory}")
        for env_index, values in sorted(grouped.items()):
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "range": range_key,
                    "environment_index": env_index,
                    "seed_count": len(array),
                    "repeat_count_mean": array.mean() + 1,
                    "repeat_count_std": array.std(),
                    "raw_final_values": ";".join(f"{value:g}" for value in array),
                }
            )
    print_rows(rows)
    write_csv(rows, args.output_dir / "table_repeat_counts.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ResultsError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
