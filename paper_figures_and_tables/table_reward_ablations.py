#!/usr/bin/env python3
"""Write last-five reward summaries for reported ablation tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from reporting_common import (
    RANGES,
    ResultsError,
    bag_number_specs,
    expert_check_specs,
    load_reward_runs,
    pi_test_specs,
    print_rows,
    relabel_specs,
    summarize_reward_methods,
    topk_specs,
    write_csv,
)


BUILDERS = {
    "bag-number": bag_number_specs,
    "relabel": relabel_specs,
    "top-k": topk_specs,
    "expert-check": expert_check_specs,
    "target-pi": pi_test_specs,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", choices=(*BUILDERS, "all"), default="all")
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    range_keys = RANGES if args.range == "all" else (args.range,)
    groups = BUILDERS if args.group == "all" else (args.group,)
    all_rows = []
    for group in groups:
        rows = []
        for range_key in range_keys:
            config = RANGES[range_key]
            methods = [(label, load_reward_runs(args.results_root, spec)) for label, _, spec in BUILDERS[group](config)]
            group_rows = summarize_reward_methods(methods, ("Ant-v4",), range_key)
            for row in group_rows:
                row = {"ablation": group, **row}
                rows.append(row)
                all_rows.append(row)
        print(f"\n[{group}]")
        print_rows(rows)
        write_csv(rows, args.output_dir / f"table_reward_ablation_{group.replace('-', '_')}.csv")
    if len(groups) > 1:
        write_csv(all_rows, args.output_dir / "table_reward_ablations_all.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultsError as exc:
        raise SystemExit(f"error: {exc}") from exc
