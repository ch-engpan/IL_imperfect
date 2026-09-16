#!/usr/bin/env python3
"""Recompute the reported BC/ILEED scoring table from offline evaluation pickles."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from reporting_common import RANGES, ResultsError, load_pickle, print_rows, require_directory, write_csv


METRICS = (
    "best_true_eval",
    "best_true_log",
    "best_est_eval",
    "best_est_log",
    "best_final_eval",
    "best_final_log",
    "best_middle_eval",
    "best_middle_log",
    "best_last5_mean_eval",
    "best_last5_mean_log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    parser.add_argument("--rl-seed-min", type=int, default=0)
    parser.add_argument("--rl-seed-max", type=int, default=19)
    return parser.parse_args()


def evaluate_artifact(path: Path) -> dict[str, float]:
    data = load_pickle(path)
    rewards = np.asarray([entry[0] for entry in data["all_evals"]], dtype=float)
    logs = np.asarray(data["all_log_likelihood"], dtype=float)
    if len(rewards) <= 100 or len(logs) <= 100:
        raise ResultsError(f"Offline evaluation requires indices 0..100: {path}")
    true_index = int(np.argmax(rewards[:100]))
    estimated_index = int(np.argmin(logs[:100]))
    return {
        "best_true_eval": rewards[true_index],
        "best_true_log": logs[true_index],
        "best_est_eval": rewards[estimated_index],
        "best_est_log": logs[estimated_index],
        "best_final_eval": rewards[100],
        "best_final_log": logs[100],
        "best_middle_eval": rewards[100],
        "best_middle_log": logs[100],
        "best_last5_mean_eval": rewards[95:100].mean(),
        "best_last5_mean_log": logs[95:100].mean(),
    }


def load_runs(log_root: Path, method: str, seed_min: int, seed_max: int):
    require_directory(log_root, f"{method} offline log directory")
    pattern = re.compile(
        rf"^(?P<env>[A-Za-z0-9-]+)_{method}_SAC_NetAgentSeed_(?P<net>\d+-\d+)_RLseed_(?P<rl>\d+)_noisy-un_stepnum30e5_both$"
    )
    grouped = defaultdict(lambda: defaultdict(dict))
    for directory in sorted(log_root.iterdir()):
        if not directory.is_dir():
            continue
        match = pattern.match(directory.name)
        if not match:
            continue
        rl_seed = int(match.group("rl"))
        artifact = directory / "ileed_scoring_model_evals.pkl"
        if seed_min <= rl_seed <= seed_max and artifact.is_file():
            grouped[match.group("env")][match.group("net")][rl_seed] = evaluate_artifact(artifact)
    if not grouped:
        raise ResultsError(f"No {method} offline runs matched {log_root}")
    return grouped


def summarize(grouped, range_key: str, method: str) -> list[dict]:
    rows = []
    for environment, net_seeds in sorted(grouped.items()):
        selected_by_net = []
        for _net_seed, rl_runs in sorted(net_seeds.items()):
            selected = {
                "best_true_eval": max(run["best_true_eval"] for run in rl_runs.values()),
            }
            for stem in ("best_est", "best_final", "best_middle", "best_last5_mean"):
                best_run = min(rl_runs.values(), key=lambda run: run[f"{stem}_log"])
                selected[f"{stem}_eval"] = best_run[f"{stem}_eval"]
                selected[f"{stem}_log"] = best_run[f"{stem}_log"]
            true_run = max(rl_runs.values(), key=lambda run: run["best_true_eval"])
            selected["best_true_log"] = true_run["best_true_log"]
            selected_by_net.append(selected)
        for metric in METRICS:
            values = np.asarray([selected[metric] for selected in selected_by_net], dtype=float)
            rows.append(
                {
                    "range": range_key,
                    "method": method,
                    "environment": environment,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "net_seed_count": len(values),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    if args.rl_seed_min > args.rl_seed_max:
        raise ResultsError("--rl-seed-min must be no greater than --rl-seed-max")
    range_keys = RANGES if args.range == "all" else (args.range,)
    rows = []
    for range_key in range_keys:
        config = RANGES[range_key]
        for method, log_name in (("ILEED", "logs_ileed_noisy"), ("BC", "logs_ileed_bc_noisy")):
            grouped = load_runs(
                args.results_root / config.directory / log_name,
                method,
                args.rl_seed_min,
                args.rl_seed_max,
            )
            rows.extend(summarize(grouped, range_key, method))
    print_rows([row for row in rows if row["metric"] == "best_last5_mean_eval"])
    write_csv(rows, args.output_dir / "table_offline_bc_ileed.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ResultsError, KeyError, IndexError) as exc:
        raise SystemExit(f"error: {exc}") from exc
