#!/usr/bin/env python3
"""Generate reported analysis, bag-count, relabel, and top-k figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reporting_plotting import plot_reward_axis, save_separate_reward_figures
from reporting_common import (
    PAPER_ENVIRONMENTS,
    RANGES,
    ResultsError,
    analysis_specs,
    bag_number_specs,
    load_reward_runs,
    relabel_specs,
    topk_specs,
)


def load_methods(results_root: Path, specs):
    return [(label, color, load_reward_runs(results_root, spec)) for label, color, spec in specs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", choices=("analysis", "bag-number", "relabel-topk", "all"), default="all")
    parser.add_argument("--range", choices=("general", "low", "all"), default="all")
    return parser.parse_args()


def make_composite(results_root: Path, output_dir: Path) -> None:
    panel_definitions = [
        (RANGES["general"], relabel_specs, "(a) General expertise test", True),
        (RANGES["low"], relabel_specs, "(b) Low expertise test", False),
        (RANGES["general"], topk_specs, "(c) General expertise test", True),
        (RANGES["low"], topk_specs, "(d) Low expertise test", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(25, 5.2))
    for axis, (config, spec_builder, caption, legend) in zip(axes, panel_definitions):
        methods = load_methods(results_root, spec_builder(config))
        plot_reward_axis(axis, methods, "Ant-v4", show_legend=legend, fontsize=15)
        axis.text(0.5, -0.25, caption, transform=axis.transAxes, ha="center", va="top", fontsize=15)
    fig.subplots_adjust(bottom=0.27, wspace=0.26)
    output = output_dir / "abl_figs.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    range_keys = RANGES if args.range == "all" else (args.range,)
    for range_key in range_keys:
        config = RANGES[range_key]
        if args.group in ("analysis", "all"):
            methods = load_methods(args.results_root, analysis_specs(config))
            save_separate_reward_figures(methods, PAPER_ENVIRONMENTS, args.output_dir, f"{config.label}_analysis")
        if args.group in ("bag-number", "all"):
            methods = load_methods(args.results_root, bag_number_specs(config))
            save_separate_reward_figures(methods, ("Ant-v4",), args.output_dir, f"{config.label}_ablations_bagNum")
        if args.group in ("relabel-topk", "all"):
            relabel = load_methods(args.results_root, relabel_specs(config))
            save_separate_reward_figures(relabel, ("Ant-v4",), args.output_dir, f"{config.label}_ablations_relabel_earlyStop")
            topk = load_methods(args.results_root, topk_specs(config))
            save_separate_reward_figures(topk, ("Ant-v4",), args.output_dir, f"{config.label}_ablations_topk_values")
    if args.group in ("relabel-topk", "all") and args.range == "all":
        make_composite(args.results_root, args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultsError as exc:
        raise SystemExit(f"error: {exc}") from exc
