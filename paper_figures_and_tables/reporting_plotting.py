"""Internal plotting helpers used by the paper figure command-line scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reporting_common import ResultsError, aggregate_reward_curve


LoadedMethod = tuple[str, str, Mapping[str, Mapping[str, Mapping[str, Any]]]]


def plot_reward_axis(
    axis: plt.Axes,
    methods: Sequence[LoadedMethod],
    environment: str,
    *,
    show_legend: bool,
    title: str | None = None,
    fontsize: int = 18,
) -> None:
    plotted = 0
    for label, color, data in methods:
        if environment not in data:
            continue
        steps, mean, std = aggregate_reward_curve(data[environment])
        axis.plot(steps, mean, label=label, linewidth=2, color=color, zorder=10 - plotted)
        axis.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color, zorder=9 - plotted)
        plotted += 1
    if not plotted:
        raise ResultsError(f"No data available for {environment}")
    axis.set_xlabel("Training Steps", fontsize=fontsize - 2)
    axis.set_ylabel("Reward", fontsize=fontsize - 2)
    axis.set_title(title or environment, fontsize=fontsize)
    axis.tick_params(labelsize=fontsize - 4)
    axis.grid(True)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    if max(abs(value) for value in axis.get_ylim()) > 1000:
        axis.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    if show_legend:
        axis.legend(fontsize=fontsize - 6, loc="upper left")


def add_offline_line(
    axis: plt.Axes,
    steps: np.ndarray,
    *,
    label: str,
    color: str,
    mean: float,
    std: float,
    zorder: int,
) -> None:
    line = np.full(len(steps), mean)
    axis.plot(steps, line, label=label, linewidth=2, color=color, zorder=zorder)
    axis.fill_between(steps, line - std, line + std, alpha=0.2, color=color, zorder=zorder - 1)


def save_separate_reward_figures(
    methods: Sequence[LoadedMethod],
    environments: Sequence[str],
    output_dir: Path,
    filename_suffix: str,
    *,
    legend_on_ant: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for environment in environments:
        fig, axis = plt.subplots(figsize=(8, 6))
        plot_reward_axis(axis, methods, environment, show_legend=legend_on_ant and environment == "Ant-v4", fontsize=26)
        fig.tight_layout()
        path = output_dir / f"reward_{environment}__{filename_suffix}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
        print(f"Wrote {path}")
    return outputs
