"""Shared loaders and aggregation helpers for paper figure/table scripts."""

from __future__ import annotations

import csv
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ENVIRONMENTS = ("Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4")
PAPER_ENVIRONMENTS = ("Ant-v4", "HalfCheetah-v4", "Swimmer-v4", "Walker2d-v4")


@dataclass(frozen=True)
class RangeConfig:
    key: str
    directory: str
    alpha_token: str
    topk_token: str
    label: str


RANGES = {
    "general": RangeConfig(
        key="general",
        directory="alphaEst_randomPi_PiRange01-09",
        alpha_token="alpha_05",
        topk_token="05",
        label="PiRange01-09",
    ),
    "low": RangeConfig(
        key="low",
        directory="alphaEst_randomPi_PiRange005-015",
        alpha_token="alpha_01",
        topk_token="01",
        label="PiRange005-015",
    ),
}


# Exact best-last-five summaries embedded in plot_reward_epochs_bc_ileed.ipynb.
# table_offline_bc_ileed.py recomputes these when raw offline artifacts exist.
PUBLISHED_OFFLINE = {
    ("general", "ILEED", "Ant-v4"): (1428.239931707118, 80.30208621932131),
    ("general", "ILEED", "HalfCheetah-v4"): (1526.2424895429529, 1020.4384258611778),
    ("general", "ILEED", "Swimmer-v4"): (16.812320047701633, 20.535220085379475),
    ("general", "ILEED", "Walker2d-v4"): (440.31112935443053, 312.97313098436393),
    ("general", "BC", "Ant-v4"): (1341.8273696857707, 164.4891692321331),
    ("general", "BC", "HalfCheetah-v4"): (2318.051715729079, 521.2398302610937),
    ("general", "BC", "Swimmer-v4"): (21.29507695493383, 23.02536672396615),
    ("general", "BC", "Walker2d-v4"): (308.4402455086914, 111.54240686111682),
    ("low", "ILEED", "Ant-v4"): (1376.528214931588, 62.08062803887376),
    ("low", "ILEED", "HalfCheetah-v4"): (3767.587939550842, 176.90741593505325),
    ("low", "ILEED", "Swimmer-v4"): (34.016429230939785, 2.407262536965019),
    ("low", "ILEED", "Walker2d-v4"): (605.2535607809853, 94.53407393079202),
    ("low", "BC", "Ant-v4"): (1473.3630826529673, 97.65838281651175),
    ("low", "BC", "HalfCheetah-v4"): (3753.929670089999, 220.46981591480647),
    ("low", "BC", "Swimmer-v4"): (27.81180407887758, 11.568241530154388),
    ("low", "BC", "Walker2d-v4"): (436.59350968351856, 69.05891317649167),
}


@dataclass(frozen=True)
class RewardRunSpec:
    """Location and filename filters for one plotted method."""

    root_parts: tuple[str, ...]
    log_directory: str
    required_tokens: tuple[str, ...]
    forbidden_tokens: tuple[str, ...] = ()


class ResultsError(RuntimeError):
    """Raised when a requested report cannot be built from the supplied results."""


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise ResultsError(f"Missing {description}: {path}")


def load_pickle(path: Path) -> Any:
    """Load a trusted experiment pickle.

    Pickle files can execute code while loading. These scripts are intended only
    for artifacts produced by this project or another trusted source.
    """

    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # pragma: no cover - exact pickle failures vary
        raise ResultsError(f"Could not load trusted pickle {path}: {exc}") from exc


def _environment_from_name(name: str) -> str | None:
    return next((env for env in ENVIRONMENTS if name.startswith(f"{env}_")), None)


def _net_seed_from_name(name: str) -> str | None:
    match = re.search(r"NetAgentSeed_(\d+-\d+)", name)
    return match.group(1) if match else None


def load_reward_runs(results_root: Path, spec: RewardRunSpec) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Load reward-curve pickles matching a notebook experiment definition."""

    log_root = results_root.joinpath(*spec.root_parts, spec.log_directory)
    require_directory(log_root, "experiment directory")

    loaded: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for run_dir in sorted(log_root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if not all(token in name for token in spec.required_tokens):
            continue
        if any(token in name for token in spec.forbidden_tokens):
            continue
        env = _environment_from_name(name)
        seed = _net_seed_from_name(name)
        if env is None or seed is None:
            continue
        artifact = run_dir / "eval_data_reward_step_after_training.pkl"
        if not artifact.is_file():
            continue
        if seed in loaded[env]:
            raise ResultsError(f"Multiple matching reward runs for {env}, seed {seed}, and {spec}")
        loaded[env][seed] = load_pickle(artifact)

    if not loaded:
        tokens = ", ".join(spec.required_tokens) or "<none>"
        raise ResultsError(f"No reward runs matched {log_root}; required filename tokens: {tokens}")
    return dict(loaded)


def aggregate_reward_curve(seed_data: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steps, across-seed mean, and across-seed standard deviation."""

    if not seed_data:
        raise ResultsError("Cannot aggregate an empty set of reward seeds")

    step_arrays: list[np.ndarray] = []
    reward_arrays: list[np.ndarray] = []
    for seed in sorted(seed_data, key=lambda value: int(value.split("-")[0])):
        data = seed_data[seed]
        try:
            steps = np.asarray(data["stepnum_list"], dtype=float)
            rewards = np.asarray(data["mean_reward_list"], dtype=float)
        except KeyError as exc:
            raise ResultsError(f"Reward artifact is missing key {exc.args[0]!r}") from exc
        if steps.ndim != 1 or rewards.ndim != 1 or len(steps) != len(rewards):
            raise ResultsError(f"Invalid one-dimensional reward curve for seed {seed}")
        step_arrays.append(steps)
        reward_arrays.append(rewards)

    reference = step_arrays[0]
    if any(not np.array_equal(reference, steps) for steps in step_arrays[1:]):
        raise ResultsError("Reward seeds use different training-step grids")
    stacked = np.stack(reward_arrays)
    return reference, stacked.mean(axis=0), stacked.std(axis=0)


def last_five_seed_values(seed_data: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    """Match the notebooks: average each seed's final five evaluations, then round."""

    values = []
    for seed in sorted(seed_data, key=lambda value: int(value.split("-")[0])):
        rewards = np.asarray(seed_data[seed]["mean_reward_list"], dtype=float)
        if rewards.size < 5:
            raise ResultsError(f"Seed {seed} has fewer than five reward evaluations")
        values.append(round(float(rewards[-5:].mean())))
    return np.asarray(values, dtype=float)


def summarize_reward_methods(
    methods: Sequence[tuple[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]],
    environments: Iterable[str],
    range_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env in environments:
        for method, data in methods:
            if env not in data:
                continue
            values = last_five_seed_values(data[env])
            rows.append(
                {
                    "range": range_name,
                    "environment": env,
                    "method": method,
                    "seed_count": len(values),
                    "mean_last_five": float(values.mean()),
                    "std_last_five": float(values.std()),
                    "seed_values": ";".join(f"{value:g}" for value in values),
                }
            )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ResultsError(f"No rows were produced for {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


def print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        print("No rows")
        return
    columns = list(rows[0].keys())
    widths = {column: max(len(column), *(len(str(row.get(column, ""))) for row in rows)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def baseline_spec(config: RangeConfig) -> RewardRunSpec:
    return RewardRunSpec(
        root_parts=(config.directory,),
        log_directory="logs_scoring_sin-train_noReplacing",
        required_tokens=(
            "_scoring_SAC_",
            "RLseed_0_",
            config.alpha_token,
            f"topk_expTopk_{config.topk_token}_earlyStop30num",
        ),
    )


def main_reward_specs(config: RangeConfig) -> list[tuple[str, str, RewardRunSpec]]:
    root = (config.directory,)
    return [
        ("Scoring (ours)", "blue", baseline_spec(config)),
        (
            "GAIL o.s.",
            "C1",
            RewardRunSpec(root, "logs_gails_clean", ("_GAIL_SAC_", "RLseed_0_", "gail_discrm_500_iters-un")),
        ),
        (
            "GAIL",
            "C3",
            RewardRunSpec(root, "logs_gails_noisy", ("_GAIL_SAC_", "RLseed_0_", "gail_discrm_500_iters_noisy-un")),
        ),
        (
            "RIL",
            "C2",
            RewardRunSpec(root, "logs_ril_noisy", ("_RIL_SAC_", "RLseed_0_", "ril_discrm_500_iters_noisy-un")),
        ),
        (
            "WGAIL",
            "C4",
            RewardRunSpec(root, "logs_wgail_noisy", ("_WGAIL_SAC_", "RLseed_0_", "wgail_discrm_500_iters_noisy-un")),
        ),
    ]


def analysis_specs(config: RangeConfig) -> list[tuple[str, str, RewardRunSpec]]:
    root = (config.directory,)
    common = (
        "_scoring_SAC_",
        "RLseed_0_",
        config.alpha_token,
    )
    return [
        ("Scoring", "blue", baseline_spec(config)),
        (
            "Scoring PN",
            "C2",
            RewardRunSpec(
                root,
                "abl_logs_scoring_sin-train_noReplacing_uu10",
                common + (f"topk_expTopk_{config.topk_token}_earlyStop30num_expLossPN",),
            ),
        ),
        (
            "Scoring exp. known",
            "C3",
            RewardRunSpec(
                root,
                "logs_scoring_sin-train_noReplacing_piGiven",
                common + (f"topk_expTopk_{config.topk_token}_earlyStop30num",),
            ),
        ),
        (
            "Scoring opt. known",
            "C4",
            RewardRunSpec(
                root,
                "logs_scoring_sin-train_noReplacing_oracle_pre_uu",
                common + ("noRelabel_expFineTrueLabels",),
            ),
        ),
    ]


def bag_number_specs(config: RangeConfig) -> list[tuple[str, str, RewardRunSpec]]:
    root = (config.directory,)
    required = ("_scoring_SAC_", "RLseed_0_", config.alpha_token)
    return [
        ("num 2", "paleturquoise", RewardRunSpec(root, "abl_logs_scoring_sin-train_noReplacing_bagNum2", required)),
        ("num 4", "dodgerblue", RewardRunSpec(root, "abl_logs_scoring_sin-train_noReplacing_bagNum4", required)),
        ("num 6", "blue", baseline_spec(config)),
        ("num 8", "navy", RewardRunSpec(root, "abl_logs_scoring_sin-train_noReplacing_bagNum8", required)),
    ]


def relabel_specs(config: RangeConfig) -> list[tuple[str, str, RewardRunSpec]]:
    root = (config.directory,)
    common = ("_scoring_SAC_", "RLseed_0_", config.alpha_token)
    return [
        ("Scoring", "blue", baseline_spec(config)),
        ("Scoring no relabel", "C1", RewardRunSpec(root, "abl_logs_scoring_sin-train_noReplacing_noRelabel", common + ("noRelabel",))),
        (
            "Scoring no early stop",
            "C3",
            RewardRunSpec(
                root,
                "abl_logs_scoring_sin-train_noReplacing_noEarlyStop",
                common + (f"topk_expTopk_{config.topk_token}",),
                forbidden_tokens=("earlyStop",),
            ),
        ),
    ]


def topk_specs(config: RangeConfig) -> list[tuple[str, str, RewardRunSpec]]:
    root = (config.directory,)

    def variant(token: str, label: str, color: str) -> tuple[str, str, RewardRunSpec]:
        return (
            label,
            color,
            RewardRunSpec(
                root,
                f"abl_logs_scoring_sin-train_noReplacing_topk_{token}",
                ("_scoring_SAC_", "RLseed_0_", config.alpha_token, f"topk_expTopk_{token}_earlyStop30num"),
            ),
        )

    if config.key == "general":
        return [
            variant("01", "topk 0.1", "paleturquoise"),
            variant("03", "topk 0.3", "dodgerblue"),
            ("topk 0.5", "blue", baseline_spec(config)),
            variant("055", "topk 0.55", "#66cc66"),
            variant("06", "topk 0.6", "#339966"),
            variant("07", "topk 0.7", "green"),
        ]
    return [
        variant("005", "topk 0.05", "dodgerblue"),
        ("topk 0.1", "blue", baseline_spec(config)),
        variant("015", "topk 0.15", "#66cc66"),
        variant("02", "topk 0.2", "green"),
    ]


def expert_check_specs(config: RangeConfig) -> list[tuple[str, str, RewardRunSpec]]:
    root = (config.directory,)
    specs: list[tuple[str, str, RewardRunSpec]] = []
    for number, color in ((1, "lightblue"), (3, "dodgerblue"), (5, "blue"), (7, "navy"), (9, "black")):
        directory = "logs_scoring_sin-train_noReplacing" if number == 5 else f"logs_scoring_sin-train_noReplacing_expCheckNum{number}"
        rl_seed = "RLseed_0_" if number == 5 else "RLseed_100_"
        specs.append(
            (
                f"Query {number}",
                color,
                RewardRunSpec(root, directory, ("_scoring_SAC_", rl_seed, config.alpha_token)),
            )
        )
    return specs


def pi_test_specs(config: RangeConfig) -> list[tuple[str, str, RewardRunSpec]]:
    if config.key == "general":
        values = ((0.1, "red"), (0.2, "orange"), (0.3, "olive"), (0.4, "green"), (0.6, "blue"), (0.7, "dodgerblue"), (0.8, "lightblue"), (0.9, "black"))
        baseline = (0.5, "navy")
    else:
        values = ((0.45, "red"), (0.55, "orange"), (0.6, "green"), (0.7, "blue"), (0.9, "black"))
        baseline = (0.5, "purple")

    specs: list[tuple[str, str, RewardRunSpec]] = []
    for value, color in values:
        suffix = f"_expCheckNum3_piTest{value:g}"
        root_name = f"{config.directory}{suffix}"
        specs.append(
            (
                f"α'={value:g}",
                color,
                RewardRunSpec(
                    ("rebt_ablation", root_name),
                    f"logs_scoring_sin-train_noReplacing{suffix}",
                    ("_scoring_SAC_", "RLseed_100_", config.alpha_token),
                ),
            )
        )
    baseline_value, baseline_color = baseline
    specs.append((f"α'={baseline_value:g}", baseline_color, baseline_spec(config)))
    return sorted(specs, key=lambda item: float(item[0].split("=")[1]))


@dataclass(frozen=True)
class ClassificationRecord:
    environment: str
    seed: int
    values: Mapping[str, Any]
    source: Path


def parse_classification_file(path: Path) -> ClassificationRecord:
    match = re.search(r"trajs_(.+?)_ExpertSeed_(\d+)_", path.name)
    if not match:
        raise ResultsError(f"Cannot parse environment and seed from {path.name}")
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        size_match = re.fullmatch(r"(\w+): torch\.Size\(\[(.*?)\]\)", line)
        if size_match:
            shape_text = size_match.group(2).strip()
            values[size_match.group(1)] = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
            continue
        tuple_match = re.fullmatch(r"(\w+): \((.*?)\)", line)
        if tuple_match:
            values[tuple_match.group(1)] = tuple(int(part.strip()) for part in tuple_match.group(2).split(",") if part.strip())
            continue
        array_match = re.fullmatch(r"([\w ]+): \[(.*?)\]", line)
        if array_match:
            values[array_match.group(1).strip()] = np.fromstring(array_match.group(2), sep=" ")
    return ClassificationRecord(match.group(1), int(match.group(2)), values, path)


def load_classification_records(directory: Path) -> list[ClassificationRecord]:
    require_directory(directory, "classification-info directory")
    records = [parse_classification_file(path) for path in sorted(directory.glob("opt_nonopt_trajs_*.txt"))]
    if not records:
        raise ResultsError(f"No classification text files found in {directory}")
    return records


def scalar_count(value: Any, key: str) -> float:
    if isinstance(value, tuple):
        if not value:
            raise ResultsError(f"Empty tuple for {key}")
        return float(value[0])
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ResultsError(f"Empty value for {key}")
    return float(array[0])


def relabel_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_pickle(path)
    required = ("TP_exp", "TN_exp", "FP_exp", "FN_exp", "recall_exp", "unprecision_exp", "relabel_earlyStop")
    missing = [key for key in required if key not in data]
    if missing:
        raise ResultsError(f"{path} is missing relabel keys: {', '.join(missing)}")
    early_stop = np.asarray(data["relabel_earlyStop"])
    transitions = np.flatnonzero((early_stop[:-1] == 0) & (early_stop[1:] == 1))
    if not len(transitions):
        raise ResultsError(f"No relabel early-stop transition found in {path}")
    stop_index = int(transitions[0] + 1)

    def at(index: int) -> np.ndarray:
        return np.asarray(
            [
                data["TP_exp"][index],
                data["TN_exp"][index],
                data["FP_exp"][index],
                data["FN_exp"][index],
                data["recall_exp"][index],
                1.0 - data["unprecision_exp"][index],
            ],
            dtype=float,
        )

    return at(0), at(stop_index)


def discover_relabel_runs(log_directory: Path, required_tokens: Sequence[str]) -> list[tuple[str, str, Path]]:
    require_directory(log_directory, "relabel experiment directory")
    discovered = []
    for run_dir in sorted(log_directory.iterdir()):
        if not run_dir.is_dir() or not all(token in run_dir.name for token in required_tokens):
            continue
        env = _environment_from_name(run_dir.name)
        seed = _net_seed_from_name(run_dir.name)
        artifact = run_dir / "rl_scoring_uu_learning_data.pkl"
        if env and seed and artifact.is_file():
            discovered.append((env, seed, artifact))
    if not discovered:
        raise ResultsError(f"No relabel artifacts matched {log_directory}")
    return discovered
