# Imitating the Imperfect

Code for **Imitating the Imperfect: Offline-to-Online Robust Imitation
Learning from Heterogeneous Demonstrators**.

This guide covers the MuJoCo experiments in this repository. The Robomimic
scripts are not covered because they require a separate Robomimic installation,
datasets, and source files that are not included here.

## 1. Repository layout and workflow

The MuJoCo workflow has two main stages:

1. `mujo_02_trajs-Udata_alpha_est_entropy_main.py` reads precomputed optimal
   and non-optimal trajectories, creates heterogeneous unlabeled bags, estimates
   their expertise levels (`Pi`), and saves scoring models and U-data.
2. The main method or a baseline consumes those generated artifacts and trains
   a policy.

The scripts use paths relative to the repository and write results to
`../results/`. Run all commands from the repository root.

Supported environments:

| `--env` | Gymnasium environment |
| ---: | --- |
| `0` | `Ant-v4` |
| `1` | `HalfCheetah-v4` |
| `2` | `Hopper-v4` |
| `3` | `Swimmer-v4` |
| `4` | `Walker2d-v4` |

The default `--env 9` in several scripts is not valid. Always pass an
environment index from `0` to `4`.

## 2. Installation and dependencies

The supplied environments were exported on Linux. The more portable file omits
Conda build identifiers and is recommended for a new installation:

```bash
git clone https://github.com/ch-engpan/IL_imperfect.git
cd IL_imperfect
conda env create --file env.yml
conda activate example_rl
```

For the exact Linux package builds used in the original environment, use:

```bash
conda env create --file env_exact.yml
conda activate example_rl
```

Both files use the general environment name `example_rl` and contain no
machine-specific installation prefix. Pass `--name another_name` when creating
the environment if you prefer a different local name.

Important pinned packages are:

| Dependency | Version |
| --- | ---: |
| Python | `3.9.19` |
| PyTorch | `2.4.1` |
| Gymnasium | `0.29.1` |
| MuJoCo | `2.3.3` |
| Stable-Baselines3 | `2.2.1` |
| sb3-contrib | `2.2.1` |
| NumPy | `1.26.4` |
| SciPy | `1.13.1` |
| scikit-learn | `1.4.1.post1` |
| Matplotlib | `3.9.2` |

The recommended full dependency specification is `env.yml`. The
`env_exact.yml` file is retained as an exact Linux build export for archival
reproduction. The current MuJoCo-v4 code uses the official `mujoco` package;
it does not import the legacy `mujoco_py` package.

Check the installation with:

```bash
python3 -c "import gymnasium, mujoco, numpy, sklearn, stable_baselines3, torch; print('gymnasium', gymnasium.__version__); print('mujoco', mujoco.__version__); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('stable_baselines3', stable_baselines3.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

CUDA is used when available; otherwise the scripts select CPU. Environment
simulation and parallel rollout collection are CPU-heavy, so multiple CPU cores
are useful even when a GPU is available.

## 3. Required trajectory data

The initial trajectory datasets are not included in this repository. Replace
every placeholder such as `{dataset_path}` with the correct path on your
machine.

For each environment, the expertise-estimation script expects:

```text
../results/opt_nonopt_trajs_noReplacing/
  opt_nonopt_trajs_{environment}_sacExpertSeed_0_newneg_fullDataset.pkl
```

Each pickle must contain:

```text
opt_traj_s_set, opt_traj_a_set,
nonopt_traj_s_set, nonopt_traj_a_set,
scaler_s, scaler_a, opt_start_steps
```

For example, install the Ant-v4 input as follows:

```bash
mkdir -p ../results/opt_nonopt_trajs_noReplacing
cp {dataset_path}/opt_nonopt_trajs_Ant-v4_sacExpertSeed_0_newneg_fullDataset.pkl \
  ../results/opt_nonopt_trajs_noReplacing/
```

To install all five datasets from one directory:

```bash
mkdir -p ../results/opt_nonopt_trajs_noReplacing
for environment in Ant-v4 HalfCheetah-v4 Hopper-v4 Swimmer-v4 Walker2d-v4; do
  cp "{dataset_path}/opt_nonopt_trajs_${environment}_sacExpertSeed_0_newneg_fullDataset.pkl" \
    ../results/opt_nonopt_trajs_noReplacing/
done
```

## 4. Generate U-data and estimate expertise levels

The following canonical example uses Ant-v4, estimator seed `0`, six bags, Pi
range `[0.1, 0.9]`, five expert checks, and test Pi `0.5`:

```bash
python3 mujo_02_trajs-Udata_alpha_est_entropy_main.py \
  --env 0 \
  --uu_seed 0 \
  --method 5 \
  --Pi_range_select 0 \
  --pi_given False \
  --bag_num 6 \
  --exp_check_num 5 \
  --pi_test 0.5
```

`--method 5` is required by the active training path. The main arguments are:

| Argument | Meaning |
| --- | --- |
| `--uu_seed` | Seed used for data splitting, sampling, and the scoring network |
| `--Pi_range_select` | Index into the training-bag Pi-range table below |
| `--pi_given` | `True` uses known Pi; `False` estimates it |
| `--bag_num` | Number of training bags; common values are `2`, `4`, `6`, and `8` |
| `--exp_check_num` | Number of expert-check samples used by Pi estimation |
| `--pi_test` | Positive ratio used to build the test bag |

Pi ranges implemented by the estimator are:

| Index | Pi range | Index | Pi range |
| ---: | --- | ---: | --- |
| `0` | `[0.10, 0.90]` | `8` | `[0.05, 0.40]` |
| `1` | `[0.05, 0.15]` | `9` | `[0.05, 0.50]` |
| `2` | `[0.45, 0.55]` | `10` | `[0.40, 0.60]` |
| `3` | `[0.85, 0.95]` | `11` | `[0.30, 0.70]` |
| `4` | `[0.05, 0.25]` | `12` | `[0.80, 0.95]` |
| `5` | `[0.40, 0.60]` | `13` | `[0.70, 0.95]` |
| `6` | `[0.75, 0.95]` | `14` | `[0.60, 0.95]` |
| `7` | `[0.20, 0.80]` | `15` | `[0.50, 0.95]` |

The downstream training scripts only initialize their expected mean-alpha value
for Pi-range indices `0`, `1`, `2`, and `4`. Use one of these indices for an
end-to-end run unless you also update the downstream alpha mapping.

For the command above, outputs are written below:

```text
../results/alphaEst_randomPi_PiRange01-09_expCheckNum5_piTest0.5/
  Pi_estimation_csv/
  scoring_model_alphaEst/
  opt_nonopt_trajs_alphaEst/
  noisy_data/
```

The generated NPZ stores the train/test state-action bags, bag labels, true
binary labels, Pi values, and fitted state/action scalers.

### Run environments and seeds sequentially

This example generates data for every environment and seeds `0` through `4`:

```bash
for env_index in 0 1 2 3 4; do
  for seed in 0 1 2 3 4; do
    python3 mujo_02_trajs-Udata_alpha_est_entropy_main.py \
      --env "$env_index" \
      --uu_seed "$seed" \
      --method 5 \
      --Pi_range_select 0 \
      --pi_given False \
      --bag_num 6 \
      --exp_check_num 5 \
      --pi_test 0.5
  done
done
```

The processes are expensive. Run several independent terminal sessions only if
the machine has enough CPU and GPU capacity.

## 5. Normalize generated filenames

The current generator writes an extra underscore before the extension for its
NPZ, scoring-model PKL, and pseudo-labeled PKL artifacts. Current downstream
loaders expect the same filenames without that final underscore. Preserve the
original files and create compatible copies with:

```bash
alpha_root=../results/alphaEst_randomPi_PiRange01-09_expCheckNum5_piTest0.5
find "$alpha_root" -type f \( -name '*_.npz' -o -name '*_.pkl' \) -print0 | \
  while IFS= read -r -d '' source_file; do
    destination_file="${source_file%_*}.${source_file##*.}"
    cp -- "$source_file" "$destination_file"
  done
```

For example, this creates:

```text
..._newneg.npz       from ..._newneg_.npz
..._NetSeed_0.pkl    from ..._NetSeed_0_.pkl
..._alpha_05.pkl     from ..._alpha_05_.pkl
```

## 6. Run the main AlphaEst-guided method

Run this only after completing Sections 4 and 5 with matching values for
`--env`, `--Expert_idx`/`--uu_seed`, `--Pi_range_select`, `--bag_num`,
`--exp_check_num`, and `--pi_test`.

Canonical Ant-v4 command:

```bash
python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py \
  --new_scormodel false \
  --env 0 \
  --Expert_idx 0 \
  --total_steps 3e6 \
  --data_augm both \
  --pre_label True \
  --rl_seed 0 \
  --uuLoss_str uuloss_Nagent \
  --agentReplay_fine True \
  --expertDemo_fine True \
  --expertDemo_label_method topk \
  --topk_k 0.5 \
  --Pi_range_select 0 \
  --bag_num 6 \
  --exp_check_num 5 \
  --pi_test 0.5 \
  --num_envs 32
```

Important options:

| Argument | Values and behavior |
| --- | --- |
| `--Expert_idx` | Selects the U-data and scoring-network seed; use a single digit (`0`-`9`) because the parser converts the value to a character list |
| `--uuLoss_str` | `uuloss_Nagent`, `uuloss_all`, or `None` |
| `--data_augm` | `normal`, `mixup`, or `both` |
| `--expertDemo_fine` | Enable expert-demonstration relabeling |
| `--agentReplay_fine` | Enable agent replay relabeling |
| `--expertDemo_label_method` | `topk` or `threshold` |
| `--topk_k` | Fraction selected as optimal when using `topk`; a non-positive value uses estimated mean Pi |
| `--exp_opt_threshold` | Classification threshold when using `threshold` |
| `--early_stop` | Enable or disable relabeling early stopping |
| `--exp_loss` | `uu_loss` or `PN_loss` |
| `--num_envs` | Number of parallel rollout environments; reduce this on smaller machines |

Outputs are created inside the matching AlphaEst root under
`logs_scoring_sin-train_noReplacing.../`. Each run includes checkpoints,
`best_model.zip`, `evaluations.npz`, serialized scoring/training data, and PDF
plots.

## 7. Baselines

The baseline scripts look for generated NPZ files in the older result root that
does not contain `_expCheckNum5_piTest0.5`. Prepare that compatibility location
once after Sections 4 and 5:

```bash
alpha_root=../results/alphaEst_randomPi_PiRange01-09_expCheckNum5_piTest0.5
baseline_root=../results/alphaEst_randomPi_PiRange01-09
mkdir -p "$baseline_root/noisy_data"
cp "$alpha_root"/noisy_data/*.npz "$baseline_root/noisy_data/"
```

Replace the environment, expert seed, RL seed, and step count as needed, but
keep `--Pi_range_select` consistent with the data-generation stage.

### GAIL

```bash
python3 mujo_10_gail_paral.py \
  --Expert_idx 0 \
  --total_steps 3e6 \
  --env 0 \
  --noisy False \
  --data_augm both \
  --rl_seed 0 \
  --Pi_range_select 0
```

`--noisy False` keeps only samples carrying a true optimal label;
`--noisy True` trains on the mixed U-data.

### RIL

```bash
python3 mujo_20_ril_paral.py \
  --Expert_idx 0 \
  --total_steps 3e6 \
  --env 0 \
  --data_augm both \
  --rl_seed 0 \
  --Pi_range_select 0
```

### WGAIL

```bash
python3 mujo_20_wgail_paral.py \
  --Expert_idx 0 \
  --total_steps 3e6 \
  --env 0 \
  --data_augm both \
  --rl_seed 0 \
  --Pi_range_select 0
```

### iLEED

```bash
python3 mujo_10_ileed_offline.py \
  --Expert_idx 0 \
  --total_steps 3e6 \
  --env 0 \
  --noisy True \
  --data_augm both \
  --rl_seed 0 \
  --Pi_range_select 0 \
  --embed_dim 10 \
  --hidden_size 256
```

### iLEED-BC

```bash
python3 mujo_10_ileed_bc_offline.py \
  --Expert_idx 0 \
  --total_steps 3e6 \
  --env 0 \
  --noisy True \
  --data_augm both \
  --rl_seed 0 \
  --Pi_range_select 0 \
  --embed_dim 10 \
  --hidden_size 256
```

Baseline outputs are saved under the compatibility root in method-specific
folders such as `logs_gails_clean/`, `logs_gails_noisy/`, `logs_ril_noisy/`,
`logs_wgail_noisy/`, `logs_ileed_noisy/`, and `logs_ileed_bc_noisy/`.

### Baseline seed sweep

For example, run noisy GAIL with five RL seeds against expertise seed `0`:

```bash
for rl_seed in 0 1 2 3 4; do
  python3 mujo_10_gail_paral.py \
    --Expert_idx 0 \
    --total_steps 3e6 \
    --env 0 \
    --noisy True \
    --data_augm both \
    --rl_seed "$rl_seed" \
    --Pi_range_select 0
done
```

## 8. Optional 10%-oracle ablation

Generate the oracle variant with:

```bash
python3 mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py \
  --env 0 \
  --uu_seed 0 \
  --method 5 \
  --Pi_range_select 0 \
  --pi_given True \
  --bag_num 6
```

This generator writes to:

```text
../results/alphaEst_randomPi_PiRange01-09_piGiven_uu10orcl/
```

The main training script additionally expects the suffix
`_expCheckNum5_piTest0.5`. Create the expected compatibility directory, then
normalize its filenames:

```bash
oracle_source=../results/alphaEst_randomPi_PiRange01-09_piGiven_uu10orcl
oracle_target=../results/alphaEst_randomPi_PiRange01-09_piGiven_uu10orcl_expCheckNum5_piTest0.5
mkdir -p "$oracle_target"
cp -a "$oracle_source"/. "$oracle_target"/
find "$oracle_target" -type f \( -name '*_.npz' -o -name '*_.pkl' \) -print0 | \
  while IFS= read -r -d '' source_file; do
    destination_file="${source_file%_*}.${source_file##*.}"
    cp -- "$source_file" "$destination_file"
  done
```

Then run:

```bash
python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py \
  --new_scormodel false \
  --env 0 \
  --Expert_idx 0 \
  --total_steps 3e6 \
  --data_augm both \
  --pre_label True \
  --rl_seed 0 \
  --uuLoss_str uuloss_Nagent \
  --agentReplay_fine False \
  --expertDemo_fine False \
  --expertDemo_label_method topk \
  --topk_k 0.5 \
  --early_stop False \
  --Pi_range_select 0 \
  --pi_given True \
  --exp_loss PN_loss \
  --uu10_orcl True \
  --exp_check_num 5 \
  --pi_test 0.5
```

## 9. Evaluation and generated files

The main and online-baseline scripts evaluate checkpoints and generate plots at
the end of training. Typical run directories contain:

```text
checkpoints/
best_model.zip
evaluations.npz
*_model.pkl
eval_data_reward_step_*.pkl
*.pdf
```

To regenerate a reward plot from an `evaluations.npz` file without relying on
the older hardcoded paths in the standalone entry point, run:

```bash
python3 -c "from mujo_04_EVAL_results_npz import plot_results_npz; plot_results_npz('Ant-v4', '{run_output_dir}/', 'manual')"
```

Replace `{run_output_dir}` with the directory containing `evaluations.npz` and
keep the trailing `/`.

`mujo_04_EVAL.py`, the command-line section of
`mujo_04_EVAL_results_npz.py`, and the command-line section of
`mujo_10_EVAL_discriminator.py` retain older hardcoded result layouts. Edit
their result paths before using them as standalone scripts. Their helper
functions are already called by the current training scripts.

## 10. Reproduce reported figures and tables

The cleaned reporting programs are in `paper_figures_and_tables/`. Files whose
names begin with `figure_` write PDF figures; files beginning with `table_`
print a summary and write CSV tables. They replace the original notebooks'
hardcoded `/home/robo/...` paths with explicit command-line paths.

The reporting-only dependencies are a small subset of the full environment:

```bash
python3 -m pip install -r paper_figures_and_tables/requirements.txt
```

They are pinned to NumPy `1.26.4` and Matplotlib `3.9.2`, matching the supplied
Conda environments. The table scripts use the Python standard library for CSV
files. PyTorch is not required merely to parse the textual `torch.Size(...)`
records produced by the experiments.

Set these placeholders to your own paths. `{results_path}` must be the parent
of result directories such as `alphaEst_randomPi_PiRange01-09/`,
`alphaEst_randomPi_PiRange005-015/`, and, for target-Pi experiments,
`rebt_ablation/`.

```bash
mkdir -p {figure_output_path} {table_output_path}
```

Only load pickle artifacts that were produced by this project or another
trusted source. Python pickle files can execute code while being loaded.

### 10.1 Generate the paper figures

Main comparison curves, including the constant BC and ILEED lines used in the
paper:

```bash
python3 paper_figures_and_tables/figure_main_reward_curves.py \
  --results-root {results_path} \
  --output-dir {figure_output_path}
```

This writes the eight reported PDFs for Ant, HalfCheetah, Swimmer, and Walker
under the two expertise ranges. Add `--include-hopper` to reproduce the two
Hopper PDFs that exist in the original figure directory but are not included
in the paper.

Analysis, bag-number, relabel/early-stop, and top-k plots:

```bash
python3 paper_figures_and_tables/figure_analysis_ablation_curves.py \
  --results-root {results_path} \
  --output-dir {figure_output_path} \
  --group all
```

The command also creates `abl_figs.pdf` directly as a four-panel Python figure.
The original panels were generated by `plot_reward_epochs.ipynb`, but the final
four-panel PDF was assembled in Apple Keynote. The new script removes that
manual assembly step.

Expert-query-count and target-Pi ablations:

```bash
python3 paper_figures_and_tables/figure_query_target_pi_ablations.py \
  --results-root {results_path} \
  --output-dir {figure_output_path} \
  --group all
```

Each figure program accepts `--range general`, `--range low`, or the default
`--range all`. The two ablation programs also accept one specific `--group` if
only part of the result tree is available; see `python3 SCRIPT.py --help`.

To recompute the offline BC/ILEED lines instead of using the exact constants
embedded in the original figure notebook, first generate their table and pass
it back to the main figure command:

```bash
python3 paper_figures_and_tables/table_offline_bc_ileed.py \
  --results-root {results_path} \
  --output-dir {table_output_path}

python3 paper_figures_and_tables/figure_main_reward_curves.py \
  --results-root {results_path} \
  --output-dir {figure_output_path} \
  --offline-summary {table_output_path}/table_offline_bc_ileed.csv
```

### 10.2 Generate the paper tables

Run the table programs independently according to the artifacts you have:

```bash
python3 paper_figures_and_tables/table_main_performance.py \
  --results-root {results_path} \
  --output-dir {table_output_path}

python3 paper_figures_and_tables/table_reward_ablations.py \
  --results-root {results_path} \
  --output-dir {table_output_path} \
  --group all

python3 paper_figures_and_tables/table_expertise_estimation.py \
  --results-root {results_path} \
  --output-dir {table_output_path} \
  --include-bag-ablation

python3 paper_figures_and_tables/table_relabel_metrics.py \
  --results-root {results_path} \
  --output-dir {table_output_path} \
  --include-query-variants

python3 paper_figures_and_tables/table_repeat_counts.py \
  --results-root {results_path} \
  --output-dir {table_output_path}

python3 paper_figures_and_tables/table_offline_bc_ileed.py \
  --results-root {results_path} \
  --output-dir {table_output_path} \
  --rl-seed-min 0 \
  --rl-seed-max 19

python3 paper_figures_and_tables/table_target_pi_classification.py \
  --results-root {results_path} \
  --output-dir {table_output_path}
```

The repeat-count script intentionally preserves the notebook convention: it
reads the first column of the last row in each `Pi_estimation_csv/*.csv` file
and reports `mean + 1` with the population standard deviation.

The original checkpoint-reward notebook stored its raw rewards directly in a
notebook rather than loading a result artifact. The cleaned replacement accepts
a CSV with columns `environment`, `quality`, and `reward`, where `quality` is
`optimal` or `suboptimal`:

```text
environment,quality,reward
Ant-v4,optimal,6997.4117
Ant-v4,suboptimal,113.4584
```

Generate the optimal, suboptimal, 50/50, and 10/90 dataset summaries with:

```bash
python3 paper_figures_and_tables/table_policy_dataset_rewards.py \
  --rewards-csv {policy_reward_csv} \
  --output-dir {table_output_path}
```

### 10.3 Required result artifacts

| Report type | Artifact read by the script |
| --- | --- |
| Online reward figures and reward tables | `eval_data_reward_step_after_training.pkl` inside each run directory |
| Offline BC/ILEED table | `ileed_scoring_model_evals.pkl` |
| Relabel tables | `rl_scoring_uu_learning_data.pkl` |
| Expertise and target-Pi classification tables | `opt_nonopt_trajs_alphaEst/*.txt` |
| Repeat-count table | `Pi_estimation_csv/*.csv` |
| Policy/dataset reward table | User-supplied `environment,quality,reward` CSV |

Missing directories, unmatched run names, incompatible training-step grids, and
missing artifact keys produce an error that identifies the expected path or
input. A table script writes CSV only after it has produced at least one row.

Run the reporting smoke tests from the repository root with:

```bash
python3 -m unittest discover -s tests -v
```

### 10.4 Mapping from scripts to reported figures

| Paper result | Generated output | Public script | Original source |
| --- | --- | --- | --- |
| Main reward curves (`fig:main_combined` and appendix panels) | `reward_{environment}__PiRange01-09.pdf`, `reward_{environment}__PiRange005-015.pdf` | `figure_main_reward_curves.py` | `plot_reward_epochs_bc_ileed.ipynb` |
| Relabel/early-stop and top-k composite (`fig:abl_rel_early_topk`) | `abl_figs.pdf` plus the four component PDFs | `figure_analysis_ablation_curves.py --group relabel-topk` | `plot_reward_epochs.ipynb`; final original composite was made in Keynote |
| Bag-number ablation (`fig:abl_bagNum`) | `reward_Ant-v4__*_ablations_bagNum.pdf` | `figure_analysis_ablation_curves.py --group bag-number` | `plot_reward_epochs.ipynb` |
| Method analysis (`fig:analy_combined`) | Eight `reward_{environment}__*_analysis.pdf` files | `figure_analysis_ablation_curves.py --group analysis` | `plot_reward_epochs.ipynb` |
| Expert-query ablation (`fig:abl_expCheckNum`) | `reward_Ant-v4__*_ablations_expCheckNum.pdf` | `figure_query_target_pi_ablations.py --group expert-check` | `plot_reward_epochs_rebt.ipynb` |
| Target-Pi ablation (`fig:abl_testPi`) | `reward_Ant-v4__*_ablations_piTest.pdf` | `figure_query_target_pi_ablations.py --group target-pi` | `plot_reward_epochs_rebt.ipynb` |

The original `plot_reward_epochs_bc_ileed_10.ipynb` is byte-for-byte identical
to `plot_reward_epochs_bc_ileed.ipynb`, so it is not copied as another script.
The three `*_temp.pdf` files and the trajectory/reward and ILEED-scatter
diagnostics were not reported and are intentionally excluded.

### 10.5 Mapping from scripts to reported tables

| Paper table label(s) | Public script | Original source |
| --- | --- | --- |
| `tab:main_exp` | `table_main_performance.py` and `table_offline_bc_ileed.py` | `plot_reward_epochs_bc_ileed.ipynb`, `plot_reward_epochs_rebt_ileed_10.ipynb` |
| `tab:error_only`, `tab:expertise_est`, `tab:expertise_est_details`, `tab:demo_num_expertise` | `table_expertise_estimation.py` | `plot_expertise_level.ipynb` |
| `tab:acc_prec_stages`, `tab:acc_label_improvement` | `table_relabel_metrics.py` | `plot_relabel.ipynb` |
| `tab:acc_prec_stages_exp_ask_num` | `table_relabel_metrics.py --include-query-variants` | `plot_relabel_acc_rebt.ipynb` |
| `tab:demo_num_r`, `tab:ab_relabel_early`, `tab:ab_topk` | `table_reward_ablations.py` | `plot_reward_epochs.ipynb` |
| `tab:query_ablation`, `tab:alpha_ablation` | `table_reward_ablations.py` | `plot_reward_epochs_rebt.ipynb` |
| `tab:repeat_num` | `table_repeat_counts.py` | `plot_repeat_num.ipynb` |
| `tab:bc_ileed_scoring` | `table_offline_bc_ileed.py` | final 20-seed `plot_reward_epochs_rebt_ileed_10.ipynb` |
| `tab:antv4_pi_test_results_acc_prec` | `table_target_pi_classification.py` | `mujo_02_trajs_uu_classinfo_rebt_normal.ipynb`, `mujo_02_trajs_uu_classinfo_rebt_low.ipynb` |
| `tab:rl_policy` and source-dataset statistics used by `tab:main_exp` | `table_policy_dataset_rewards.py` | `check_points_rewards.ipynb` |

`tab:exp_true`, conceptual method-comparison tables, and training-parameter
tables are analytical or manually authored LaTeX tables; no plotting or result
aggregation script generated them.

## 11. Common problems

- **`FileNotFoundError` for the original trajectory pickle:** replace
  `{dataset_path}` and confirm the file is in
  `../results/opt_nonopt_trajs_noReplacing/` with the exact expected name.
- **`FileNotFoundError` ending in `_newneg.npz`, `_NetSeed_0.pkl`, or
  `_alpha_05.pkl`:** run the filename-normalization command in Section 5.
- **A baseline cannot find AlphaEst data:** copy the NPZ files to the older
  compatibility root as shown in Section 7.
- **An index/default error:** pass `--env 0` through `--env 4` and use
  `--method 5` for expertise estimation.
- **Out-of-memory or process errors:** reduce `--num_envs` from `32` to `16`,
  `8`, `4`, `2`, or `1`, and avoid running too many experiments concurrently.
- **No CUDA device:** the code falls back to CPU, but training will take longer.
