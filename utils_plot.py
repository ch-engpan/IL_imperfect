import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None):
    x, y, yerr = np.array(x), np.array(y), np.array(yerr)
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def plot_reward_stepLen(stepnum_list, mean_reward_list, std_reward_list, mean_stepLen_list, std_stepLen_list, env_name, save_dir, file_suffix_name):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # First subplot: Mean Reward with Standard Deviation
    axs[0].set_title('Reward - Mean with Standard Deviation ---' + env_name)
    plot_errorfill(stepnum_list, mean_reward_list, std_reward_list, color="blue", ax=axs[0])
    axs[0].set_xlabel('Timesteps')
    axs[0].set_ylabel('Reward')
    axs[0].grid(True)

    # Second subplot: Mean Step Length with Standard Deviation
    axs[1].set_title('Step Length - Mean with Standard Deviation ---' + env_name)
    plot_errorfill(stepnum_list, mean_stepLen_list, std_stepLen_list, color="blue", ax=axs[1])
    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Step Length')
    axs[1].grid(True)

    plt.tight_layout()
    save_dir = Path(save_dir + "/reward_stepLen_plots_" + file_suffix_name + ".pdf")
    print("save_dir: ", save_dir)
    plt.savefig(save_dir)
    # plt.show()
    plt.close()


def plot_evals_loglik_offlineMethods(
        all_evals, all_log_likelihood,
        best_eval, best_log,
        true_best_eval, true_best_log,
        save_dir, file_suffix_name):

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # ============================
    # Subplot 1: Scatter Plot
    # ============================
    axs[0].set_title("Scatter: all_evals vs all_log_likelihood")
    print("all_evals:", np.array(all_evals).shape)
    print("all_log_likelihood:", np.array(all_log_likelihood).shape)
    axs[0].scatter(all_evals, all_log_likelihood, alpha=0.6)
    axs[0].set_xlabel("all_evals")
    axs[0].set_ylabel("all_log_likelihood")
    axs[0].grid(True)

    # ---- Highlight best_eval & best_log ----
    axs[0].scatter([best_eval], [best_log], s=120, marker="o", color="red")
    axs[0].text(
        best_eval, 
        best_log + 0.02 * (max(all_log_likelihood) - min(all_log_likelihood)),   # offset upward
        f"best_eval={best_eval:.4f}\nbest_log={best_log:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="red",
        weight="bold"
    )

    # ---- Highlight true_best_eval & true_best_log (text BELOW) ----
    axs[0].scatter([true_best_eval], [true_best_log], s=120, marker="x", color="green")
    axs[0].text(
        true_best_eval,
        true_best_log - 0.02 * (max(all_log_likelihood) - min(all_log_likelihood)),   # offset downward
        f"true_best_eval={true_best_eval:.4f}\ntrue_best_log={true_best_log:.4f}",
        ha="center",
        va="top",
        fontsize=9,
        color="green",
        weight="bold"
    )

    # ============================
    # Subplot 2: Line Plot
    # ============================
    axs[1].set_title("Line Plot: all_evals and all_log_likelihood")
    axs[1].plot(all_evals, label="all_evals")
    axs[1].plot(all_log_likelihood, label="all_log_likelihood")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    # ============================
    # Save Figure
    # ============================
    plt.tight_layout()
    save_path = Path(save_dir) / f"evals_loglik_plot_{file_suffix_name}.pdf"
    print("save_dir:", save_path)
    plt.savefig(save_path)
    plt.close()

