"""
Metrics processing, statistics calculation, and plotting utilities.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_percentage_change(baseline_val: float, test_val: float, lower_is_better: bool = True) -> float:
    """
    Computes improvement percentage.
    If lower_is_better (e.g. waiting time, queue): (baseline - test) / baseline * 100.
    If higher_is_better (e.g. reward, throughput): (test - baseline) / abs(baseline) * 100.
    """
    if abs(baseline_val) < 1e-6:
        return 0.0
    if lower_is_better:
        return ((baseline_val - test_val) / baseline_val) * 100.0
    else:
        return ((test_val - baseline_val) / abs(baseline_val)) * 100.0


def create_comparison_table(baseline_results: Dict, rl_results: Dict) -> pd.DataFrame:
    """Creates a side-by-side performance summary DataFrame."""
    metrics = [
        ("Average Waiting Time (veh-steps)", "mean_waiting_time", "std_waiting_time", True),
        ("Average Queue Length (vehicles)", "mean_avg_queue", "std_avg_queue", True),
        ("Total Vehicles Served (throughput)", "mean_served", "std_served", False),
        ("Cumulative Reward", "mean_reward", "std_reward", False),
        ("Signal Switches per Episode", "mean_switches", "std_switches", True)
    ]

    records = []
    for name, mean_key, std_key, lower_better in metrics:
        b_mean = baseline_results[mean_key]
        b_std = baseline_results[std_key]
        rl_mean = rl_results[mean_key]
        rl_std = rl_results[std_key]
        pct = compute_percentage_change(b_mean, rl_mean, lower_is_better=lower_better)

        records.append({
            "Metric": name,
            "Fixed-Time Baseline": f"{b_mean:.2f} ± {b_std:.2f}",
            "Q-Learning Agent": f"{rl_mean:.2f} ± {rl_std:.2f}",
            "Improvement (%)": f"{'+' if pct > 0 else ''}{pct:.1f}%"
        })

    return pd.DataFrame(records)


def plot_training_curves(training_history: Dict) -> plt.Figure:
    """Generates a professional 2x2 multi-panel training progress figure."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    episodes = np.arange(1, len(training_history["episode_rewards"]) + 1)
    rewards = np.array(training_history["episode_rewards"])

    # 1. Episode Reward with Moving Average
    ax1 = axs[0, 0]
    ax1.plot(episodes, rewards, color="#94A3B8", alpha=0.5, label="Raw Reward")
    if len(rewards) >= 10:
        window = min(15, len(rewards))
        ma = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        ax1.plot(episodes, ma, color="#1E3A8A", linewidth=2.2, label=f"{window}-Ep Moving Avg")
    ax1.set_title("Episode Cumulative Reward", fontsize=11, fontweight="bold", color="#0F172A")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 2. Average Queue Length
    ax2 = axs[0, 1]
    queues = training_history["avg_queues"]
    ax2.plot(episodes, queues, color="#DC2626", linewidth=1.8)
    ax2.set_title("Average Intersection Queue Length", fontsize=11, fontweight="bold", color="#0F172A")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Queue Length (vehicles)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Cumulative Waiting Time
    ax3 = axs[1, 0]
    waiting = training_history["avg_waiting_times"]
    ax3.plot(episodes, waiting, color="#D97706", linewidth=1.8)
    ax3.set_title("Total Cumulative Delay / Waiting Time", fontsize=11, fontweight="bold", color="#0F172A")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Delay (veh-steps)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Epsilon Exploration Decay
    ax4 = axs[1, 1]
    epsilons = training_history["epsilons"]
    ax4.plot(episodes, epsilons, color="#059669", linewidth=1.8)
    ax4.set_title("Exploration Rate (ε) Decay", fontsize=11, fontweight="bold", color="#0F172A")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon (ε)")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_comparison_bar(baseline_results: Dict, rl_results: Dict) -> plt.Figure:
    """Generates side-by-side bar chart comparison with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    controllers = ["Fixed-Time Baseline", "Trained Q-Learning"]
    colors = ["#64748B", "#2563EB"]

    # Waiting Time Comparison
    wt_means = [baseline_results["mean_waiting_time"], rl_results["mean_waiting_time"]]
    wt_stds = [baseline_results["std_waiting_time"], rl_results["std_waiting_time"]]
    bars1 = ax1.bar(controllers, wt_means, yerr=wt_stds, capsize=6, color=colors, width=0.55)
    ax1.set_title("Mean Total Waiting Time (Lower is Better)", fontsize=11, fontweight="bold", color="#0F172A")
    ax1.set_ylabel("Waiting Time (veh-steps)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval/2.0, f"{yval:.1f}", ha="center", va="center", color="white", fontweight="bold")

    # Queue Length Comparison
    q_means = [baseline_results["mean_avg_queue"], rl_results["mean_avg_queue"]]
    q_stds = [baseline_results["std_avg_queue"], rl_results["std_avg_queue"]]
    bars2 = ax2.bar(controllers, q_means, yerr=q_stds, capsize=6, color=colors, width=0.55)
    ax2.set_title("Mean Queue Length (Lower is Better)", fontsize=11, fontweight="bold", color="#0F172A")
    ax2.set_ylabel("Queue Length (vehicles)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval/2.0, f"{yval:.1f}", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    return fig
