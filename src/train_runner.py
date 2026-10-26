"""
Pre-trains a high-quality Q-Learning agent and generates baseline comparison results.
Saves model to models/trained_agent.json and results to outputs/results/benchmark_results.json.
"""

import os
import json
import numpy as np

from src.environment import TrafficEnvironment
from src.q_learning import QLearningAgent
from src.baseline import FixedTimeController, HeuristicQueueController, evaluate_controller
from src.metrics import create_comparison_table, plot_training_curves, plot_comparison_bar


def run_pretraining_and_benchmarks():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("docs/images", exist_ok=True)

    print("Training Q-Learning agent...")
    env = TrafficEnvironment(max_steps=120, seed=42, scenario="balanced")
    agent = QLearningAgent(
        learning_rate=0.12,
        discount_factor=0.90,
        epsilon=1.0,
        epsilon_min=0.04,
        epsilon_decay=0.985,
        seed=42
    )

    history = agent.train(env, num_episodes=200)
    agent.save_model("models/trained_agent.json")
    print("Agent trained and saved to models/trained_agent.json")

    # Run Benchmark Evaluation
    print("Running evaluation benchmarks (Fixed-Time vs Q-Learning)...")
    eval_env = TrafficEnvironment(max_steps=120, scenario="balanced")
    baseline_ctrl = FixedTimeController(green_duration=6)

    baseline_metrics = evaluate_controller(baseline_ctrl, eval_env, num_episodes=30, base_seed=1000)
    rl_metrics = evaluate_controller(agent, eval_env, num_episodes=30, base_seed=1000)

    comparison_df = create_comparison_table(baseline_metrics, rl_metrics)
    print("\n--- Benchmark Comparison Table ---")
    print(comparison_df.to_string(index=False))

    benchmark_data = {
        "baseline": baseline_metrics,
        "rl": rl_metrics,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "training_history": history
    }

    with open("outputs/results/benchmark_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print("Saved benchmark results to outputs/results/benchmark_results.json")

    # Save benchmark charts
    fig_train = plot_training_curves(history)
    fig_train.savefig("outputs/results/training_curves.png", bbox_inches="tight")
    print("Saved outputs/results/training_curves.png")

    fig_comp = plot_comparison_bar(baseline_metrics, rl_metrics)
    fig_comp.savefig("outputs/results/comparison_bar.png", bbox_inches="tight")
    print("Saved outputs/results/comparison_bar.png")


if __name__ == "__main__":
    run_pretraining_and_benchmarks()
