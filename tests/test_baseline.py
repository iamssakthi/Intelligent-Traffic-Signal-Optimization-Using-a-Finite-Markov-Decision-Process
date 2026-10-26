"""
Unit tests for Baseline controllers and benchmarking framework.
"""

import pytest
import numpy as np
from src.baseline import FixedTimeController, HeuristicQueueController, evaluate_controller
from src.environment import TrafficEnvironment
from src.mdp_model import ACTION_KEEP, ACTION_SWITCH


def test_fixed_time_controller_timing():
    """Verify FixedTimeController requests SWITCH exactly every green_duration steps."""
    ctrl = FixedTimeController(green_duration=4)

    assert ctrl.select_action(0) == ACTION_KEEP
    assert ctrl.select_action(0) == ACTION_KEEP
    assert ctrl.select_action(0) == ACTION_KEEP
    assert ctrl.select_action(0) == ACTION_SWITCH
    # resets and repeats
    assert ctrl.select_action(0) == ACTION_KEEP


def test_evaluation_pipeline():
    """Verify evaluation pipeline runs across multiple episodes and produces valid statistics."""
    env = TrafficEnvironment(max_steps=20, seed=42)
    ctrl = FixedTimeController(green_duration=5)
    results = evaluate_controller(ctrl, env, num_episodes=5, base_seed=100)

    assert "mean_waiting_time" in results
    assert "mean_avg_queue" in results
    assert "mean_served" in results
    assert "mean_reward" in results
    assert results["mean_waiting_time"] > 0
    assert results["mean_avg_queue"] >= 0
    assert len(results["raw_waiting_times"]) == 5
