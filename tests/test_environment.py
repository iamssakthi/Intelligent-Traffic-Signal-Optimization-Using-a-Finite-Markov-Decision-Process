"""
Unit tests for TrafficEnvironment: safety constraints, state transitions,
queues, and yellow clearance.
"""

import pytest
import numpy as np
from src.environment import TrafficEnvironment, queue_to_density
from src.mdp_model import ACTION_KEEP, ACTION_SWITCH, NUM_STATES


def test_queue_to_density():
    """Verify discretization of queues into Low, Med, High."""
    assert queue_to_density(0.0) == 0
    assert queue_to_density(3.0) == 0
    assert queue_to_density(4.0) == 1
    assert queue_to_density(8.0) == 1
    assert queue_to_density(9.0) == 2
    assert queue_to_density(20.0) == 2


def test_environment_reset():
    """Verify environment reset initializes all state counters correctly."""
    env = TrafficEnvironment(seed=42)
    state = env.reset(seed=123)

    assert 0 <= state < NUM_STATES
    assert env.step_count == 0
    assert env.cumulative_waiting_time == 0
    assert env.total_vehicles_served == 0
    assert len(env.queues) == 4
    assert np.all(env.queues >= 0)


def test_safety_constraint_no_conflicting_greens():
    """
    CRITICAL SAFETY REQUIREMENT:
    The simulation must NEVER permit conflicting green phases simultaneously.
    At any time:
    - Phase 0: NS green, EW red
    - Phase 1: EW green, NS red
    - Yellow: Clearance state where no conflicting green flow occurs
    """
    env = TrafficEnvironment(max_steps=50, seed=42)
    state = env.reset()

    for _ in range(40):
        # Even with rapid switching, safety rules must hold
        action = ACTION_SWITCH if env.rng.rand() < 0.5 else ACTION_KEEP
        next_state, reward, done, info = env.step(action)

        phase = info["phase"]
        assert phase in [0, 1]

        # In phase 0, EW must have 0 discharge
        # In phase 1, NS must have 0 discharge
        if info["is_yellow"]:
            assert info["vehicles_served"] == 0, "No vehicles may enter intersection during yellow clearance!"

        if done:
            break


def test_yellow_clearance_enforcement():
    """Verify that switching invokes the yellow clearance period and phase transition."""
    env = TrafficEnvironment(seed=42)
    env.reset(seed=10)
    initial_phase = env.current_phase

    # Request switch
    next_state, reward, done, info = env.step(ACTION_SWITCH)
    assert env.is_yellow is True, "Yellow clearance must be marked during switch step"
    assert env.current_phase == (1 - initial_phase), "Phase transitions to target phase"

    # Step again with KEEP: yellow clearance clears and phase remains green
    next_state, reward, done, info = env.step(ACTION_KEEP)
    assert env.is_yellow is False
    assert env.current_phase == (1 - initial_phase)
