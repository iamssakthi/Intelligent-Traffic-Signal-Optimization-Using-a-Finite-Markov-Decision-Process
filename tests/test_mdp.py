"""
Unit tests for Finite MDP model, state encoding, transition matrix normalization,
and Bellman equation solvers.
"""

import pytest
import numpy as np
from src.mdp_model import (
    TrafficMDP, encode_state, decode_state, state_to_string,
    NUM_STATES, NUM_ACTIONS, NUM_PHASES, NUM_DENSITY_LEVELS,
    ACTION_KEEP, ACTION_SWITCH
)


def test_state_encoding_decoding_bijection():
    """Verify that encode_state and decode_state form a complete bijection for all 18 states."""
    seen_states = set()
    for phase in range(NUM_PHASES):
        for act_d in range(NUM_DENSITY_LEVELS):
            for opp_d in range(NUM_DENSITY_LEVELS):
                s = encode_state(phase, act_d, opp_d)
                assert 0 <= s < NUM_STATES
                p_dec, a_dec, o_dec = decode_state(s)
                assert (p_dec, a_dec, o_dec) == (phase, act_d, opp_d)
                seen_states.add(s)

    assert len(seen_states) == NUM_STATES == 18


def test_transition_probability_normalization():
    """Verify that every row of P(s' | s, a) sums to exactly 1.0 within numerical tolerance."""
    mdp = TrafficMDP()
    assert mdp.P.shape == (NUM_STATES, NUM_ACTIONS, NUM_STATES)

    for s in range(NUM_STATES):
        for a in range(NUM_ACTIONS):
            row_sum = np.sum(mdp.P[s, a])
            assert np.isclose(row_sum, 1.0, atol=1e-5), f"State {s}, Action {a} row sum is {row_sum}"
            # Probabilities must be non-negative
            assert np.all(mdp.P[s, a] >= 0.0)


def test_bellman_value_iteration():
    """Verify Value Iteration solves Bellman Optimality Equation and returns valid values."""
    mdp = TrafficMDP(gamma=0.90)
    V_star, Q_star, pi_star = mdp.value_iteration(theta=1e-6)

    assert V_star.shape == (NUM_STATES,)
    assert Q_star.shape == (NUM_STATES, NUM_ACTIONS)
    assert pi_star.shape == (NUM_STATES,)

    # V*(s) must equal max_a Q*(s, a)
    assert np.allclose(V_star, np.max(Q_star, axis=1), atol=1e-5)

    # In traffic control with penalties, state values should be negative (costs)
    assert np.all(V_star < 0.0)


def test_bellman_breakdown():
    """Verify Bellman breakdown returns required educational fields for UI explorer."""
    mdp = TrafficMDP()
    breakdown = mdp.get_bellman_breakdown(state_idx=0)

    assert "state_idx" in breakdown
    assert "actions" in breakdown
    assert ACTION_KEEP in breakdown["actions"]
    assert ACTION_SWITCH in breakdown["actions"]
    assert "best_action" in breakdown
    assert "decision_justification" in breakdown

    # Check breakdown consistency for action 0:
    act0 = breakdown["actions"][ACTION_KEEP]
    assert np.isclose(
        act0["total_action_value"],
        act0["immediate_reward"] + act0["discounted_future_value"],
        atol=1e-4
    )
