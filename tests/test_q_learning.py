"""
Unit tests for Q-Learning agent: update formula, terminal states,
exploration decay, policy derivation, and model saving/loading.
"""

import os
import tempfile
import pytest
import numpy as np
from src.q_learning import QLearningAgent
from src.environment import TrafficEnvironment
from src.mdp_model import NUM_STATES, NUM_ACTIONS


def test_q_table_initialization():
    """Verify Q-table is properly initialized to zeros."""
    agent = QLearningAgent(num_states=18, num_actions=2)
    assert agent.q_table.shape == (18, 2)
    assert np.all(agent.q_table == 0.0)


def test_q_learning_update_formula():
    """Verify Q(s,a) update against hand-calculated Bellman temporal-difference target."""
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9, epsilon=0.0)
    # Set known Q values
    agent.q_table[0, 0] = 2.0
    agent.q_table[1, 0] = 4.0
    agent.q_table[1, 1] = 6.0  # max Q(s', a') = 6.0

    state = 0
    action = 0
    reward = 1.0
    next_state = 1
    done = False

    # Target = reward + gamma * max_a' Q(s', a') = 1.0 + 0.9 * 6.0 = 6.4
    # TD error = 6.4 - 2.0 = 4.4
    # New Q = 2.0 + 0.5 * 4.4 = 4.2
    td_error = agent.update(state, action, reward, next_state, done)

    assert np.isclose(td_error, 4.4)
    assert np.isclose(agent.q_table[0, 0], 4.2)


def test_terminal_state_update():
    """Verify terminal transition uses future value = 0 in TD update."""
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9)
    agent.q_table[0, 0] = 2.0
    agent.q_table[1, 1] = 100.0  # Should be ignored because done=True

    state = 0
    action = 0
    reward = -5.0
    next_state = 1
    done = True

    # Target = reward = -5.0
    # TD error = -5.0 - 2.0 = -7.0
    # New Q = 2.0 + 0.5 * (-7.0) = -1.5
    td_error = agent.update(state, action, reward, next_state, done)

    assert np.isclose(td_error, -7.0)
    assert np.isclose(agent.q_table[0, 0], -1.5)


def test_epsilon_decay():
    """Verify epsilon decays correctly down to epsilon_min."""
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.5)
    agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.25)
    agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.125)
    agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.1)  # clamped at epsilon_min


def test_model_save_and_load():
    """Verify model persistence to JSON and restoration."""
    agent1 = QLearningAgent(learning_rate=0.15, discount_factor=0.85)
    agent1.q_table[2, 1] = -12.34
    agent1.q_table[5, 0] = 7.89

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        agent1.save_model(tmp_path)
        agent2 = QLearningAgent()
        agent2.load_model(tmp_path)

        assert np.isclose(agent2.alpha, 0.15)
        assert np.isclose(agent2.gamma, 0.85)
        assert np.isclose(agent2.q_table[2, 1], -12.34)
        assert np.isclose(agent2.q_table[5, 0], 7.89)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
