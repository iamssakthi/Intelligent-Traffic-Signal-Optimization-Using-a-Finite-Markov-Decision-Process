"""
Tabular Q-Learning Agent for Intelligent Traffic Signal Control.
Implements epsilon-greedy exploration, Bellman temporal-difference updates,
exploration decay, policy extraction, and model persistence.
"""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.mdp_model import NUM_STATES, NUM_ACTIONS, ACTION_KEEP, ACTION_SWITCH, state_to_string


class QLearningAgent:
    """
    Tabular Q-Learning Agent.
    Updates state-action values according to:
      Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s,a) ]
    """

    def __init__(
        self,
        num_states: int = NUM_STATES,
        num_actions: int = NUM_ACTIONS,
        learning_rate: float = 0.10,
        discount_factor: float = 0.90,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.RandomState(seed)

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)

        # Training history logs
        self.training_history = {
            "episode_rewards": [],
            "avg_queues": [],
            "avg_waiting_times": [],
            "epsilons": [],
            "vehicles_served": [],
            "switches": []
        }

    def select_action(self, state: int, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        If explore=False, selects greedily argmax_a Q(s, a).
        """
        if explore and self.rng.rand() < self.epsilon:
            return int(self.rng.choice(self.num_actions))

        # Break ties randomly among max Q-values
        q_s = self.q_table[state]
        max_q = np.max(q_s)
        best_actions = np.where(q_s == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Perform single Q-learning TD update.
        Returns the TD error.
        """
        current_q = self.q_table[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state_values(self) -> np.ndarray:
        """Derive state-value function V(s) = max_a Q(s, a)."""
        return np.max(self.q_table, axis=1)

    def get_policy(self) -> np.ndarray:
        """Derive greedy deterministic policy pi(s) = argmax_a Q(s, a)."""
        return np.argmax(self.q_table, axis=1)

    def train_episode(self, env) -> Dict:
        """Run and train on a single episode in the environment."""
        state = env.reset()
        done = False
        ep_reward = 0.0
        queue_samples = []

        while not done:
            action = self.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)

            self.update(state, action, reward, next_state, done)

            ep_reward += reward
            queue_samples.append(np.sum(info["queues"]))
            state = next_state

        self.decay_epsilon()

        avg_queue = float(np.mean(queue_samples)) if queue_samples else 0.0
        summary = {
            "reward": float(ep_reward),
            "avg_queue": avg_queue,
            "cumulative_waiting_time": int(env.cumulative_waiting_time),
            "vehicles_served": int(env.total_vehicles_served),
            "switches": int(env.switch_count),
            "epsilon": float(self.epsilon)
        }

        self.training_history["episode_rewards"].append(summary["reward"])
        self.training_history["avg_queues"].append(summary["avg_queue"])
        self.training_history["avg_waiting_times"].append(summary["cumulative_waiting_time"])
        self.training_history["epsilons"].append(summary["epsilon"])
        self.training_history["vehicles_served"].append(summary["vehicles_served"])
        self.training_history["switches"].append(summary["switches"])

        return summary

    def train(self, env, num_episodes: int = 150, callback=None) -> Dict:
        """
        Train agent for num_episodes.
        Optionally takes a callback(ep, summary) to report progress to Streamlit or CLI.
        """
        for ep in range(1, num_episodes + 1):
            summary = self.train_episode(env)
            if callback is not None:
                callback(ep, summary)
        return self.training_history

    def save_model(self, filepath: str):
        """Save Q-table and hyperparameters to a JSON file."""
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table": self.q_table.tolist(),
            "training_history": self.training_history
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_model(self, filepath: str):
        """Load Q-table and hyperparameters from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_min = data.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.q_table = np.array(data["q_table"], dtype=np.float64)
        if "training_history" in data:
            self.training_history = data["training_history"]
