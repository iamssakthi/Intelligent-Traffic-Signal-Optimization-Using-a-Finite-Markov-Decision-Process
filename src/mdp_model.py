"""
Finite Markov Decision Process (MDP) model for Intelligent Traffic Signal Optimization.
Defines the state space, action space, transition probabilities P(s'|s,a),
reward function R(s,a), and Bellman equation solvers.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


# Constant definitions
NUM_PHASES = 2         # 0: North-South Green (NS), 1: East-West Green (EW)
NUM_DENSITY_LEVELS = 3 # 0: Low, 1: Medium, 2: High
NUM_STATES = NUM_PHASES * NUM_DENSITY_LEVELS * NUM_DENSITY_LEVELS # 18 states
NUM_ACTIONS = 2        # 0: KEEP current green, 1: SWITCH to other phase

ACTION_KEEP = 0
ACTION_SWITCH = 1
ACTION_NAMES = {0: "KEEP (Stay Green)", 1: "SWITCH (Change Phase)"}

PHASE_NAMES = {0: "North-South Green (EW Red)", 1: "East-West Green (NS Red)"}
PHASE_SHORT = {0: "NS", 1: "EW"}
DENSITY_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def encode_state(phase: int, active_density: int, opposing_density: int) -> int:
    """Encode (phase, active_density, opposing_density) into state index 0..17."""
    return int(phase * 9 + active_density * 3 + opposing_density)


def decode_state(state_idx: int) -> Tuple[int, int, int]:
    """Decode state index 0..17 into (phase, active_density, opposing_density)."""
    phase = state_idx // 9
    rem = state_idx % 9
    active_density = rem // 3
    opposing_density = rem % 3
    return phase, active_density, opposing_density


def state_to_string(state_idx: int) -> str:
    """Return a descriptive human-readable label for the state."""
    phase, act_d, opp_d = decode_state(state_idx)
    return f"[{state_idx:02d}] {PHASE_SHORT[phase]} Green | Active: {DENSITY_NAMES[act_d]} | Opposing: {DENSITY_NAMES[opp_d]}"


class TrafficMDP:
    """
    Finite Markov Decision Process M = (S, A, P, R, gamma) for a 4-way intersection.
    """

    def __init__(self, gamma: float = 0.90, switch_penalty: float = 2.0):
        self.num_states = NUM_STATES
        self.num_actions = NUM_ACTIONS
        self.gamma = gamma
        self.switch_penalty = switch_penalty

        # Theoretical Transition Matrix: P[s, a, s']
        self.P = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)
        # Expected Reward Matrix: R[s, a]
        self.R = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        self._build_transition_and_reward_models()

    def _build_transition_and_reward_models(self):
        """
        Constructs the documented stochastic transition model and reward function.
        Ensures valid probabilities: sum_{s'} P(s' | s, a) = 1.0 for all s, a.
        """
        for s in range(self.num_states):
            phase, act_d, opp_d = decode_state(s)

            # Base queue penalty: heavier penalty for vehicles forced to wait on red
            # plus total congestion cost
            congestion_cost = -(act_d * 1.5 + opp_d * 3.0)

            # -------------------------------------------------------------
            # ACTION 0: KEEP CURRENT GREEN PHASE
            # -------------------------------------------------------------
            # Phase remains the same.
            # Active direction is moving: density tends to decrease or stay low.
            # Opposing direction is stopped: density tends to increase or stay high.
            next_phase_keep = phase

            # Active density transitions: [P(stay/decrease), P(increase)]
            p_act_keep = np.zeros(NUM_DENSITY_LEVELS)
            if act_d == 0:    # Low
                p_act_keep[0] = 0.75  # stays Low
                p_act_keep[1] = 0.20  # moves to Med
                p_act_keep[2] = 0.05  # jumps to High
            elif act_d == 1:  # Medium
                p_act_keep[0] = 0.55  # drains to Low
                p_act_keep[1] = 0.35  # stays Med
                p_act_keep[2] = 0.10  # grows to High
            else:             # High
                p_act_keep[0] = 0.15  # drains to Low
                p_act_keep[1] = 0.60  # drains to Med
                p_act_keep[2] = 0.25  # stays High

            # Opposing density transitions (stopped at red):
            p_opp_keep = np.zeros(NUM_DENSITY_LEVELS)
            if opp_d == 0:    # Low
                p_opp_keep[0] = 0.40  # stays Low
                p_opp_keep[1] = 0.45  # accumulates to Med
                p_opp_keep[2] = 0.15  # accumulates to High
            elif opp_d == 1:  # Medium
                p_opp_keep[0] = 0.05  # rarely decreases
                p_opp_keep[1] = 0.45  # stays Med
                p_opp_keep[2] = 0.50  # grows to High
            else:             # High
                p_opp_keep[0] = 0.00
                p_opp_keep[1] = 0.15
                p_opp_keep[2] = 0.85  # stays saturated High

            # Combine independent distributions for P(s' | s, KEEP)
            for next_act in range(NUM_DENSITY_LEVELS):
                for next_opp in range(NUM_DENSITY_LEVELS):
                    prob = p_act_keep[next_act] * p_opp_keep[next_opp]
                    s_prime = encode_state(next_phase_keep, next_act, next_opp)
                    self.P[s, ACTION_KEEP, s_prime] += prob

            # Reward for KEEP:
            # High reward if active traffic is heavy and opposing traffic is low.
            # Severe penalty if opposing traffic is high while active is empty (wasting green).
            throughput_bonus = 2.0 if act_d > 0 else -1.0
            wasted_green_penalty = -4.0 if (act_d == 0 and opp_d == 2) else 0.0
            self.R[s, ACTION_KEEP] = congestion_cost + throughput_bonus + wasted_green_penalty

            # -------------------------------------------------------------
            # ACTION 1: SWITCH TO OPPOSITE GREEN PHASE
            # -------------------------------------------------------------
            # Phase flips: 0 -> 1 or 1 -> 0.
            # In the new phase:
            # New active was previously opposing (opp_d)
            # New opposing was previously active (act_d)
            # During switch, a yellow transition occurs, slightly reducing immediate throughput
            next_phase_switch = 1 - phase

            # New active (was opposing):
            p_act_switch = np.zeros(NUM_DENSITY_LEVELS)
            if opp_d == 0:    # was Low
                p_act_switch[0] = 0.70
                p_act_switch[1] = 0.25
                p_act_switch[2] = 0.05
            elif opp_d == 1:  # was Med
                p_act_switch[0] = 0.35
                p_act_switch[1] = 0.50
                p_act_switch[2] = 0.15
            else:             # was High
                p_act_switch[0] = 0.10
                p_act_switch[1] = 0.50
                p_act_switch[2] = 0.40

            # New opposing (was active, now stopped at red):
            p_opp_switch = np.zeros(NUM_DENSITY_LEVELS)
            if act_d == 0:    # was Low
                p_opp_switch[0] = 0.60
                p_opp_switch[1] = 0.30
                p_opp_switch[2] = 0.10
            elif act_d == 1:  # was Med
                p_opp_switch[0] = 0.10
                p_opp_switch[1] = 0.60
                p_opp_switch[2] = 0.30
            else:             # was High
                p_opp_switch[0] = 0.05
                p_opp_switch[1] = 0.25
                p_opp_switch[2] = 0.70

            for next_act in range(NUM_DENSITY_LEVELS):
                for next_opp in range(NUM_DENSITY_LEVELS):
                    prob = p_act_switch[next_act] * p_opp_switch[next_opp]
                    s_prime = encode_state(next_phase_switch, next_act, next_opp)
                    self.P[s, ACTION_SWITCH, s_prime] += prob

            # Reward for SWITCH:
            # Incurs switch penalty (yellow clearance / delay cost)
            # Desirable if opposing traffic was High and active was Low
            relief_bonus = 3.5 if (opp_d == 2 and act_d <= 1) else (1.5 if opp_d > act_d else -2.5)
            self.R[s, ACTION_SWITCH] = congestion_cost - self.switch_penalty + relief_bonus

        # Normalize any numerical float imprecision to guarantee sum = 1.0
        for s in range(self.num_states):
            for a in range(self.num_actions):
                row_sum = np.sum(self.P[s, a])
                if row_sum > 0:
                    self.P[s, a] /= row_sum

    def bellman_expectation_evaluation(self, policy: np.ndarray, theta: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
        """
        Solves the Bellman Expectation Equation iteratively for a given policy pi(a|s):
        V^{pi}(s) = sum_a pi(a|s) [ R(s,a) + gamma sum_{s'} P(s'|s,a) V^{pi}(s') ]
        """
        V = np.zeros(self.num_states, dtype=np.float64)
        for _ in range(max_iter):
            delta = 0.0
            V_new = np.zeros_like(V)
            for s in range(self.num_states):
                v_s = 0.0
                for a in range(self.num_actions):
                    prob_a = policy[s, a]
                    if prob_a > 0:
                        future_val = np.sum(self.P[s, a] * V)
                        q_sa = self.R[s, a] + self.gamma * future_val
                        v_s += prob_a * q_sa
                delta = max(delta, abs(v_s - V[s]))
                V_new[s] = v_s
            V = V_new
            if delta < theta:
                break
        return V

    def value_iteration(self, theta: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solves the Bellman Optimality Equation via Value Iteration:
        V*(s) = max_a [ R(s,a) + gamma sum_{s'} P(s'|s,a) V*(s') ]
        Returns:
            V_star: optimal state values (num_states,)
            Q_star: optimal state-action values (num_states, num_actions)
            pi_star: optimal deterministic policy (num_states,)
        """
        V = np.zeros(self.num_states, dtype=np.float64)
        Q = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        for _ in range(max_iter):
            delta = 0.0
            V_new = np.zeros_like(V)
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    future_val = np.sum(self.P[s, a] * V)
                    Q[s, a] = self.R[s, a] + self.gamma * future_val
                best_val = np.max(Q[s])
                delta = max(delta, abs(best_val - V[s]))
                V_new[s] = best_val
            V = V_new
            if delta < theta:
                break

        # Recompute exact final Q
        for s in range(self.num_states):
            for a in range(self.num_actions):
                Q[s, a] = self.R[s, a] + self.gamma * np.sum(self.P[s, a] * V)

        pi_star = np.argmax(Q, axis=1)
        return V, Q, pi_star

    def get_bellman_breakdown(self, state_idx: int, V_estimates: np.ndarray = None) -> Dict:
        """
        Computes a step-by-step breakdown of the Bellman equation for a specific state.
        Allows interactive inspection of immediate reward, transition branches,
        discounted future value, and decision justification.
        """
        if V_estimates is None:
            V_estimates, _, _ = self.value_iteration()

        breakdown = {
            "state_idx": state_idx,
            "state_label": state_to_string(state_idx),
            "gamma": self.gamma,
            "actions": {}
        }

        q_values = []
        for a in range(self.num_actions):
            r_sa = self.R[state_idx, a]
            transitions = []
            future_expected = 0.0

            # Find all non-zero probability transitions
            for s_prime in range(self.num_states):
                prob = self.P[state_idx, a, s_prime]
                if prob > 1e-4:
                    v_prime = V_estimates[s_prime]
                    future_expected += prob * v_prime
                    transitions.append({
                        "next_state_idx": s_prime,
                        "next_state_label": state_to_string(s_prime),
                        "prob": float(prob),
                        "v_s_prime": float(v_prime),
                        "weighted_value": float(prob * v_prime)
                    })

            # Sort transitions by probability descending
            transitions.sort(key=lambda x: x["prob"], reverse=True)

            discounted_future = self.gamma * future_expected
            total_q = r_sa + discounted_future
            q_values.append(total_q)

            breakdown["actions"][a] = {
                "action_name": ACTION_NAMES[a],
                "immediate_reward": float(r_sa),
                "future_expected_value": float(future_expected),
                "discounted_future_value": float(discounted_future),
                "total_action_value": float(total_q),
                "transitions": transitions
            }

        best_action = int(np.argmax(q_values))
        breakdown["best_action"] = best_action
        breakdown["best_action_name"] = ACTION_NAMES[best_action]
        breakdown["q_difference"] = float(abs(q_values[0] - q_values[1]))

        # Generate a human-readable justification
        phase, act_d, opp_d = decode_state(state_idx)
        if best_action == ACTION_KEEP:
            reason = (
                f"Action KEEP is preferred (Q = {q_values[0]:.2f} vs {q_values[1]:.2f}) because the active "
                f"green approach has {DENSITY_NAMES[act_d]} density, and opposing density is {DENSITY_NAMES[opp_d]}. "
                f"Staying green avoids the switch penalty ({self.switch_penalty:.1f}) and maximizes active throughput."
            )
        else:
            reason = (
                f"Action SWITCH is preferred (Q = {q_values[1]:.2f} vs {q_values[0]:.2f}) because the opposing "
                f"queue is heavily accumulated ({DENSITY_NAMES[opp_d]}) while the active approach is {DENSITY_NAMES[act_d]}. "
                f"The long-term return gained by serving the starving opposing queue outweighs the switch penalty."
            )
        breakdown["decision_justification"] = reason

        return breakdown
