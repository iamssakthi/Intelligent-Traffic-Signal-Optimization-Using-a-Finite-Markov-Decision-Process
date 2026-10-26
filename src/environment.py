"""
Simulation environment for a 4-way traffic intersection.
Provides discrete state observation compatible with the Finite MDP,
stochastic vehicle arrivals, queue discharge, yellow clearance,
and performance metric tracking.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from src.mdp_model import (
    NUM_STATES, NUM_ACTIONS, ACTION_KEEP, ACTION_SWITCH,
    encode_state, decode_state, state_to_string
)


def queue_to_density(q: float) -> int:
    """Discretize continuous/integer queue count into Low (0), Medium (1), or High (2)."""
    if q <= 3.0:
        return 0  # Low
    elif q <= 8.0:
        return 1  # Medium
    else:
        return 2  # High


class TrafficEnvironment:
    """
    4-Way Traffic Intersection Simulation Environment.
    Phases:
      0: North-South Green, East-West Red
      1: East-West Green, North-South Red
    Transition:
      Yellow clearance phase occurs whenever switching to prevent accidents.
    """

    def __init__(
        self,
        max_steps: int = 100,
        max_queue: int = 25,
        discharge_rate: int = 2,
        yellow_duration: int = 1,
        switch_penalty: float = 2.0,
        scenario: str = "balanced",
        seed: Optional[int] = None
    ):
        self.max_steps = max_steps
        self.max_queue = max_queue
        self.discharge_rate = discharge_rate
        self.yellow_duration = yellow_duration
        self.switch_penalty = switch_penalty
        self.scenario = scenario

        self.rng = np.random.RandomState(seed)

        # Signal state
        self.current_phase = 0       # 0: NS green, 1: EW green
        self.is_yellow = False       # True during phase clearance
        self.yellow_timer = 0
        self.pending_phase = 0
        self.green_duration = 0      # Steps current green has been held

        # Traffic Queues [North, South, East, West]
        self.queues = np.zeros(4, dtype=np.int32)

        # Simulation step counter
        self.step_count = 0

        # Cumulative Metrics
        self.total_vehicles_arrived = 0
        self.total_vehicles_served = 0
        self.cumulative_waiting_time = 0
        self.switch_count = 0
        self.cumulative_reward = 0.0

        # Configure arrival rates for scenarios
        self.set_scenario(scenario)

    def set_scenario(self, scenario: str):
        """Configure vehicle arrival probabilities per step based on traffic scenario."""
        self.scenario = scenario
        if scenario == "rush_hour_ns":
            self.arrival_rate_ns = 0.65  # heavy NS
            self.arrival_rate_ew = 0.20  # light EW
        elif scenario == "rush_hour_ew":
            self.arrival_rate_ns = 0.20  # light NS
            self.arrival_rate_ew = 0.65  # heavy EW
        elif scenario == "heavy_all":
            self.arrival_rate_ns = 0.60
            self.arrival_rate_ew = 0.60
        else:  # "balanced"
            self.arrival_rate_ns = 0.40
            self.arrival_rate_ew = 0.40

    def reset(self, seed: Optional[int] = None, scenario: Optional[str] = None) -> int:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        if scenario is not None:
            self.set_scenario(scenario)

        self.current_phase = self.rng.choice([0, 1])
        self.is_yellow = False
        self.yellow_timer = 0
        self.pending_phase = self.current_phase
        self.green_duration = 0

        # Start with realistic small initial random queues (0 to 4 vehicles)
        self.queues = self.rng.randint(0, 5, size=4).astype(np.int32)

        self.step_count = 0
        self.total_vehicles_arrived = int(np.sum(self.queues))
        self.total_vehicles_served = 0
        self.cumulative_waiting_time = 0
        self.switch_count = 0
        self.cumulative_reward = 0.0

        return self.get_state()

    def get_active_and_opposing_queues(self) -> Tuple[float, float]:
        """Returns average queue length for (active_approach, opposing_approach)."""
        if self.current_phase == 0:
            active_q = (self.queues[0] + self.queues[1]) / 2.0
            opp_q = (self.queues[2] + self.queues[3]) / 2.0
        else:
            active_q = (self.queues[2] + self.queues[3]) / 2.0
            opp_q = (self.queues[0] + self.queues[1]) / 2.0
        return active_q, opp_q

    def get_state(self) -> int:
        """Returns the current discrete state index in [0, 17]."""
        active_q, opp_q = self.get_active_and_opposing_queues()
        act_d = queue_to_density(active_q)
        opp_d = queue_to_density(opp_q)
        return encode_state(self.current_phase, act_d, opp_d)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute one decision interval.
        Action:
          0: KEEP current green
          1: SWITCH to opposite phase
        """
        assert action in [ACTION_KEEP, ACTION_SWITCH], f"Invalid action: {action}"

        self.step_count += 1
        action_switched = (action == ACTION_SWITCH)

        if action_switched:
            self.switch_count += 1
            self.is_yellow = True
            # Transition phase
            self.current_phase = 1 - self.current_phase
            self.green_duration = 0
        else:
            self.is_yellow = False
            self.green_duration += 1

        # 1. Vehicle Arrivals (Stochastic Poisson arrivals)
        arrivals_ns = self.rng.poisson(self.arrival_rate_ns, size=2)
        arrivals_ew = self.rng.poisson(self.arrival_rate_ew, size=2)

        self.queues[0] = min(self.max_queue, self.queues[0] + arrivals_ns[0])
        self.queues[1] = min(self.max_queue, self.queues[1] + arrivals_ns[1])
        self.queues[2] = min(self.max_queue, self.queues[2] + arrivals_ew[0])
        self.queues[3] = min(self.max_queue, self.queues[3] + arrivals_ew[1])

        new_arrivals = int(arrivals_ns[0] + arrivals_ns[1] + arrivals_ew[0] + arrivals_ew[1])
        self.total_vehicles_arrived += new_arrivals

        # 2. Vehicle Departures (Discharge)
        # If action was SWITCH, intersection is clearing (yellow phase) -> 0 vehicles discharged
        # If action was KEEP, full green discharge occurs
        effective_discharge = 0 if action_switched else self.discharge_rate

        vehicles_served_this_step = 0
        if self.current_phase == 0:  # NS Green
            served_n = min(self.queues[0], effective_discharge)
            served_s = min(self.queues[1], effective_discharge)
            self.queues[0] -= served_n
            self.queues[1] -= served_s
            vehicles_served_this_step = int(served_n + served_s)
        else:  # EW Green
            served_e = min(self.queues[2], effective_discharge)
            served_w = min(self.queues[3], effective_discharge)
            self.queues[2] -= served_e
            self.queues[3] -= served_w
            vehicles_served_this_step = int(served_e + served_w)

        self.total_vehicles_served += vehicles_served_this_step

        # 3. Waiting Time accumulation
        step_waiting = int(np.sum(self.queues))
        self.cumulative_waiting_time += step_waiting

        # 4. Reward Computation
        # Penalty for queue delay, bonus for throughput, modest penalty for switching
        queue_penalty = - (0.35 * step_waiting)
        throughput_reward = + (1.2 * vehicles_served_this_step)
        switch_cost = - self.switch_penalty if action_switched else 0.0

        step_reward = queue_penalty + throughput_reward + switch_cost
        self.cumulative_reward += step_reward

        # Next state
        next_state = self.get_state()
        done = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "phase": self.current_phase,
            "phase_name": "NS" if self.current_phase == 0 else "EW",
            "is_yellow": self.is_yellow,
            "queues": self.queues.tolist(),  # [N, S, E, W]
            "active_queue": float(np.mean(self.queues[0:2] if self.current_phase == 0 else self.queues[2:4])),
            "opposing_queue": float(np.mean(self.queues[2:4] if self.current_phase == 0 else self.queues[0:2])),
            "vehicles_served": vehicles_served_this_step,
            "total_served": self.total_vehicles_served,
            "total_arrived": self.total_vehicles_arrived,
            "cumulative_waiting_time": self.cumulative_waiting_time,
            "switches": self.switch_count,
            "reward": step_reward
        }

        return next_state, step_reward, done, info
