"""
Baseline Controllers for Traffic Signal Comparison:
- FixedTimeController: alternates signal phases after a fixed time window.
- HeuristicGreedyController: switches phases when opposing queue exceeds active queue.
Also provides standard evaluation routines for fair benchmarking.
"""

from typing import Dict, List, Optional
import numpy as np
from src.mdp_model import ACTION_KEEP, ACTION_SWITCH


class FixedTimeController:
    """
    Traditional Fixed-Time Traffic Signal Controller.
    Holds each phase green for a predetermined fixed duration (e.g. 6 decision steps),
    then triggers a switch, without sensing or adapting to current queues.
    """

    def __init__(self, green_duration: int = 6):
        self.green_duration = green_duration
        self.timer = 0

    def reset(self):
        self.timer = 0

    def select_action(self, state: int, env=None) -> int:
        self.timer += 1
        if self.timer >= self.green_duration:
            self.timer = 0
            return ACTION_SWITCH
        return ACTION_KEEP


class HeuristicQueueController:
    """
    Sensing baseline: switches if opposing queue significantly exceeds active queue.
    """

    def __init__(self, switch_threshold: int = 3, min_green: int = 2):
        self.switch_threshold = switch_threshold
        self.min_green = min_green
        self.steps_in_phase = 0

    def reset(self):
        self.steps_in_phase = 0

    def select_action(self, state: int, env) -> int:
        self.steps_in_phase += 1
        if self.steps_in_phase < self.min_green:
            return ACTION_KEEP

        active_q, opp_q = env.get_active_and_opposing_queues()
        if opp_q > active_q + self.switch_threshold:
            self.steps_in_phase = 0
            return ACTION_SWITCH
        return ACTION_KEEP


def evaluate_controller(
    controller,
    env,
    num_episodes: int = 20,
    base_seed: int = 42
) -> Dict[str, float]:
    """
    Evaluates a controller across multiple episodes under identical traffic seeds.
    Returns mean and standard deviation of key performance indicators.
    """
    ep_rewards = []
    ep_waiting_times = []
    ep_avg_queues = []
    ep_served = []
    ep_switches = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        state = env.reset(seed=seed)
        if hasattr(controller, "reset"):
            controller.reset()

        done = False
        step_queues = []

        while not done:
            # RL agent select_action has explore=False, baseline does not
            if hasattr(controller, "select_action"):
                import inspect
                sig = inspect.signature(controller.select_action)
                if "explore" in sig.parameters:
                    action = controller.select_action(state, explore=False)
                elif "env" in sig.parameters:
                    action = controller.select_action(state, env=env)
                else:
                    action = controller.select_action(state)
            else:
                action = 0

            next_state, reward, done, info = env.step(action)
            step_queues.append(np.sum(info["queues"]))
            state = next_state

        ep_rewards.append(env.cumulative_reward)
        ep_waiting_times.append(env.cumulative_waiting_time)
        ep_avg_queues.append(float(np.mean(step_queues)))
        ep_served.append(env.total_vehicles_served)
        ep_switches.append(env.switch_count)

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_waiting_time": float(np.mean(ep_waiting_times)),
        "std_waiting_time": float(np.std(ep_waiting_times)),
        "mean_avg_queue": float(np.mean(ep_avg_queues)),
        "std_avg_queue": float(np.std(ep_avg_queues)),
        "mean_served": float(np.mean(ep_served)),
        "std_served": float(np.std(ep_served)),
        "mean_switches": float(np.mean(ep_switches)),
        "std_switches": float(np.std(ep_switches)),
        "raw_waiting_times": ep_waiting_times,
        "raw_avg_queues": ep_avg_queues,
        "raw_rewards": ep_rewards
    }
