"""
Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process
Main Streamlit Application
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.mdp_model import (
    TrafficMDP, NUM_STATES, NUM_ACTIONS, ACTION_KEEP, ACTION_SWITCH,
    ACTION_NAMES, PHASE_NAMES, PHASE_SHORT, DENSITY_NAMES,
    encode_state, decode_state, state_to_string
)
from src.environment import TrafficEnvironment
from src.q_learning import QLearningAgent
from src.baseline import FixedTimeController, HeuristicQueueController, evaluate_controller
from src.metrics import create_comparison_table, plot_training_curves, plot_comparison_bar

# Page configuration
st.set_page_config(
    page_title="Traffic Signal Optimization | Finite MDP",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Restrained Academic Custom CSS
st.markdown("""
<style>
    /* Academic Typography and Restrained Styling */
    .main-title {
        color: #0F172A;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        font-weight: 700;
        font-size: 1.85rem;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #475569;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        color: #1E3A8A;
        font-weight: 600;
        font-size: 1.3rem;
        margin-top: 1.0rem;
        margin-bottom: 0.75rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.3rem;
    }
    .academic-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.75rem;
        border-radius: 4px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# Persistent session state initialization
if "env" not in st.session_state:
    st.session_state.env = TrafficEnvironment(max_steps=120, seed=42)
    st.session_state.sim_state = st.session_state.env.reset()
    st.session_state.sim_info = {
        "step": 0, "phase": st.session_state.env.current_phase,
        "is_yellow": False, "queues": st.session_state.env.queues.tolist(),
        "vehicles_served": 0, "total_served": 0, "cumulative_waiting_time": 0,
        "switches": 0, "reward": 0.0
    }
    st.session_state.sim_history = []

if "mdp" not in st.session_state:
    st.session_state.mdp = TrafficMDP(gamma=0.90)

if "agent" not in st.session_state:
    agent = QLearningAgent(seed=42)
    model_path = "models/trained_agent.json"
    if os.path.exists(model_path):
        try:
            agent.load_model(model_path)
        except Exception:
            pass
    st.session_state.agent = agent

if "baseline_ctrl" not in st.session_state:
    st.session_state.baseline_ctrl = FixedTimeController(green_duration=6)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚦 Traffic Signal MDP")
    st.markdown("**Reinforcement Learning Project**")
    st.markdown("---")

    page = st.radio(
        "Navigation Menu",
        [
            "1. Project Overview",
            "2. Finite MDP Design",
            "3. Live Traffic Simulation",
            "4. RL Training Dashboard",
            "5. Bellman Equation Explorer",
            "6. Performance Comparison",
            "7. Report & Deliverables"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### System State")
    st.caption(f"**State Space |S|**: {NUM_STATES} states")
    st.caption(f"**Action Space |A|**: {NUM_ACTIONS} actions")
    st.caption(f"**Discount (γ)**: {st.session_state.mdp.gamma:.2f}")
    st.caption(f"**Agent Epsilon (ε)**: {st.session_state.agent.epsilon:.3f}")

    st.markdown("---")
    st.caption("Undergraduate AIML Academic Prototype")


# =============================================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================================
if page == "1. Project Overview":
    st.markdown('<div class="main-title">Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Decoding Real-World Sequential Decision-Making Through Tabular Reinforcement Learning</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Problem Statement & Motivation</div>', unsafe_allow_html=True)
        st.markdown("""
        Urban road intersections are critical bottlenecks in transportation networks. Conventional **fixed-time signal controllers** operate on predetermined, rigid cycle lengths regardless of fluctuating traffic volumes.
        
        **Limitations of Conventional Control:**
        * Continues granting green signals to empty approaches while opposing queues build up.
        * Cannot adapt dynamically to rush-hour asymmetry or stochastic vehicle surges.
        * Induces unnecessary vehicle idling, fuel wastage, and elevated carbon emissions.
        
        **The Proposed Reinforcement Learning Solution:**
        By formulating traffic control as a **Finite Markov Decision Process (MDP)**, an intelligent software agent continuously observes traffic density, evaluates delayed consequences via value functions, and learns a sequential policy that minimizes cumulative vehicle waiting times.
        """)

        st.markdown('<div class="section-header">The Agent-Environment Interaction Cycle</div>', unsafe_allow_html=True)
        st.markdown("""
        ```
        +-------------------------------------------------------------+
        |                           AGENT                             |
        |  Intelligent Controller (Tabular Q-Learning / Policy π)     |
        +-----------------------------+-------------------------------+
                      |               ^
             Action   |               | Observation (State s_t)
             a_t ∈ A  |               | & Feedback (Reward r_t)
                      v               |
        +-----------------------------+-------------------------------+
        |                        ENVIRONMENT                          |
        | 4-Way Intersection Dynamics (Queues, Clearance, Departures) |
        +-------------------------------------------------------------+
        ```
        """)

    with col2:
        st.markdown('<div class="section-header">Project Specifications</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="academic-card">
            <strong>Formal MDP Tuple:</strong><br>
            <code>M = (S, A, P, R, γ)</code><br><br>
            <strong>State Space |S|:</strong> 18 Discrete States<br>
            <strong>Action Space |A|:</strong> 2 Actions (KEEP, SWITCH)<br>
            <strong>Algorithm:</strong> Tabular Q-Learning<br>
            <strong>Evaluation:</strong> Bellman Equation Analysis<br>
            <strong>Baseline:</strong> Fixed-Time Cycle Controller<br>
            <strong>Tech Stack:</strong> Python, Streamlit, NumPy, Pandas, Matplotlib
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Key Objectives</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Model Formulation:** Represent continuous intersection dynamics as a rigorous finite state MDP.
        2. **Safety Enforcement:** Maintain non-conflicting phases and guarantee mandatory yellow clearance.
        3. **Bellman Analysis:** Deconstruct how delayed expected return governs immediate decisions.
        4. **Benchmark Verification:** Quantitatively compare the RL agent against fixed-time controllers under identical seeds.
        """)


# =============================================================================
# PAGE 2: FINITE MDP DESIGN
# =============================================================================
elif page == "2. Finite MDP Design":
    st.markdown('<div class="main-title">Formal Finite Markov Decision Process Formulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Complete Mathematical Specification of M = (S, A, P, R, γ)</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["1. MDP Components & Tables", "2. Formal Definitions", "3. Transition Probability Model"])

    with tab1:
        st.markdown('<div class="section-header">State Space Representation S</div>', unsafe_allow_html=True)
        st.markdown("""
        To guarantee a **Finite MDP**, each traffic condition is represented by a discrete 3-tuple:
        $$s = (\\text{current\\_phase},\\, \\text{active\\_density},\\, \\text{opposing\\_density})$$
        * **Phase ($p \\in \\{0, 1\\}$):** $0$ for North-South Green (East-West Red); $1$ for East-West Green (North-South Red).
        * **Active Density ($d_{act} \\in \\{0, 1, 2\\}$):** Low ($\\le 3$ veh), Medium ($4-8$ veh), High ($9+$ veh).
        * **Opposing Density ($d_{opp} \\in \\{0, 1, 2\\}$):** Low ($\\le 3$ veh), Medium ($4-8$ veh), High ($9+$ veh).
        
        Total Finite States: $|S| = 2 \\times 3 \\times 3 = 18$ states.
        """)

        # Generate State Space Table
        state_rows = []
        for s in range(NUM_STATES):
            p, a_d, o_d = decode_state(s)
            state_rows.append({
                "State Index (s)": s,
                "Current Phase": PHASE_NAMES[p],
                "Active Density": DENSITY_NAMES[a_d],
                "Opposing Density": DENSITY_NAMES[o_d],
                "State Descriptor": state_to_string(s)
            })
        st.dataframe(pd.DataFrame(state_rows), use_container_width=True, height=280)

        st.markdown('<div class="section-header">Action Space Representation A</div>', unsafe_allow_html=True)
        action_rows = [
            {"Action Code": 0, "Action Name": "KEEP (Stay Green)", "Operational Meaning": "Maintain current phase green; active traffic continues discharging."},
            {"Action Code": 1, "Action Name": "SWITCH (Change Phase)", "Operational Meaning": "Trigger phase switch; initiates mandatory yellow clearance and shifts green to opposing approach."}
        ]
        st.table(pd.DataFrame(action_rows))

    with tab2:
        st.markdown('<div class="section-header">Detailed MDP Elements</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **1. Agent:**
            The autonomous traffic signal controller that observes state $s_t$ at discrete decision step $t$ and chooses action $a_t$.
            
            **2. Environment:**
            A 4-way intersection characterized by stochastic Poisson vehicle arrivals, saturation discharge rates, and clearance constraints.
            
            **3. Reward Function $R(s, a)$:**
            Encourages queue clearance and discourages excessive waiting delay and phase chattering:
            $$R_t = - c_{\\text{wait}} \\sum_{i} q_{i} + c_{\\text{serve}} v_{\\text{served}} - c_{\\text{switch}} \\mathbb{I}(a = \\text{SWITCH})$$
            * $c_{\\text{wait}} = 0.35$: Delay penalty per waiting vehicle.
            * $c_{\\text{serve}} = 1.20$: Throughput incentive for discharged vehicles.
            * $c_{\\text{switch}} = 1.00$: Cost incurred for yellow clearance phase transition.
            """)
        with c2:
            st.markdown("""
            **4. Goals & Discounted Return $G_t$:**
            The agent aims to maximize expected discounted cumulative reward:
            $$G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}$$
            With discount factor $\\gamma = 0.90$, immediate congestion is prioritized while future queue buildup is valued.
            
            **5. Policy $\\pi(a \\mid s)$:**
            A mapping from states to probabilities of selecting each action. Under training, the agent follows an $\\epsilon$-greedy policy:
            $$\\pi(a \\mid s) = \\begin{cases} 1 - \\epsilon + \\frac{\\epsilon}{|A|} & \\text{if } a = \\arg\\max Q(s, a) \\\\ \\frac{\\epsilon}{|A|} & \\text{otherwise} \\end{cases}$$
            
            **6. State-Value Function $V(s)$:**
            Estimates expected long-term return from state $s$: $V^\\pi(s) = \\mathbb{E}_\\pi [G_t \\mid S_t = s]$.
            """)

    with tab3:
        st.markdown('<div class="section-header">Theoretical Transition Probability Model P(s\' | s, a)</div>', unsafe_allow_html=True)
        st.markdown("""
        Because vehicle arrivals are stochastic, transitions between states are probabilistic.
        The theoretical model satisfies the Markov property:
        $$\\sum_{s' \\in S} P(s' \\mid s, a) = 1.0 \\quad \\forall s \\in S, \\, a \\in A$$
        """)

        # Interactive State Transition Probability Inspector
        inspect_state = st.selectbox(
            "Select State s to Inspect Transition Probabilities P(s' | s, a):",
            options=list(range(NUM_STATES)),
            format_func=lambda s: state_to_string(s)
        )
        inspect_action = st.radio("Select Action a:", [0, 1], format_func=lambda a: ACTION_NAMES[a], horizontal=True)

        mdp = st.session_state.mdp
        probs = mdp.P[inspect_state, inspect_action]
        active_transitions = []
        for s_prime in range(NUM_STATES):
            if probs[s_prime] > 0.001:
                active_transitions.append({
                    "Next State s'": s_prime,
                    "Next State Description": state_to_string(s_prime),
                    "Transition Probability P(s'|s,a)": f"{probs[s_prime]:.4f}",
                    "Percentage": f"{probs[s_prime] * 100:.1f}%"
                })
        st.dataframe(pd.DataFrame(active_transitions), use_container_width=True)
        st.caption(f"Sum of transition probabilities for state {inspect_state}, action {inspect_action}: {np.sum(probs):.5f}")


# =============================================================================
# PAGE 3: LIVE TRAFFIC SIMULATION
# =============================================================================
elif page == "3. Live Traffic Simulation":
    st.markdown('<div class="main-title">Interactive 4-Way Intersection Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Direct Visual Demonstration of Signal Control and Real-Time Queue Dynamics</div>', unsafe_allow_html=True)

    # Simulation Controls Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 2, 2])
    with ctrl_col1:
        controller_choice = st.selectbox(
            "Signal Controller:",
            ["Trained Q-Learning Agent", "Fixed-Time Baseline (6-Step Cycle)"]
        )
    with ctrl_col2:
        scenario_choice = st.selectbox(
            "Traffic Demand Scenario:",
            ["balanced", "rush_hour_ns", "rush_hour_ew"]
        )
    with ctrl_col3:
        step_btn = st.button("▶ Step Forward (1 Interval)", use_container_width=True)
        step_5_btn = st.button("⏩ Step Forward (5 Intervals)", use_container_width=True)
    with ctrl_col4:
        reset_btn = st.button("🔄 Reset Simulation", use_container_width=True)

    env = st.session_state.env
    agent = st.session_state.agent
    baseline = st.session_state.baseline_ctrl

    # Handle Reset
    if reset_btn:
        env.reset(scenario=scenario_choice)
        baseline.reset()
        st.session_state.sim_state = env.get_state()
        st.session_state.sim_info = {
            "step": 0, "phase": env.current_phase, "is_yellow": False,
            "queues": env.queues.tolist(), "vehicles_served": 0,
            "total_served": 0, "cumulative_waiting_time": 0,
            "switches": 0, "reward": 0.0
        }
        st.session_state.sim_history = []
        st.rerun()

    # Handle Step
    def execute_simulation_steps(steps_to_run: int):
        for _ in range(steps_to_run):
            current_s = env.get_state()
            if "Q-Learning" in controller_choice:
                action = agent.select_action(current_s, explore=False)
            else:
                action = baseline.select_action(current_s, env=env)

            next_s, reward, done, info = env.step(action)
            st.session_state.sim_state = next_s
            st.session_state.sim_info = info
            st.session_state.sim_history.append({
                "step": env.step_count,
                "action": ACTION_NAMES[action],
                "phase": info["phase_name"],
                "queues": info["queues"],
                "served": info["vehicles_served"],
                "reward": reward
            })
            if done:
                st.warning("Episode maximum steps reached. Please reset to start a new run.")
                break

    if step_btn:
        execute_simulation_steps(1)
    elif step_5_btn:
        execute_simulation_steps(5)

    info = st.session_state.sim_info
    phase = info["phase"]
    is_yellow = info["is_yellow"]
    queues = info["queues"]  # [N, S, E, W]

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{info["step"]} / {env.max_steps}</div><div class="metric-label">Step Number</div></div>', unsafe_allow_html=True)
    with m2:
        status_text = "YELLOW CLEARANCE" if is_yellow else ("NORTH-SOUTH GREEN" if phase == 0 else "EAST-WEST GREEN")
        status_color = "#D97706" if is_yellow else ("#059669" if phase == 0 else "#2563EB")
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{status_color}; font-size:1.1rem;">{status_text}</div><div class="metric-label">Signal Phase</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sum(queues)}</div><div class="metric-label">Total Queued (Veh)</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{info["total_served"]}</div><div class="metric-label">Total Served</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{info["cumulative_waiting_time"]}</div><div class="metric-label">Total Delay (veh-step)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TOP-DOWN 4-WAY INTERSECTION VISUALIZATION (SVG)
    # -------------------------------------------------------------------------
    ns_color = "#F59E0B" if is_yellow else ("#10B981" if phase == 0 else "#EF4444")
    ew_color = "#F59E0B" if is_yellow else ("#EF4444" if phase == 0 else "#10B981")

    # Queue representations (vehicle rectangles)
    def render_veh_boxes(count: int, max_show: int = 8):
        return min(count, max_show)

    svg_content = f"""
    <svg width="100%" height="420" viewBox="0 0 600 420" xmlns="http://www.w3.org/2000/svg" style="background:#F1F5F9; border-radius:8px; border:1px solid #CBD5E1;">
        <!-- Grass / Ground background -->
        <rect width="600" height="420" fill="#E2E8F0"/>
        
        <!-- Roads (Asphalt) -->
        <!-- North-South Road -->
        <rect x="230" y="0" width="140" height="420" fill="#334155"/>
        <!-- East-West Road -->
        <rect x="0" y="140" width="600" height="140" fill="#334155"/>
        
        <!-- Center Intersection Area -->
        <rect x="230" y="140" width="140" height="140" fill="#1E293B"/>

        <!-- Dashed Centerlines -->
        <!-- North arm -->
        <line x1="300" y1="0" x2="300" y2="140" stroke="#F8FAFC" stroke-width="2.5" stroke-dasharray="10,8"/>
        <!-- South arm -->
        <line x1="300" y1="280" x2="300" y2="420" stroke="#F8FAFC" stroke-width="2.5" stroke-dasharray="10,8"/>
        <!-- West arm -->
        <line x1="0" y1="210" x2="230" y2="210" stroke="#F8FAFC" stroke-width="2.5" stroke-dasharray="10,8"/>
        <!-- East arm -->
        <line x1="370" y1="210" x2="600" y2="210" stroke="#F8FAFC" stroke-width="2.5" stroke-dasharray="10,8"/>

        <!-- Stop Lines -->
        <line x1="230" y1="138" x2="300" y2="138" stroke="#FFFFFF" stroke-width="4"/>
        <line x1="300" y1="282" x2="370" y2="282" stroke="#FFFFFF" stroke-width="4"/>
        <line x1="228" y1="210" x2="228" y2="280" stroke="#FFFFFF" stroke-width="4"/>
        <line x1="372" y1="140" x2="372" y2="210" stroke="#FFFFFF" stroke-width="4"/>

        <!-- Traffic Signals Indicators -->
        <!-- North Signal Head -->
        <rect x="180" y="105" width="40" height="30" rx="4" fill="#0F172A"/>
        <circle cx="200" cy="120" r="9" fill="{ns_color}"/>
        <text x="200" y="98" fill="#0F172A" font-size="11" font-weight="bold" text-anchor="middle">NS Signal</text>

        <!-- South Signal Head -->
        <rect x="380" y="285" width="40" height="30" rx="4" fill="#0F172A"/>
        <circle cx="400" cy="300" r="9" fill="{ns_color}"/>
        <text x="400" y="332" fill="#0F172A" font-size="11" font-weight="bold" text-anchor="middle">NS Signal</text>

        <!-- West Signal Head -->
        <rect x="185" y="290" width="30" height="40" rx="4" fill="#0F172A"/>
        <circle cx="200" cy="310" r="9" fill="{ew_color}"/>
        <text x="160" y="315" fill="#0F172A" font-size="11" font-weight="bold" text-anchor="end">EW Signal</text>

        <!-- East Signal Head -->
        <rect x="385" y="90" width="30" height="40" rx="4" fill="#0F172A"/>
        <circle cx="400" cy="110" r="9" fill="{ew_color}"/>
        <text x="440" y="115" fill="#0F172A" font-size="11" font-weight="bold" text-anchor="start">EW Signal</text>

        <!-- Queue Badges & Labels -->
        <!-- North Approach (Queue 0) -->
        <rect x="238" y="15" width="55" height="26" rx="4" fill="#1E3A8A"/>
        <text x="265" y="32" fill="#FFFFFF" font-size="12" font-weight="bold" text-anchor="middle">N: {queues[0]} veh</text>

        <!-- South Approach (Queue 1) -->
        <rect x="307" y="375" width="55" height="26" rx="4" fill="#1E3A8A"/>
        <text x="334" y="392" fill="#FFFFFF" font-size="12" font-weight="bold" text-anchor="middle">S: {queues[1]} veh</text>

        <!-- West Approach (Queue 3) -->
        <rect x="15" y="222" width="55" height="26" rx="4" fill="#1E3A8A"/>
        <text x="42" y="239" fill="#FFFFFF" font-size="12" font-weight="bold" text-anchor="middle">W: {queues[3]} veh</text>

        <!-- East Approach (Queue 2) -->
        <rect x="525" y="172" width="55" height="26" rx="4" fill="#1E3A8A"/>
        <text x="552" y="189" fill="#FFFFFF" font-size="12" font-weight="bold" text-anchor="middle">E: {queues[2]} veh</text>

        <!-- Visualized Queued Vehicles (Blocks) -->
        <!-- North lane queued vehicles -->
        {''.join([f'<rect x="250" y="{110 - i*16}" width="25" height="12" rx="2" fill="#38BDF8"/>' for i in range(render_veh_boxes(queues[0]))])}
        <!-- South lane queued vehicles -->
        {''.join([f'<rect x="325" y="{295 + i*16}" width="25" height="12" rx="2" fill="#38BDF8"/>' for i in range(render_veh_boxes(queues[1]))])}
        <!-- West lane queued vehicles -->
        {''.join([f'<rect x="{200 - i*16}" y="235" width="12" height="25" rx="2" fill="#38BDF8"/>' for i in range(render_veh_boxes(queues[3]))])}
        <!-- East lane queued vehicles -->
        {''.join([f'<rect x="{385 + i*16}" y="160" width="12" height="25" rx="2" fill="#38BDF8"/>' for i in range(render_veh_boxes(queues[2]))])}

        <!-- Direction Indicators -->
        <text x="300" y="215" fill="#64748B" font-size="14" font-weight="bold" text-anchor="middle">4-WAY INTERSECTION</text>
    </svg>
    """
    st.markdown(svg_content, unsafe_allow_html=True)

    # Simulation Event Log
    if st.session_state.sim_history:
        st.markdown('<div class="section-header">Recent Step Log</div>', unsafe_allow_html=True)
        recent_log = pd.DataFrame(st.session_state.sim_history[-6:])
        st.dataframe(recent_log, use_container_width=True)


# =============================================================================
# PAGE 4: RL TRAINING
# =============================================================================
elif page == "4. RL Training Dashboard":
    st.markdown('<div class="main-title">Tabular Q-Learning Agent Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive Hyperparameter Tuning, Training Execution, and Convergence Monitoring</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-header">Hyperparameters</div>', unsafe_allow_html=True)
        train_episodes = st.slider("Number of Episodes:", min_value=50, max_value=400, value=150, step=25)
        learning_rate = st.slider("Learning Rate (α):", min_value=0.01, max_value=0.50, value=0.12, step=0.01)
        discount_factor = st.slider("Discount Factor (γ):", min_value=0.50, max_value=0.99, value=0.90, step=0.01)
        initial_epsilon = st.slider("Initial Exploration (ε):", min_value=0.20, max_value=1.00, value=1.00, step=0.05)
        epsilon_decay = st.slider("Exploration Decay Rate:", min_value=0.950, max_value=0.999, value=0.985, step=0.005)

        start_train_btn = st.button("🚀 Start Training Session", use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Temporal-Difference Q-Learning Formulation</div>', unsafe_allow_html=True)
        st.markdown("""
        The agent estimates state-action values $Q(s, a)$ iteratively using Bellman's equation:
        $$Q(s_t, a_t) \\leftarrow Q(s_t, a_t) + \\alpha \\left[ R_{t+1} + \\gamma \\max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \\right]$$
        * **Terminal State Handling:** When episode ends ($s_{t+1}$ is terminal), target is purely $R_{t+1}$.
        * **Exploration Schedule:** $\\epsilon \\leftarrow \\max(\\epsilon_{min}, \\epsilon \\cdot \\text{decay})$.
        * **Greedy Value Derivation:** $V(s) = \\max_{a} Q(s, a)$.
        """)

    if start_train_btn:
        train_env = TrafficEnvironment(max_steps=120, scenario="balanced")
        agent = QLearningAgent(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            seed=int(time.time())
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        for ep in range(1, train_episodes + 1):
            agent.train_episode(train_env)
            progress_bar.progress(ep / train_episodes)
            if ep % 10 == 0 or ep == train_episodes:
                status_text.text(f"Training Progress: Episode {ep}/{train_episodes} | Epsilon: {agent.epsilon:.3f}")

        agent.save_model("models/trained_agent.json")
        st.session_state.agent = agent
        st.success(f"Training finished successfully across {train_episodes} episodes! Model saved.")

    # Display training curves
    agent = st.session_state.agent
    if agent.training_history and len(agent.training_history["episode_rewards"]) > 0:
        st.markdown('<div class="section-header">Empirical Training Curves</div>', unsafe_allow_html=True)
        fig = plot_training_curves(agent.training_history)
        st.pyplot(fig)

        st.markdown('<div class="section-header">Learned Q-Table & Derived State Values V(s)</div>', unsafe_allow_html=True)
        q_rows = []
        for s in range(NUM_STATES):
            q_rows.append({
                "State Index": s,
                "State Description": state_to_string(s),
                "Q(s, KEEP)": f"{agent.q_table[s, ACTION_KEEP]:.2f}",
                "Q(s, SWITCH)": f"{agent.q_table[s, ACTION_SWITCH]:.2f}",
                "V(s) = max Q": f"{np.max(agent.q_table[s]):.2f}",
                "Learned Policy π*(s)": ACTION_NAMES[np.argmax(agent.q_table[s])]
            })
        st.dataframe(pd.DataFrame(q_rows), use_container_width=True, height=260)


# =============================================================================
# PAGE 5: BELLMAN EQUATION EXPLORER
# =============================================================================
elif page == "5. Bellman Equation Explorer":
    st.markdown('<div class="main-title">Bellman Equation Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Step-by-Step Deconstruction of Expected Return and Sequential Decision Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    The Bellman equation decomposes the value of a state into the **immediate reward** plus the **discounted expected value of successor states**:
    """)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("""
        <div class="academic-card">
            <strong>Bellman Expectation Equation:</strong><br>
            $$V^{\\pi}(s) = \\sum_{a} \\pi(a|s) \\left[ R(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V^{\\pi}(s') \\right]$$
            Evaluates the expected return under a specific stationary policy $\\pi$.
        </div>
        """, unsafe_allow_html=True)
    with b_col2:
        st.markdown("""
        <div class="academic-card">
            <strong>Bellman Optimality Equation:</strong><br>
            $$V^*(s) = \\max_{a} \\left[ R(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V^*(s') \\right]$$
            Characterizes the optimal state-value achieved by the best sequential action.
        </div>
        """, unsafe_allow_html=True)

    mdp = st.session_state.mdp
    V_opt, Q_opt, pi_opt = mdp.value_iteration()

    st.markdown('<div class="section-header">Interactive Numerical Substitution</div>', unsafe_allow_html=True)
    selected_s = st.selectbox(
        "Select a State s to Deconstruct the Bellman Calculation:",
        options=list(range(NUM_STATES)),
        format_func=lambda s: state_to_string(s)
    )

    breakdown = mdp.get_bellman_breakdown(selected_s, V_opt)

    c_keep, c_switch = st.columns(2)
    with c_keep:
        act0 = breakdown["actions"][ACTION_KEEP]
        st.markdown(f"""
        <div class="academic-card">
            <h4>Action 0: {act0['action_name']}</h4>
            <strong>Immediate Expected Reward R(s, KEEP):</strong> {act0['immediate_reward']:.2f}<br>
            <strong>Expected Future Value Σ P(s'|s,a) V*(s'):</strong> {act0['future_expected_value']:.2f}<br>
            <strong>Discounted Future Value γ Σ P V*:</strong> {act0['discounted_future_value']:.2f} (γ = {breakdown['gamma']})<br>
            <hr style="margin:8px 0;">
            <h4 style="color:#1E3A8A; margin:0;">Total Q*(s, KEEP) = {act0['total_action_value']:.2f}</h4>
        </div>
        """, unsafe_allow_html=True)

    with c_switch:
        act1 = breakdown["actions"][ACTION_SWITCH]
        st.markdown(f"""
        <div class="academic-card">
            <h4>Action 1: {act1['action_name']}</h4>
            <strong>Immediate Expected Reward R(s, SWITCH):</strong> {act1['immediate_reward']:.2f}<br>
            <strong>Expected Future Value Σ P(s'|s,a) V*(s'):</strong> {act1['future_expected_value']:.2f}<br>
            <strong>Discounted Future Value γ Σ P V*:</strong> {act1['discounted_future_value']:.2f} (γ = {breakdown['gamma']})<br>
            <hr style="margin:8px 0;">
            <h4 style="color:#1E3A8A; margin:0;">Total Q*(s, SWITCH) = {act1['total_action_value']:.2f}</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color:#EFF6FF; border:1px solid #93C5FD; padding:1rem; border-radius:4px; margin-top:0.5rem;">
        <strong style="color:#1E40AF;">Optimal Decision Justification:</strong><br>
        {breakdown['decision_justification']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Successor State Transition Distribution for Selected State</div>', unsafe_allow_html=True)
    sub_tab1, sub_tab2 = st.tabs(["Transitions under KEEP", "Transitions under SWITCH"])
    with sub_tab1:
        st.dataframe(pd.DataFrame(act0["transitions"]), use_container_width=True)
    with sub_tab2:
        st.dataframe(pd.DataFrame(act1["transitions"]), use_container_width=True)


# =============================================================================
# PAGE 6: PERFORMANCE COMPARISON
# =============================================================================
elif page == "6. Performance Comparison":
    st.markdown('<div class="main-title">Empirical Benchmark & Baseline Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Rigorous Quantitative Evaluation Under Identical Stochastic Seeds</div>', unsafe_allow_html=True)

    b_col1, b_col2, b_col3 = st.columns([2, 2, 2])
    with b_col1:
        eval_scenario = st.selectbox(
            "Evaluation Traffic Scenario:",
            ["balanced", "rush_hour_ns", "rush_hour_ew"]
        )
    with b_col2:
        eval_episodes = st.slider("Evaluation Episodes:", min_value=10, max_value=60, value=30, step=5)
    with b_col3:
        run_bench_btn = st.button("📊 Run Live Benchmark", use_container_width=True)

    benchmark_file = "outputs/results/benchmark_results.json"
    benchmark_data = None
    if os.path.exists(benchmark_file):
        try:
            with open(benchmark_file, "r") as f:
                benchmark_data = json.load(f)
        except Exception:
            pass

    if run_bench_btn:
        with st.spinner("Executing comparative simulation episodes under identical arrival seeds..."):
            bench_env = TrafficEnvironment(max_steps=120, scenario=eval_scenario)
            base_ctrl = FixedTimeController(green_duration=6)
            rl_agent = st.session_state.agent

            base_res = evaluate_controller(base_ctrl, bench_env, num_episodes=eval_episodes, base_seed=5000)
            rl_res = evaluate_controller(rl_agent, bench_env, num_episodes=eval_episodes, base_seed=5000)

            comp_df = create_comparison_table(base_res, rl_res)
            benchmark_data = {
                "baseline": base_res,
                "rl": rl_res,
                "comparison_table": comp_df.to_dict(orient="records"),
                "scenario": eval_scenario
            }
            with open(benchmark_file, "w") as f:
                json.dump(benchmark_data, f, indent=2)

            st.success("Benchmark completed and recorded!")

    if benchmark_data:
        st.markdown('<div class="section-header">Statistical Performance Summary Table</div>', unsafe_allow_html=True)
        table_df = pd.DataFrame(benchmark_data["comparison_table"])
        st.table(table_df)

        st.markdown('<div class="section-header">Performance Comparison Visualizations</div>', unsafe_allow_html=True)
        fig_bar = plot_comparison_bar(benchmark_data["baseline"], benchmark_data["rl"])
        st.pyplot(fig_bar)

        st.markdown('<div class="section-header">Academic Analysis & Discussion</div>', unsafe_allow_html=True)
        st.markdown("""
        **Scientific Observations:**
        1. **Cycle Efficiency vs Responsiveness:** Fixed-time control performs reasonably well when traffic is strictly symmetric and balanced because cyclic phases match arrival rates by design.
        2. **Asymmetric Traffic Advantage:** Under rush-hour asymmetry (e.g., heavy NS flow and sparse EW flow), the Q-learning agent prevents wasting green phases on empty approaches, yielding significant waiting time reductions.
        3. **Switching Frequency:** The learned policy switches less frequently than fixed-time controllers when queues are low, eliminating unnecessary yellow clearance delays.
        """)
    else:
        st.info("Click 'Run Live Benchmark' to generate empirical evaluation results.")


# =============================================================================
# PAGE 7: REPORT AND PROJECT INFORMATION
# =============================================================================
elif page == "7. Report & Deliverables":
    st.markdown('<div class="main-title">Academic Project Report & Deliverables</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Complete Documentation, Formal Report Downloads, and Repository Information</div>', unsafe_allow_html=True)

    st.markdown("""
    This project includes a comprehensive, 10-chapter academic report formatted in **DOCX (editable)**, **PDF (submission-ready)**, and **Markdown (GitHub-compatible)**.
    """)

    rep_col1, rep_col2, rep_col3 = st.columns(3)
    with rep_col1:
        docx_path = "docs/report.docx"
        if os.path.exists(docx_path):
            with open(docx_path, "rb") as f:
                st.download_button(
                    label="📄 Download DOCX Report",
                    data=f.read(),
                    file_name="Intelligent_Traffic_Signal_MDP_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
        else:
            st.button("DOCX Report (Generating...)", disabled=True, use_container_width=True)

    with rep_col2:
        pdf_path = "docs/report.pdf"
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📕 Download PDF Report",
                    data=f.read(),
                    file_name="Intelligent_Traffic_Signal_MDP_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.button("PDF Report (Generating...)", disabled=True, use_container_width=True)

    with rep_col3:
        md_path = "docs/report.md"
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📝 Download Markdown Report",
                    data=f.read(),
                    file_name="report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        else:
            st.button("Markdown Report (Generating...)", disabled=True, use_container_width=True)

    st.markdown('<div class="section-header">Project Summary & Evaluation Highlights</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Project Title:** Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process
    * **Domain:** Sequential Decision-Making & Reinforcement Learning (AIML)
    * **Key Formulation:** Finite MDP $M = (S, A, P, R, \\gamma)$ with $|S| = 18$, $|A| = 2$.
    * **Primary Algorithm:** Tabular Q-Learning with $\\epsilon$-greedy exploration and temporal-difference updates.
    * **Mathematical Validation:** Value Iteration and Bellman optimality decomposition.
    * **Safety Guarantees:** Strict enforcement of non-conflicting green phases and yellow clearance periods.
    """)

    st.markdown('<div class="section-header">Git & Version Control Status</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Local Repository Status:** Git repository initialized and fully structured.
    * **Publishing Instructions:** To host on GitHub:
      ```bash
      git remote add origin https://github.com/<your-username>/intelligent-traffic-signal-mdp.git
      git branch -M main
      git push -u origin main
      ```
    * **Repository URL:** *(Pending user GitHub authorization / publication)*
    """)
