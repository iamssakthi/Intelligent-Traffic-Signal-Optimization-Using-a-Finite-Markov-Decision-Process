# System Architecture

## Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process

This document details the architectural design and component interactions of the academic prototype developed for real-world sequential decision-making in traffic control.

---

## 1. Architectural Overview

The system is organized into modular Python components adhering to the classical agent-environment paradigm of Reinforcement Learning (RL):

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|                       Streamlit Web Dashboard (app.py)                            |
|  - Page 1: Overview & Theory        - Page 5: Bellman Equation Explorer          |
|  - Page 2: Finite MDP Formal Design  - Page 6: Performance Benchmark              |
|  - Page 3: Live 4-Way Simulation     - Page 7: Academic Report Generator          |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+------------------------------------------+----------------------------------------+
|                          EXPERIMENT & CONTROL LAYER                               |
|                  (src/train_runner.py & src/baseline.py)                           |
|  - Fixed-Time Baseline Controller    - Heuristic Actuated Controller              |
|  - Tabular Q-Learning Agent Trainer  - Comparative Evaluation Pipeline            |
+---------------------+---------------------------------------+---------------------+
                      |                                       |
                      v                                       v
+---------------------+-----------------+   +-----------------+---------------------+
|        FINITE MDP & THEORY            |   |     ENVIRONMENT SIMULATION            |
|        (src/mdp_model.py)             |   |     (src/environment.py)              |
|  - States: |S| = 18                   |   |  - 4 Approaches (North, South, E, W)  |
|  - Actions: |A| = 2 (KEEP, SWITCH)    |   |  - Stochastic Arrivals (Poisson)      |
|  - Theoretical Model: P(s'|s,a)       |   |  - Saturation Flow Discharge          |
|  - Bellman Expectation & Optimality   |   |  - Enforced Yellow Clearance Phase    |
|  - Value Iteration Solver             |   |  - Non-conflicting Signal Phases      |
+---------------------+-----------------+   +-----------------+---------------------+
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        METRICS & REPORTING SUBSYSTEM                              |
|           (src/metrics.py, src/report_generator.py, outputs/results/)             |
|  - Statistical KPI Evaluation (Waiting Time, Queue Lengths, Throughput, Switches) |
|  - Auto-generated Artifacts: report.docx, report.pdf, report.md                   |
|  - High-resolution Plots & Genuine Application Screenshots                        |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Descriptions

### 2.1 Finite Markov Decision Process (`src/mdp_model.py`)
Encapsulates the mathematical definition of the finite MDP:
* **State Space ($S$)**: 18 discrete states encoded as $(p, d_{active}, d_{opposing})$, where $p \in \{0: \text{NS}, 1: \text{EW}\}$, and $d \in \{0: \text{Low}, 1: \text{Medium}, 2: \text{High}\}$.
* **Action Space ($A$)**: Two discrete decisions:
  * `0: KEEP`: Stay in current green phase.
  * `1: SWITCH`: Safely transition to opposing green phase.
* **Transition Probabilities ($P(s' \mid s, a)$)**: Documented stochastic transition tensor of shape $(18, 2, 18)$ satisfying $\sum_{s'} P(s' \mid s, a) = 1.0$ for all $(s, a)$.
* **Reward Function ($R(s, a)$)**: Balances waiting delay penalties, throughput incentives, and phase-switch costs.
* **Bellman Solvers**:
  * Policy Evaluation for finding $V^\pi(s)$.
  * Value Iteration for computing $V^*(s)$ and optimal greedy policy $\pi^*(s)$.
  * Step-by-step breakdown explorer providing numerical substitutions.

### 2.2 Environment Simulation (`src/environment.py`)
Simulates physical 4-way intersection dynamics:
* Enforces strict safety: opposing directions cannot both receive green signals simultaneously.
* Simulates yellow clearance upon switching to guarantee safe vehicular passage before conflicting green.
* Manages continuous queue counts and discretizes them into the finite MDP state observation.

### 2.3 Tabular Q-Learning Agent (`src/q_learning.py`)
Implements temporal difference learning:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
Supports $\epsilon$-greedy exploration with decay, policy extraction, and model persistence (`JSON`).

### 2.4 Baseline Controllers (`src/baseline.py`)
Provides deterministic fixed-time signal cycles (alternating phases every $K$ steps) and actuated heuristic control to enable empirical benchmarking under identical stochastic seeds.

### 2.5 User Interface Dashboard (`app.py`)
Streamlit-powered interactive interface designed for faculty reviews and mentor demonstrations, containing dedicated pages for problem presentation, MDP formulation, live top-down visual simulation, interactive agent training, Bellman explorer, benchmarking, and report generation.
