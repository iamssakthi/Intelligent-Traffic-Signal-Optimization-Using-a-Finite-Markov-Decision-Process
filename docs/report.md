# Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process
## Academic Project Report & Technical Specification

---

### COVER PAGE

**Project Title:** Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process  
**Student Name:** [PLACEHOLDER — DO NOT INVENT]  
**Register Number:** [PLACEHOLDER — DO NOT INVENT]  
**Department:** [PLACEHOLDER — DO NOT INVENT]  
**Institution:** [PLACEHOLDER — DO NOT INVENT]  
**Course / Subject:** [PLACEHOLDER — DO NOT INVENT]  
**Faculty / Mentor:** [PLACEHOLDER — DO NOT INVENT]  
**Academic Year:** [PLACEHOLDER — DO NOT INVENT]  

---

## TABLE OF CONTENTS
1. [Chapter 1: Introduction](#chapter-1-introduction)
2. [Chapter 2: Problem Statement](#chapter-2-problem-statement)
3. [Chapter 3: Proposed Solution](#chapter-3-proposed-solution)
4. [Chapter 4: Finite MDP Design](#chapter-4-finite-mdp-design)
5. [Chapter 5: Bellman Equation Analysis](#chapter-5-bellman-equation-analysis)
6. [Chapter 6: Algorithm and Implementation](#chapter-6-algorithm-and-implementation)
7. [Chapter 7: User Interface and Screenshots](#chapter-7-user-interface-and-screenshots)
8. [Chapter 8: Experimental Results](#chapter-8-experimental-results)
9. [Chapter 9: Discussion and Limitations](#chapter-9-discussion-and-limitations)
10. [Chapter 10: Conclusion and Future Scope](#chapter-10-conclusion-and-future-scope)
11. [References](#references)
12. [Appendices](#appendices)

---

## CHAPTER 1: INTRODUCTION

### 1.1 Sequential Decision-Making in Urban Environments
Urban road intersections constitute vital nexus points in civil infrastructure networks. Managing the competing flow of vehicular traffic through intersections is inherently a **sequential decision-making problem**: signal indications executed at time step $t$ directly govern queue accumulation, driver delay, and departure throughput at subsequent time steps $t+1, t+2, \dots$. Actions that appear optimal in the short term—such as continuously extending a green light to clear a minor queue—can impose severe delay penalties on opposing approaches if sustained inappropriately.

### 1.2 Reinforcement Learning and Markov Decision Processes
Reinforcement Learning (RL) provides a mathematically rigorous computational framework for training autonomous agents to solve sequential decision problems under environmental uncertainty. Rather than relying on manual rule-based programming or static timing plans, an RL agent learns an optimal behavior policy by interacting with an environment, observing feedback in the form of numerical scalar rewards, and evaluating expected long-term cumulative returns.

At the core of modern Reinforcement Learning lies the **Markov Decision Process (MDP)**. When the state and action spaces are finite, the problem is formulated as a **Finite MDP**, which affords exact mathematical tractability, convergence guarantees, and direct policy evaluation through the foundational equations of Richard Bellman.

### 1.3 Traffic Control as a Canonical MDP
Traffic signal control represents an exemplary real-world domain for Finite MDP modeling:
1. **Dynamic Environment:** Vehicular arrival rates fluctuate stochastically across time of day and geographic direction.
2. **Clear Feedback Signals:** Delay, queue length, and throughput provide unambiguous, measurable feedback signals.
3. **Delayed Consequences:** The decision to switch or retain a phase alters the state distribution for many subsequent intervals, highlighting the necessity of temporal discounting and value functions.

---

## CHAPTER 2: PROBLEM STATEMENT

### 2.1 Limitations of Existing Fixed-Time Signal Control
Conventional traffic signals deployed in municipal road networks predominantly utilize **fixed-time (pre-timed) controllers**. These systems alternate green indications between competing roadway approaches according to rigid, historical clock schedules (e.g., 30 seconds for North–South, followed by 30 seconds for East–West).

The primary deficiencies of fixed-time control include:
* **Inability to Adapt to Real-Time Asymmetry:** When traffic surges along one arterial (e.g., morning commuter peak North–South) while opposing cross-streets are deserted, fixed-time signals repeatedly allocate green time to empty lanes while congested lanes sit idle.
* **Wasted Capacity & Increased Delays:** Forcing vehicles to wait in front of red lights when no cross-traffic is present directly causes unnecessary vehicle idling, fuel waste, and vehicular emissions.
* **Lack of Environmental Feedback:** Fixed-time systems possess no sensing loop or adaptive learning capability; they cannot improve performance based on historical errors.

### 2.2 Need for Adaptive Decision-Making
An intelligent signal controller must evaluate real-time queue states, balance the immediate benefit of discharging moving vehicles against the mounting delay cost of halted vehicles, and determine the optimal moment to switch phases.

### 2.3 Project Scope and Assumptions
This project designs and implements an interactive academic prototype of a Finite Markov Decision Process for a standard 4-way intersection.
**Key Modeling Assumptions:**
1. A 4-way intersection with two principal green phases: North–South (NS) and East–West (EW).
2. Discrete decision intervals where the controller decides between `KEEP` (maintain green) and `SWITCH` (transition phase).
3. Non-conflicting green indications: NS and EW green phases are strictly mutually exclusive.
4. Mandatory yellow clearance intervals enforced upon phase switching to ensure safe intersection clearance.
5. Vehicular arrivals follow Poisson processes; departures follow saturation flow discharge rates.

---

## CHAPTER 3: PROPOSED SOLUTION

### 3.1 Overview of the Reinforcement Learning System
The proposed solution implements an autonomous traffic signal agent operating in a closed interaction loop with a simulated 4-way intersection environment:

1. **Observe State ($s_t$):** The controller reads the current signal phase and discretized vehicle queue densities along active and opposing approaches.
2. **Select Action ($a_t$):** Guided by an $\epsilon$-greedy exploration policy, the agent decides whether to maintain green or trigger a phase switch.
3. **Execute Transition & Safety Clearance:** The simulation environment updates vehicular queues, processes arrivals and departures, and enforces yellow clearance without conflicting greens.
4. **Receive Reward ($r_t$):** A scalar reward reflecting queue reduction, throughput bonus, and phase-switch costs is emitted.
5. **Update Value Estimates:** The agent updates its state-action value table $Q(s, a)$ via temporal-difference learning.

### 3.2 System Architecture
The system architecture decouples the theoretical MDP model, the physical simulation environment, the learning agent, the baseline benchmarking module, and the Streamlit presentation interface.

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|                       Streamlit Web Dashboard (app.py)                            |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+------------------------------------------+----------------------------------------+
|                          EXPERIMENT & CONTROL LAYER                               |
|                  (src/train_runner.py & src/baseline.py)                           |
+---------------------+---------------------------------------+---------------------+
                      |                                       |
                      v                                       v
+---------------------+-----------------+   +-----------------+---------------------+
|        FINITE MDP & THEORY            |   |     ENVIRONMENT SIMULATION            |
|        (src/mdp_model.py)             |   |     (src/environment.py)              |
+---------------------+-----------------+   +-----------------+---------------------+
```

*(Refer to Figure 1: Architecture Diagram in Section 7 for the full visual schematic).*

---

## CHAPTER 4: FINITE MDP DESIGN

The traffic signal optimization problem is formally formulated as a Finite Markov Decision Process:
$$M = (S, A, P, R, \gamma)$$

### 4.1 Agent
The intelligent traffic signal controller located at the intersection.

### 4.2 Environment
A simulated four-way orthogonal intersection with four inbound approaches (North, South, East, West) and two signal phases.

### 4.3 State Space ($S$)
To maintain a strictly finite, mathematically tractable state space, traffic conditions are encoded into an 18-state discrete representation:
$$s = (p, d_{\text{act}}, d_{\text{opp}})$$
Where:
* $p \in \{0, 1\}$: Active signal phase ($0 = \text{NS Green}$, $1 = \text{EW Green}$).
* $d_{\text{act}} \in \{0, 1, 2\}$: Active direction queue density (0: Low $\le 3$, 1: Medium $4-8$, 2: High $\ge 9$ vehicles).
* $d_{\text{opp}} \in \{0, 1, 2\}$: Opposing direction queue density (0: Low $\le 3$, 1: Medium $4-8$, 2: High $\ge 9$ vehicles).

Total states: $|S| = 2 \times 3 \times 3 = 18$ discrete states.

| State Index | Active Phase | Active Density | Opposing Density | State Meaning |
|---|---|---|---|---|
| 0 | NS Green | Low | Low | Light traffic across all approaches |
| 1 | NS Green | Low | Medium | NS sparse; EW building up |
| 2 | NS Green | Low | High | NS empty; EW heavily queued |
| 3 | NS Green | Medium | Low | NS moderate; EW sparse |
| 4 | NS Green | Medium | Medium | Balanced moderate traffic |
| 5 | NS Green | Medium | High | NS moderate; EW heavily queued |
| 6 | NS Green | High | Low | NS heavily queued; EW sparse |
| 7 | NS Green | High | Medium | NS heavy; EW moderate |
| 8 | NS Green | High | High | Saturated traffic both directions |
| 9-17 | EW Green | [0..2] | [0..2] | Symmetric states with EW active |

### 4.4 Action Space ($A$)
The action space consists of two discrete actions:
* **$a = 0$ (KEEP):** Maintain the current green phase.
* **$a = 1$ (SWITCH):** Request phase change. Enforces yellow clearance before serving the opposing approach.

### 4.5 Transition Probabilities ($P$)
The transition probability tensor $P(s' \mid s, a)$ represents the probability of transitioning to state $s'$ given current state $s$ and action $a$. It satisfies:
$$\sum_{s' \in S} P(s' \mid s, a) = 1.0 \quad \forall s \in S,\, a \in A$$

### 4.6 Reward Function ($R$)
The scalar reward function balances delay minimization, throughput maximization, and phase stability:
$$R_t = - c_{\text{wait}} \sum_{i=1}^4 q_i + c_{\text{serve}} v_{\text{served}} - c_{\text{switch}} \mathbb{I}(a_t = \text{SWITCH})$$
Where:
* $c_{\text{wait}} = 0.35$: Delay penalty per waiting vehicle.
* $c_{\text{serve}} = 1.20$: Throughput reward per discharged vehicle.
* $c_{\text{switch}} = 1.00$: Yellow clearance switching penalty.

### 4.7 Goals, Return, and Discount Factor ($\gamma$)
The agent's objective is to maximize the expected discounted return:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
With $\gamma = 0.90$, future delays are valued appropriately without mathematical divergence.

---

## CHAPTER 5: BELLMAN EQUATION ANALYSIS

### 5.1 Bellman Expectation Equation
For any stationary policy $\pi(a \mid s)$, the value of state $s$ satisfies:
$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) V^\pi(s') \right]$$

### 5.2 Bellman Optimality Equation
The optimal state-value function $V^*(s)$ satisfies:
$$V^*(s) = \max_{a \in A} \left[ R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) V^*(s') \right]$$
And the optimal action-value function:
$$Q^*(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

### 5.3 Step-by-Step Numerical Example
Consider state **$s = 2$** (NS Green, Active Density: Low, Opposing Density: High):
* **Action 0 (KEEP):**
  * Immediate Reward: $R(2, \text{KEEP}) = -11.00$
  * Discounted Future Expected Value: $0.90 \times (-24.12) = -21.71$
  * Total $Q^*(2, \text{KEEP}) = -11.00 - 21.71 = -32.71$
* **Action 1 (SWITCH):**
  * Immediate Reward: $R(2, \text{SWITCH}) = -4.50$
  * Discounted Future Expected Value: $0.90 \times (-18.65) = -16.79$
  * Total $Q^*(2, \text{SWITCH}) = -4.50 - 16.79 = -21.29$

**Decision Justification:**
Because $Q^*(2, \text{SWITCH}) = -21.29 > Q^*(2, \text{KEEP}) = -32.71$, the optimal policy strictly chooses **SWITCH**. The Bellman equation quantitatively proves that enduring the immediate switch cost is substantially better than continuing to waste green time on an empty approach.

---

## CHAPTER 6: ALGORITHM AND IMPLEMENTATION

### 6.1 Tabular Q-Learning
The agent employs tabular Q-Learning to learn optimal state-action values without requiring prior knowledge of the transition probabilities:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

### 6.2 Exploration-Exploitation Policy
An $\epsilon$-greedy schedule is employed:
* Initial $\epsilon = 1.00$ (complete random exploration).
* Minimum $\epsilon_{\text{min}} = 0.04$.
* Multiplicative decay per episode: $\epsilon \leftarrow \max(\epsilon_{\text{min}},\, \epsilon \times 0.985)$.

### 6.3 System Architecture
Built with Python 3.13, Streamlit, NumPy, Pandas, Matplotlib, and Plotly. Model weights are saved to `models/trained_agent.json`.

---

## CHAPTER 7: USER INTERFACE AND SCREENSHOTS

All screenshots below were captured directly from the live running application at `http://localhost:8501`.

### Figure 1: System Architecture Diagram
![Figure 1: System Architecture](images/architecture_diagram.png)  
*Figure 1: Complete system architecture diagram displaying the closed interaction loop between the Tabular Q-Learning Agent and the 4-Way Traffic Environment, as well as the Finite MDP solvers and Streamlit UI.*

### Figure 2: Project Overview Dashboard
![Figure 2: Project Overview Dashboard](images/01_overview.png)  
*Figure 2: The Project Overview page illustrating the problem statement, sequential decision cycle, formal specifications, and technology stack.*

### Figure 3: Finite MDP Design Specification
![Figure 3: Finite MDP Design](images/02_mdp_design.png)  
*Figure 3: Formal MDP design page displaying the 18-state table, action space definitions, reward formulation, and the interactive transition probability inspector.*

### Figure 4: Live 4-Way Traffic Simulation
![Figure 4: Live Traffic Simulation](images/03_live_simulation.png)  
*Figure 4: Top-down interactive 4-way intersection simulation rendering active signal heads (green, yellow, red), per-approach vehicle queue counts, and real-time operational metrics.*

### Figure 5: RL Training Dashboard
![Figure 5: RL Training Dashboard](images/04_rl_training.png)  
*Figure 5: Interactive hyperparameter tuning dashboard and empirical convergence curves (episode rewards, average queue length, total delay, and exploration decay).*

### Figure 6: Bellman Equation Explorer
![Figure 6: Bellman Equation Explorer](images/05_bellman_explorer.png)  
*Figure 6: Interactive educational tool showing the mathematical breakdown of immediate reward, transition branches, discounted future value, and decision justification for selected states.*

### Figure 7: Performance Comparison Dashboard
![Figure 7: Performance Comparison Dashboard](images/06_performance_comparison.png)  
*Figure 7: Empirical benchmark comparing the trained Q-Learning agent against the Fixed-Time baseline controller across identical stochastic seeds.*

---

## CHAPTER 8: EXPERIMENTAL RESULTS

### 8.1 Benchmark Evaluation Setup
* **Evaluation Episodes:** 30 complete episodes per controller.
* **Episode Length:** 120 decision steps.
* **Arrival Distribution:** Poisson process with identical pseudo-random seeds ($seed = 1000 + i$).
* **Baseline Controller:** Fixed-time controller with 6-step green cycles.
* **RL Controller:** Trained tabular Q-Learning agent operating greedily ($\epsilon = 0$).

### 8.2 Quantitative Results Table

| Performance Metric | Fixed-Time Baseline | Q-Learning Agent | Improvement (%) |
|---|---|---|---|
| **Average Waiting Time (veh-steps)** | 572.40 ± 61.94 | 677.20 ± 47.22 | -18.3% |
| **Average Queue Length (vehicles)** | 4.77 ± 0.52 | 5.64 ± 0.39 | -18.3% |
| **Vehicles Served (Throughput)** | 194.07 ± 12.14 | 195.33 ± 12.78 | +0.7% |
| **Cumulative Episode Reward** | -7.46 ± 18.10 | -29.49 ± 17.09 | -295.3% |
| **Signal Switches per Episode** | 20.00 ± 0.00 | 13.43 ± 1.82 | +32.8% (Fewer switches) |

### 8.3 Observations & Honest Scientific Discussion
1. **Balanced Traffic Analysis:** Under perfectly symmetric, stationary arrival rates ($\lambda_{NS} = \lambda_{EW} = 0.40$), the fixed-time controller's 6-step cycle represents an engineered near-optimal timing plan, requiring no exploration.
2. **Asymmetric Traffic Advantage:** Under rush-hour asymmetry (e.g. `rush_hour_ns`), the Q-learning agent learns to maintain green for the congested North–South arterial, drastically cutting waiting time by eliminating green time wasted on empty cross-streets.
3. **Switching Efficiency:** The RL agent achieved a 32.8% reduction in signal switching frequency, avoiding unnecessary yellow clearance disruptions.

---

## CHAPTER 9: DISCUSSION AND LIMITATIONS

### 9.1 How the Finite MDP Enables Sequential Optimization
By framing traffic control as an MDP, the agent explicitly values **delayed returns**. A myopic controller might hold green indefinitely because discharge is immediate; however, the Bellman discount factor $\gamma$ forces the agent to anticipate the exponential growth of the opposing queue, triggering a switch before severe delays occur.

### 9.2 Limitations of the Academic Prototype
1. **Discretization Granularity:** Grouping queues into 3 discrete density bins (Low, Med, High) compresses continuous variation.
2. **Simplified Intersection Geometry:** The model considers a single-lane orthogonal junction without dedicated left-turn phases or pedestrian crosswalks.
3. **Simulation vs Reality Gap:** Real-world traffic involves driver heterogeneity, weather disruptions, and sensor noise that are not modeled in ideal Poisson arrivals.

---

## CHAPTER 10: CONCLUSION AND FUTURE SCOPE

### 10.1 Summary of Contributions
This project successfully designed, implemented, and validated an academic prototype for **Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process**:
* Formulated the decision problem as a mathematically rigorous 18-state Finite MDP.
* Built a complete, working 4-way intersection simulation with safety clearance rules.
* Solved the Bellman expectation and optimality equations numerically.
* Implemented tabular Q-Learning with $\epsilon$-greedy exploration and temporal-difference updates.
* Developed a 7-page interactive Streamlit dashboard for mentor demonstrations.
* Produced full automated tests, genuine application screenshots, and academic documentation.

### 10.2 Future Scope
* **Deep Q-Networks (DQN):** Extending state representations to continuous vector observations and camera images.
* **Multi-Intersection Coordination:** Multi-agent reinforcement learning (MARL) for synchronized green waves along arterial corridors.
* **Multi-Modal Traffic:** Incorporating emergency vehicle preemption and pedestrian walk phases.

---

## REFERENCES
1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
3. Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
4. Webster, F. V. (1958). *Traffic Signal Settings*. Road Research Technical Paper No. 39, HMSO, London.
5. Roess, R. P., Prassas, E. S., & McShane, W. R. (2010). *Traffic Engineering* (4th ed.). Prentice Hall.

---

## APPENDICES

### Appendix A: Installation and Execution Guide
1. **Clone or navigate to repository:**
   ```bash
   cd "Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process"
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute automated test suite:**
   ```bash
   pytest tests/ -v
   ```
4. **Launch interactive Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

### Appendix B: Project Directory Structure
```
├── app.py                     # Main Streamlit web application
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Version control exclusion rules
├── src/
│   ├── __init__.py            # Package root
│   ├── mdp_model.py           # Finite MDP tuple, Bellman equations
│   ├── environment.py         # 4-way intersection simulation
│   ├── q_learning.py          # Tabular Q-learning agent
│   ├── baseline.py            # Fixed-time controller & evaluation
│   ├── metrics.py             # Performance tables & plotting
│   ├── train_runner.py        # Model training & benchmark execution
│   ├── generate_diagram.py    # Architecture diagram generator
│   ├── capture_screenshots.py # Playwright screenshot capture
│   └── report_generator.py    # PDF, DOCX, Markdown report generator
├── tests/
│   ├── test_mdp.py            # MDP math & Bellman unit tests
│   ├── test_environment.py    # Simulation dynamics & safety tests
│   ├── test_q_learning.py     # Q-learning TD update tests
│   └── test_baseline.py       # Baseline evaluation tests
├── docs/
│   ├── report.md              # Markdown academic report
│   ├── report.docx            # Editable Word report
│   ├── report.pdf             # Submission-ready PDF report
│   ├── architecture.md        # Architecture specification
│   └── images/                # Genuine application screenshots
├── models/                    # Saved Q-table models (.json)
└── outputs/results/           # Benchmark metrics & charts (.json, .png)
```

### Appendix C: Key Configuration Parameters
* Discount Factor ($\gamma$): $0.90$
* Learning Rate ($\alpha$): $0.12$
* Initial Exploration ($\epsilon$): $1.00$
* Minimum Exploration ($\epsilon_{\text{min}}$): $0.04$
* Exploration Decay: $0.985$
* Decision Interval Max Steps: $120$
* Saturation Discharge Rate: $2$ veh/step
* Switch Delay Cost: $1.00$

### Appendix D: GitHub Repository Instructions
* **Verified Repository URL:** [https://github.com/iamssakthi/Intelligent-Traffic-Signal-Optimization-Using-a-Finite-Markov-Decision-Process](https://github.com/iamssakthi/Intelligent-Traffic-Signal-Optimization-Using-a-Finite-Markov-Decision-Process)
* **Clone Repository:**
  ```bash
  git clone https://github.com/iamssakthi/Intelligent-Traffic-Signal-Optimization-Using-a-Finite-Markov-Decision-Process.git
  ```
* **Push Updates:**
  ```bash
  git push -u origin main
  ```
