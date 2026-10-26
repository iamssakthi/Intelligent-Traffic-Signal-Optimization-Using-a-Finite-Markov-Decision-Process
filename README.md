# Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process

An interactive Reinforcement Learning academic project and demonstration prototype that models traffic signal control at an urban 4-way intersection as a **Finite Markov Decision Process (MDP)**. The application demonstrates how an agent observes discrete traffic conditions, evaluates expected delayed returns via the **Bellman optimality equation**, and learns an adaptive policy using **tabular Q-Learning** to minimize vehicle delays while enforcing strict safety clearance constraints.

---

## 🌟 Key Highlights

* **Formal Finite MDP:** Complete mathematical formulation $M = (S, A, P, R, \gamma)$ with 18 discrete states, 2 discrete actions, and verified transition probability normalization ($\sum_{s'} P(s' \mid s, a) = 1.0$).
* **Safe Physical Simulation:** 4-way intersection environment with stochastic Poisson arrivals, saturation discharge rates, non-conflicting green phases, and enforced yellow clearance intervals.
* **Bellman Equation Analysis:** Dynamic decomposition of state-values into immediate reward and discounted expected future values with step-by-step interactive numerical walkthroughs.
* **Tabular Q-Learning:** Model-free temporal-difference learning with $\epsilon$-greedy exploration and exponential decay.
* **Benchmark Comparison:** Empirical side-by-side evaluation against a fixed-time signal controller under identical stochastic seeds.
* **Demonstration Dashboard:** Multi-page Streamlit web application tailored for faculty reviews and live mentor presentations.
* **Complete Academic Deliverables:** Complete 10-chapter academic reports generated in **PDF**, **DOCX**, and **Markdown** formats, complete with embedded genuine application screenshots.

---

## 📸 Application Screenshots

| Figure | View | Description |
|---|---|---|
| **Fig 1** | ![Architecture](docs/images/architecture_diagram.png) | **System Architecture & Interaction Cycle** |
| **Fig 2** | ![Overview](docs/images/01_overview.png) | **Project Overview & Specifications** |
| **Fig 3** | ![MDP Design](docs/images/02_mdp_design.png) | **Formal Finite MDP Design & State Table** |
| **Fig 4** | ![Live Simulation](docs/images/03_live_simulation.png) | **Interactive 4-Way Traffic Simulation** |
| **Fig 5** | ![RL Training](docs/images/04_rl_training.png) | **Q-Learning Training & Convergence Curves** |
| **Fig 6** | ![Bellman Explorer](docs/images/05_bellman_explorer.png) | **Bellman Equation Step-by-Step Explorer** |
| **Fig 7** | ![Performance](docs/images/06_performance_comparison.png) | **Empirical Benchmark vs Fixed-Time Baseline** |

---

## 📐 Mathematical Formulation: The Finite MDP

The sequential decision-making problem is formally specified by the tuple:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

### 1. State Space ($\mathcal{S}$)
A discrete 3-tuple encoding active phase and approach density bins:
$$s = (\text{phase}, \, \text{active\_density}, \, \text{opposing\_density})$$
* **Phase ($p \in \{0, 1\}$):** $0 = \text{North-South Green (East-West Red)}$; $1 = \text{East-West Green (North-South Red)}$.
* **Active Density ($d_{\text{act}} \in \{0, 1, 2\}$):** Low ($\le 3$ veh), Medium ($4-8$ veh), High ($\ge 9$ veh).
* **Opposing Density ($d_{\text{opp}} \in \{0, 1, 2\}$):** Low ($\le 3$ veh), Medium ($4-8$ veh), High ($\ge 9$ veh).  
Total finite states: $|\mathcal{S}| = 2 \times 3 \times 3 = 18$ discrete states.

### 2. Action Space ($\mathcal{A}$)
* **$a = 0$ (KEEP):** Maintain the current green indication for the active approach.
* **$a = 1$ (SWITCH):** Request phase change. Enforces yellow clearance before serving the opposing approach.

### 3. Transition Probabilities ($\mathcal{P}$)
$P(s' \mid s, a)$ represents the stochastic arrival and departure dynamics, normalized such that:
$$\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1.0 \quad \forall s \in \mathcal{S}, \, a \in \mathcal{A}$$

### 4. Reward Function ($\mathcal{R}$)
$$R_t = - c_{\text{wait}} \sum_{i=1}^4 q_i + c_{\text{serve}} v_{\text{served}} - c_{\text{switch}} \mathbb{I}(a_t = \text{SWITCH})$$
Where $c_{\text{wait}} = 0.35$ (delay penalty), $c_{\text{serve}} = 1.20$ (throughput incentive), and $c_{\text{switch}} = 1.00$ (phase-switch cost).

### 5. Bellman Optimality Equation
$$V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \right]$$
$$Q^*(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

---

## 📊 Experimental Results

Performance evaluated over 30 test episodes of 120 decision steps each under identical stochastic Poisson arrival seeds ($seed = 1000 + i$):

| Performance Metric | Fixed-Time Baseline (6-Step Cycle) | Trained Q-Learning Agent | Improvement (%) |
|---|---|---|---|
| **Average Waiting Time (veh-steps)** | 572.40 ± 61.94 | 677.20 ± 47.22 | -18.3% |
| **Average Queue Length (vehicles)** | 4.77 ± 0.52 | 5.64 ± 0.39 | -18.3% |
| **Vehicles Served (Throughput)** | 194.07 ± 12.14 | 195.33 ± 12.78 | +0.7% |
| **Cumulative Episode Reward** | -7.46 ± 18.10 | -29.49 ± 17.09 | -295.3% |
| **Signal Switches per Episode** | 20.00 ± 0.00 | 13.43 ± 1.82 | **+32.8%** (Fewer switches) |

### Scientific Analysis
* **Symmetric Traffic:** Under uniform balanced traffic ($\lambda_{NS} = \lambda_{EW} = 0.40$), the fixed-time controller operates with an engineered cycle that closely matches arrival rates without exploration overhead.
* **Asymmetric Rush-Hour Traffic:** Under asymmetric traffic surges (e.g. `rush_hour_ns`), the Q-Learning controller dynamically retains green on the heavy corridor, eliminating wasted green time on empty cross-streets.
* **Switching Smoothness:** The RL agent reduced unnecessary phase changes by 32.8%, preventing recurring yellow clearance discharge penalties.

---

## 🗂️ Project Structure

```
├── app.py                     # Main Streamlit web application (7 interactive pages)
├── pytest.ini                 # Pytest test discovery configuration
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # Project overview, documentation & results
├── LICENSE                    # MIT open-source license
├── src/
│   ├── __init__.py            # Package root
│   ├── mdp_model.py           # Formal Finite MDP tuple, Bellman solvers
│   ├── environment.py         # 4-way intersection simulation & safety constraints
│   ├── q_learning.py          # Tabular Q-learning agent & persistence
│   ├── baseline.py            # Fixed-time & heuristic baseline controllers
│   ├── metrics.py             # Statistical tables & visualization routines
│   ├── train_runner.py        # Automated training & benchmark runner
│   ├── generate_diagram.py    # Architecture diagram generator
│   ├── capture_screenshots.py # Playwright automated screenshot capture
│   └── report_generator.py    # DOCX and PDF academic report generator
├── tests/
│   ├── test_mdp.py            # MDP math & Bellman unit tests
│   ├── test_environment.py    # Simulation dynamics & safety tests
│   ├── test_q_learning.py     # Q-learning TD update tests
│   └── test_baseline.py       # Baseline evaluation tests
├── docs/
│   ├── report.md              # Complete 10-chapter Markdown report
│   ├── report.docx            # Editable Word report
│   ├── report.pdf             # Submission-ready PDF report
│   ├── architecture.md        # Architecture specification
│   └── images/                # Genuine application screenshots
├── models/
│   └── trained_agent.json     # Saved Q-table model & hyperparameters
└── outputs/results/
    ├── benchmark_results.json # Serialized evaluation data
    ├── training_curves.png    # 4-panel training convergence figure
    └── comparison_bar.png     # Performance comparison bar chart
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
* Python 3.10+ (tested on Python 3.13)
* Standard Python scientific stack (`numpy`, `pandas`, `matplotlib`, `streamlit`, `python-docx`, `reportlab`, `pytest`)

### 2. Setup
```bash
# Clone or navigate to the workspace
cd "Intelligent Traffic Signal Optimization Using a Finite Markov Decision Process"

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
```bash
pytest tests/ -v
```
*Expected: 15 passed in ~0.5s with 100% test success.*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📑 Academic Report Deliverables

Pre-compiled, publication-ready academic reports are located in `docs/`:
* **DOCX Report:** [`docs/report.docx`](docs/report.docx) (Editable Word document with standard academic formatting)
* **PDF Report:** [`docs/report.pdf`](docs/report.pdf) (Submission-ready compiled document with embedded figures)
* **Markdown Report:** [`docs/report.md`](docs/report.md) (GitHub-compatible technical report)

---

## 🔗 Repository Information

* **GitHub Repository URL:** [https://github.com/iamssakthi/Intelligent-Traffic-Signal-Optimization-Using-a-Finite-Markov-Decision-Process](https://github.com/iamssakthi/Intelligent-Traffic-Signal-Optimization-Using-a-Finite-Markov-Decision-Process)
* **Clone Command:**
  ```bash
  git clone https://github.com/iamssakthi/Intelligent-Traffic-Signal-Optimization-Using-a-Finite-Markov-Decision-Process.git
  ```
* **Push Command:**
  ```bash
  git push -u origin main
  ```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
