"""
Generates the professional Architecture Diagram for the Finite MDP Traffic Signal System.
Saves high-resolution PNG to docs/images/architecture_diagram.png.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_architecture_diagram():
    os.makedirs("docs/images", exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.axis("off")

    # Title Banner
    ax.text(6.0, 7.1, "INTELLIGENT TRAFFIC SIGNAL OPTIMIZATION — FINITE MDP ARCHITECTURE",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A",
            fontfamily="sans-serif")
    ax.text(6.0, 6.75, "System Components, Closed-Loop RL Interaction, and Theoretical MDP Solvers",
            ha="center", va="center", fontsize=9.5, color="#475569")

    # Box Styles
    box_blue = dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.5)
    box_green = dict(boxstyle="round,pad=0.5", facecolor="#ECFDF5", edgecolor="#10B981", linewidth=1.5)
    box_purple = dict(boxstyle="round,pad=0.5", facecolor="#F5F3FF", edgecolor="#8B5CF6", linewidth=1.5)
    box_amber = dict(boxstyle="round,pad=0.5", facecolor="#FFFBEB", edgecolor="#F59E0B", linewidth=1.5)
    box_slate = dict(boxstyle="round,pad=0.5", facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.5)

    # 1. RL Agent Box (Left)
    ax.text(2.5, 4.8,
            "REINFORCEMENT LEARNING AGENT\n\n"
            "• Tabular Q-Learning Agent\n"
            "• Q-Table: |S|=18 × |A|=2\n"
            "• ε-Greedy Action Selection\n"
            "• Bellman TD Update:\n"
            "  Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q]\n"
            "• Policy Extraction: π*(s) = argmax Q(s,a)\n"
            "• State-Value: V(s) = max_a Q(s,a)",
            ha="center", va="center", fontsize=8.5, color="#1E3A8A", bbox=box_blue)

    # 2. Environment Box (Right)
    ax.text(9.5, 4.8,
            "TRAFFIC ENVIRONMENT SIMULATOR\n\n"
            "• 4-Way Intersection (N, S, E, W)\n"
            "• Phases: NS Green (0) vs EW Green (1)\n"
            "• Queue Dynamics: Arrivals (Poisson)\n"
            "  Discharge: Saturation Flow\n"
            "• Safety Enforcement:\n"
            "  - No conflicting green phases\n"
            "  - Yellow clearance transition\n"
            "• Step metrics: delay, queues, served",
            ha="center", va="center", fontsize=8.5, color="#065F46", bbox=box_green)

    # Connecting Arrows between Agent and Environment
    # Action Arrow: Agent -> Environment (Top)
    ax.annotate("", xy=(7.2, 5.5), xytext=(4.8, 5.5),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#2563EB"))
    ax.text(6.0, 5.75, "Action a_t ∈ {0: KEEP, 1: SWITCH}",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#1D4ED8")

    # State & Reward Arrow: Environment -> Agent (Bottom)
    ax.annotate("", xy=(4.8, 4.1), xytext=(7.2, 4.1),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#059669"))
    ax.text(6.0, 3.8, "Reward r_t & Next State s_{t+1} = (Phase, ActDensity, OppDensity)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#047857")

    # 3. Finite MDP Theoretical Model Box (Bottom Left)
    ax.text(3.0, 1.8,
            "FINITE MDP THEORETICAL FORMULATION\n\n"
            "• Tuple: M = (S, A, P, R, γ)\n"
            "• State Space S: 2 × 3 × 3 = 18 States\n"
            "• Action Space A: {KEEP, SWITCH}\n"
            "• Transition Model P(s'|s,a): Σ P = 1.0\n"
            "• Reward Function R(s,a)\n"
            "• Bellman Optimality: V*(s) = max_a [R + γ Σ P V*]\n"
            "• Value Iteration & Policy Evaluation",
            ha="center", va="center", fontsize=8.2, color="#5B21B6", bbox=box_purple)

    # 4. Evaluation & UI Dashboard (Bottom Right)
    ax.text(9.0, 1.8,
            "STREAMLIT INTERACTIVE DASHBOARD & BENCHMARKS\n\n"
            "• Page 1: Problem Overview & Objectives\n"
            "• Page 2: Finite MDP Formal Design & Tables\n"
            "• Page 3: Live 4-Way Traffic Simulation\n"
            "• Page 4: Interactive Q-Learning Training\n"
            "• Page 5: Bellman Equation Step-by-Step Explorer\n"
            "• Page 6: Performance Benchmark (vs Fixed-Time)\n"
            "• Page 7: Academic Report Generator & Export",
            ha="center", va="center", fontsize=8.2, color="#92400E", bbox=box_amber)

    # Connector Lines
    ax.annotate("", xy=(3.0, 3.3), xytext=(3.0, 3.0),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#64748B"))
    ax.annotate("", xy=(9.0, 3.3), xytext=(9.0, 3.0),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#64748B"))

    plt.tight_layout()
    output_path = "docs/images/architecture_diagram.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram saved to {output_path}")


if __name__ == "__main__":
    generate_architecture_diagram()
