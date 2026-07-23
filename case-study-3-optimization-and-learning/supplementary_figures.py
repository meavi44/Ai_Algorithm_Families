"""
=====================================================================
ExtraGraphs.py
=====================================================================
Project : AI-Powered Autonomous Smart Agriculture
          Swarm Intelligence + Reinforcement Learning
Purpose : SUPPLEMENTARY figures that reflect the *whole system* and the
          theory behind it, for the lab demonstration / IEEE-style report.
---------------------------------------------------------------------
This file adds seven extra visualisations on top of the three core
algorithm files.  It does NOT re-implement any algorithm -- it imports
the existing (already-verified) code and the saved PSO/VI/QL results so
everything stays reproducible and consistent.

Generated figures
    1. system_pipeline.png        -- end-to-end architecture diagram
    2. vi_vs_ql_trajectory.png    -- optimal vs learned path (same grid)
    3. vi_vs_ql_policy.png        -- optimal policy vs learned policy
    4. vi_value_propagation.png   -- Bellman value spreading over iters
    5. ql_epsilon_decay.png       -- exploration -> exploitation schedule
    6. all_robots_navigation.png  -- the full multi-robot mission

RUN ORDER : run PSO.py, ValueIteration.py, QLearning.py first.
=====================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Reuse the already-implemented, verified code.
from value_iteration import (
    GridWorld, value_iteration, extract_policy, run_mission,
    ARROW, CSV_DIR, SEED,
)
from q_learning import train_q_learning, evaluate_policy

np.random.seed(SEED)

# These supplementary figures are filed under the algorithm they belong to:
#   System/     -> whole-system diagrams
#   ValueIteration/, QLearning/ -> per-algorithm supplements
#   Comparison/ -> VI-vs-QL cross-algorithm figures
FIG_SYS = os.path.join("outputs", "figures", "System")
FIG_VI = os.path.join("outputs", "figures", "ValueIteration")
FIG_QL = os.path.join("outputs", "figures", "QLearning")
FIG_CMP = os.path.join("outputs", "figures", "Comparison")
for _d in (FIG_SYS, FIG_VI, FIG_QL, FIG_CMP):
    os.makedirs(_d, exist_ok=True)


# =====================================================================
# small shared helper for grid plots
# =====================================================================
def _draw_grid_markers(ax, gw):
    """Draw obstacles / flowers / charging / start on a grid axis."""
    for (r, c) in gw.obstacles:
        ax.scatter(c, r, marker="s", c="black", s=110)
    for (r, c) in gw.flowers:
        ax.scatter(c, r, marker="*", c="magenta", s=190, edgecolors="k",
                   zorder=5)
    for (r, c) in gw.charging:
        ax.scatter(c, r, marker="P", c="cyan", s=150, edgecolors="k",
                   zorder=5)
    ax.scatter(gw.start[1], gw.start[0], marker="s", c="lime", s=130,
               edgecolors="k", zorder=5)


# =====================================================================
# 1. SYSTEM PIPELINE / ARCHITECTURE DIAGRAM
# =====================================================================
def plot_pipeline():
    """
    Block diagram showing how the two AI paradigms connect:
    global optimisation (PSO) feeds local sequential decision making
    (Value Iteration / Q-learning).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("Autonomous Pollination System -- End-to-End Pipeline",
                 fontsize=15, fontweight="bold", pad=12)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.05,rounding_size=0.15",
                     linewidth=1.5, edgecolor="black", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.5, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=18,
                     linewidth=1.8, color="#444"))

    # --- top row: task allocation (population-based optimization) -----
    box(0.2, 6.9, 2.5, 1.5, "Synthetic\nGreenhouse\n(flowers, robots,\nobstacles)",
        "#dbeafe")
    box(3.0, 6.9, 2.6, 1.5, "PHASE 1\nTask Allocation\nPSO  vs  GA", "#93c5fd")
    box(5.9, 6.9, 2.5, 1.5, "Assigned Flower\nClusters\n(flower -> robot)",
        "#dbeafe")

    arrow(2.7, 7.65, 3.0, 7.65)
    arrow(5.6, 7.65, 5.9, 7.65)
    arrow(7.15, 6.9, 7.15, 5.8)         # down to the shared environment

    # --- middle: shared environment ----------------------------------
    box(5.9, 4.3, 2.6, 1.4, "Shared Grid-World\nMDP (one robot)", "#bbf7d0")

    # --- navigation branches (dynamic programming / RL) --------------
    box(3.4, 2.0, 2.8, 1.5, "PHASE 2a\nValue Iteration\n(model-based)", "#86efac")
    box(8.2, 2.0, 2.8, 1.5, "PHASE 2b\nQ-learning\n(model-free)", "#fdba74")

    arrow(6.6, 4.3, 5.3, 3.5)           # grid-world -> VI
    arrow(7.8, 4.3, 9.1, 3.5)           # grid-world -> QL

    # --- outcome (diamond re-joins under both branches) --------------
    box(5.85, 0.1, 2.7, 1.2, "Completed\nPollination", "#fde68a")
    arrow(5.3, 2.0, 6.7, 1.3)           # VI -> completed
    arrow(9.1, 2.0, 7.7, 1.3)           # QL -> completed

    # side captions
    ax.text(4.3, 6.2, "GLOBAL OPTIMIZATION\n(Population-based)",
            ha="center", fontsize=9, color="#1d4ed8", style="italic")
    ax.text(12.3, 4.7, "SEQUENTIAL\nDECISION MAKING\n(Dynamic Programming / RL)",
            ha="center", fontsize=9, color="#15803d", style="italic")

    fig.savefig(os.path.join(FIG_SYS, "system_pipeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 2 & 3.  need VI + QL results on the shared environment
# =====================================================================
def compute_navigation_artifacts():
    """Rebuild the shared env; get VI + QL trajectories and policies."""
    gw = GridWorld()

    # --- Value Iteration: optimal value, policy and mission path -------
    U, _, _ = value_iteration(gw, set(gw.flowers))
    vi_policy = extract_policy(gw, U, set(gw.flowers))
    vi_mission = run_mission(gw)

    # --- Q-learning: train, then take one greedy trajectory + policy --
    Q, _, fb, fm = train_q_learning(gw)
    ql_test = evaluate_policy(gw, Q, fb, fm)
    # learned policy for the "no flower collected yet" state (mask = 0)
    ql_policy = np.argmax(Q[:, :, 0, :], axis=2)

    return gw, U, vi_policy, vi_mission, ql_policy, ql_test


def plot_vi_vs_ql_trajectory(gw, vi_mission, ql_test):
    """Overlay the optimal (VI) and learned (QL) robot paths on one grid."""
    n = gw.n
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.grid(alpha=0.3)

    vt = np.array(vi_mission["trajectory"])
    ax.plot(vt[:, 1], vt[:, 0], "-o", c="#1f77b4", ms=3, lw=2,
            label=f"Value Iteration ({vi_mission['steps']} steps)")
    if ql_test["trajectory"] is not None:
        qt = np.array(ql_test["trajectory"])
        ax.plot(qt[:, 1], qt[:, 0], "-s", c="#ff7f0e", ms=3, lw=2,
                alpha=0.8,
                label=f"Q-learning ({len(qt)-1} steps)")

    _draw_grid_markers(ax, gw)
    ax.set_title("Optimal (VI) vs Learned (Q-learning) Robot Path",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_CMP, "vi_vs_ql_trajectory.png"), dpi=150)
    plt.close(fig)


def plot_vi_vs_ql_policy(gw, vi_policy, ql_policy):
    """Side-by-side arrow maps: VI optimal policy vs QL learned policy."""
    n = gw.n
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    fig.suptitle("Optimal Policy (Value Iteration) vs Learned Policy "
                 "(Q-learning)", fontsize=14, fontweight="bold")

    for ax, policy, title in [
            (axes[0], vi_policy, "Value Iteration (model-based)"),
            (axes[1], ql_policy, "Q-learning (model-free, mask=0)")]:
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.grid(alpha=0.3)
        for r in range(n):
            for c in range(n):
                if (r, c) in gw.obstacles or (r, c) in gw.flowers \
                        or (r, c) in gw.charging:
                    continue
                a = int(policy[r, c])
                if a >= 0:
                    ax.text(c, r, ARROW[a], ha="center", va="center",
                            fontsize=9, color="gray")
        _draw_grid_markers(ax, gw)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_CMP, "vi_vs_ql_policy.png"), dpi=150)
    plt.close(fig)


# =====================================================================
# 5. VALUE PROPAGATION SNAPSHOTS  (Bellman backups spreading over time)
# =====================================================================
def plot_value_propagation(gw):
    """Value-function heatmaps after 1, 5, 15 and (converged) iterations."""
    snaps = [1, 5, 15, 1000]                 # 1000 -> runs until convergence
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Value Iteration -- Reward Propagating Outward from Flowers "
                 "(Bellman backups)", fontsize=14, fontweight="bold")

    for ax, k in zip(axes, snaps):
        U, deltas, n_iter = value_iteration(gw, set(gw.flowers), max_iter=k)
        im = ax.imshow(U, origin="lower", cmap="viridis", vmin=0, vmax=10)
        for (r, c) in gw.obstacles:
            ax.scatter(c, r, marker="x", c="red", s=25)
        for (r, c) in gw.flowers:
            ax.scatter(c, r, marker="*", c="magenta", s=90, edgecolors="k")
        label = f"converged ({n_iter} iters)" if k == 1000 else f"iteration {k}"
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.02, label="U(s)")
    fig.savefig(os.path.join(FIG_VI, "vi_value_propagation.png"), dpi=150)
    plt.close(fig)


# =====================================================================
# 6. EPSILON DECAY CURVE  (exploration -> exploitation)
# =====================================================================
def plot_epsilon_decay():
    """Plot the epsilon schedule recorded during Q-learning training."""
    hist = pd.read_csv(os.path.join(CSV_DIR, "ql_training_history.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist["episode"], hist["epsilon"], color="#d62728", lw=2)
    ax.fill_between(hist["episode"], hist["epsilon"], alpha=0.15,
                    color="#d62728")
    ax.set_title("Q-learning -- Epsilon Decay (Exploration -> Exploitation)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("episode"); ax.set_ylabel("epsilon (exploration rate)")
    ax.grid(alpha=0.3)
    ax.text(0.55, 0.75, "high epsilon = explore\nlow epsilon = exploit",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_QL, "ql_epsilon_decay.png"), dpi=150)
    plt.close(fig)


# =====================================================================
# 7. ALL-ROBOTS NAVIGATION  (the full multi-robot mission)
# =====================================================================
def plot_all_robots_navigation():
    """Run the VI mission for every robot and plot all paths on one grid."""
    layout = pd.read_csv(os.path.join(CSV_DIR, "greenhouse_layout.csv"))
    n_robots = int((layout["type"] == "robot_depot").sum())
    colours = plt.cm.tab10(np.linspace(0, 1, n_robots))

    fig, ax = plt.subplots(figsize=(9, 9))
    base = GridWorld(robot_id=0)
    n = base.n
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.grid(alpha=0.3)

    # shared static elements
    for (r, c) in base.obstacles:
        ax.scatter(c, r, marker="s", c="black", s=110, zorder=4)
    for (r, c) in base.charging:
        ax.scatter(c, r, marker="P", c="cyan", s=160, edgecolors="k", zorder=6)

    total_steps = 0
    for rid in range(n_robots):
        gw = GridWorld(robot_id=rid)
        mission = run_mission(gw)
        total_steps += mission["steps"]
        tr = np.array(mission["trajectory"])
        ax.plot(tr[:, 1], tr[:, 0], "-", color=colours[rid], lw=1.8,
                alpha=0.85, label=f"Robot {rid+1} ({mission['steps']} steps)")
        for (fr, fc) in gw.flowers:
            ax.scatter(fc, fr, marker="*", color=colours[rid], s=170,
                       edgecolors="k", zorder=5)
        ax.scatter(gw.start[1], gw.start[0], marker="s",
                   color=colours[rid], s=120, edgecolors="k", zorder=6)

    ax.set_title(f"Full Multi-Robot Pollination Mission "
                 f"(Value Iteration)\ntotal steps across all robots "
                 f"= {total_steps}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_SYS, "all_robots_navigation.png"), dpi=150)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 65)
    print(" GENERATING SUPPLEMENTARY FIGURES")
    print("=" * 65)

    need = ["pso_flower_assignment.csv", "greenhouse_layout.csv",
            "ql_training_history.csv", "vi_results.csv"]
    for f in need:
        if not os.path.exists(os.path.join(CSV_DIR, f)):
            print(f"[!] missing {f}. Run PSO.py, ValueIteration.py and "
                  f"QLearning.py first.")
            return

    print("[1/6] system pipeline diagram ...")
    plot_pipeline()

    print("[2-3/6] computing VI + QL navigation artifacts ...")
    gw, U, vi_policy, vi_mission, ql_policy, ql_test = \
        compute_navigation_artifacts()
    plot_vi_vs_ql_trajectory(gw, vi_mission, ql_test)
    plot_vi_vs_ql_policy(gw, vi_policy, ql_policy)

    print("[4/6] value propagation snapshots ...")
    plot_value_propagation(gw)
    print("[5/6] epsilon decay curve ...")
    plot_epsilon_decay()
    print("[6/6] all-robots navigation ...")
    plot_all_robots_navigation()

    print("\n" + "=" * 65)
    print(" SUPPLEMENTARY FIGURES COMPLETE")
    print(f"  saved to -> figures/(System, ValueIteration, QLearning, Comparison)")
    print("=" * 65)


if __name__ == "__main__":
    main()
