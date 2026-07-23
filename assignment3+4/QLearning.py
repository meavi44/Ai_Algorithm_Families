"""
=====================================================================
QLearning.py
=====================================================================
Project : AI-Powered Autonomous Smart Agriculture
          Swarm Intelligence + Reinforcement Learning
Phase   : 2 -- LOCAL NAVIGATION  (Q-learning, model-free)
Author  : AI Laboratory (CSE 4101)
---------------------------------------------------------------------
ROLE OF THIS FILE
---------------------------------------------------------------------
This file solves the SAME navigation problem as ValueIteration.py --
guide one robot to pollinate all its PSO-assigned flowers and return
to a charging station -- but now the environment is treated as UNKNOWN.

The robot has no transition model.  It learns purely from trial-and-
error interaction using **Q-learning** with an epsilon-greedy policy:

    Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

To visit ALL flowers, the state is augmented with a bitmask recording
which flowers have already been pollinated, so the memory-less robot
can still complete the full multi-flower mission.

The GridWorld environment is imported from ValueIteration.py, so Value
Iteration and Q-learning share the identical grid, obstacles, flowers
and reward constants -- only the learning method differs.  At the end
this file produces the final Value-Iteration vs Q-learning comparison.
---------------------------------------------------------------------
DEPENDENCIES : Python, NumPy, Matplotlib, Pandas
REPRODUCIBLE : fixed seed, deterministic environment from PSO output
=====================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the SHARED environment and constants so both RL algorithms are
# evaluated on exactly the same problem.
from ValueIteration import (
    GridWorld, ROBOT_ID, GRID,
    ACTIONS, DELTA, SLIP, MOVE_PROB, SLIP_PROB,
    GOAL_REWARD, CHARGE_REWARD, STEP_COST, COLLISION_PENALTY,
    ENERGY_PER_STEP, CSV_DIR, SEED,
)


# =====================================================================
# 0. REPRODUCIBILITY
# =====================================================================
np.random.seed(SEED)

# Figures are organised per algorithm: Q-learning's own figures go to
# figures/QLearning, and the VI-vs-QL comparison goes to figures/Comparison.
FIG_QL = os.path.join("outputs", "figures", "QLearning")
FIG_CMP = os.path.join("outputs", "figures", "Comparison")
for _d in (FIG_QL, FIG_CMP):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------
# Q-learning hyperparameters (defaults; also tuned later).
# ---------------------------------------------------------------------
ALPHA = 0.15          # learning rate
GAMMA = 0.95          # discount factor
EPS_START = 1.0       # initial exploration rate
EPS_MIN = 0.05        # minimum exploration rate
EPS_DECAY = 0.9990    # multiplicative epsilon decay per episode
N_EPISODES = 6000     # number of training episodes
MAX_STEPS = 300       # step cap per episode
N_TEST = 200          # greedy test episodes for evaluation


# =====================================================================
# 1. ENVIRONMENT TRANSITION SAMPLER (model-free view)
# =====================================================================
def sample_move(gw, cell, action, rng):
    """
    Sample one stochastic transition (actuator noise) and report whether
    the robot collided with a wall/obstacle.  This is the ONLY way the
    Q-learning agent interacts with the world -- it never sees T(s,a,s').
    """
    p = rng.random()
    if p < MOVE_PROB:                       # intended move
        act = action
    elif p < MOVE_PROB + SLIP_PROB:         # slip to one side
        act = SLIP[action][0]
    else:                                   # slip to the other side
        act = SLIP[action][1]

    dr, dc = DELTA[act]
    target = (cell[0] + dr, cell[1] + dc)
    if gw.passable(target):
        return target, False
    return cell, True                       # blocked -> stay, collision


# =====================================================================
# 2. Q-LEARNING (from scratch)
# =====================================================================
def build_flower_index(gw):
    """Map each target-flower cell to a bit index 0..N-1."""
    return {cell: i for i, cell in enumerate(gw.flowers)}


def env_reset(gw):
    """Start of an episode: robot at its depot, no flowers pollinated."""
    return gw.start, 0                      # (cell, visited-bitmask)


def env_step(gw, cell, mask, action, flower_bit, full_mask, rng):
    """
    Execute one action and return (next_cell, next_mask, reward, done,
    collided).  Reward logic (shared conceptually with Value Iteration):

        * land on an un-pollinated flower   -> +GOAL_REWARD, set its bit
        * all flowers done & reach charging -> +CHARGE_REWARD, done
        * collision with obstacle/wall      -> COLLISION_PENALTY
        * otherwise                         -> STEP_COST
    """
    nxt, collided = sample_move(gw, cell, action, rng)

    if collided:
        return cell, mask, COLLISION_PENALTY, False, True

    # newly pollinated flower?
    if nxt in flower_bit and not (mask >> flower_bit[nxt]) & 1:
        new_mask = mask | (1 << flower_bit[nxt])
        return nxt, new_mask, GOAL_REWARD, False, False

    # safe return once every flower is pollinated
    if mask == full_mask and nxt in gw.charging:
        return nxt, mask, CHARGE_REWARD, True, False

    return nxt, mask, STEP_COST, False, False


def train_q_learning(gw, alpha=ALPHA, gamma=GAMMA,
                     eps_start=EPS_START, eps_min=EPS_MIN,
                     eps_decay=EPS_DECAY, n_episodes=N_EPISODES,
                     max_steps=MAX_STEPS, seed=SEED, record=True):
    """
    Train a Q-table over the augmented state (row, col, visited-mask).

    Returns the learned Q-table plus per-episode training histories that
    are used to draw the learning curves.
    """
    rng = np.random.RandomState(seed)
    n = gw.n
    N = len(gw.flowers)
    full_mask = (1 << N) - 1
    flower_bit = build_flower_index(gw)

    # Q-table: (row, col, mask, action).  Small enough to store directly.
    Q = np.zeros((n, n, 1 << N, len(ACTIONS)))

    hist = {"reward": [], "length": [], "success": [],
            "collisions": [], "battery": [], "epsilon": []}

    eps = eps_start
    for ep in range(n_episodes):
        cell, mask = env_reset(gw)
        ep_reward = 0.0
        ep_collisions = 0
        done = False
        step = 0

        for step in range(max_steps):
            # --- epsilon-greedy action selection -----------------------
            if rng.random() < eps:
                action = rng.randint(len(ACTIONS))           # explore
            else:
                action = int(np.argmax(Q[cell[0], cell[1], mask]))  # exploit

            nxt, new_mask, reward, done, collided = env_step(
                gw, cell, mask, action, flower_bit, full_mask, rng)

            # --- Q-learning update (off-policy TD) ---------------------
            best_next = 0.0 if done else np.max(Q[nxt[0], nxt[1], new_mask])
            td_target = reward + gamma * best_next
            Q[cell[0], cell[1], mask, action] += \
                alpha * (td_target - Q[cell[0], cell[1], mask, action])

            cell, mask = nxt, new_mask
            ep_reward += reward
            ep_collisions += int(collided)
            if done:
                break

        if record:
            hist["reward"].append(ep_reward)
            hist["length"].append(step + 1)
            hist["success"].append(int(done))
            hist["collisions"].append(ep_collisions)
            hist["battery"].append((step + 1) * ENERGY_PER_STEP)
            hist["epsilon"].append(eps)

        eps = max(eps_min, eps * eps_decay)                  # decay epsilon

    return Q, hist, flower_bit, full_mask


# =====================================================================
# 3. GREEDY EVALUATION (testing loop)
# =====================================================================
def evaluate_policy(gw, Q, flower_bit, full_mask,
                    n_test=N_TEST, max_steps=MAX_STEPS, seed=SEED + 7):
    """
    Run the learned greedy policy (epsilon = 0) for several stochastic
    test episodes and average the performance metrics.
    """
    rng = np.random.RandomState(seed)
    rewards, steps_list, successes, collisions_list, battery_list = \
        [], [], [], [], []
    best_traj = None

    for t in range(n_test):
        cell, mask = env_reset(gw)
        traj = [cell]
        ep_reward = 0.0
        ep_coll = 0
        done = False
        step = 0
        for step in range(max_steps):
            action = int(np.argmax(Q[cell[0], cell[1], mask]))   # greedy
            nxt, new_mask, reward, done, collided = env_step(
                gw, cell, mask, action, flower_bit, full_mask, rng)
            cell, mask = nxt, new_mask
            traj.append(cell)
            ep_reward += reward
            ep_coll += int(collided)
            if done:
                break
        rewards.append(ep_reward)
        steps_list.append(step + 1)
        successes.append(int(done))
        collisions_list.append(ep_coll)
        battery_list.append((step + 1) * ENERGY_PER_STEP)
        if done and best_traj is None:
            best_traj = traj                       # keep one successful run

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_battery": float(np.mean(battery_list)),
        "avg_collisions": float(np.mean(collisions_list)),
        "trajectory": best_traj,
    }


# =====================================================================
# 4. HELPER: moving average & convergence speed
# =====================================================================
def moving_average(x, w=100):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def episodes_to_converge(rewards, w=100, tol=0.95):
    """First episode where the smoothed reward reaches tol*final level."""
    ma = moving_average(rewards, w)
    if len(ma) == 0:
        return len(rewards)
    final = ma[-1]
    if final <= 0:
        return len(rewards)
    target = tol * final
    idx = np.argmax(ma >= target)
    return int(idx + w)


# =====================================================================
# 5. HYPERPARAMETER TUNING
# =====================================================================
def hyperparameter_study(gw):
    """Vary learning rate, discount and exploration decay."""
    print("\n[Hyperparameter study] Q-learning ...")
    configs = [
        # (alpha, gamma, eps_decay, episodes)
        (0.10, 0.95, 0.9990, 3000),
        (0.15, 0.95, 0.9990, 3000),
        (0.30, 0.95, 0.9990, 3000),
        (0.15, 0.90, 0.9990, 3000),
        (0.15, 0.99, 0.9990, 3000),
        (0.15, 0.95, 0.9980, 3000),
        (0.15, 0.95, 0.9995, 3000),
    ]
    rows = []
    for (a, g, d, ep) in configs:
        Q, _, fb, fm = train_q_learning(gw, alpha=a, gamma=g, eps_decay=d,
                                        n_episodes=ep, record=False)
        res = evaluate_policy(gw, Q, fb, fm, n_test=100)
        rows.append({
            "alpha": a, "gamma": g, "eps_decay": d, "episodes": ep,
            "success_rate": round(res["success_rate"], 3),
            "avg_reward": round(res["avg_reward"], 2),
            "avg_steps": round(res["avg_steps"], 1),
        })
        print(f"  alpha={a:.2f} gamma={g:.2f} decay={d:.4f} "
              f"-> success={res['success_rate']*100:5.1f}% "
              f"reward={res['avg_reward']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, "ql_hyperparameter_study.csv"), index=False)
    return df


# =====================================================================
# 6. Q-LEARNING TRAINING VISUALISATION
# =====================================================================
def plot_learning_curves(hist):
    """Reward, learning curve, episode length, success, collisions, battery."""
    rewards = hist["reward"]
    ep = np.arange(1, len(rewards) + 1)
    ma = moving_average(rewards, 100)
    ma_ep = np.arange(100, 100 + len(ma))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Q-learning Training Behaviour", fontsize=15, fontweight="bold")

    ax[0, 0].plot(ep, rewards, color="#c6dbef", lw=0.7)
    ax[0, 0].plot(ma_ep, ma, color="#08519c", lw=2, label="100-ep avg")
    ax[0, 0].set_title("Reward vs Episode")
    ax[0, 0].set_xlabel("episode"); ax[0, 0].set_ylabel("total reward")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(ma_ep, ma, color="#08519c", lw=2)
    ax[0, 1].set_title("Learning Curve (smoothed reward)")
    ax[0, 1].set_xlabel("episode"); ax[0, 1].set_ylabel("avg reward")
    ax[0, 1].grid(alpha=0.3)

    succ_ma = moving_average(hist["success"], 100) * 100
    ax[0, 2].plot(np.arange(100, 100 + len(succ_ma)), succ_ma,
                  color="#2ca02c", lw=2)
    ax[0, 2].set_title("Success Rate (rolling %)")
    ax[0, 2].set_xlabel("episode"); ax[0, 2].set_ylabel("success (%)")
    ax[0, 2].grid(alpha=0.3)

    len_ma = moving_average(hist["length"], 100)
    ax[1, 0].plot(np.arange(100, 100 + len(len_ma)), len_ma,
                  color="#9467bd", lw=2)
    ax[1, 0].set_title("Episode Length (rolling)")
    ax[1, 0].set_xlabel("episode"); ax[1, 0].set_ylabel("steps")
    ax[1, 0].grid(alpha=0.3)

    coll_ma = moving_average(hist["collisions"], 100)
    ax[1, 1].plot(np.arange(100, 100 + len(coll_ma)), coll_ma,
                  color="#d62728", lw=2)
    ax[1, 1].set_title("Collision Count (rolling)")
    ax[1, 1].set_xlabel("episode"); ax[1, 1].set_ylabel("collisions")
    ax[1, 1].grid(alpha=0.3)

    batt_ma = moving_average(hist["battery"], 100)
    ax[1, 2].plot(np.arange(100, 100 + len(batt_ma)), batt_ma,
                  color="#ff7f0e", lw=2)
    ax[1, 2].set_title("Battery Usage (rolling)")
    ax[1, 2].set_xlabel("episode"); ax[1, 2].set_ylabel("battery units")
    ax[1, 2].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_QL, "ql_training_curves.png"), dpi=150)
    plt.close(fig)


def plot_hyperparameters(df):
    """Success rate for each Q-learning hyperparameter configuration."""
    labels = [f"a{r.alpha}\ng{r.gamma}\nd{r.eps_decay}"
              for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, df["success_rate"] * 100, color="#17becf")
    ax.set_title("Q-learning Hyperparameter Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("greedy success rate (%)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_QL, "ql_hyperparameter_comparison.png"),
                dpi=150)
    plt.close(fig)


# =====================================================================
# 7. FINAL COMPARISON : VALUE ITERATION vs Q-LEARNING
# =====================================================================
def final_comparison(gw, ql_hist, ql_test, ql_time, ql_conv_episodes):
    """
    Build the comparison table, grouped bar chart and convergence
    comparison between the two navigation algorithms.
    """
    # --- load Value Iteration results produced by ValueIteration.py ----
    vi = pd.read_csv(os.path.join(CSV_DIR, "vi_results.csv")).iloc[0]
    vi_conv = pd.read_csv(os.path.join(CSV_DIR, "vi_convergence.csv"))

    # --- comparison table ---------------------------------------------
    # Both algorithms are evaluated under the SAME stochastic slip model:
    # the VI columns come from evaluate_vi_policy (optimal policy rolled out
    # with actuator noise), so this is a genuine like-for-like comparison.
    table = pd.DataFrame([
        {"metric": "Success Rate",
         "ValueIteration": round(float(vi["success_rate"]), 3),
         "QLearning": round(ql_test["success_rate"], 3)},
        {"metric": "Average Reward",
         "ValueIteration": round(float(vi["avg_reward"]), 2),
         "QLearning": round(ql_test["avg_reward"], 2)},
        {"metric": "Average Steps",
         "ValueIteration": round(float(vi["avg_steps"]), 1),
         "QLearning": round(ql_test["avg_steps"], 1)},
        {"metric": "Battery Usage",
         "ValueIteration": round(float(vi["avg_battery"]), 1),
         "QLearning": round(ql_test["avg_battery"], 1)},
        {"metric": "Execution Time (s)",
         "ValueIteration": float(vi["execution_time_s"]),
         "QLearning": round(ql_time, 3)},
        {"metric": "Convergence Speed",
         "ValueIteration": int(vi["iterations_to_converge"]),
         "QLearning": int(ql_conv_episodes)},
    ])
    table.to_csv(os.path.join(CSV_DIR, "vi_vs_ql_comparison.csv"), index=False)
    print("\n--- FINAL COMPARISON TABLE ---")
    print(table.to_string(index=False))

    # --- grouped bar chart (small multiples: different scales) ---------
    # (Execution time is intentionally NOT shown here: it is wall-clock and
    #  hardware/load dependent, so it is the only non-reproducible metric.
    #  It remains available in ql_results.csv / vi_results.csv.)
    # (Convergence speed is deliberately NOT shown here: VI counts Bellman
    #  iterations while QL counts training episodes, so a single bar would mix
    #  units.  The like-for-like convergence view is comparison_convergence.)
    metrics = ["Success Rate", "Average Reward", "Average Steps",
               "Battery Usage"]
    units = ["rate", "reward", "steps", "battery units"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Value Iteration vs Q-learning -- Performance Comparison",
                 fontsize=14, fontweight="bold")
    for i, m in enumerate(metrics):
        row = table[table["metric"] == m].iloc[0]
        a = ax[i // 2, i % 2]
        vals = [row["ValueIteration"], row["QLearning"]]
        labels = ["Value\nIteration", "Q-learning"]
        a.bar(labels, vals, color=["#1f77b4", "#ff7f0e"])
        # value labels on top of each bar for exact reading
        for j, v in enumerate(vals):
            txt = f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
            a.text(j, v, txt, ha="center", va="bottom",
                   fontsize=10, fontweight="bold")
        a.set_title(m); a.set_ylabel(units[i])
        a.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_CMP, "comparison_bar.png"), dpi=150)
    plt.close(fig)

    # --- convergence comparison (different units, side by side) --------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergence Comparison", fontsize=14, fontweight="bold")

    ax[0].plot(vi_conv["iteration"], vi_conv["max_delta"],
               color="#1f77b4", lw=2)
    ax[0].set_yscale("log")
    ax[0].set_title("Value Iteration (model-based)")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("max |U change| (log)")
    ax[0].grid(alpha=0.3)

    ma = moving_average(ql_hist["reward"], 100)
    ax[1].plot(np.arange(100, 100 + len(ma)), ma, color="#ff7f0e", lw=2)
    ax[1].axvline(ql_conv_episodes, color="k", ls="--", lw=1,
                  label=f"converged ~ep {ql_conv_episodes}")
    ax[1].set_title("Q-learning (model-free)")
    ax[1].set_xlabel("episode"); ax[1].set_ylabel("smoothed reward")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG_CMP, "comparison_convergence.png"), dpi=150)
    plt.close(fig)


# =====================================================================
# 8. MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 65)
    print(" PHASE 2 : LOCAL NAVIGATION using Q-LEARNING")
    print("=" * 65)

    # 1) rebuild the SAME shared environment used by Value Iteration
    gw = GridWorld(robot_id=ROBOT_ID)
    print(f"Grid {GRID}x{GRID} | robot {ROBOT_ID} | "
          f"{len(gw.flowers)} target flowers | "
          f"{len(gw.obstacles)} obstacles | start={gw.start}")

    # 2) train Q-learning
    print(f"\n[Training] {N_EPISODES} episodes, epsilon-greedy ...")
    t0 = time.time()
    Q, hist, fb, fm = train_q_learning(gw)
    ql_time = time.time() - t0
    print(f"  training finished in {ql_time:.2f}s")

    # 3) greedy evaluation
    print("\n[Testing] evaluating learned greedy policy ...")
    test = evaluate_policy(gw, Q, fb, fm)
    conv_ep = episodes_to_converge(hist["reward"])
    print(f"  success rate : {test['success_rate']*100:.1f}%")
    print(f"  avg reward   : {test['avg_reward']:.2f}")
    print(f"  avg steps    : {test['avg_steps']:.1f}")
    print(f"  avg battery  : {test['avg_battery']:.1f}")
    print(f"  converged    : ~episode {conv_ep}")

    # 4) hyperparameter study
    hp_df = hyperparameter_study(gw)

    # 5) figures
    print("\n[Plots] generating figures ...")
    plot_learning_curves(hist)
    plot_hyperparameters(hp_df)

    # 6) save Q-learning results
    print("[Export] writing CSV result files ...")
    pd.DataFrame({
        "episode": np.arange(1, len(hist["reward"]) + 1),
        "reward": hist["reward"],
        "length": hist["length"],
        "success": hist["success"],
        "collisions": hist["collisions"],
        "battery": hist["battery"],
        "epsilon": hist["epsilon"],
    }).to_csv(os.path.join(CSV_DIR, "ql_training_history.csv"), index=False)

    pd.DataFrame([{
        "algorithm": "QLearning",
        "episodes": N_EPISODES,
        "execution_time_s": round(ql_time, 5),
        "success_rate": round(test["success_rate"], 3),
        "avg_reward": round(test["avg_reward"], 3),
        "avg_steps": round(test["avg_steps"], 2),
        "avg_battery": round(test["avg_battery"], 2),
        "avg_collisions": round(test["avg_collisions"], 3),
        "episodes_to_converge": conv_ep,
    }]).to_csv(os.path.join(CSV_DIR, "ql_results.csv"), index=False)

    # 7) FINAL COMPARISON  (requires ValueIteration.py to have run first)
    vi_path = os.path.join(CSV_DIR, "vi_results.csv")
    if os.path.exists(vi_path):
        final_comparison(gw, hist, test, ql_time, conv_ep)
    else:
        print("\n[!] vi_results.csv not found -- run ValueIteration.py first "
              "to generate the VI-vs-QL comparison.")

    print("\n" + "=" * 65)
    print(" PHASE 2 (Q-learning) COMPLETE")
    print(f"  Figures  -> {FIG_QL} , {FIG_CMP}")
    print(f"  CSV data -> {CSV_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
