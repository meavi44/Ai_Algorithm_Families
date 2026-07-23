"""
=====================================================================
ValueIteration.py
=====================================================================
Project : AI-Powered Autonomous Smart Agriculture
          Swarm Intelligence + Reinforcement Learning
Phase   : 2 -- LOCAL NAVIGATION  (Value Iteration, model-based)
Author  : AI Laboratory (CSE 4101)
---------------------------------------------------------------------
ROLE OF THIS FILE
---------------------------------------------------------------------
Phase 1 (PSO.py) already decided WHICH flowers each robot owns.  This
file answers the SECOND, different question for ONE robot:

    "How should the robot move inside the greenhouse to reach all its
     assigned flowers while avoiding obstacles, minimising battery
     usage, and returning safely to a charging station?"

This is a Sequential Decision-Making problem, modelled as a Markov
Decision Process (MDP) on a discrete grid world.  Because the whole
environment is assumed KNOWN, we solve it with **Value Iteration**
(Bellman updates) and extract the optimal policy.

The GridWorld class defined here is the SHARED environment: QLearning.py
imports it so that Value Iteration and Q-learning run on exactly the
same grid, same obstacles, same assigned flowers and same rewards --
only the learning method differs.
---------------------------------------------------------------------
DEPENDENCIES : Python, NumPy, Matplotlib, Pandas
REPRODUCIBLE : fixed seed, deterministic grid built from PSO output
=====================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 0. CONFIGURATION & REPRODUCIBILITY
# =====================================================================
SEED = 42
np.random.seed(SEED)

CSV_DIR = os.path.join("outputs", "csv")
# Figures are organised per algorithm: this file writes to figures/ValueIteration.
FIG_DIR = os.path.join("outputs", "figures", "ValueIteration")
for _d in (CSV_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# Which robot from the PSO allocation do we navigate?
ROBOT_ID = 0
# Discrete grid resolution (GRID x GRID cells covering the greenhouse).
GRID = 20
# Continuous greenhouse size (must match PSO.py so coordinates map correctly).
GH_W, GH_H = 100.0, 100.0
# For a clean, evaluable navigation demo we select a small, well-spread
# set of the robot's assigned flowers as navigation targets (the prompt's
# example lists ~5 target flowers for a robot).
N_TARGETS = 6

# ---------------------------------------------------------------------
# MDP reward parameters (shared by Value Iteration AND Q-learning).
# ---------------------------------------------------------------------
GOAL_REWARD = 10.0        # reward for reaching a flower (pollination)
CHARGE_REWARD = 10.0      # reward for reaching the charging station (safe return)
STEP_COST = -0.1          # small per-step cost (time / battery)
COLLISION_PENALTY = -5.0  # penalty for bumping into an obstacle
MOVE_PROB = 0.8           # probability the intended move succeeds
SLIP_PROB = 0.1           # probability of slipping to each perpendicular side

# Battery model (used to report energy usage during a mission).
BATTERY_CAPACITY = 200.0
ENERGY_PER_STEP = 1.0

# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTIONS = [0, 1, 2, 3]
ACTION_NAME = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
# (row, col) displacement for each action.  row increases upward because
# plots use origin='lower'.
DELTA = {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}
# perpendicular "slip" actions for each intended action (stochastic MDP)
SLIP = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}
# arrow glyphs for the policy plot
ARROW = {0: "^", 1: "v", 2: "<", 3: ">"}


# =====================================================================
# 1. SHARED GRID-WORLD ENVIRONMENT
# =====================================================================
def _to_cell(x, y, grid=GRID, w=GH_W, h=GH_H):
    """Map a continuous greenhouse point (x, y) to a discrete grid cell."""
    c = int(np.clip(x / w * grid, 0, grid - 1))    # column from x
    r = int(np.clip(y / h * grid, 0, grid - 1))    # row from y
    return (r, c)


class GridWorld:
    """
    Discrete grid-world MDP for a single robot's pollination mission.

    Cell contents
        obstacles : impassable cells (trees / rocks / water channels)
        flowers   : navigation-target cells the robot must pollinate
        charging  : safe return cell(s)
        start     : the robot's starting cell (its depot)

    The class provides BOTH:
        * a known transition model  -> used by Value Iteration
        * a step(state, action) sampler -> used by model-free Q-learning
    so the two algorithms share one identical environment.
    """

    def __init__(self, robot_id=ROBOT_ID, grid=GRID, n_targets=N_TARGETS,
                 seed=SEED):
        self.n = grid
        self.rng = np.random.RandomState(seed)

        # --- load the PSO outputs (assignment + greenhouse layout) -----
        assign = pd.read_csv(os.path.join(CSV_DIR, "pso_flower_assignment.csv"))
        layout = pd.read_csv(os.path.join(CSV_DIR, "greenhouse_layout.csv"))

        obstacles_xy = layout[layout["type"] == "obstacle"][["x", "y"]].values
        charging_xy = layout[layout["type"] == "charging"][["x", "y"]].values
        depot_xy = layout[layout["type"] == "robot_depot"][["x", "y"]].values

        # obstacles as a set of blocked cells
        self.obstacles = set(_to_cell(x, y) for x, y in obstacles_xy)

        # charging stations (cannot sit on an obstacle)
        self.charging = set(_to_cell(x, y) for x, y in charging_xy)
        self.charging -= self.obstacles

        # robot start cell (must be free)
        self.start = _to_cell(*depot_xy[robot_id])
        if self.start in self.obstacles:
            self.obstacles.discard(self.start)

        # --- pick navigation-target flowers for this robot -------------
        mine = assign[assign["assigned_robot"] == robot_id][["x", "y"]].values
        cells = []
        for x, y in mine:
            cell = _to_cell(x, y)
            if cell not in self.obstacles and cell != self.start \
                    and cell not in self.charging and cell not in cells:
                cells.append(cell)
        self.flowers = self._farthest_point_sample(cells, n_targets)

    # -----------------------------------------------------------------
    def _farthest_point_sample(self, cells, k):
        """Choose k well-spread target cells (deterministic, seed-free)."""
        if len(cells) <= k:
            return list(cells)
        chosen = [cells[0]]
        while len(chosen) < k:
            # add the cell that is farthest from the already-chosen set
            best, best_d = None, -1
            for c in cells:
                if c in chosen:
                    continue
                d = min((c[0] - s[0]) ** 2 + (c[1] - s[1]) ** 2 for s in chosen)
                if d > best_d:
                    best, best_d = c, d
            chosen.append(best)
        return chosen

    # -----------------------------------------------------------------
    def passable(self, cell):
        """True if a cell is inside the grid and not an obstacle."""
        r, c = cell
        return 0 <= r < self.n and 0 <= c < self.n and cell not in self.obstacles

    def _move(self, cell, action):
        """Deterministic outcome of one action (stay if blocked)."""
        dr, dc = DELTA[action]
        nxt = (cell[0] + dr, cell[1] + dc)
        return nxt if self.passable(nxt) else cell

    # -----------------------------------------------------------------
    def step(self, state, action, goals):
        """
        Model-FREE interface used by Q-learning.  Executes one stochastic
        step and returns (next_state, reward, done, collided).

        `goals` is the current set of flower cells still to be pollinated.
        """
        # sample the realised action (actuator noise)
        p = self.rng.random()
        if p < MOVE_PROB:
            act = action
        elif p < MOVE_PROB + SLIP_PROB:
            act = SLIP[action][0]
        else:
            act = SLIP[action][1]

        dr, dc = DELTA[act]
        target = (state[0] + dr, state[1] + dc)

        collided = not self.passable(target)
        nxt = state if collided else target

        # reward on arrival
        if nxt in goals:
            reward, done = GOAL_REWARD, True
        elif collided:
            reward, done = COLLISION_PENALTY, False
        else:
            reward, done = STEP_COST, False
        return nxt, reward, done, collided


# =====================================================================
# 2. VALUE ITERATION  (model-based, Bellman updates)
# =====================================================================
def _action_q_values(U, passable, rr, cc, n, gamma):
    """
    Vectorised expected action-values Q(s, a) for every cell and all four
    actions, under the stochastic slip model AND the shared TRANSITION
    rewards:

        * a normal move          -> STEP_COST
        * a move blocked by a
          wall / grid boundary    -> COLLISION_PENALTY  (robot stays put)

    Modelling the collision penalty *inside* the Bellman expectation makes
    Value Iteration optimise the EXACT SAME reward constants that
    Q-learning experiences (STEP_COST, COLLISION_PENALTY, GOAL_REWARD), so
    the two navigation methods are theoretically comparable.

    Returns an array of shape (4, n, n): q[a] = expected value of action a.
    """
    def outcome(dr, dc):
        """(reward, next_value) arrays for attempting move (dr, dc)."""
        nr, nc = rr + dr, cc + dc
        inb = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)
        nr_c, nc_c = np.clip(nr, 0, n - 1), np.clip(nc, 0, n - 1)
        can_go = inb & passable[nr_c, nc_c]
        nxt = U.copy()                                # blocked -> stay (own U)
        nxt[can_go] = U[nr_c[can_go], nc_c[can_go]]
        rew = np.where(can_go, STEP_COST, COLLISION_PENALTY)
        return rew, nxt

    qs = []
    for a in ACTIONS:
        dr, dc = DELTA[a]
        (s1r, s1c), (s2r, s2c) = DELTA[SLIP[a][0]], DELTA[SLIP[a][1]]
        r0, v0 = outcome(dr, dc)                       # intended  (prob 0.8)
        r1, v1 = outcome(s1r, s1c)                     # slip side (prob 0.1)
        r2, v2 = outcome(s2r, s2c)                     # slip side (prob 0.1)
        q = (MOVE_PROB * (r0 + gamma * v0)
             + SLIP_PROB * (r1 + gamma * v1)
             + SLIP_PROB * (r2 + gamma * v2))
        qs.append(q)
    return np.stack(qs, axis=0)


def value_iteration(gw, goals, gamma=0.9, theta=1e-4, max_iter=1000):
    """
    Compute the optimal value function U(s) for reaching any cell in
    `goals`, using the Bellman optimality update with transition rewards:

        U(s) = max_a  sum_s' T(s,a,s') [ R(s,a,s') + gamma * U(s') ]

    where R is STEP_COST for a normal move and COLLISION_PENALTY when the
    sampled move is blocked by a wall (see _action_q_values).  The goal
    cells are absorbing (terminal) with value GOAL_REWARD; obstacles are
    walls (never entered).  Returns:
        U        : (n, n) value function
        deltas   : per-iteration max change (convergence curve)
        n_iter   : iterations until convergence
    """
    n = gw.n
    U = np.zeros((n, n))

    # passability mask (True = the robot may occupy this cell)
    passable = np.ones((n, n), dtype=bool)
    for (r, c) in gw.obstacles:
        passable[r, c] = False

    goal_mask = np.zeros((n, n), dtype=bool)
    for (r, c) in goals:
        goal_mask[r, c] = True

    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    deltas = []
    for it in range(max_iter):
        q = _action_q_values(U, passable, rr, cc, n, gamma)
        best = np.max(q, axis=0)                   # greedy over actions

        U_new = best                               # reward already inside q
        U_new[goal_mask] = GOAL_REWARD             # terminal goals fixed
        U_new[~passable] = 0.0                     # obstacles inert

        delta = float(np.max(np.abs(U_new - U)))
        deltas.append(delta)
        U = U_new
        if delta < theta:
            break

    return U, deltas, it + 1


def extract_policy(gw, U, goals, gamma=0.9):
    """
    Greedy policy: for every free, non-goal cell choose the action with
    the highest expected value under the known transition model (using the
    same transition-reward expectation as value_iteration, so the policy is
    consistent with U).  Returns an (n, n) int array of action ids
    (-1 = obstacle/goal).
    """
    n = gw.n
    passable = np.ones((n, n), dtype=bool)
    for (r, c) in gw.obstacles:
        passable[r, c] = False

    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    q = _action_q_values(U, passable, rr, cc, n, gamma)
    policy = np.argmax(q, axis=0)

    for (r, c) in gw.obstacles:
        policy[r, c] = -1
    for (r, c) in goals:
        policy[r, c] = -1
    return policy


# =====================================================================
# 3. MISSION SIMULATION  (sequential re-planning to visit ALL flowers)
# =====================================================================
def run_mission(gw, gamma=0.9, theta=1e-4):
    """
    The robot must pollinate every target flower and then return to a
    charging station.  Because Value Iteration solves a single goal set,
    we re-plan after each flower is reached (classic sequential decision
    making).  The greedy optimal policy is executed under the intended
    (deterministic) transitions to trace the planned optimal path.

    Returns a dictionary of trajectory + performance metrics.
    """
    pos = gw.start
    remaining = list(gw.flowers)
    trajectory = [pos]
    total_reward = 0.0
    steps = 0
    collisions = 0
    cap = 400                                  # safety cap on total steps

    # visit every flower (nearest-first, via re-planned value functions)
    while remaining and steps < cap:
        U, _, _ = value_iteration(gw, set(remaining), gamma, theta)
        policy = extract_policy(gw, U, set(remaining), gamma)
        # walk greedily until we land on a remaining flower
        while pos not in remaining and steps < cap:
            a = int(policy[pos])
            nxt = gw._move(pos, a)
            collided = (nxt == pos)
            pos = nxt
            trajectory.append(pos)
            steps += 1
            # same transition rewards as the MDP / Q-learning
            total_reward += COLLISION_PENALTY if collided else STEP_COST
            if collided:
                collisions += 1
        if pos in remaining:
            remaining.remove(pos)
            total_reward += GOAL_REWARD

    # finally, return to the nearest charging station
    if gw.charging:
        U, _, _ = value_iteration(gw, set(gw.charging), gamma, theta)
        policy = extract_policy(gw, U, set(gw.charging), gamma)
        while pos not in gw.charging and steps < cap:
            a = int(policy[pos])
            nxt = gw._move(pos, a)
            collided = (nxt == pos)
            pos = nxt
            trajectory.append(pos)
            steps += 1
            total_reward += COLLISION_PENALTY if collided else STEP_COST
            if collided:
                collisions += 1
        if pos in gw.charging:
            total_reward += CHARGE_REWARD

    success = (len(remaining) == 0) and (pos in gw.charging)
    battery_used = steps * ENERGY_PER_STEP
    return {
        "trajectory": trajectory,
        "steps": steps,
        "total_reward": total_reward,
        "collisions": collisions,
        "battery_used": battery_used,
        "success": success,
        "flowers_done": len(gw.flowers) - len(remaining),
    }


def evaluate_vi_policy(gw, n_test=200, gamma=0.9, theta=1e-4,
                       seed=SEED + 7, max_steps=400):
    """
    Roll out the Value-Iteration-optimal policy under the SAME stochastic
    slip model (MOVE_PROB / SLIP_PROB) and the SAME transition rewards that
    are used to evaluate Q-learning, then average over `n_test` episodes.

    This gives a like-for-like comparison: the deterministic run_mission()
    above traces the *planned* optimal path (used for the policy figure),
    whereas this function measures how the optimal policy actually performs
    when actuator noise is present -- exactly the regime Q-learning is
    tested in.  Because VI already accounts for slip inside the Bellman
    expectation, its policy is optimal for this stochastic environment.
    """
    rng = np.random.RandomState(seed)

    # The policy for a given remaining-flower set is identical across
    # episodes, so cache it (keyed by the frozen goal set) to stay fast.
    policy_cache = {}

    def get_policy(goalset):
        key = frozenset(goalset)
        if key not in policy_cache:
            U, _, _ = value_iteration(gw, set(goalset), gamma, theta)
            policy_cache[key] = extract_policy(gw, U, set(goalset), gamma)
        return policy_cache[key]

    def stoch_move(pos, a):
        """One noisy move; returns (next_cell, collided)."""
        p = rng.random()
        if p < MOVE_PROB:
            act = a
        elif p < MOVE_PROB + SLIP_PROB:
            act = SLIP[a][0]
        else:
            act = SLIP[a][1]
        dr, dc = DELTA[act]
        target = (pos[0] + dr, pos[1] + dc)
        return (pos, True) if not gw.passable(target) else (target, False)

    successes, steps_list, rewards, battery, coll_list = [], [], [], [], []
    for _ in range(n_test):
        pos = gw.start
        remaining = list(gw.flowers)
        steps, total_r, coll = 0, 0.0, 0

        # --- pollinate every flower --------------------------------------
        while remaining and steps < max_steps:
            policy = get_policy(remaining)
            while pos not in remaining and steps < max_steps:
                nxt, collided = stoch_move(pos, int(policy[pos]))
                pos = nxt
                steps += 1
                if collided:
                    total_r += COLLISION_PENALTY
                    coll += 1
                elif pos in remaining:
                    total_r += GOAL_REWARD           # arrival reward
                else:
                    total_r += STEP_COST
            if pos in remaining:
                remaining.remove(pos)

        # --- return to the nearest charging station ----------------------
        if gw.charging:
            policy = get_policy(gw.charging)
            while pos not in gw.charging and steps < max_steps:
                nxt, collided = stoch_move(pos, int(policy[pos]))
                pos = nxt
                steps += 1
                if collided:
                    total_r += COLLISION_PENALTY
                    coll += 1
                elif pos in gw.charging:
                    total_r += CHARGE_REWARD
                else:
                    total_r += STEP_COST

        success = (len(remaining) == 0) and (pos in gw.charging)
        successes.append(int(success))
        steps_list.append(steps)
        rewards.append(total_r)
        battery.append(steps * ENERGY_PER_STEP)
        coll_list.append(coll)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_battery": float(np.mean(battery)),
        "avg_collisions": float(np.mean(coll_list)),
    }


# =====================================================================
# 4. HYPERPARAMETER TUNING (discount factor & convergence threshold)
# =====================================================================
def hyperparameter_study(gw):
    """Vary gamma and theta; record iterations-to-converge and steps."""
    print("\n[Hyperparameter study] Value Iteration ...")
    rows = []
    for gamma in [0.80, 0.90, 0.95, 0.99]:
        for theta in [1e-2, 1e-4, 1e-6]:
            _, deltas, n_iter = value_iteration(gw, set(gw.flowers),
                                                gamma, theta)
            mission = run_mission(gw, gamma, theta)
            rows.append({
                "gamma": gamma, "theta": theta,
                "iterations_to_converge": n_iter,
                "steps_to_goal": mission["steps"],
                "total_reward": round(mission["total_reward"], 3),
                "collisions": mission["collisions"],
            })
            print(f"  gamma={gamma:.2f} theta={theta:.0e} "
                  f"-> iters={n_iter:3d}, steps={mission['steps']:3d}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, "vi_hyperparameter_study.csv"), index=False)
    return df


# =====================================================================
# 5. VISUALISATION
# =====================================================================
def plot_value_heatmap(gw, U):
    """Heatmap of the optimal value function over the grid."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(U, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="U(s)  (optimal value)")

    for (r, c) in gw.obstacles:
        ax.scatter(c, r, marker="x", c="red", s=40)
    for (r, c) in gw.flowers:
        ax.scatter(c, r, marker="*", c="magenta", s=180, edgecolors="k")
    for (r, c) in gw.charging:
        ax.scatter(c, r, marker="P", c="cyan", s=150, edgecolors="k")
    ax.scatter(gw.start[1], gw.start[0], marker="s", c="white",
               s=120, edgecolors="k")

    ax.set_title("Value Iteration -- Optimal Value Function Heatmap",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("grid X"); ax.set_ylabel("grid Y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vi_value_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_policy(gw, policy, mission):
    """Arrow map of the optimal policy plus the executed trajectory."""
    n = gw.n
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.grid(True, alpha=0.3)

    for r in range(n):
        for c in range(n):
            if (r, c) in gw.obstacles or (r, c) in gw.flowers \
                    or (r, c) in gw.charging:
                continue
            a = policy[r, c]
            if a >= 0:
                ax.text(c, r, ARROW[a], ha="center", va="center",
                        fontsize=9, color="gray")

    # executed trajectory
    tr = np.array(mission["trajectory"])
    ax.plot(tr[:, 1], tr[:, 0], "-o", c="orange", ms=3, lw=1.5,
            label="robot path")

    for (r, c) in gw.obstacles:
        ax.scatter(c, r, marker="s", c="black", s=120)
    for (r, c) in gw.flowers:
        ax.scatter(c, r, marker="*", c="magenta", s=200, edgecolors="k")
    for (r, c) in gw.charging:
        ax.scatter(c, r, marker="P", c="cyan", s=160, edgecolors="k")
    ax.scatter(gw.start[1], gw.start[0], marker="s", c="lime",
               s=140, edgecolors="k", label="start")

    ax.set_title("Value Iteration -- Optimal Policy & Robot Path",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vi_policy.png"), dpi=150)
    plt.close(fig)


def plot_convergence(deltas):
    """Convergence curve: max value change vs iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(deltas) + 1), deltas, color="#1f77b4", lw=2)
    ax.set_yscale("log")
    ax.set_title("Value Iteration -- Convergence Curve",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |U change|  (log scale)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vi_convergence.png"), dpi=150)
    plt.close(fig)


def plot_hyperparameters(df):
    """Iterations-to-converge and steps-to-goal across gamma / theta."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Value Iteration -- Hyperparameter Study",
                 fontsize=13, fontweight="bold")

    for theta in sorted(df["theta"].unique()):
        sub = df[df["theta"] == theta]
        ax[0].plot(sub["gamma"], sub["iterations_to_converge"],
                   "-o", label=f"theta={theta:.0e}")
        ax[1].plot(sub["gamma"], sub["steps_to_goal"],
                   "-o", label=f"theta={theta:.0e}")

    ax[0].set_title("Iterations to Converge vs Discount Factor")
    ax[0].set_xlabel("gamma"); ax[0].set_ylabel("iterations")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].set_title("Steps to Goal vs Discount Factor")
    ax[1].set_xlabel("gamma"); ax[1].set_ylabel("steps")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "vi_hyperparameters.png"), dpi=150)
    plt.close(fig)


# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 65)
    print(" PHASE 2 : LOCAL NAVIGATION using VALUE ITERATION")
    print("=" * 65)

    # 1) build the shared grid-world from the PSO allocation
    gw = GridWorld(robot_id=ROBOT_ID)
    print(f"Grid {GRID}x{GRID} | robot {ROBOT_ID} | "
          f"{len(gw.flowers)} target flowers | "
          f"{len(gw.obstacles)} obstacles | "
          f"{len(gw.charging)} charging | start={gw.start}")
    print(f"Target flowers: {gw.flowers}")

    # 2) representative value iteration (all targets as goals)
    print("\n[Value Iteration] solving the MDP ...")
    t0 = time.time()
    U, deltas, n_iter = value_iteration(gw, set(gw.flowers))
    exec_time = time.time() - t0
    policy = extract_policy(gw, U, set(gw.flowers))
    print(f"  converged in {n_iter} iterations ({exec_time*1000:.1f} ms)")

    # 3) full mission simulation (visit every flower, return to charge)
    #    This is the DETERMINISTIC planned optimal path (used for the policy
    #    figure and the planned-path CSV).
    print("\n[Mission] executing optimal policy (sequential re-planning) ...")
    mission = run_mission(gw)
    print(f"  flowers pollinated : {mission['flowers_done']}/{len(gw.flowers)}")
    print(f"  planned steps      : {mission['steps']}")
    print(f"  planned reward     : {mission['total_reward']:.2f}")
    print(f"  battery used       : {mission['battery_used']:.1f}")
    print(f"  collisions         : {mission['collisions']}")
    print(f"  success            : {mission['success']}")

    # 3b) STOCHASTIC evaluation of the optimal policy under actuator noise,
    #     using the SAME slip model + rewards Q-learning is tested with, so
    #     the Phase-2 comparison is genuinely like-for-like.
    print("\n[Evaluation] rolling out the optimal policy under slip noise ...")
    vi_eval = evaluate_vi_policy(gw)
    print(f"  success rate : {vi_eval['success_rate']*100:.1f}%")
    print(f"  avg reward   : {vi_eval['avg_reward']:.2f}")
    print(f"  avg steps    : {vi_eval['avg_steps']:.1f}")
    print(f"  avg battery  : {vi_eval['avg_battery']:.1f}")

    # 4) hyperparameter study
    hp_df = hyperparameter_study(gw)

    # 5) figures
    print("\n[Plots] generating figures ...")
    plot_value_heatmap(gw, U)
    plot_policy(gw, policy, mission)
    plot_convergence(deltas)
    plot_hyperparameters(hp_df)

    # 6) CSV results
    print("[Export] writing CSV result files ...")
    pd.DataFrame({"iteration": range(1, len(deltas) + 1),
                  "max_delta": deltas}).to_csv(
        os.path.join(CSV_DIR, "vi_convergence.csv"), index=False)

    pd.DataFrame([{
        "algorithm": "ValueIteration",
        "iterations_to_converge": n_iter,
        "execution_time_s": round(exec_time, 5),
        # --- fair, stochastic evaluation (comparable to Q-learning) -------
        "success_rate": round(vi_eval["success_rate"], 3),
        "avg_steps": round(vi_eval["avg_steps"], 2),
        "avg_reward": round(vi_eval["avg_reward"], 3),
        "avg_battery": round(vi_eval["avg_battery"], 2),
        "avg_collisions": round(vi_eval["avg_collisions"], 3),
        # --- deterministic planned optimal path (for reference/figure) ----
        "planned_steps": mission["steps"],
        "planned_reward": round(mission["total_reward"], 3),
        "flowers_done": mission["flowers_done"],
    }]).to_csv(os.path.join(CSV_DIR, "vi_results.csv"), index=False)

    tr = np.array(mission["trajectory"])
    pd.DataFrame({"step": range(len(tr)),
                  "row": tr[:, 0], "col": tr[:, 1]}).to_csv(
        os.path.join(CSV_DIR, "vi_trajectory.csv"), index=False)

    print("\n" + "=" * 65)
    print(" PHASE 2 (Value Iteration) COMPLETE")
    print(f"  Figures  -> {FIG_DIR}")
    print(f"  CSV data -> {CSV_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
