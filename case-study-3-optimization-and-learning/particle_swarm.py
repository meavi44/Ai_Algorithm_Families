"""
=====================================================================
PSO.py
=====================================================================
Project : AI-Powered Autonomous Smart Agriculture
          Swarm Intelligence + Reinforcement Learning
Phase   : 1 -- GLOBAL TASK ALLOCATION (Particle Swarm Optimization)
Author  : AI Laboratory (CSE 4101)
---------------------------------------------------------------------
ROLE OF THIS FILE
---------------------------------------------------------------------
This file solves ONLY the global task-allocation problem:

    "How should all flowers be distributed among all pollination
     robots so that every flower is pollinated while minimizing
     total travel distance, workload imbalance, battery consumption
     and completion time?"

Particle Swarm Optimization (PSO) is used to decide WHICH robot is
responsible for WHICH flowers.  PSO does NOT navigate robots, does
NOT move them step-by-step, and does NOT decide low-level actions.
Its responsibility ends once the optimal flower->robot assignment
is found.  The resulting assignment is exported to CSV so that the
navigation phase (ValueIteration.py / QLearning.py) can consume it.
---------------------------------------------------------------------
DEPENDENCIES : Python, NumPy, Matplotlib, Pandas  (nothing else)
REPRODUCIBLE : fixed global seed, deterministic synthetic dataset
=====================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 0. GLOBAL CONFIGURATION & REPRODUCIBILITY
# =====================================================================
# A single fixed seed makes every run identical: same greenhouse,
# same swarm initialisation, same final assignment.
SEED = 42
np.random.seed(SEED)

# Output folders (created automatically). Figures and CSV results are
# written here so they are directly usable in a research paper.
OUT_DIR = "outputs"
# Figures are organised per algorithm: this file writes to figures/PSO.
FIG_DIR = os.path.join(OUT_DIR, "figures", "PSO")
CSV_DIR = os.path.join(OUT_DIR, "csv")
for _d in (OUT_DIR, FIG_DIR, CSV_DIR):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------
# Greenhouse / robot physical parameters (the synthetic environment).
# These are intentionally simple, fixed constants -> deterministic.
# ---------------------------------------------------------------------
GREENHOUSE = {
    "width": 100.0,          # greenhouse width  (metres)
    "height": 100.0,         # greenhouse height (metres)
    "n_flowers": 400,        # total number of flowers to pollinate
    "n_robots": 5,           # number of autonomous pollination robots
    "n_charging": 3,         # number of charging stations
    "n_obstacles": 25,       # trees / rocks / water channels / obstacles
}

ROBOT = {
    "battery_capacity": 100.0,   # abstract battery units
    "max_distance": 600.0,       # maximum flying distance per robot (m)
    "flying_speed": 5.0,         # metres per second
    "pollinate_time": 2.0,       # seconds spent pollinating one flower
    "energy_per_metre": 0.10,    # battery units consumed per metre flown
}


# =====================================================================
# 1. SYNTHETIC GREENHOUSE DATASET GENERATION
# =====================================================================
def generate_greenhouse(cfg=GREENHOUSE, seed=SEED):
    """
    Build a deterministic synthetic greenhouse.

    Returns a dictionary containing:
        flowers   : (n_flowers, 2) flower coordinates
        robots    : (n_robots, 2)  robot start (depot) coordinates
        charging  : (n_charging, 2) charging-station coordinates
        obstacles : (n_obstacles, 2) obstacle coordinates
    Coordinates are continuous positions inside the greenhouse.
    """
    rng = np.random.RandomState(seed)     # local RNG -> isolated & reproducible
    w, h = cfg["width"], cfg["height"]

    # Flowers are grouped into a few natural clusters (like crop rows /
    # planting beds) so that a good allocation is spatially meaningful.
    n_clusters = 6
    centres = rng.uniform([5, 5], [w - 5, h - 5], size=(n_clusters, 2))
    flowers = []
    for i in range(cfg["n_flowers"]):
        c = centres[i % n_clusters]
        pt = c + rng.normal(0, 6.0, size=2)          # scatter around centre
        pt = np.clip(pt, 0, [w, h])                  # keep inside greenhouse
        flowers.append(pt)
    flowers = np.array(flowers)

    # Robots start from evenly spread depots along the bottom edge.
    robots = np.column_stack([
        np.linspace(w * 0.1, w * 0.9, cfg["n_robots"]),
        np.full(cfg["n_robots"], 2.0),
    ])

    # Charging stations spread across the greenhouse.
    charging = rng.uniform([5, 5], [w - 5, h - 5], size=(cfg["n_charging"], 2))

    # Static obstacles (trees, rocks, water channels).
    obstacles = rng.uniform([0, 0], [w, h], size=(cfg["n_obstacles"], 2))

    return {
        "flowers": flowers,
        "robots": robots,
        "charging": charging,
        "obstacles": obstacles,
    }


# =====================================================================
# 2. ROUTE / DISTANCE HELPERS
# =====================================================================
def _euclid(a, b):
    """Euclidean distance between two 2-D points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def nearest_neighbour_route(start, points, charging):
    """
    Estimate the travel distance for a single robot that starts at
    `start`, visits every flower in `points` using a greedy
    nearest-neighbour tour, and finally returns to the closest
    charging station.

    We use nearest-neighbour (not exact TSP) because PSO only needs a
    consistent, fast surrogate for route cost -- exact routing is the
    job of the navigation phase, not the allocator.  The per-step
    distance computation is vectorised with NumPy so the fitness
    function stays fast enough for a full swarm search.

    Returns
        total_distance, n_served
    where n_served counts the flowers reached before the robot's
    maximum flying distance (battery range) is exhausted.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0

    points = np.asarray(points, dtype=float)
    visited = np.zeros(n, dtype=bool)          # which flowers are done
    current = np.asarray(start, dtype=float)
    total = 0.0
    served = 0

    for _ in range(n):
        # distance from the current position to every flower at once
        diff = points - current
        d = np.hypot(diff[:, 0], diff[:, 1])
        d[visited] = np.inf                    # ignore visited flowers
        idx = int(np.argmin(d))
        step = d[idx]

        # stop serving once the battery flying-range would be exceeded
        if total + step > ROBOT["max_distance"]:
            break

        total += step
        current = points[idx]
        visited[idx] = True
        served += 1

    # add the trip back to the nearest charging station
    if served > 0:
        cd = np.asarray(charging, dtype=float)
        back = float(np.min(np.hypot(cd[:, 0] - current[0],
                                     cd[:, 1] - current[1])))
        total += back

    return total, served


# =====================================================================
# 3. PARTICLE DECODING  (continuous vector  ->  discrete assignment)
# =====================================================================
def decode(position, n_robots):
    """
    A particle position is a continuous vector of length n_flowers.
    Element i in [0, n_robots) is floored to obtain the robot index
    that flower i is assigned to.  This is the standard continuous-PSO
    encoding for a discrete assignment problem.
    """
    assign = np.floor(position).astype(int)
    return np.clip(assign, 0, n_robots - 1)


# =====================================================================
# 4. FITNESS FUNCTION
# =====================================================================
# Weights for the six competing objectives.  They express the relative
# importance of each term and are kept fixed for reproducibility.
FITNESS_WEIGHTS = {
    "coverage": 5.0,      # strongly reward pollinating every flower
    "distance": 1.0,      # minimise total travel distance
    "energy": 1.0,        # minimise battery consumption
    "time": 1.0,          # minimise completion time (makespan)
    "balance": 1.0,       # minimise workload imbalance between robots
    "utilisation": 0.5,   # reward using all available robots
}


def evaluate(position, env):
    """
    Compute the fitness (to be MINIMISED) and a metrics dictionary for
    one particle / assignment.

    Objectives considered:
        * Pollination coverage   (fraction of flowers actually served)
        * Total travel distance
        * Energy consumption
        * Completion time (makespan across robots)
        * Workload balance        (std-dev of per-robot flower counts)
        * Robot utilisation       (fraction of robots given work)
    """
    flowers = env["flowers"]
    robots = env["robots"]
    charging = env["charging"]
    n_robots = len(robots)
    n_flowers = len(flowers)

    assign = decode(position, n_robots)

    robot_distance = np.zeros(n_robots)
    robot_served = np.zeros(n_robots, dtype=int)
    robot_time = np.zeros(n_robots)
    robot_load = np.zeros(n_robots, dtype=int)

    for r in range(n_robots):
        pts = flowers[assign == r]
        robot_load[r] = len(pts)
        dist, served = nearest_neighbour_route(robots[r], pts, charging)
        robot_distance[r] = dist
        robot_served[r] = served
        # completion time = flight time + pollination time for served flowers
        robot_time[r] = dist / ROBOT["flying_speed"] + served * ROBOT["pollinate_time"]

    total_distance = float(robot_distance.sum())
    total_served = int(robot_served.sum())
    total_energy = total_distance * ROBOT["energy_per_metre"]
    completion_time = float(robot_time.max())          # makespan
    coverage = total_served / n_flowers
    workload_std = float(robot_load.std())
    utilisation = float(np.count_nonzero(robot_load) / n_robots)

    # --- normalisation reference scales (keep every term ~O(1)) --------
    ref_distance = n_flowers * 10.0
    ref_energy = ref_distance * ROBOT["energy_per_metre"]
    ref_time = ref_distance / ROBOT["flying_speed"]
    ref_balance = n_flowers / n_robots

    w = FITNESS_WEIGHTS
    fitness = (
        w["coverage"] * (1.0 - coverage)
        + w["distance"] * (total_distance / ref_distance)
        + w["energy"] * (total_energy / ref_energy)
        + w["time"] * (completion_time / ref_time)
        + w["balance"] * (workload_std / ref_balance)
        + w["utilisation"] * (1.0 - utilisation)
    )

    metrics = {
        "fitness": fitness,
        "coverage": coverage,
        "total_distance": total_distance,
        "total_energy": total_energy,
        "completion_time": completion_time,
        "workload_std": workload_std,
        "utilisation": utilisation,
        "robot_load": robot_load,
        "robot_distance": robot_distance,
        "assign": assign,
    }
    return fitness, metrics


# =====================================================================
# 5. PARTICLE SWARM OPTIMIZATION  (from scratch)
# =====================================================================
def run_pso(env,
            swarm_size=40,
            inertia=0.7,
            c1=1.5,
            c2=1.5,
            max_iter=120,
            vmax_frac=0.2,
            patience=30,
            seed=SEED,
            verbose=True):
    """
    Standard global-best PSO for the flower-allocation problem.

    Parameters (hyperparameters that can be tuned)
        swarm_size : number of particles
        inertia    : inertia weight  w
        c1         : cognitive (personal) coefficient
        c2         : social (global) coefficient
        max_iter   : maximum number of iterations
        vmax_frac  : velocity clamp as a fraction of the search range
        patience   : early-stop if gbest does not improve for this many
                     iterations (stopping criterion)

    Returns
        best_metrics : metrics dict of the best assignment found
        history      : per-iteration history (for plotting)
    """
    rng = np.random.RandomState(seed)
    n_flowers = len(env["flowers"])
    n_robots = len(env["robots"])

    lo, hi = 0.0, float(n_robots)           # search-space bounds per dimension
    vmax = vmax_frac * (hi - lo)

    # --- particle initialisation --------------------------------------
    pos = rng.uniform(lo, hi, size=(swarm_size, n_flowers))
    vel = rng.uniform(-vmax, vmax, size=(swarm_size, n_flowers))

    # personal bests
    pbest_pos = pos.copy()
    pbest_val = np.full(swarm_size, np.inf)

    # global best
    gbest_pos = None
    gbest_val = np.inf
    gbest_metrics = None

    # evaluate the initial swarm
    for i in range(swarm_size):
        f, m = evaluate(pos[i], env)
        pbest_val[i] = f
        if f < gbest_val:
            gbest_val, gbest_pos, gbest_metrics = f, pos[i].copy(), m

    # per-iteration history for publication-quality graphs
    history = {k: [] for k in
               ["fitness", "coverage", "total_energy", "completion_time",
                "total_distance", "workload_std"]}

    no_improve = 0

    # --- main optimisation loop ---------------------------------------
    for it in range(max_iter):
        for i in range(swarm_size):
            r1 = rng.random(n_flowers)
            r2 = rng.random(n_flowers)

            # velocity update:  inertia + cognitive + social
            vel[i] = (inertia * vel[i]
                      + c1 * r1 * (pbest_pos[i] - pos[i])
                      + c2 * r2 * (gbest_pos - pos[i]))
            vel[i] = np.clip(vel[i], -vmax, vmax)        # velocity clamp

            # position update
            pos[i] = pos[i] + vel[i]

            # constraint handling: keep particles inside the search space
            # (reflect at the boundaries so information is not lost)
            below = pos[i] < lo
            above = pos[i] >= hi
            pos[i] = np.clip(pos[i], lo, hi - 1e-9)
            vel[i][below | above] *= -0.5                # damp on bounce

            # evaluate and update personal / global bests
            f, m = evaluate(pos[i], env)
            if f < pbest_val[i]:
                pbest_val[i] = f
                pbest_pos[i] = pos[i].copy()
            if f < gbest_val:
                gbest_val, gbest_pos, gbest_metrics = f, pos[i].copy(), m
                no_improve = -1                          # reset (becomes 0 below)

        no_improve += 1

        # record best-so-far metrics
        history["fitness"].append(gbest_metrics["fitness"])
        history["coverage"].append(gbest_metrics["coverage"])
        history["total_energy"].append(gbest_metrics["total_energy"])
        history["completion_time"].append(gbest_metrics["completion_time"])
        history["total_distance"].append(gbest_metrics["total_distance"])
        history["workload_std"].append(gbest_metrics["workload_std"])

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"  iter {it:3d} | fitness={gbest_val:7.4f} "
                  f"| coverage={gbest_metrics['coverage']*100:5.1f}% "
                  f"| dist={gbest_metrics['total_distance']:8.1f}")

        # stopping criterion: no improvement for `patience` iterations
        if no_improve >= patience:
            if verbose:
                print(f"  early stop at iter {it} (no improvement "
                      f"for {patience} iters)")
            break

    return gbest_metrics, history


# =====================================================================
# 6. RANDOM-ASSIGNMENT BASELINE
# =====================================================================
def random_assignment(env, seed=SEED):
    """
    Naive baseline: assign each flower to a robot uniformly at random.
    Used to demonstrate the benefit of PSO.
    """
    rng = np.random.RandomState(seed + 1)
    n_flowers = len(env["flowers"])
    n_robots = len(env["robots"])
    position = rng.uniform(0, n_robots, size=n_flowers)
    _, metrics = evaluate(position, env)
    return metrics


# =====================================================================
# 7. HYPERPARAMETER TUNING
# =====================================================================
def hyperparameter_study(env):
    """
    Explore a small grid of key PSO hyperparameters and record the
    best fitness / coverage reached by each configuration.
    Returns a pandas DataFrame (also saved to CSV).
    """
    print("\n[Hyperparameter study] running configurations ...")
    configs = [
        # (swarm_size, inertia, c1, c2, max_iter)
        (20, 0.7, 1.5, 1.5, 80),
        (40, 0.7, 1.5, 1.5, 80),
        (60, 0.7, 1.5, 1.5, 80),
        (40, 0.4, 1.5, 1.5, 80),
        (40, 0.9, 1.5, 1.5, 80),
        (40, 0.7, 2.0, 1.0, 80),
        (40, 0.7, 1.0, 2.0, 80),
        (40, 0.7, 1.5, 1.5, 150),
    ]

    rows = []
    for (sw, w, c1, c2, mi) in configs:
        m, _ = run_pso(env, swarm_size=sw, inertia=w, c1=c1, c2=c2,
                       max_iter=mi, verbose=False)
        rows.append({
            "swarm_size": sw, "inertia": w, "c1": c1, "c2": c2,
            "max_iter": mi, "fitness": round(m["fitness"], 4),
            "coverage": round(m["coverage"], 4),
            "total_distance": round(m["total_distance"], 2),
            "total_energy": round(m["total_energy"], 2),
            "completion_time": round(m["completion_time"], 2),
        })
        print(f"  swarm={sw:3d} w={w:.1f} c1={c1:.1f} c2={c2:.1f} "
              f"iter={mi:3d} -> fitness={m['fitness']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, "pso_hyperparameter_study.csv"),
              index=False)
    return df


# =====================================================================
# 8. VISUALISATION  (publication-quality graphs)
# =====================================================================
def plot_convergence(history):
    """Fitness / Total-distance / Energy / Completion-time vs iteration."""
    it = range(1, len(history["fitness"]) + 1)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("PSO Convergence Behaviour", fontsize=14, fontweight="bold")

    ax[0, 0].plot(it, history["fitness"], color="#1f77b4", lw=2)
    ax[0, 0].set_title("Fitness vs Iteration")
    ax[0, 0].set_xlabel("Iteration"); ax[0, 0].set_ylabel("Best fitness")
    ax[0, 0].grid(alpha=0.3)

    # Total travel distance is the primary objective PSO minimises.
    # (Coverage is not plotted here: it stays at 100% every iteration,
    #  so a coverage curve would be an uninformative flat line.)
    ax[0, 1].plot(it, history["total_distance"], color="#2ca02c", lw=2)
    ax[0, 1].set_title("Total Distance vs Iteration")
    ax[0, 1].set_xlabel("Iteration"); ax[0, 1].set_ylabel("Total distance (m)")
    ax[0, 1].grid(alpha=0.3)

    # Workload balance (std-dev of flowers per robot): a distinct objective.
    # (Energy is not plotted here -- it is exactly proportional to total
    #  distance, so it would duplicate the panel above.)
    ax[1, 0].plot(it, history["workload_std"], color="#d62728", lw=2)
    ax[1, 0].set_title("Workload Imbalance vs Iteration")
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("Std-dev of flowers/robot")
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(it, history["completion_time"], color="#9467bd", lw=2)
    ax[1, 1].set_title("Completion Time vs Iteration")
    ax[1, 1].set_xlabel("Iteration"); ax[1, 1].set_ylabel("Time (s)")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "pso_convergence.png"), dpi=150)
    plt.close(fig)


def plot_workload_and_utilisation(metrics):
    """Bar charts: workload distribution and robot utilisation."""
    load = metrics["robot_load"]
    dist = metrics["robot_distance"]
    robots = [f"R{i+1}" for i in range(len(load))]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Workload & Robot Utilisation (PSO)",
                 fontsize=14, fontweight="bold")

    ax[0].bar(robots, load, color="#1f77b4")
    ax[0].axhline(load.mean(), color="k", ls="--", lw=1,
                  label=f"mean = {load.mean():.0f}")
    ax[0].set_title("Workload Distribution (flowers per robot)")
    ax[0].set_ylabel("Assigned flowers"); ax[0].legend()
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].bar(robots, dist, color="#ff7f0e")
    ax[1].set_title("Travel Distance per Robot (utilisation)")
    ax[1].set_ylabel("Route distance (m)")
    ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "pso_workload_utilisation.png"), dpi=150)
    plt.close(fig)


def plot_allocation_map(env, metrics):
    """Scatter map of the greenhouse coloured by robot assignment."""
    flowers = env["flowers"]
    assign = metrics["assign"]
    n_robots = len(env["robots"])
    colours = plt.cm.tab10(np.linspace(0, 1, n_robots))

    fig, ax = plt.subplots(figsize=(8, 8))
    for r in range(n_robots):
        pts = flowers[assign == r]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=colours[r],
                   label=f"Robot {r+1} ({len(pts)})")
    ax.scatter(env["robots"][:, 0], env["robots"][:, 1],
               marker="s", s=120, c="black", label="Robot depot")
    ax.scatter(env["charging"][:, 0], env["charging"][:, 1],
               marker="P", s=160, c="lime", edgecolors="k", label="Charging")
    ax.scatter(env["obstacles"][:, 0], env["obstacles"][:, 1],
               marker="x", s=60, c="red", label="Obstacle")

    ax.set_title("PSO Flower-to-Robot Allocation Map",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pso_allocation_map.png"), dpi=150)
    plt.close(fig)


def plot_hyperparameter_study(df):
    """Bar chart of best fitness for each hyperparameter configuration."""
    labels = [f"S{r.swarm_size}\nw{r.inertia}\nc{r.c1}/{r.c2}\ni{r.max_iter}"
              for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, df["fitness"], color="#17becf")
    ax.set_title("Hyperparameter Comparison (lower fitness = better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Best fitness")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pso_hyperparameter_comparison.png"),
                dpi=150)
    plt.close(fig)


# =====================================================================
# 9. EXPORT RESULTS  (CSV files consumed by the navigation phase)
# =====================================================================
def export_results(env, pso_m, rand_m):
    """Save the greenhouse, the flower assignment, and the comparison."""
    flowers = env["flowers"]
    assign = pso_m["assign"]

    # --- (a) the flower->robot assignment (used by navigation files) --
    assign_df = pd.DataFrame({
        "flower_id": np.arange(len(flowers)),
        "x": flowers[:, 0],
        "y": flowers[:, 1],
        "assigned_robot": assign,
    })
    assign_df.to_csv(os.path.join(CSV_DIR, "pso_flower_assignment.csv"),
                     index=False)

    # --- (b) the environment layout (robots / charging / obstacles) ---
    def _layout(name, arr):
        return pd.DataFrame({"type": name, "x": arr[:, 0], "y": arr[:, 1]})

    layout_df = pd.concat([
        _layout("robot_depot", env["robots"]),
        _layout("charging", env["charging"]),
        _layout("obstacle", env["obstacles"]),
    ], ignore_index=True)
    layout_df.to_csv(os.path.join(CSV_DIR, "greenhouse_layout.csv"),
                     index=False)

    # --- (c) per-robot summary ---------------------------------------
    summary_df = pd.DataFrame({
        "robot_id": np.arange(len(env["robots"])),
        "assigned_flowers": pso_m["robot_load"],
        "route_distance": np.round(pso_m["robot_distance"], 2),
    })
    summary_df.to_csv(os.path.join(CSV_DIR, "pso_robot_summary.csv"),
                      index=False)

    # --- (d) random-vs-PSO comparison --------------------------------
    cmp_df = pd.DataFrame([
        {"method": "Random", "coverage": rand_m["coverage"],
         "total_distance": rand_m["total_distance"],
         "total_energy": rand_m["total_energy"],
         "completion_time": rand_m["completion_time"],
         "workload_std": rand_m["workload_std"],
         "utilisation": rand_m["utilisation"]},
        {"method": "PSO", "coverage": pso_m["coverage"],
         "total_distance": pso_m["total_distance"],
         "total_energy": pso_m["total_energy"],
         "completion_time": pso_m["completion_time"],
         "workload_std": pso_m["workload_std"],
         "utilisation": pso_m["utilisation"]},
    ])
    cmp_df.to_csv(os.path.join(CSV_DIR, "pso_vs_random.csv"), index=False)

    return assign_df, cmp_df


# =====================================================================
# 10. MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 65)
    print(" PHASE 1 : GLOBAL TASK ALLOCATION using PSO")
    print("=" * 65)

    # 1) build the deterministic greenhouse
    env = generate_greenhouse()
    print(f"Greenhouse: {len(env['flowers'])} flowers | "
          f"{len(env['robots'])} robots | "
          f"{len(env['charging'])} charging stations | "
          f"{len(env['obstacles'])} obstacles")

    # 2) random baseline
    print("\n[Baseline] random flower assignment ...")
    rand_m = random_assignment(env)
    print(f"  coverage={rand_m['coverage']*100:.1f}% | "
          f"distance={rand_m['total_distance']:.1f} | "
          f"energy={rand_m['total_energy']:.1f}")

    # 3) main PSO optimisation
    print("\n[PSO] optimising flower->robot allocation ...")
    t0 = time.time()
    pso_m, history = run_pso(env)
    t1 = time.time()
    print(f"  done in {t1 - t0:.2f}s | "
          f"coverage={pso_m['coverage']*100:.1f}% | "
          f"distance={pso_m['total_distance']:.1f} | "
          f"energy={pso_m['total_energy']:.1f}")

    # 4) hyperparameter study
    hp_df = hyperparameter_study(env)

    # 5) graphs
    print("\n[Plots] generating publication-quality figures ...")
    plot_convergence(history)
    plot_workload_and_utilisation(pso_m)
    plot_allocation_map(env, pso_m)
    plot_hyperparameter_study(hp_df)

    # 6) export CSV results
    print("[Export] writing CSV result files ...")
    export_results(env, pso_m, rand_m)

    print("\n" + "=" * 65)
    print(" PHASE 1 COMPLETE")
    print(f"  Figures  -> {FIG_DIR}")
    print(f"  CSV data -> {CSV_DIR}")
    print("  Key output for navigation phase:")
    print("     outputs/csv/pso_flower_assignment.csv")
    print("     outputs/csv/greenhouse_layout.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
