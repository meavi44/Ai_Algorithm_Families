"""
=====================================================================
GeneticAlgorithm.py
=====================================================================
Project : AI-Powered Autonomous Smart Agriculture
          Swarm Intelligence + Reinforcement Learning
Purpose : GENETIC ALGORITHM as a TASK ALLOCATOR, compared head-to-head
          with PSO.
Author  : AI Laboratory (CSE 4101)
---------------------------------------------------------------------
WHY THIS FILE
---------------------------------------------------------------------
PSO.py solves the flower->robot allocation with a continuous encoding.
A Genetic Algorithm (GA) solves the SAME allocation problem with a
natural DISCRETE encoding inspired by biological evolution:

    * chromosome  = an array of 400 genes; gene i is the robot index
      (0..4) assigned to flower i,
    * fitness     = the SAME multi-objective cost from PSO.evaluate
      (lower is better),
    * selection   = tournament selection (fitter chromosomes are more
      likely to become parents),
    * crossover   = uniform crossover (mix two parents gene by gene),
    * mutation    = random-reset (occasionally reassign a flower to a
      random robot) -> keeps diversity, avoids premature convergence,
    * elitism     = the best few chromosomes always survive unchanged.

To give evolution a strong starting point we seed a fraction of the
initial population with a distance-greedy assignment (each flower to
its NEAREST robot depot) and let crossover/mutation balance and refine
it.  This heuristic initialisation is a standard GA technique and gives
the GA the spatial awareness that a blind 400-dimensional search lacks.

To keep the comparison perfectly fair, this file imports the SAME
greenhouse generator and the SAME multi-objective fitness function from
PSO.py, and runs PSO, GA and a Random baseline on one environment.
---------------------------------------------------------------------
DEPENDENCIES : Python, NumPy, Matplotlib, Pandas
REPRODUCIBLE : fixed seed, deterministic greenhouse
=====================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse PSO's environment + fitness so both allocators are judged identically.
from particle_swarm import (
    generate_greenhouse, evaluate, run_pso, random_assignment,
    CSV_DIR, SEED,
)

np.random.seed(SEED)

# Figures are organised per algorithm: GA's own figures go to figures/GA,
# while the PSO-vs-GA comparison goes to figures/Comparison.
FIG_GA = os.path.join("outputs", "figures", "GA")
FIG_CMP = os.path.join("outputs", "figures", "Comparison")
for _d in (FIG_GA, FIG_CMP):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------
# GA hyperparameters (also tuned later).
# ---------------------------------------------------------------------
POP_SIZE = 40         # number of chromosomes in the population
N_GEN = 120           # number of generations
TOURNAMENT_K = 3      # tournament size for parent selection
CX_RATE = 0.9         # probability that two parents are crossed over
MUT_RATE = 0.02       # per-gene mutation probability
ELITE = 2             # number of best chromosomes copied verbatim
SEED_FRAC = 0.30      # fraction of initial population seeded from heuristic
SEED_JITTER = 0.30    # fraction of genes randomised in each seeded chromosome


# =====================================================================
# 1. FITNESS + HEURISTIC HELPERS
# =====================================================================
def fitness_of(assign, env):
    """
    Score a chromosome with PSO's identical multi-objective fitness.
    A chromosome is an int array (robot per flower); evaluate() expects a
    continuous vector, so we add 0.5 to land in the middle of each robot
    index's [r, r+1) decoding bin.  Returns (fitness, metrics).
    """
    position = assign.astype(float) + 0.5
    return evaluate(position, env)


def nearest_depot_assignment(env):
    """
    Greedy heuristic: assign every flower to its NEAREST robot depot.
    Produces spatially-coherent (but possibly unbalanced) clusters that
    seed the GA with good structure to evolve from.
    """
    flowers = env["flowers"]                                  # (F, 2)
    depots = env["robots"]                                    # (R, 2)
    diff = flowers[:, None, :] - depots[None, :, :]           # (F, R, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))                   # (F, R)
    return dist.argmin(axis=1).astype(int)                   # nearest depot


# =====================================================================
# 2. GENETIC OPERATORS
# =====================================================================
def init_population(env, pop_size, n_robots, rng,
                    seed_frac=SEED_FRAC, jitter=SEED_JITTER):
    """
    Build the initial population.  A `seed_frac` fraction of chromosomes
    are heuristic-seeded (greedy nearest-depot, then partly randomised for
    diversity); the rest are fully random.
    """
    n_flowers = len(env["flowers"])
    greedy = nearest_depot_assignment(env)
    n_seed = int(seed_frac * pop_size)

    pop = np.empty((pop_size, n_flowers), dtype=int)
    for k in range(pop_size):
        if k < n_seed:
            child = greedy.copy()
            # randomise a fraction of genes so seeded individuals differ
            mask = rng.random(n_flowers) < jitter
            child[mask] = rng.randint(0, n_robots, size=int(mask.sum()))
            pop[k] = child
        else:
            pop[k] = rng.randint(0, n_robots, size=n_flowers)
    return pop


def tournament_select(fitnesses, k, rng):
    """
    Tournament selection for a MINIMISATION problem: pick k random
    chromosomes and return the index of the one with the LOWEST fitness.
    """
    idx = rng.randint(0, len(fitnesses), size=k)
    return idx[np.argmin(fitnesses[idx])]


def uniform_crossover(p1, p2, rng):
    """
    Uniform crossover: each gene is taken from parent 1 or parent 2 with
    equal probability.  Well suited to assignment chromosomes where gene
    position (flower) is meaningful but order is not.
    """
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child.astype(int)


def mutate(child, mut_rate, n_robots, rng):
    """
    Random-reset mutation: each gene, with probability mut_rate, is
    reassigned to a random robot.  Maintains diversity / exploration.
    """
    mask = rng.random(len(child)) < mut_rate
    if mask.any():
        child = child.copy()
        child[mask] = rng.randint(0, n_robots, size=int(mask.sum()))
    return child


# =====================================================================
# 3. GENETIC ALGORITHM  (from scratch)
# =====================================================================
def run_ga_allocation(env, pop_size=POP_SIZE, n_gen=N_GEN,
                      tournament_k=TOURNAMENT_K, cx_rate=CX_RATE,
                      mut_rate=MUT_RATE, elite=ELITE, seed=SEED,
                      record=True):
    """
    Steady evolutionary loop for flower->robot allocation.

    Each generation:
        1. evaluate every chromosome with PSO's identical fitness,
        2. carry the `elite` best chromosomes forward unchanged,
        3. fill the rest of the next generation by tournament-selecting
           two parents, crossing them over (prob cx_rate) and mutating.

    Returns best_metrics (from PSO.evaluate) and the best-fitness history.
    """
    rng = np.random.RandomState(seed)
    n_robots = len(env["robots"])

    pop = init_population(env, pop_size, n_robots, rng)

    best_fit, best_metrics = np.inf, None
    history = []

    for gen in range(n_gen):
        # --- evaluate the whole population ---------------------------
        fits = np.empty(pop_size)
        metrics_list = [None] * pop_size
        for i in range(pop_size):
            f, m = fitness_of(pop[i], env)
            fits[i] = f
            metrics_list[i] = m

        # --- track the global best -----------------------------------
        gi = int(np.argmin(fits))
        if fits[gi] < best_fit:
            best_fit, best_metrics = fits[gi], metrics_list[gi]
        if record:
            history.append(best_fit)

        # --- elitism: keep the best `elite` chromosomes --------------
        elite_idx = np.argsort(fits)[:elite]
        new_pop = [pop[i].copy() for i in elite_idx]

        # --- breed the rest ------------------------------------------
        while len(new_pop) < pop_size:
            p1 = pop[tournament_select(fits, tournament_k, rng)]
            p2 = pop[tournament_select(fits, tournament_k, rng)]
            if rng.random() < cx_rate:
                child = uniform_crossover(p1, p2, rng)
            else:
                child = p1.copy()
            child = mutate(child, mut_rate, n_robots, rng)
            new_pop.append(child)

        pop = np.array(new_pop[:pop_size])

    return best_metrics, history


# =====================================================================
# 4. HYPERPARAMETER TUNING
# =====================================================================
def hyperparameter_study(env):
    """Vary population size, crossover rate and mutation rate for the GA."""
    print("\n[Hyperparameter study] GA allocation ...")
    configs = [
        # (pop_size, cx_rate, mut_rate)
        (40, 0.9, 0.02),
        (40, 0.7, 0.02),
        (40, 0.9, 0.01),
        (40, 0.9, 0.05),
        (30, 0.9, 0.02),
        (60, 0.9, 0.02),
        (40, 1.0, 0.02),
    ]
    rows = []
    for (ps, cx, mu) in configs:
        m, _ = run_ga_allocation(env, pop_size=ps, n_gen=60, cx_rate=cx,
                                 mut_rate=mu, record=False)
        rows.append({"pop_size": ps, "cx_rate": cx, "mut_rate": mu,
                     "fitness": round(m["fitness"], 4),
                     "total_distance": round(m["total_distance"], 2)})
        print(f"  pop={ps:2d} cx={cx:.1f} mut={mu:.2f} "
              f"-> fitness={m['fitness']:.4f} dist={m['total_distance']:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, "ga_alloc_hyperparameter_study.csv"),
              index=False)
    return df


# =====================================================================
# 5. VISUALISATION
# =====================================================================
def plot_convergence(history):
    """GA allocation best-fitness vs generation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, color="#2ca02c", lw=2)
    ax.set_title("GA Allocation -- Best Fitness vs Generation",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best fitness")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_GA, "ga_alloc_convergence.png"), dpi=150)
    plt.close(fig)


def plot_allocation_map(env, metrics):
    """Greenhouse coloured by the GA's flower->robot assignment."""
    flowers = env["flowers"]
    assign = metrics["assign"]
    n_robots = len(env["robots"])
    colours = plt.cm.tab10(np.linspace(0, 1, n_robots))

    fig, ax = plt.subplots(figsize=(8, 8))
    for r in range(n_robots):
        pts = flowers[assign == r]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=colours[r],
                   label=f"Robot {r+1} ({len(pts)})")
    ax.scatter(env["robots"][:, 0], env["robots"][:, 1], marker="s",
               s=120, c="black", label="Robot depot")
    ax.scatter(env["charging"][:, 0], env["charging"][:, 1], marker="P",
               s=160, c="lime", edgecolors="k", label="Charging")
    ax.scatter(env["obstacles"][:, 0], env["obstacles"][:, 1], marker="x",
               s=60, c="red", label="Obstacle")

    ax.set_title("GA Flower-to-Robot Allocation Map\n"
                 "(evolved for short routes + balanced load: ~80 flowers each)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_GA, "ga_alloc_map.png"), dpi=150)
    plt.close(fig)


def plot_pso_vs_ga(rand_m, pso_m, ga_m):
    """Grouped comparison of Random vs PSO vs GA across all objectives."""
    metrics = ["coverage", "total_distance", "total_energy",
               "completion_time", "workload_std", "utilisation"]
    titles = ["Coverage (%)", "Total distance (m)", "Energy (units)",
              "Completion time (s)", "Workload imbalance (std)",
              "Utilisation"]
    scale = [100, 1, 1, 1, 1, 1]

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Task Allocation -- Random vs PSO vs GA",
                 fontsize=15, fontweight="bold")
    for i, m in enumerate(metrics):
        a = ax[i // 3, i % 3]
        vals = [rand_m[m] * scale[i], pso_m[m] * scale[i], ga_m[m] * scale[i]]
        bars = a.bar(["Random", "PSO", "GA"], vals,
                     color=["#8c564b", "#1f77b4", "#2ca02c"])
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, b.get_height(),
                   f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        a.set_title(titles[i])
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_CMP, "pso_vs_ga_allocation.png"), dpi=150)
    plt.close(fig)


def plot_hyperparameters(df):
    """Best fitness per GA-allocation hyperparameter configuration."""
    labels = [f"pop{r.pop_size}\ncx{r.cx_rate}\nmut{r.mut_rate}"
              for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, df["fitness"], color="#17becf")
    ax.set_title("GA Allocation -- Hyperparameter Comparison "
                 "(lower fitness = better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("best fitness")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_GA, "ga_alloc_hyperparameter_comparison.png"),
                dpi=150)
    plt.close(fig)


# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 65)
    print(" TASK ALLOCATION -- PSO vs GA comparison")
    print("=" * 65)

    # identical greenhouse for every method
    env = generate_greenhouse()
    print(f"Greenhouse: {len(env['flowers'])} flowers | "
          f"{len(env['robots'])} robots")

    # --- Random baseline ---------------------------------------------
    rand_m = random_assignment(env)
    print(f"\n[Random] fitness={rand_m['fitness']:.4f} "
          f"dist={rand_m['total_distance']:.1f}")

    # --- PSO allocator (identical fitness) ---------------------------
    print("[PSO]   optimising allocation ...")
    t0 = time.time()
    pso_m, _ = run_pso(env, verbose=False)
    pso_time = time.time() - t0
    print(f"        fitness={pso_m['fitness']:.4f} "
          f"dist={pso_m['total_distance']:.1f} ({pso_time:.2f}s)")

    # --- GA allocator -------------------------------------------------
    print("[GA]    evolving allocation ...")
    t0 = time.time()
    ga_m, history = run_ga_allocation(env)
    ga_time = time.time() - t0
    print(f"        fitness={ga_m['fitness']:.4f} "
          f"dist={ga_m['total_distance']:.1f} ({ga_time:.2f}s)")

    # --- summary table ------------------------------------------------
    print("\n--- ALLOCATION COMPARISON ---")
    print(f"{'metric':<20}{'Random':>12}{'PSO':>12}{'GA':>12}")
    for k, name in [("coverage", "coverage"),
                    ("total_distance", "total_distance"),
                    ("total_energy", "total_energy"),
                    ("completion_time", "completion_time"),
                    ("workload_std", "workload_std"),
                    ("utilisation", "utilisation")]:
        print(f"{name:<20}{rand_m[k]:>12.3f}{pso_m[k]:>12.3f}{ga_m[k]:>12.3f}")

    # --- hyperparameter study ----------------------------------------
    hp_df = hyperparameter_study(env)

    # --- figures ------------------------------------------------------
    print("\n[Plots] generating figures ...")
    plot_convergence(history)
    plot_allocation_map(env, ga_m)
    plot_pso_vs_ga(rand_m, pso_m, ga_m)
    plot_hyperparameters(hp_df)

    # --- CSV results --------------------------------------------------
    print("[Export] writing CSV result files ...")
    pd.DataFrame({"generation": range(1, len(history) + 1),
                  "best_fitness": history}).to_csv(
        os.path.join(CSV_DIR, "ga_alloc_convergence.csv"), index=False)

    rows = []
    for method, m, t in [("Random", rand_m, ""), ("PSO", pso_m, pso_time),
                         ("GA", ga_m, ga_time)]:
        rows.append({"method": method,
                     "fitness": round(m["fitness"], 4),
                     "coverage": round(m["coverage"], 4),
                     "total_distance": round(m["total_distance"], 2),
                     "total_energy": round(m["total_energy"], 2),
                     "completion_time": round(m["completion_time"], 2),
                     "workload_std": round(m["workload_std"], 3),
                     "utilisation": round(m["utilisation"], 3),
                     "time_s": t})
    pd.DataFrame(rows).to_csv(
        os.path.join(CSV_DIR, "pso_vs_ga_allocation.csv"), index=False)

    print("\n" + "=" * 65)
    print(" PSO vs GA ALLOCATION COMPARISON COMPLETE")
    print(f"  Figures  -> {FIG_GA} , {FIG_CMP}")
    print(f"  CSV data -> {CSV_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
