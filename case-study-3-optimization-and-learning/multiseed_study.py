"""Multi-seed robustness study for CS-3 (the stochastic methods).

The paper reports the allocation and navigation results as mean +/- std over
10 seeds, not as a single run. The per-algorithm scripts (particle_swarm.py,
genetic_algorithm.py, q_learning.py, value_iteration.py) each execute ONE fixed
seed for their figures; this driver re-runs them across seeds 0..9 on the SAME
fixed environment and aggregates the spread, reproducing the numbers quoted in
Tables for Case Study 3.

Run from anywhere:  python multiseed_study.py
Outputs: outputs/multiseed_results.json, outputs/multiseed_vi.json
"""
import os
import sys
import json
import subprocess
import numpy as np

# resolve relative outputs/ paths against this file's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("outputs", exist_ok=True)

# The navigation MDP (GridWorld) is built from the seed-42 PSO allocation, which
# particle_swarm.py writes to outputs/csv/. Generate it once if absent.
if not os.path.exists(os.path.join("outputs", "csv", "pso_flower_assignment.csv")):
    print("Generating PSO pipeline CSVs (one-time prerequisite) ...")
    subprocess.run([sys.executable, "particle_swarm.py"], check=True,
                   stdout=subprocess.DEVNULL)

from particle_swarm import generate_greenhouse, run_pso
from genetic_algorithm import run_ga_allocation
from value_iteration import GridWorld, evaluate_vi_policy
from q_learning import train_q_learning, evaluate_policy, episodes_to_converge

SEEDS = list(range(10))


def ms(x):
    a = np.asarray(x, float)
    return [float(a.mean()), float(a.std())]


# ---- Phase 1: allocation (PSO vs GA), vary optimizer seed --------------------
env = generate_greenhouse()                 # one fixed greenhouse for all seeds
pso = {"fitness": [], "total_distance": [], "workload_std": []}
ga = {"fitness": [], "total_distance": [], "workload_std": []}
for s in SEEDS:
    m, _ = run_pso(env, seed=s, verbose=False)
    for k in pso:
        pso[k].append(m[k])
    m, _ = run_ga_allocation(env, seed=s, record=False)
    for k in ga:
        ga[k].append(m[k])
    print(f"[alloc] seed {s}: PSO fit={pso['fitness'][-1]:.3f}  GA fit={ga['fitness'][-1]:.3f}")

# ---- Phase 2a: value iteration -- deterministic policy, stochastic eval ------
gw = GridWorld(robot_id=0)
vi = {"success_rate": [], "avg_reward": [], "avg_steps": [], "avg_collisions": []}
for s in SEEDS:
    r = evaluate_vi_policy(gw, seed=s + 100)
    for k in vi:
        vi[k].append(r[k])

# ---- Phase 2b: Q-learning, vary training seed on the same grid ---------------
ql = {"avg_reward": [], "avg_steps": [], "success_rate": [], "conv": []}
for s in SEEDS:
    Q, hist, fb, fm = train_q_learning(gw, seed=s, record=True)
    res = evaluate_policy(gw, Q, fb, fm, seed=s + 100)
    ql["avg_reward"].append(res["avg_reward"])
    ql["avg_steps"].append(res["avg_steps"])
    ql["success_rate"].append(res["success_rate"])
    ql["conv"].append(episodes_to_converge(hist["reward"]))
    print(f"[nav]   seed {s}: VI reward={vi['avg_reward'][s]:.2f}  QL reward={res['avg_reward']:.2f}")

alloc = {"n_seeds": len(SEEDS),
         "PSO": {k: ms(v) for k, v in pso.items()},
         "GA": {k: ms(v) for k, v in ga.items()},
         "QL": {k: ms(v) for k, v in ql.items()}}
vires = {"n_seeds": len(SEEDS), "VI_eval": {k: ms(v) for k, v in vi.items()}}

with open(os.path.join("outputs", "multiseed_results.json"), "w") as f:
    json.dump(alloc, f, indent=2)
with open(os.path.join("outputs", "multiseed_vi.json"), "w") as f:
    json.dump(vires, f, indent=2)

print("\n===== MULTI-SEED SUMMARY (mean +/- std over %d seeds) =====" % len(SEEDS))
for algo in ("PSO", "GA", "QL"):
    print(algo)
    for k, (mu, sd) in alloc[algo].items():
        print(f"  {k:16s} {mu:8.3f} +/- {sd:.3f}")
print("VI_eval")
for k, (mu, sd) in vires["VI_eval"].items():
    print(f"  {k:16s} {mu:8.3f} +/- {sd:.3f}")
