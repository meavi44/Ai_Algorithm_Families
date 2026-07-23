# Matching Algorithm Families to Problem Structure

Code and experiment scripts for the preprint *"Matching Algorithm Families to
Problem Structure: An Empirical Comparison of Search, Constraint Satisfaction,
and Learning-Based Optimization."*

**Author:** Shraban Karmoker Avi — Department of Computer Science and
Engineering, University of Dhaka.

The central thesis: *the structure of a problem — the size and observability of
its state space, the presence of hard constraints, the shape of its objective,
and whether outcomes are deterministic — dictates which family of algorithms is
appropriate.* Three from-scratch case studies test it.

## Layout

| Folder | Case study | Methods |
|--------|-----------|---------|
| `case-study-1-search-navigation/` | CS-1: urban route finding (state-space search) | BFS, DFS, UCS, DLS, IDS, bidirectional, GBFS, A* |
| `case-study-2-csp-fuel-rationing/` | CS-2: fuel rationing (constraint satisfaction) | backtracking, MRV, LCV, forward checking, AC-3, min-conflicts |
| `case-study-3-optimization-and-learning/` | CS-3: smart agriculture (optimization + learning) | PSO, GA; value iteration, Q-learning |

The write-up of these experiments is the preprint above (available on arXiv); it
is not part of this code repository.

## Reproducing the results

Every experiment fixes the NumPy/Python random seed, so reruns are identical
(wall-clock timings aside). Requirements: Python 3 with NumPy, Matplotlib, pandas
and NetworkX.

```bash
pip install -r requirements.txt

# CS-1: route finding
python case-study-1-search-navigation/route_search.py

# CS-2: fuel rationing CSP
python case-study-2-csp-fuel-rationing/fuel_crisis_csp.py

# CS-3: run in order — each stage writes CSVs the next stage reads
cd case-study-3-optimization-and-learning
python particle_swarm.py        # task allocation (particle swarm)
python genetic_algorithm.py     # task allocation (genetic algorithm)
python value_iteration.py       # navigation policy (dynamic programming)
python q_learning.py            # navigation policy (reinforcement learning)
python supplementary_figures.py # supplementary figures
```

No external solver, optimizer, or deep-learning framework is used: every
algorithm is implemented directly so that measured behaviour reflects the
algorithm itself. Hard-constraint results (CS-2) are re-checked by an independent
verifier that re-evaluates all 11 constraints from the raw problem data.
