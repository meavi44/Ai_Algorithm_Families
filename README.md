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
| `assignment1/` | CS-1: urban route finding (state-space search) | BFS, DFS, UCS, DLS, IDS, bidirectional, GBFS, A* |
| `assignment2/` | CS-2: fuel rationing (constraint satisfaction) | backtracking, MRV, LCV, forward checking, AC-3, min-conflicts |
| `assignment3+4/` | CS-3: smart agriculture (optimization + learning) | PSO, GA; value iteration, Q-learning |
| `paper/` | LaTeX source, figures, and the compiled preprint | — |

## Reproducing the results

Every experiment fixes the NumPy/Python random seed, so reruns are identical
(wall-clock timings aside). Requirements: Python 3, NumPy, and Matplotlib.

```bash
pip install numpy matplotlib

# CS-1: route finding
python assignment1/ai5.py

# CS-2: fuel rationing CSP
python assignment2/fuel_crisis_csp.py

# CS-3: allocation (PSO / GA) and navigation (VI / Q-learning)
python assignment3+4/PSO.py
python assignment3+4/GeneticAlgorithm.py
python assignment3+4/ValueIteration.py
python assignment3+4/QLearning.py
```

No external solver, optimizer, or deep-learning framework is used: every
algorithm is implemented directly so that measured behaviour reflects the
algorithm itself. Hard-constraint results (CS-2) are re-checked by an independent
verifier that re-evaluates all 11 constraints from the raw problem data.
