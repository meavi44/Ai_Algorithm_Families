# arXiv submission pack

Everything you need to move from `paper.tex` to a posted preprint. Fill the
`[bracketed]` fields first.

---

## 1. Metadata to paste into the arXiv submission form

**Title**
```
Matching Algorithm Families to Problem Structure: An Empirical Comparison of
Search, Constraint Satisfaction, and Learning-Based Optimization
```

**Authors**
```
Anindya Kundu, [Supervisor Full Name]
```

**Primary category:** `cs.AI` (Artificial Intelligence)
**Cross-list (optional):** `cs.LG` (Machine Learning)

**Comments field** (recommended — signals scope and reproducibility):
```
11 figures, 6 tables. Reproducible from-scratch implementations
(NumPy only); code, fixed seeds and CSV outputs included.
```

**ACM classification (optional):** `I.2.8` (Problem Solving, Control Methods,
and Search)

**Abstract (plain text — arXiv strips most LaTeX, so paste this version):**
```
Modern artificial intelligence offers a wide toolbox - uninformed and informed
state-space search, constraint satisfaction, population-based metaheuristics,
dynamic programming and reinforcement learning - yet practitioners often reach
for a familiar algorithm rather than the one the problem calls for. This paper
argues, and empirically demonstrates, a single organizing principle: the
structural properties of a problem determine which family of algorithms is
appropriate. We test this thesis on three deliberately dissimilar decision
problems, each implemented from scratch and evaluated under a common,
reproducible protocol. (i) An urban route-finding task exposes the gap between
uninformed and informed search: A* attains the optimal path cost while expanding
up to 19% fewer nodes than uniform-cost search, whereas greedy best-first search
is cheapest to run but loses optimality when edge weights shift. (ii) A
fuel-rationing task, cast as a constraint satisfaction problem with 11
constraints and a 720^n raw search space, shows a sharp phase transition: at a
saturated instance naive backtracking exhausts its node budget after 2461
backtracks, while adding the minimum-remaining-values heuristic solves the same
instance in 41 nodes with zero backtracking. (iii) A smart-agriculture pipeline
combines combinatorial optimization (assigning 400 flowers to 5 robots, a 5^400
space) with sequential decision-making under uncertainty. A genetic algorithm
outperforms particle swarm optimization on the discrete allocation, and a
model-free Q-learning agent rediscovers a near-optimal navigation policy that
value iteration computes far faster when a model is available. Across all three
studies the same lesson holds: exact methods dominate when the space is small
and known, inference and heuristics rescue tractability at the constraint-
satisfaction phase transition, and population-based or learning methods become
the only viable option once the space is enormous or the environment is unknown.
We release all code, fixed-seed scripts and figures.
```

---

## 2. Endorsement / co-authorship note to your supervisor

arXiv requires a first-time cs.AI submitter to be **endorsed**. The cleanest route
is your supervisor as **co-author** (endorsement then follows). Send them the PDF
plus this note:

> Subject: arXiv preprint — request to co-author / endorse
>
> Dear [Supervisor Name],
>
> I have written up our AI coursework as a single unified research preprint,
> "Matching Algorithm Families to Problem Structure," and I would like to post it
> to arXiv (cs.AI). It reframes the three lab problems (route search, a fuel-
> rationing CSP, and a swarm-intelligence + reinforcement-learning agriculture
> pipeline) as three controlled case studies around one thesis: that a problem's
> structure should dictate the algorithm family. Every result is reproducible
> from fixed-seed, from-scratch code, and I have added a multi-seed robustness
> study and a full related-work section with citations.
>
> arXiv needs a first-time submitter to be endorsed. Would you be willing to be
> listed as co-author (which resolves the endorsement), or, if you prefer, to
> endorse me as an existing arXiv author? I have attached the current PDF and can
> incorporate any changes you suggest before submission.
>
> Thank you,
> Anindya Kundu

---

## 3. Submission-day checklist

- [ ] Fill `[Supervisor Name]` and `[University Name]` in `paper.tex`.
- [ ] Compile locally or on Overleaf; confirm all 11 figures render.
- [ ] Add the real page count to the Comments field above (unknown until compiled).
- [ ] Upload `figures/` only — do **not** include `figures_unused/`.
- [ ] Supervisor confirms co-authorship / endorsement.
- [ ] Zip the `paper/` folder (`paper.tex` + `figures/`). Submit the **source**,
      not just the PDF — arXiv recompiles it.
- [ ] On arXiv: New Submission → upload zip → set category `cs.AI` (+ `cs.LG`) →
      paste title / authors / abstract / comments above.
- [ ] Preview the arXiv-generated PDF; check figures and references.
- [ ] Submit. Moderation typically clears within 1 business day.

## 4. If a moderator puts it on hold
This is usually a reclassification, not a rejection. Your supervisor (as an
affiliated co-author) can reply to the moderation email clarifying that it is an
original comparative study with reproducible experiments. The related-work
section, the multi-seed study, and the institutional affiliation are your main
evidence that it is research, not coursework.
