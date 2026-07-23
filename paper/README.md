# Paper: "Matching Algorithm Families to Problem Structure"

Unified arXiv (cs.AI) preprint drawing together the three assignments:
- **CS-1** — urban route finding (state-space search)   [assignment1/ai5.py]
- **CS-2** — fuel rationing (constraint satisfaction)     [assignment2/fuel_crisis_csp.py]
- **CS-3** — smart agriculture (PSO/GA + Value Iteration/Q-Learning)  [assignment3+4/]

## Files
- `paper.tex`        — full self-contained paper (two-column, embedded bibliography)
- `references.bib`   — optional BibTeX source (not needed for the default build)
- `figures/`         — the 11 figures used by the paper (copied from the assignments)
- `figures_unused/`  — figures cut during editing; **not** part of the submission
  (`cs2_solution` = solution-verification panel, now stated in prose;
   `cs1_network` = superseded by the labelled route atlas;
   `cs3_vi_ql_bar` = duplicated Table 5)

## Build
The paper needs **no bibtex** (bibliography is embedded). Two passes for refs/floats:

```
pdflatex paper.tex
pdflatex paper.tex
```

Easiest path if you don't have LaTeX installed: upload `paper/` to **Overleaf**
(New Project → Upload Project → zip of this folder) and click Recompile.

**Important:** zip the folder so that `figures/` sits *next to* `paper.tex`.
If you upload `paper.tex` alone, every image comes out blank/missing.
Use **pdfLaTeX** (not `latex`→dvi) — the figures are PNG, which dvi cannot embed.

## Numbers in the paper are real
Every table/figure comes from the actual scripts (fixed seeds):
- CS-1 from `python assignment1/ai5.py` console table
- CS-2 phase transition from `CSP_PROBE=30,36,40 python assignment2/fuel_crisis_csp.py`
- CS-3 from the CSVs in `assignment3+4/outputs/csv/`

## Before submitting to arXiv (checklist)
1. **Fill in author + affiliation** in `paper.tex` (`[Supervisor Name]`, `[University Name]`).
2. **Supervisor as co-author/endorser** — required for a first cs.AI submission.
3. Multi-seed mean±std for PSO/GA/Q-learning is **already included** (10 seeds;
   raw numbers in `multiseed_results.json` and `multiseed_vi.json`). Optional next
   step: randomize the *instances* too (many greenhouses/graphs), not just seeds.
4. Submit the **source** (this folder zipped), not just the PDF; arXiv compiles it.
5. Suggested category: **cs.AI** (cross-list **cs.LG** optional).
