"""
=====================================================================
build_report_pdf.py
=====================================================================
Project : AI-Powered Autonomous Smart Agriculture
          Population-based Optimization (PSO, GA) + Reinforcement
          Learning (Value Iteration, Q-learning)
Purpose : Assemble ALL generated figures and their DETAILED, beginner
          explanations into a single self-contained PDF -- WITHOUT a
          LaTeX engine.  It mirrors report_figures.tex: a one-time
          notation key, then every figure with a "how to read it"
          breakdown of its axes, curves, colours and symbols.
          Uses only Matplotlib (an allowed project dependency).

          Run AFTER PSO.py, GeneticAlgorithm.py, ValueIteration.py,
          QLearning.py and ExtraGraphs.py have produced the figures.

Output  : report_figures.pdf  (in the project root)
=====================================================================
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

FIG_ROOT = os.path.join("outputs", "figures")
OUT_PDF = "report_figures.pdf"

PAGE_W, PAGE_H = 8.27, 11.69          # A4 portrait, inches
GREEN = "#14532d"

# ---------------------------------------------------------------------
# NOTATION KEY (mirrors the "How to Read These Figures" section)
# ---------------------------------------------------------------------
MARKERS = [
    ("Coloured dots",
     "Flowers. On the allocation maps the colour tells you WHICH robot is "
     "responsible for that flower."),
    ("Magenta star",
     "A target flower the robot must reach and pollinate (a 'goal' cell) on "
     "the navigation grids."),
    ("Black square",
     "An obstacle -- an impassable cell (tree, rock, water). The robot can "
     "never enter it."),
    ("Green / lime square",
     "The robot's START cell (its depot / parking spot)."),
    ("'P' marker (cyan or lime)",
     "A charging station -- where the robot must return once every flower is "
     "done."),
    ("Red x",
     "An obstacle on the allocation maps."),
    ("Small grey arrows (up/down/left/right)",
     "The POLICY: the single best direction to move from that cell. A whole "
     "field of arrows is a complete strategy."),
    ("Coloured line with dots",
     "The robot's actual PATH (trajectory) as it moves from cell to cell."),
]

SYMBOLS = [
    ("Fitness",
     "A single 'score' blending all six allocation objectives into one "
     "number. It is a COST, so LOWER IS BETTER."),
    ("Coverage",
     "Fraction of flowers actually pollinated (100% = every flower served). "
     "Higher is better."),
    ("Total distance",
     "Sum of all robots' flight distances, in metres."),
    ("Energy",
     "Battery units used -- directly proportional to distance."),
    ("Completion time / makespan",
     "The moment the LAST (slowest) robot finishes; the fleet is only done "
     "when everyone is done."),
    ("Workload std. dev. (sigma)",
     "How UNEVEN the flower counts are across robots. 0 = every robot has the "
     "same number (perfectly balanced); large = some robots overloaded."),
    ("Utilization",
     "Fraction of robots given any work at all."),
    ("w, c1, c2  (PSO)",
     "The three weights in PSO's velocity rule: w = inertia (keep going the "
     "same way), c1 = cognitive (pull to my own best), c2 = social (pull to "
     "the swarm's best)."),
    ("pop, cx, mut  (GA)",
     "GA settings: population size, crossover rate (how often two parents are "
     "mixed), mutation rate (how often a gene is randomly changed)."),
    ("U(s)",
     "The VALUE of grid cell s: the total future reward the robot can expect "
     "if it starts there and acts optimally. High near flowers, low far away."),
    ("gamma",
     "Discount factor (0..1): how much a reward LATER is worth vs NOW. Near 1 "
     "= far-sighted."),
    ("theta",
     "Convergence threshold: stop iterating once values change by less than "
     "this tiny amount."),
    ("epsilon",
     "Exploration rate in Q-learning: probability of trying a RANDOM move "
     "instead of the current best."),
    ("alpha",
     "Learning rate in Q-learning: how big a step each new experience takes "
     "when updating a value."),
    ("'smoothed' / 'rolling' / '100-ep average'",
     "A moving average that hides episode-to-episode noise so the underlying "
     "TREND is visible."),
]

# ---------------------------------------------------------------------
# REPORT CONTENT
# Each section: (title, intro, [ (image, title, [how-to-read bullets]) ])
# ---------------------------------------------------------------------
SECTIONS = [
    ("1.  System Overview",
     "Global optimization (PSO and a Genetic Algorithm) assigns flowers to "
     "robots; reinforcement learning (Value Iteration and Q-learning) then "
     "navigates a single robot through a grid-world MDP.",
     [
        ("System/system_pipeline.png", "End-to-end pipeline", [
            "Top row (blue boxes): Phase 1. The greenhouse feeds the allocator "
            "('PSO vs GA'), which outputs flower clusters -- which robot owns "
            "which flowers.",
            "Middle (green box): those clusters define one shared grid-world "
            "MDP for a single robot to navigate.",
            "Two branches below: Phase 2 solves that grid two ways -- Value "
            "Iteration (left, model-based) and Q-learning (right, model-free).",
            "Bottom (yellow box): both branches re-join at 'Completed "
            "Pollination' -- either method finishes the mission.",
            "Italic side captions name the paradigm of each half.",
        ]),
        ("System/all_robots_navigation.png", "Full multi-robot mission", [
            "Axes: grid X and Y -- the greenhouse discretized into a 20x20 "
            "lattice of cells.",
            "Each colour = one robot. Its coloured line is its path; the "
            "coloured stars are its assigned flowers; the coloured square is "
            "where it starts.",
            "Shared black squares are obstacles; cyan 'P' are charging "
            "stations (common to all robots).",
            "Title reports total steps summed over the whole fleet -- evidence "
            "the COMPLETE system finishes, not just one robot.",
        ]),
     ]),

    ("2.  Phase 1 -- PSO Task Allocation",
     "PSO encodes an assignment as a continuous vector (one dimension per "
     "flower, floored to a robot index) and minimizes the weighted "
     "multi-objective fitness from the notation key.",
     [
        ("PSO/pso_convergence.png", "PSO convergence behaviour", [
            "Four panels, all sharing the x-axis 'iteration' (one full swarm "
            "update). Every panel plots the BEST-SO-FAR value.",
            "Top-left, Fitness vs Iteration: the overall cost. Falls then "
            "flattens (improving, then converging). Lower = better.",
            "Top-right, Total Distance (m): total flight distance; may wobble "
            "since it is only one of six competing terms.",
            "Bottom-left, Workload Imbalance (sigma): how unevenly flowers are "
            "split. Falling = fairer.",
            "Bottom-right, Completion Time (s): the makespan (slowest robot's "
            "finish). Also trends down.",
            "Take-away: fitness dropping WHILE imbalance drops = PSO trades a "
            "little distance for a much fairer, cheaper plan.",
        ]),
        ("PSO/pso_workload_utilisation.png", "Workload and utilization", [
            "Two bar charts. 'R1'...'R5' on the x-axis are the five robots.",
            "Left, Workload Distribution: bar height = flowers assigned to each "
            "robot. Dashed line = the mean (~80). Bars near it = balanced.",
            "Right, Travel Distance per Robot: bar height = how far that robot "
            "flies. Similar heights = no robot overworked.",
            "Take-away: PSO gives every robot work and keeps counts AND "
            "distances even.",
        ]),
        ("PSO/pso_allocation_map.png", "PSO allocation map", [
            "Axes: X and Y in metres -- real positions in the 100x100 m "
            "greenhouse.",
            "Small dots = flowers, coloured by the robot assigned to them; "
            "legend gives each robot's flower count.",
            "Black squares = robot depots; lime 'P' = charging; red x = "
            "obstacles.",
            "Take-away: PSO's colours stay intermixed and it is only modestly "
            "better than random -- which motivates the GA next.",
        ]),
        ("PSO/pso_hyperparameter_comparison.png", "PSO hyperparameter study", [
            "Each bar = one PSO setting. The x-label encodes it: S=swarm size, "
            "w=inertia, c=c1/c2, i=iterations.",
            "Bar height = best fitness reached (lower is better).",
            "Take-away: flat-ish bars show PSO is fairly robust; the lowest bar "
            "justifies the default configuration used elsewhere.",
        ]),
     ]),

    ("3.  Phase 1 -- GA Task Allocation (vs. PSO)",
     "A Genetic Algorithm solves the SAME problem with a DISCRETE encoding: a "
     "chromosome is 400 genes (one robot index per flower), evolved with "
     "tournament selection, uniform crossover, random-reset mutation and "
     "elitism, seeded with a distance-greedy assignment.",
     [
        ("GA/ga_alloc_convergence.png", "GA allocation convergence", [
            "x-axis: generation (one cycle of select -> crossover -> mutate). "
            "y-axis: best fitness so far.",
            "The curve only ever goes DOWN because elitism always keeps the "
            "best chromosome -- the population can never get worse.",
            "Numbers: it falls from ~1.6 to ~1.43, landing BELOW PSO's ~1.59 "
            "-- the GA finds a better plan.",
        ]),
        ("GA/ga_alloc_map.png", "GA allocation map", [
            "Read the markers exactly as the PSO map (dots = flowers by robot, "
            "black squares = depots, lime 'P' = charging, red x = obstacles).",
            "Legend counts: every robot gets 79-82 of the 400 flowers -- an "
            "almost perfectly even split.",
            "Honesty note: the colours look as intermixed as PSO's. Expected -- "
            "the 400 flowers sit in dense clusters that MUST be shared to "
            "balance loads. The GA's win is QUANTITATIVE (shorter routes, "
            "tighter balance, lower fitness), not a neater-looking map.",
        ]),
        ("Comparison/pso_vs_ga_allocation.png", "Random vs PSO vs GA", [
            "Six panels, one per objective. In each, the three bars are "
            "Random, PSO and GA; labels print the exact value.",
            "Coverage (%) and Utilisation: higher is better (all reach 100%).",
            "Total distance, Energy, Completion time, Workload sigma: LOWER is "
            "better. The GA (rightmost bar) is lowest in all four.",
            "Take-away -- the central Phase 1 result: the GA is the strongest "
            "allocator, beating PSO which beats Random.",
        ]),
        ("GA/ga_alloc_hyperparameter_comparison.png",
         "GA hyperparameter study", [
            "x-label encodes the setting: pop=population size, cx=crossover "
            "rate, mut=mutation rate.",
            "Bar height = best fitness (lower is better).",
            "Take-away: the tall (bad) bar at high mutation (0.05) shows too "
            "much random change destroys good chromosomes -- small mutation + "
            "strong crossover is the right balance.",
        ]),
     ]),

    ("4.  Phase 2 -- Value Iteration (model-based)",
     "The environment is fully known. Value Iteration applies the Bellman "
     "update  U(s) = max_a sum_s' T(s,a,s')[ R(s,a,s') + gamma U(s') ]  until "
     "convergence, then extracts the greedy policy. R is -0.1 for a normal "
     "step and -5 for a wall-blocked move, so VI feels the same collision "
     "penalty as Q-learning.",
     [
        ("ValueIteration/vi_value_heatmap.png", "Optimal value function U(s)", [
            "Colour = value U(s) of each cell (see the colourbar): bright "
            "yellow = high value (good place), dark purple = low.",
            "Value is HIGHEST at the flowers and fades smoothly with distance "
            "-- like a hill with flowers at the peaks.",
            "Markers: magenta star = flower, red x = obstacle, cyan 'P' = "
            "charging, white square = start.",
            "Take-away: the robot only has to walk 'uphill' on this surface to "
            "reach a flower -- exactly what a correct value function looks "
            "like.",
        ]),
        ("ValueIteration/vi_policy.png", "Optimal policy and executed path", [
            "Grey arrows = the policy (best action in every cell). Follow them "
            "from any cell and you reach a flower optimally.",
            "Orange line with dots = the robot's actual executed mission "
            "(visit all flowers, then return to charge).",
            "Markers: lime square = start, stars = flowers, cyan 'P' = "
            "charging, black squares = obstacles.",
            "Take-away: the orange path faithfully follows the grey arrows -- "
            "policy and value agree.",
        ]),
        ("ValueIteration/vi_convergence.png", "Value Iteration convergence", [
            "x-axis: iteration (one Bellman sweep over all cells).",
            "y-axis (LOG scale): the biggest change in any cell's value that "
            "sweep. As it shrinks, values are settling.",
            "A straight line on a log axis = error shrinks by a constant factor "
            "each step = geometric (fast, guaranteed) convergence.",
            "Take-away: it crosses the stop threshold theta in about 36 "
            "iterations.",
        ]),
        ("ValueIteration/vi_hyperparameters.png",
         "Value Iteration hyperparameter study", [
            "Two panels vs the discount gamma (x-axis); one line per threshold "
            "theta.",
            "Left, Iterations to Converge: bigger gamma spreads value further "
            "so it needs MORE sweeps; smaller theta (stricter) also needs "
            "more.",
            "Right, Steps to Goal: the planned path length -- stays essentially "
            "FLAT at ~63 steps.",
            "Take-away: these settings change only the COMPUTE COST (left), not "
            "the QUALITY of the final path (right).",
        ]),
        ("ValueIteration/vi_value_propagation.png",
         "Value propagation (Bellman backups)", [
            "Four heat-map snapshots, left to right, after 1, 5, 15 and "
            "(converged) iterations; same colour scale (U from 0 to 10).",
            "Panel 1: only cells TOUCHING a flower are bright -- value spread "
            "one ring.",
            "Panels 2-3: the bright region grows outward, ring by ring.",
            "Panel 4 (converged): the whole reachable grid has a sensible value "
            "pointing back to the flowers.",
            "Take-away: this literally shows the Bellman update 'backing up' "
            "reward one step per sweep.",
        ]),
     ]),

    ("5.  Phase 2 -- Q-learning (model-free)",
     "The environment is treated as unknown. The agent learns by trial and "
     "error with an epsilon-greedy policy and the update  Q(s,a) <- Q(s,a) + "
     "alpha[ r + gamma max_a' Q(s',a') - Q(s,a) ].  The state carries a "
     "bitmask of already-pollinated flowers so the memoryless agent can finish "
     "the multi-flower mission.",
     [
        ("QLearning/ql_training_curves.png", "Q-learning training behaviour", [
            "Six panels, all with x-axis 'episode' (one full attempt at the "
            "mission).",
            "Top-left, Reward vs Episode: pale line = raw noisy reward; dark "
            "line = 100-episode average. Climbs from negative to positive.",
            "Top-middle, Learning Curve: just the smoothed reward, so the trend "
            "is crisp.",
            "Top-right, Success Rate (%): rolling share of episodes that finish "
            "the mission; climbs to 100%.",
            "Bottom-left, Episode Length: rolling steps per episode -- FALLS as "
            "the agent stops wandering.",
            "Bottom-middle, Collisions: rolling wall-bumps -- FALLS as it "
            "learns to avoid obstacles.",
            "Bottom-right, Battery Usage: rolling energy -- falls with shorter "
            "routes. Take-away: reward/success up, length/collisions/battery "
            "down = successful learning.",
        ]),
        ("QLearning/ql_epsilon_decay.png", "Epsilon decay schedule", [
            "x-axis: episode. y-axis: epsilon, the chance of a RANDOM move.",
            "Decays from 1.0 (start: 100% random -- pure exploration, the agent "
            "knows nothing) to a floor of 0.05 (mostly exploiting).",
            "The annotation spells it out: high epsilon = explore, low epsilon "
            "= exploit.",
            "Take-away: this schedule is WHY the reward curve can rise -- early "
            "randomness discovers good moves, later greediness locks them in.",
        ]),
        ("QLearning/ql_hyperparameter_comparison.png",
         "Q-learning hyperparameter study", [
            "x-label encodes the setting: a=alpha (learning rate), g=gamma "
            "(discount), d=epsilon-decay.",
            "Bar height = greedy success rate (%); here higher is better.",
            "Take-away: most settings reach 100%, but a too-FAST decay (stops "
            "exploring too early) visibly lowers its bar.",
        ]),
     ]),

    ("6.  Value Iteration vs. Q-learning",
     "Both methods solve the identical grid and are scored the same way -- "
     "rolled out over 200 episodes WITH the 0.8/0.1/0.1 actuator noise on -- "
     "so the comparison is strictly like-for-like.",
     [
        ("Comparison/comparison_bar.png", "VI vs Q-learning performance", [
            "Four panels; in each, the BLUE bar is Value Iteration and the "
            "ORANGE bar is Q-learning. Value labels sit on top of each bar.",
            "Success Rate: both 100%. Average Reward: VI slightly higher "
            "(better). Average Steps & Battery: VI slightly lower (better).",
            "Convergence is deliberately NOT a bar here -- VI counts Bellman "
            "iterations, QL counts episodes (different units); the like-for-"
            "like view is the next figure.",
            "Take-away: the classic model-based vs model-free trade-off -- VI "
            "is a touch better and far faster BECAUSE it already knows the "
            "map.",
        ]),
        ("Comparison/comparison_convergence.png",
         "Convergence comparison (like-for-like)", [
            "Two panels with DIFFERENT x-units, shown separately on purpose.",
            "Left, Value Iteration: max|U change| (log axis) vs iteration; "
            "plunges to the threshold in tens of iterations.",
            "Right, Q-learning: smoothed reward vs episode; the dashed vertical "
            "line marks convergence (~1900 episodes).",
            "Take-away: knowing the model (left) reaches the answer in a "
            "fraction of the effort that learning it (right) requires.",
        ]),
        ("Comparison/vi_vs_ql_trajectory.png", "Optimal vs learned path", [
            "Blue line with circles = the VI (optimal) path; orange line with "
            "squares = the Q-learning (learned) path. Legend gives each step "
            "count.",
            "Grid markers as usual: stars = flowers, black squares = "
            "obstacles, cyan 'P' = charging.",
            "Take-away: both visit every flower and return to charge; the blue "
            "path is a little shorter -- you can SEE the cost of learning "
            "without a model.",
        ]),
        ("Comparison/vi_vs_ql_policy.png", "Optimal vs learned policy", [
            "Two arrow maps side by side. Left = the VI policy. Right = the "
            "Q-learning policy for the 'no flower collected yet' state "
            "(mask=0).",
            "In both, a grey arrow in each cell is the best move; "
            "stars/squares/'P' mark flowers/obstacles/charging.",
            "Take-away: near the flowers the two arrow-fields point the same "
            "way -- model-free Q-learning REDISCOVERED essentially the optimal "
            "policy VI computed directly.",
        ]),
     ]),
]

ALLOC_TABLE = {
    "title": "Table 1.  Task allocation: Random vs. PSO vs. GA "
             "(identical greenhouse and fitness). Lower is better except the "
             "last row.",
    "cols": ["Metric", "Random", "PSO", "GA"],
    "rows": [
        ["Fitness (lower is better)", "1.77", "1.59", "1.43"],
        ["Total distance (m)", "2663", "2451", "2204"],
        ["Energy (units)", "266", "245", "220"],
        ["Completion time (s)", "282", "267", "255"],
        ["Workload std. dev.", "7.1", "2.1", "1.1"],
        ["Coverage / Utilization", "100%", "100%", "100%"],
    ],
    "best_col": 3,
}

NAV_TABLE = {
    "title": "Table 2.  Navigation: Value Iteration vs. Q-learning "
             "(same grid-world MDP, identical stochastic test).",
    "cols": ["Metric", "Value Iteration", "Q-learning"],
    "rows": [
        ["Success rate", "100%", "100%"],
        ["Average reward", "58.8", "58.4"],
        ["Average steps", "84", "87"],
        ["Battery usage", "84", "87"],
        ["Execution time (s)*", "~0.05", "~9-17"],
        ["Convergence", "36 iterations", "1885 episodes"],
    ],
    "best_col": None,
}

NAV_FOOTNOTE = ("* Wall-clock time; hardware dependent and the only "
                "non-reproducible metric. Reward/steps/battery are averaged "
                "over 200 test episodes under the identical 0.8/0.1/0.1 "
                "actuator-noise model for BOTH methods, so the comparison is "
                "strictly like-for-like and reproducible under the fixed seed.")


# =====================================================================
# RENDERING HELPERS
# =====================================================================
def _new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("white")
    return fig


def _bullets_block(bullets, width=96):
    """Wrap a list of bullet strings into one text block with hanging indent."""
    lines = []
    for b in bullets:
        wrapped = textwrap.fill(b, width=width, initial_indent="•  ",
                                subsequent_indent="    ")
        lines.append(wrapped)
    return "\n".join(lines)


def add_title_page(pdf):
    fig = _new_page()
    fig.text(0.5, 0.78, "AI-Powered Autonomous", ha="center",
             fontsize=24, fontweight="bold")
    fig.text(0.5, 0.73, "Smart Agriculture", ha="center",
             fontsize=24, fontweight="bold")
    fig.text(0.5, 0.665,
             "Swarm Intelligence and Reinforcement Learning for an\n"
             "Autonomous Pollination System  --  Figure Guide",
             ha="center", fontsize=13, style="italic")
    fig.text(0.5, 0.62, "_" * 60, ha="center", fontsize=10, color="#888")
    fig.text(0.5, 0.55,
             "Department of Computer Science and Engineering\n"
             "University of Dhaka\n"
             "CSE 4101 -- Artificial Intelligence Laboratory",
             ha="center", fontsize=12)

    abstract = (
        "This document presents and explains every figure produced by the "
        "autonomous pollination system. Global optimization (PSO and a "
        "Genetic Algorithm) assigns flowers to robots; reinforcement learning "
        "(Value Iteration and Q-learning) then navigates a grid-world MDP. "
        "Every figure has a plain-language 'how to read it' breakdown of its "
        "axes, curves, colours and symbols. Start with the Notation Key on the "
        "next page.")
    fig.text(0.12, 0.40, "Abstract", fontsize=13, fontweight="bold")
    fig.text(0.12, 0.375, textwrap.fill(abstract, width=82), fontsize=10.5,
             va="top", ha="left", linespacing=1.5)
    fig.text(0.5, 0.06,
             "Generated from outputs/figures/  (22 figures).  "
             "Mirrors report_figures.tex.",
             ha="center", fontsize=8, color="#666")
    pdf.savefig(fig)
    plt.close(fig)


def add_notation_pages(pdf, title, entries):
    """Flow a list of (term, definition) entries across as many pages as
    needed. Each term is bold; its definition is wrapped beneath it."""
    y = None
    fig = None

    def start_page(first):
        nonlocal fig, y
        fig = _new_page()
        head = title if first else title + "  (cont.)"
        fig.text(0.08, 0.94, head, fontsize=17, fontweight="bold", color=GREEN)
        fig.text(0.08, 0.925, "_" * 78, fontsize=9, color="#ccc")
        y = 0.89

    start_page(True)
    for term, definition in entries:
        wrapped = textwrap.fill(definition, width=90)
        n_lines = wrapped.count("\n") + 1
        block_h = 0.018 + n_lines * 0.016 + 0.012      # term + defn + gap
        if y - block_h < 0.06:                          # no room -> new page
            pdf.savefig(fig)
            plt.close(fig)
            start_page(False)
        fig.text(0.08, y, term, fontsize=10.5, fontweight="bold", va="top")
        fig.text(0.11, y - 0.020, wrapped, fontsize=9.5, va="top",
                 linespacing=1.4)
        y -= block_h
    pdf.savefig(fig)
    plt.close(fig)


def add_section_divider(pdf, section_title, intro):
    fig = _new_page()
    fig.text(0.10, 0.82, section_title, fontsize=20, fontweight="bold",
             color=GREEN)
    fig.text(0.10, 0.80, "_" * 70, fontsize=10, color="#bbb")
    fig.text(0.10, 0.74, textwrap.fill(intro, width=84), fontsize=11.5,
             va="top", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf, section_title, img_rel, title, bullets):
    """One figure per page: header, image (aspect-preserved), then a
    'How to read this figure' bullet breakdown."""
    fig = _new_page()
    fig.text(0.08, 0.965, section_title, fontsize=9, color="#888",
             fontweight="bold")
    fig.text(0.08, 0.935, title, fontsize=15, fontweight="bold", color="#111")

    img_path = os.path.join(FIG_ROOT, img_rel)
    # image box (figure fractions): top ~45% of the page
    bx, by, bw, bh = 0.08, 0.50, 0.84, 0.42
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ih, iw = img.shape[0], img.shape[1]
        img_aspect = iw / ih
        box_w_in, box_h_in = bw * PAGE_W, bh * PAGE_H
        box_aspect = box_w_in / box_h_in
        if img_aspect >= box_aspect:                    # width-limited
            draw_w_in = box_w_in
            draw_h_in = draw_w_in / img_aspect
        else:                                           # height-limited
            draw_h_in = box_h_in
            draw_w_in = draw_h_in * img_aspect
        cw, ch = draw_w_in / PAGE_W, draw_h_in / PAGE_H
        cx = bx + (bw - cw) / 2
        cy = by + (bh - ch)                             # top-align in box
        ax = fig.add_axes([cx, cy, cw, ch])
        ax.imshow(img)
        ax.axis("off")
    else:
        fig.text(0.5, by + bh / 2, f"[missing: {img_rel}]",
                 ha="center", color="red", fontsize=11)

    # "How to read this figure" heading + bullets
    fig.text(0.08, 0.475, "How to read this figure:", fontsize=11.5,
             fontweight="bold", color=GREEN, va="top")
    fig.text(0.08, 0.450, _bullets_block(bullets, width=96), fontsize=9.3,
             va="top", ha="left", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf, alloc, nav, footnote):
    fig = _new_page()
    fig.text(0.08, 0.95, "7.  Summary of Key Results", fontsize=16,
             fontweight="bold", color=GREEN)

    def draw_table(spec, y_top, height):
        ax = fig.add_axes([0.08, y_top - height, 0.84, height])
        ax.axis("off")
        ax.text(0, 1.06, spec["title"], fontsize=10, fontweight="bold",
                transform=ax.transAxes, va="bottom")
        tbl = ax.table(cellText=spec["rows"], colLabels=spec["cols"],
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(fontweight="bold")
            if c == 0:
                cell.set_text_props(ha="left")
                cell._text.set_x(0.04)
            if spec["best_col"] is not None and c == spec["best_col"] and r > 0:
                cell.set_facecolor("#dcfce7")
                cell.set_text_props(fontweight="bold")

    draw_table(alloc, 0.90, 0.30)
    draw_table(nav, 0.52, 0.30)
    fig.text(0.08, 0.14, textwrap.fill(footnote, width=96), fontsize=8.5,
             va="top", ha="left", color="#555", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("Building", OUT_PDF, "...")
    n_fig = 0
    with PdfPages(OUT_PDF) as pdf:
        add_title_page(pdf)
        add_notation_pages(pdf, "How to Read These Figures -- Markers",
                           MARKERS)
        add_notation_pages(pdf, "How to Read These Figures -- Symbols & Terms",
                           SYMBOLS)
        for section_title, intro, figures in SECTIONS:
            add_section_divider(pdf, section_title, intro)
            for img_rel, title, bullets in figures:
                add_figure_page(pdf, section_title, img_rel, title, bullets)
                n_fig += 1
        add_table_page(pdf, ALLOC_TABLE, NAV_TABLE, NAV_FOOTNOTE)

        d = pdf.infodict()
        d["Title"] = "AI-Powered Autonomous Smart Agriculture -- Figure Guide"
        d["Author"] = "CSE 4101 AI Laboratory"
        d["Subject"] = "PSO/GA allocation + Value Iteration/Q-learning navigation"

    print(f"Done: {OUT_PDF}  ({n_fig} figures embedded).")


if __name__ == "__main__":
    main()
