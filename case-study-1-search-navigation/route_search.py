"""
Dhaka West-Corridor Navigation Study
------------------------------------
A route-planning experiment over the Mohammadpur / Bosila road network.

The same origin-destination pair is solved under three travel priorities
(shortest, least-traffic, safest) and every classical search strategy is put
side by side so the trade-off between *search effort* and *route quality* is
made visible.

    Blind strategies : Breadth-First, Depth-First, Bidirectional,
                       Depth-Limited, Iterative-Deepening, Uniform-Cost
    Guided strategies: A* and Greedy-Best-First (three heuristics each)
"""

import os, sys, math, time, heapq, textwrap
from collections import deque, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Everything the run produces is dropped into one folder.
REPORT_DIR = "wd_navigation_report"
os.makedirs(REPORT_DIR, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np
except Exception:
    plt = LineCollection = np = None


# ══════════════════════════════════════════════════════════════════════════════
#  1. THE CITY DATA  —  places, how safe / crowded they are, and coordinates
# ══════════════════════════════════════════════════════════════════════════════

SAFETY = {
    "bosila bridge": 0.28, "basilla road": 0.30, "beribadh road": 0.32, "beribadh": 0.32,
    "embankment": 0.75, "geneva camp": 0.22, "bihari camp": 0.22, "40-feet road": 0.30,
    "40 feet road": 0.30, "nobodoy housing": 0.33, "tin rastar mor": 0.45,
    "three-road junction": 0.45, "dhaka uddan": 0.48, "chandrima model town": 0.48,
    "asad avenue": 0.40, "town hall": 0.38, "tajmahal road": 0.42, "ring road": 0.53,
    "tokyo square": 0.55, "mohammadpur krishi": 0.45, "krishi market": 0.45,
    "japan garden city": 0.85, "lalmatia": 0.80, "pisciculture": 0.80, "pc culture": 0.80,
    "baitul aman": 0.74, "st. joseph": 0.72, "residential model": 0.70,
    "zakir hossain": 0.68, "mohammadpur": 0.55, "adabar": 0.75, "shyamoli": 0.72,
    "dhanmondi": 0.82, "rayer bazar": 0.75, "bosila": 0.35, "nabinagar": 0.40, "keraniganj": 0.42,
}
CROWD = {
    "asad avenue": 0.95, "town hall": 0.93, "tajmahal road": 0.90, "ring road": 0.85,
    "tokyo square": 0.84, "mohammadpur krishi": 0.91, "krishi market": 0.91,
    "bosila bridge": 0.55, "beribadh road": 0.40, "beribadh": 0.40, "geneva camp": 0.68,
    "bihari camp": 0.68, "40-feet road": 0.60, "nobodoy housing": 0.52, "tin rastar mor": 0.65,
    "three-road junction": 0.65, "dhaka uddan": 0.62, "chandrima model town": 0.62,
    "embankment": 0.35, "japan garden city": 0.45, "lalmatia": 0.45, "pisciculture": 0.32,
    "pc culture": 0.32, "baitul aman": 0.48, "st. joseph": 0.55,
    "mohammadpur": 0.75, "adabar": 0.50, "shyamoli": 0.52, "dhanmondi": 0.28,
    "rayer bazar": 0.40, "zakir hossain": 0.48,
}
COORDS = {
    "bosila bridge": (23.7380, 90.3458), "basilla road": (23.7370, 90.3480),
    "beribadh road": (23.7450, 90.3490), "beribadh": (23.7450, 90.3490),
    "embankment": (23.7460, 90.3495), "geneva camp": (23.7660, 90.3542),
    "bihari camp": (23.7665, 90.3545), "40-feet road": (23.7420, 90.3510),
    "40 feet road": (23.7420, 90.3510), "nobodoy housing": (23.7630, 90.3580),
    "tin rastar mor": (23.7600, 90.3560), "three-road junction": (23.7590, 90.3555),
    "dhaka uddan": (23.7520, 90.3620), "chandrima model town": (23.7500, 90.3600),
    "asad avenue": (23.7620, 90.3680), "town hall": (23.7640, 90.3700),
    "tajmahal road": (23.7580, 90.3650), "ring road": (23.7490, 90.3570),
    "tokyo square": (23.7480, 90.3590), "mohammadpur krishi": (23.7620, 90.3600),
    "krishi market": (23.7625, 90.3605), "japan garden city": (23.7530, 90.3480),
    "lalmatia": (23.7540, 90.3740), "pisciculture": (23.7700, 90.3520),
    "pc culture": (23.7705, 90.3525), "baitul aman": (23.7480, 90.3500),
    # NOTE: "st joseph" (no period) used to appear here as a second entry with
    # identical coordinates, identical traffic/risk factors and an identical
    # neighbour list -- one landmark entered twice. Every route through St.
    # Joseph therefore existed twice, which inflated node-expansion counts and
    # made two copies of the same physical road look like two distinct routes.
    "st. joseph": (23.7560, 90.3760),
    "residential model": (23.7580, 90.3780), "zakir hossain": (23.7560, 90.3720),
    "mohammadpur": (23.7626, 90.3567), "adabar": (23.7560, 90.3530),
    "shyamoli": (23.7729, 90.3598), "dhanmondi": (23.7461, 90.3742),
    "rayer bazar": (23.7498, 90.3550), "bosila": (23.7385, 90.3460),
    "nabinagar": (23.7410, 90.3520), "keraniganj": (23.7300, 90.3430),
}

# Each place folds its four attributes into one record: lat, lon, safety, crowd.
PLACES = {p: (COORDS[p][0], COORDS[p][1], SAFETY.get(p, 0.5), CROWD.get(p, 0.5))
          for p in SAFETY if p in COORDS}

LINKS = {
    "keraniganj": ["bosila", "nabinagar"],
    "bosila": ["keraniganj", "bosila bridge", "basilla road", "nabinagar"],
    "nabinagar": ["keraniganj", "bosila"],
    "bosila bridge": ["bosila", "basilla road", "embankment"],
    "basilla road": ["bosila bridge", "beribadh", "40-feet road", "asad avenue"],
    "beribadh road": ["basilla road", "beribadh", "40-feet road"],
    "beribadh": ["beribadh road", "basilla road"],
    "40-feet road": ["basilla road", "beribadh road"],
    "asad avenue": ["basilla road", "krishi market", "town hall"],
    "town hall": ["asad avenue", "tajmahal road", "lalmatia"],
    "tajmahal road": ["town hall", "zakir hossain"],
    "zakir hossain": ["tajmahal road", "lalmatia", "st. joseph"],
    "st. joseph": ["zakir hossain", "lalmatia", "residential model"],
    "residential model": ["st. joseph", "dhanmondi", "rayer bazar"],
    "embankment": ["bosila bridge", "pisciculture", "tokyo square"],
    "pisciculture": ["embankment", "nobodoy housing", "pc culture", "ring road"],
    "pc culture": ["pisciculture", "geneva camp", "nobodoy housing"],
    "geneva camp": ["pc culture", "bihari camp", "nobodoy housing"],
    "bihari camp": ["geneva camp", "nobodoy housing"],
    "nobodoy housing": ["pisciculture", "pc culture", "geneva camp", "bihari camp", "tin rastar mor"],
    "tin rastar mor": ["nobodoy housing", "three-road junction", "dhaka uddan"],
    "three-road junction": ["tin rastar mor", "dhaka uddan", "chandrima model town"],
    "dhaka uddan": ["tin rastar mor", "three-road junction", "chandrima model town", "tokyo square"],
    "chandrima model town": ["three-road junction", "dhaka uddan", "tokyo square"],
    "tokyo square": ["dhaka uddan", "chandrima model town", "ring road", "embankment"],
    "ring road": ["tokyo square", "japan garden city", "baitul aman", "pisciculture"],
    "japan garden city": ["ring road", "baitul aman", "adabar"],
    "baitul aman": ["ring road", "japan garden city", "adabar"],
    "adabar": ["japan garden city", "baitul aman", "shyamoli", "residential model"],
    "shyamoli": ["adabar", "dhanmondi"],
    "dhanmondi": ["shyamoli", "residential model"],
    "rayer bazar": ["residential model"],
    "krishi market": ["asad avenue", "lalmatia"],
    "lalmatia": ["town hall", "zakir hossain", "st. joseph", "krishi market"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  2. BUILDING THE WEIGHTED GRAPH  —  edge cost blends distance, traffic, risk
# ══════════════════════════════════════════════════════════════════════════════

def great_circle_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))

# Which travel priority is active is remembered here for the heuristics to read.
ACTIVE_WEIGHTS = {"dist": 0.4, "traffic": 0.3, "risk": 0.3}

PRIORITY_PROFILES = {
    "shortest_path":  {"dist": 0.72, "traffic": 0.14, "risk": 0.14},
    "clear_traffic":  {"dist": 0.28, "traffic": 0.52, "risk": 0.20},
    "safest_route":   {"dist": 0.26, "traffic": 0.22, "risk": 0.52},
}

def assemble_graph(priority="shortest_path"):
    w = PRIORITY_PROFILES.get(priority, PRIORITY_PROFILES["shortest_path"])
    wd, wt, wr = w["dist"], w["traffic"], w["risk"]
    net = defaultdict(list)
    already = set()
    for u in LINKS:
        if u not in PLACES:
            continue
        for v in LINKS[u]:
            if v not in PLACES:
                continue
            key = tuple(sorted((u, v)))
            if key in already:
                continue
            already.add(key)
            lat1, lon1, su, cu = PLACES[u]
            lat2, lon2, sv, cv = PLACES[v]
            span = great_circle_km(lat1, lon1, lat2, lon2)
            jam = (cu + cv) / 2
            hazard = ((1 - su) * (1 + cu) + (1 - sv) * (1 + cv)) / 2
            cost = wd * span + wt * span * jam + wr * span * hazard
            info = {"dist": round(span, 3),
                    "time": round(span * (3.2 + jam * 2.4)),
                    "cong": round(jam, 3),
                    "risk": round(hazard, 3),
                    "cost": round(cost, 4)}
            net[u].append((v, info))
            net[v].append((u, info))
    return net, w


# ══════════════════════════════════════════════════════════════════════════════
#  3. HEURISTICS  —  optimistic straight-line estimates toward the goal
# ══════════════════════════════════════════════════════════════════════════════

# Scales every straight-line estimate down so it stays below the true cost.
# This must hold for ADMISSIBILITY (h <= h*) and, because a_star() keeps a
# closed set and never reopens an expanded node (graph search, not tree
# search), also for CONSISTENCY (h(n) <= c(n,n') + h(n')).  Checked against a
# backward Dijkstra over every node and edge of all three priority profiles:
# the binding constraints are `residential model` and `shyamoli` under the
# clear_traffic and safest_route weightings, which cap RELAX at 0.4682.
# 0.60 -- the earlier value -- violated both properties at those two nodes.
RELAX = 0.45  # <= 0.4682, so admissible AND consistent on every profile

def geo_gap(a, b):
    return great_circle_km(*PLACES[a][:2], *PLACES[b][:2])

def h_plain(node, goal):
    return geo_gap(node, goal) * RELAX

def h_traffic(node, goal):
    return geo_gap(node, goal) * (1 + ACTIVE_WEIGHTS.get("traffic", 0.3) * 0.30) * RELAX

def h_blend(node, goal):
    return geo_gap(node, goal) * (1 + ACTIVE_WEIGHTS.get("traffic", 0.3) * 0.30
                                  + ACTIVE_WEIGHTS.get("risk", 0.3) * 0.08) * RELAX

HEURISTIC_BANK = {
    "H1 (Straight-Line)": h_plain,
    "H2 (Traffic-Aware)": h_traffic,
    "H3 (Blended)": h_blend,
}


# ══════════════════════════════════════════════════════════════════════════════
#  4. SEARCH STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def trace_back(parent, start, goal):
    route, cur = [], goal
    while cur is not None:
        route.append(cur)
        cur = parent.get(cur)
    route.reverse()
    return route if route and route[0] == start else []

def summarize_route(route, net):
    c = d = t = r = 0
    for i in range(len(route) - 1):
        for nb, info in net[route[i]]:
            if nb == route[i + 1]:
                c += info["cost"]; d += info["dist"]; t += info["time"]; r += info["risk"]
                break
    return {"cost": round(c, 3), "dist_km": round(d, 2),
            "time_min": t, "risk": round(r, 3), "hops": len(route) - 1}

# ---- uninformed --------------------------------------------------------------

def breadth_first(net, start, goal):
    frontier = deque([start]); seen = {start}; parent = {start: None}; touched = 0
    while frontier:
        node = frontier.popleft(); touched += 1
        if node == goal:
            return trace_back(parent, start, goal), touched
        for nb, _ in net[node]:
            if nb not in seen:
                seen.add(nb); parent[nb] = node; frontier.append(nb)
    return [], touched

def depth_first(net, start, goal):
    stack = [(start, [start])]; seen = set(); touched = 0
    while stack:
        node, route = stack.pop(); touched += 1
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            return route, touched
        for nb, _ in net[node]:
            if nb not in seen:
                stack.append((nb, route + [nb]))
    return [], touched

def uniform_cost(net, start, goal):
    pq = [(0, start)]; seen = set(); parent = {start: None}; best = {start: 0}; touched = 0
    while pq:
        acc, node = heapq.heappop(pq); touched += 1
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            return trace_back(parent, start, goal), touched
        for nb, info in net[node]:
            nxt = acc + info["cost"]
            if nb not in seen and (nb not in best or nxt < best[nb]):
                best[nb] = nxt; parent[nb] = node; heapq.heappush(pq, (nxt, nb))
    return [], touched

def depth_limited(net, start, goal, ceiling=9):
    stack = [(start, 0, [start])]; seen = set(); touched = 0
    while stack:
        node, depth, route = stack.pop(); touched += 1
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            return route, touched
        if depth < ceiling:
            for nb, _ in net[node]:
                if nb not in seen:
                    stack.append((nb, depth + 1, route + [nb]))
    return [], touched

def iterative_deepening(net, start, goal, ceiling=None):
    ceiling = ceiling or len(net); tally = 0
    for limit in range(ceiling + 1):
        seen = set(); stack = [(start, 0, [start])]
        while stack:
            node, depth, route = stack.pop(); tally += 1
            if node in seen:
                continue
            seen.add(node)
            if node == goal:
                return route, tally
            if depth < limit:
                for nb, _ in net[node]:
                    if nb not in seen:
                        stack.append((nb, depth + 1, route + [nb]))
    return [], tally

def bidirectional(net, start, goal):
    if start == goal:
        return [start], 1
    fwd, bwd = deque([start]), deque([goal])
    fseen, bseen = {start: None}, {goal: None}; touched = 0
    while fwd or bwd:
        if fwd:
            node = fwd.popleft(); touched += 1
            for nb, _ in net[node]:
                if nb in bseen:
                    left = []; cur = node
                    while cur: left.append(cur); cur = fseen[cur]
                    left.reverse()
                    right = []; cur = nb
                    while cur: right.append(cur); cur = bseen[cur]
                    return left + right, touched
                if nb not in fseen:
                    fseen[nb] = node; fwd.append(nb)
        if bwd:
            node = bwd.popleft(); touched += 1
            for nb, _ in net[node]:
                if nb in fseen:
                    left = []; cur = nb
                    while cur: left.append(cur); cur = fseen[cur]
                    left.reverse()
                    right = []; cur = node
                    while cur: right.append(cur); cur = bseen[cur]
                    return left + right, touched
                if nb not in bseen:
                    bseen[nb] = node; bwd.append(nb)
    return [], touched

# ---- informed ----------------------------------------------------------------

def greedy_best_first(net, start, goal, h):
    pq = [(h(start, goal), start)]; seen = set(); parent = {start: None}; touched = 0
    while pq:
        _, node = heapq.heappop(pq); touched += 1
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            return trace_back(parent, start, goal), touched
        for nb, _ in net[node]:
            if nb not in seen:
                parent[nb] = node; heapq.heappush(pq, (h(nb, goal), nb))
    return [], touched

def a_star(net, start, goal, h):
    best = {start: 0}; pq = [(h(start, goal), 0, start)]; parent = {start: None}
    seen = set(); touched = 0
    while pq:
        _, acc, node = heapq.heappop(pq); touched += 1
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            return trace_back(parent, start, goal), touched
        for nb, info in net[node]:
            nxt = acc + info["cost"]
            if nb not in seen and (nb not in best or nxt < best[nb]):
                best[nb] = nxt; parent[nb] = node
                heapq.heappush(pq, (nxt + h(nb, goal), nxt, nb))
    return [], touched


# ══════════════════════════════════════════════════════════════════════════════
#  5. EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_priority(start, goal, priority):
    global ACTIVE_WEIGHTS
    net, weights = assemble_graph(priority)
    ACTIVE_WEIGHTS = weights
    table = []

    def record(label, algo, *extra):
        t0 = time.perf_counter()
        route, expanded = algo(net, start, goal, *extra)
        elapsed = (time.perf_counter() - t0) * 1000
        stats = summarize_route(route, net) if route else {}
        table.append({"name": label, "path": route, "expanded": expanded,
                      "time_ms": round(elapsed, 4), **stats})

    record("BFS", breadth_first)
    record("DFS", depth_first)
    record("BiDir", bidirectional)
    record("DLS", depth_limited)
    record("IDS", iterative_deepening)
    record("UCS", uniform_cost)
    for hname, hfn in HEURISTIC_BANK.items():
        record(f"A* | {hname}", a_star, hfn)
        record(f"GBFS | {hname}", greedy_best_first, hfn)
    return table, start, goal, priority, weights


# ══════════════════════════════════════════════════════════════════════════════
#  6. VISUALS  —  clean, flat styling with a single coherent palette
# ══════════════════════════════════════════════════════════════════════════════

# One design-system-style palette shared by every figure.
INK          = "#1c1c1a"
INK_SOFT     = "#6a6a63"
SURFACE      = "#fcfcfb"
GRID         = "#e4e3dc"
ZONE = {                       # place categories on the map
    "hazard":     "#d03b3b",
    "congested":  "#eda100",
    "sheltered":  "#1baf7a",
    "ordinary":   "#2a78d6",
}
STRATEGY_HUE = {               # one hue per algorithm family
    "BFS": "#2a78d6", "DFS": "#eb6834", "BiDir": "#1baf7a",
    "DLS": "#4a3aa7", "IDS": "#008300", "UCS": "#e34948",
    "A* | H1 (Straight-Line)": "#4a3aa7", "A* | H2 (Traffic-Aware)": "#2a78d6",
    "A* | H3 (Blended)": "#008300",
    "GBFS | H1 (Straight-Line)": "#4a3aa7", "GBFS | H2 (Traffic-Aware)": "#2a78d6",
    "GBFS | H3 (Blended)": "#008300",
}
PRIORITY_ORDER = ["shortest_path", "clear_traffic", "safest_route"]

def zone_of(name):
    low = name.lower()
    for tag in ["bosila", "basilla", "beribadh", "geneva camp", "bihari camp",
                "40-feet", "nobodoy", "tin rastar", "dhaka uddan", "chandrima"]:
        if tag in low:
            return "hazard"
    for tag in ["asad avenue", "town hall", "tajmahal", "ring road", "tokyo square", "krishi market"]:
        if tag in low:
            return "congested"
    for tag in ["japan garden", "lalmatia", "pisciculture", "baitul aman",
                "st. joseph", "residential model", "zakir hossain"]:
        if tag in low:
            return "sheltered"
    return "ordinary"

def _draw_base(ax, net, route=None, route_hue="#2a78d6"):
    edges = []
    for u in net:
        for v, _ in net[u]:
            if u < v:
                edges.append([(PLACES[u][1], PLACES[u][0]), (PLACES[v][1], PLACES[v][0])])
    if edges:
        ax.add_collection(LineCollection(edges, colors=GRID, linewidths=0.9, alpha=0.9, zorder=1))
    for name, (lat, lon, *_ ) in PLACES.items():
        ax.scatter([lon], [lat], s=16, color=ZONE[zone_of(name)], alpha=0.45,
                   zorder=2, edgecolors=SURFACE, linewidths=0.4)
    if route:
        xs = [PLACES[n][1] for n in route]; ys = [PLACES[n][0] for n in route]
        ax.plot(xs, ys, color=route_hue, linewidth=2.6, alpha=0.95,
                zorder=3, solid_capstyle="round")
        for n in route:
            lat, lon = PLACES[n][:2]
            ax.scatter([lon], [lat], s=34, color=ZONE[zone_of(n)], alpha=1.0,
                       zorder=4, edgecolors=SURFACE, linewidths=0.8)

def _mark_ends(ax, start, goal):
    slat, slon = PLACES[start][:2]; glat, glon = PLACES[goal][:2]
    ax.scatter([slon], [slat], s=150, color="#008300", marker="o",
               zorder=6, edgecolors=SURFACE, linewidths=1.6, label="origin")
    ax.scatter([glon], [glat], s=150, color="#d03b3b", marker="s",
               zorder=6, edgecolors=SURFACE, linewidths=1.6, label="destination")

def _bare(ax, title):
    ax.set_title(title, fontsize=8.5, fontweight="bold", color=INK, pad=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)

def figure_overview(net, start, goal):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(SURFACE)
    _draw_base(ax, net)
    _mark_ends(ax, start, goal)
    ax.set_title("West-Dhaka Road Network  ·  Mohammadpur / Bosila Corridor",
                 fontsize=14, fontweight="bold", color=INK)
    ax.text(0.5, 1.005,
            f"origin: {start}    →    destination: {goal}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color=INK_SOFT)
    ax.set_facecolor(SURFACE)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="none", label=k,
                     markerfacecolor=v, markersize=9) for k, v in ZONE.items()]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9)
    ax.tick_params(labelsize=8, colors=INK_SOFT)
    ax.grid(color=GRID, alpha=0.5, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig1_road_network.png"), dpi=150, bbox_inches="tight")
    plt.close()

def _short_name(place):
    """Compact a place name so stop labels stay readable on small maps."""
    parts = place.split()
    if len(parts) == 1:
        return parts[0][:11]
    return parts[0][:7] + " " + parts[1][:6]

def _compact_algo(name):
    """Turn 'A* | H2 (Traffic-Aware)' into a short tag like 'A*·H2'."""
    if "|" in name:
        fam, rest = name.split("|")
        return f"{fam.strip()}·{rest.strip().split()[0]}"
    return name

ROUTE_PALETTE = ["#2a78d6", "#eb6834", "#008300", "#4a3aa7", "#e34948", "#1baf7a"]

def _distinct_routes(all_tables):
    """
    Collapse every strategy's result into the handful of DISTINCT node-paths and
    give each a stable letter A, B, C… (ordered by cheapest cost). Returns the
    ordered list plus a path -> (letter, colour) map so every figure agrees.
    """
    routes = {}
    for table, pretty in all_tables:
        for r in table:
            p = tuple(r["path"])
            if not p:
                continue
            if p not in routes:
                routes[p] = {"hops": r["hops"], "by": {}, "algos": []}
            routes[p]["by"].setdefault(pretty, {"cost": r["cost"]})
            tag = _compact_algo(r["name"])
            if tag not in routes[p]["algos"]:
                routes[p]["algos"].append(tag)
    uniq = sorted(routes.items(), key=lambda kv: min(v["cost"] for v in kv[1]["by"].values()))
    letters = "ABCDEFGH"
    mapping = {p: (letters[i], ROUTE_PALETTE[i % len(ROUTE_PALETTE)])
               for i, (p, _) in enumerate(uniq)}
    return uniq, mapping

def _draw_one_route(ax, base_net, path, hue):
    """A single, clearly readable route: faded map, arrows, numbered + named stops."""
    ax.set_facecolor(SURFACE)
    ctx = [[(PLACES[u][1], PLACES[u][0]), (PLACES[v][1], PLACES[v][0])]
           for u in base_net for v, _ in base_net[u] if u < v]
    ax.add_collection(LineCollection(ctx, colors=GRID, linewidths=0.7, alpha=0.7, zorder=1))
    for _, (la, lo, *_r) in PLACES.items():
        ax.scatter([lo], [la], s=8, color="#d7d6ce", alpha=0.6, zorder=2)

    # direction arrows along the one route
    for i in range(len(path) - 1):
        y1, x1 = PLACES[path[i]][:2]
        y2, x2 = PLACES[path[i + 1]][:2]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=hue, lw=2.6,
                                    alpha=0.95, shrinkA=8, shrinkB=8), zorder=3)

    # numbered badges + a street-name label on every stop
    for i, n in enumerate(path):
        lat, lon = PLACES[n][:2]
        if i == 0:
            mc, mk, sz = "#008300", "o", 260
        elif i == len(path) - 1:
            mc, mk, sz = "#d03b3b", "s", 260
        else:
            mc, mk, sz = hue, "o", 200
        ax.scatter([lon], [lat], s=sz, color=mc, marker=mk, zorder=5,
                   edgecolors=SURFACE, linewidths=1.3)
        ax.text(lon, lat, str(i + 1), color=SURFACE, fontsize=7.5,
                fontweight="bold", ha="center", va="center", zorder=6)
        dy = 0.0016 if i % 2 == 0 else -0.0016
        ax.text(lon, lat + dy, _short_name(n), fontsize=7, color=INK,
                ha="center", va="bottom" if dy > 0 else "top", zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=0.4))

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)

def figure_route_gallery(all_tables, origin, dest):
    """
    A gallery of the DISTINCT routes (one clean map each), not a per-algorithm
    grid. Each card names the streets in order and shows the route's cost under
    every travel priority plus which strategies choose it.
    """
    if plt is None:
        return
    base_net, _ = assemble_graph("shortest_path")   # topology is the same in every priority
    prio_order = [pretty for _, pretty in all_tables]
    uniq, mapping = _distinct_routes(all_tables)
    k = len(uniq)

    ncols = 2
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.8, nrows * 5.9))
    fig.patch.set_facecolor(SURFACE)
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (path, info) in enumerate(uniq):
        ax = flat[idx]
        letter, hue = mapping[path]
        _draw_one_route(ax, base_net, list(path), hue)

        cost_bits = [f"{pr} {info['by'][pr]['cost']:.2f}"
                     for pr in prio_order if pr in info["by"]]
        cheapest = min((v["cost"] for v in info["by"].values()))
        ax.set_title(f"Route {letter}   ·   {info['hops']} hops   ·   "
                     f"cheapest {cheapest:.2f}",
                     fontsize=12, fontweight="bold", color=INK, loc="left", pad=6)

        itinerary = " → ".join(f"{i+1} {_short_name(n)}" for i, n in enumerate(path))
        caption = (
            "\n".join(textwrap.wrap(itinerary, width=58))
            + "\ncost by priority:   " + "    ".join(cost_bits)
            + "\n" + "\n".join(textwrap.wrap("chosen by:  " + "  ".join(info["algos"]), width=64))
        )
        ax.set_xlabel(caption, fontsize=7.2, color=INK_SOFT, labelpad=6, linespacing=1.5)

    for j in range(k, nrows * ncols):     # hide any leftover empty cell
        flat[j].axis("off")

    fig.suptitle(f"Distinct Route Options   ·   S {origin}  →  G {dest}",
                 fontsize=15, fontweight="bold", color=INK, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(REPORT_DIR, "fig2_route_choices.png"), dpi=150, bbox_inches="tight")
    plt.close()

def figure_choice_matrix(all_tables):
    """
    Every algorithm compared in ONE view: rows = the 12 strategies, columns = the
    three priorities. Each cell is coloured by which distinct route (A–E) that
    algorithm chose, and labelled with its cost and nodes expanded.
    """
    if plt is None:
        return
    from matplotlib.patches import Rectangle, Patch
    _uniq, mapping = _distinct_routes(all_tables)
    prio_order = [pretty for _, pretty in all_tables]
    algos = [r["name"] for r in all_tables[0][0]]          # canonical strategy order
    nrows, ncols = len(algos), len(prio_order)

    fig, ax = plt.subplots(figsize=(2.6 + 2.4 * ncols, 1.6 + 0.52 * nrows))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)

    for yi, aname in enumerate(algos):
        for xi, (table, _pretty) in enumerate(all_tables):
            r = next(rr for rr in table if rr["name"] == aname)
            p = tuple(r["path"])
            if p and p in mapping:
                letter, hue = mapping[p]
                txt = f"{letter}   cost {r['cost']:.2f}\n{r['expanded']} nodes"
                tcol = SURFACE
            else:
                hue, letter, txt, tcol = "#eeeeea", "—", "no path", INK_SOFT
            ax.add_patch(Rectangle((xi, nrows - 1 - yi), 1, 1,
                                   facecolor=hue, edgecolor=SURFACE, linewidth=2))
            ax.text(xi + 0.5, nrows - 1 - yi + 0.5, txt, ha="center", va="center",
                    fontsize=8, color=tcol, fontweight="bold")

    ax.set_xticks([i + 0.5 for i in range(ncols)])
    ax.set_xticklabels(prio_order, fontsize=10, fontweight="bold", color=INK)
    ax.set_yticks([nrows - 1 - i + 0.5 for i in range(nrows)])
    ax.set_yticklabels(algos, fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)

    legend = [Patch(facecolor=mapping[p][1], edgecolor="none",
                    label=f"Route {mapping[p][0]} · {info['hops']} hops")
              for p, info in _uniq]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=8.5, title="distinct routes",
              title_fontproperties={"weight": "bold"})

    fig.suptitle("Algorithm Comparison  ·  route chosen, cost & search effort",
                 fontsize=13, fontweight="bold", color=INK, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(REPORT_DIR, "fig5_algorithm_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

def figure_metrics(table, pretty):
    if plt is None:
        return
    named = [r for r in table if r.get("cost")]
    names = [r["name"] for r in named]
    hues = [STRATEGY_HUE.get(n, "#999") for n in names]
    series = {
        "Route cost": [r["cost"] for r in named],
        "Runtime (ms)": [r["time_ms"] for r in named],
        "Nodes expanded": [r["expanded"] for r in named],
        "Distance (km)": [r["dist_km"] for r in named],
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(SURFACE)
    for ax, (ylabel, vals) in zip(axes.flat, series.items()):
        ax.set_facecolor(SURFACE)
        ax.barh(range(len(names)), vals, color=hues, alpha=0.85,
                edgecolor=SURFACE, linewidth=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel, fontsize=10, color=INK_SOFT)
        ax.grid(axis="x", color=GRID, alpha=0.6)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT)
    fig.suptitle(f"Strategy Metrics  ·  {pretty}", fontsize=13, fontweight="bold", color=INK)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    slug = pretty.lower().replace(" ", "_")
    plt.savefig(os.path.join(REPORT_DIR, f"fig3_metrics_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close()

def figure_effort_vs_quality(all_tables):
    """NEW: the trade-off frontier — search effort (x) against route cost (y)."""
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(all_tables), figsize=(len(all_tables) * 5, 4.6), sharey=True)
    if len(all_tables) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    for ax, (table, pretty) in zip(axes, all_tables):
        ax.set_facecolor(SURFACE)
        pts = [r for r in table if r.get("cost")]
        for r in pts:
            fam = "guided" if r["name"].startswith(("A*", "GBFS")) else "blind"
            marker = "^" if fam == "guided" else "o"
            ax.scatter(r["expanded"], r["cost"], s=90, marker=marker,
                       color=STRATEGY_HUE.get(r["name"], "#999"),
                       alpha=0.9, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        # highlight the optimal-cost point reached with the least search effort:
        # cheapest cost first, then fewest nodes expanded to break the tie. On the
        # priorities where the cheapest route is not hop-minimal this is A*, which
        # attains the optimal cost that UCS does but expands fewer nodes.
        best = min(pts, key=lambda r: (r["cost"], r["expanded"]))
        ax.scatter([best["expanded"]], [best["cost"]], s=230, facecolors="none",
                   edgecolors=INK, linewidths=1.6, zorder=4)
        ax.annotate("optimal cost",
                    (best["expanded"], best["cost"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color=INK)
        ax.set_title(pretty, fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("nodes expanded  (search effort →)", fontsize=9, color=INK_SOFT)
        ax.grid(color=GRID, alpha=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=8)
    axes[0].set_ylabel("route cost  (↓ better)", fontsize=9, color=INK_SOFT)
    from matplotlib.lines import Line2D
    key = [Line2D([0], [0], marker="o", color="none", markerfacecolor=INK_SOFT,
                  markersize=9, label="blind"),
           Line2D([0], [0], marker="^", color="none", markerfacecolor=INK_SOFT,
                  markersize=9, label="guided"),
           Line2D([0], [0], marker="o", color="none", markeredgecolor=INK,
                  markerfacecolor="none", markersize=11, label="cheapest route")]
    axes[-1].legend(handles=key, loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("Search Effort vs. Route Quality", fontsize=13.5, fontweight="bold", color=INK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(REPORT_DIR, "fig4_effort_vs_cost.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ORIGIN, DESTINATION = "nabinagar", "dhanmondi"

    print("\n" + "=" * 92)
    print("  WEST-DHAKA NAVIGATION STUDY  ·  MOHAMMADPUR / BOSILA CORRIDOR")
    print("=" * 92)
    header = (f"{'Strategy':<30}{'Cost':>8}{'Dist':>7}{'Risk':>7}"
              f"{'Hops':>6}{'Exp':>6}{'ms':>9}")

    priorities = [("shortest_path", "Shortest Path"),
                  ("clear_traffic", "Clear Traffic"),
                  ("safest_route", "Safest Route")]

    collected = []
    for key, pretty in priorities:
        table, *_rest, weights = run_priority(ORIGIN, DESTINATION, key)
        print(f"\n{'-' * 92}\nPRIORITY: {pretty}   |   weights: {weights}\n{header}\n{'-' * 92}")
        cheapest = min((r["cost"] for r in table if r.get("cost")), default=None)
        for r in table:
            if r.get("cost"):
                flag = "*" if r["cost"] == cheapest else " "
                print(f"{r['name']:<30}{flag}{r['cost']:>7.3f}{r['dist_km']:>7.2f}"
                      f"{r['risk']:>7.3f}{r['hops']:>6}{r['expanded']:>6}{r['time_ms']:>9.4f}")
            else:
                print(f"{r['name']:<30}  (no path within limits)")
        collected.append((table, pretty))
    print("=" * 92)

    base_net, _ = assemble_graph()
    figure_overview(base_net, ORIGIN, DESTINATION)
    figure_route_gallery(collected, ORIGIN, DESTINATION)
    figure_choice_matrix(collected)
    for table, pretty in collected:
        figure_metrics(table, pretty)
    figure_effort_vs_quality(collected)

    print(f"\n[done] figures written to  ./{REPORT_DIR}/")
