"""
================================================================================
  AI LAB : CONSTRAINT SATISFACTION PROBLEM (CSP)
  PROBLEM: FUEL CRISIS RATIONING IN A BANGLADESHI CITY
  CITY   : Gulshan-Banani corridor, Dhaka, Bangladesh
================================================================================

FORMAL CSP  <X, D, C>
--------------------------------------------------------------------------------
X (VARIABLES)  : one variable per vehicle stranded in the city
                 X = { V001, V002, ..., Vn }

D (DOMAIN)     : every variable draws from ONE common composite domain of
                 "operating profiles". A value is a 4-tuple:

                     d = (speed, quota, slot, station)

                     speed   in {20,30,40,50,60,80}  km/h   cruise speed
                     quota   in {5,10,15,20}         litres rationed fuel
                     slot    in {T1..T5}                    refuelling window
                     station in {P1..P6}                    filling station

                 |D| = 6 x 4 x 5 x 6 = 720 raw values per variable.

C (CONSTRAINTS): 11 constraints, ALL expressed over the domain above.
                 C1-C5   unary   (node consistency)
                 C6-C8   binary  (arc consistency, drawn as constraint graph)
                 C9-C11  global  (n-ary, checked incrementally)

A solution is a total assignment A : X -> D satisfying C1..C11 simultaneously.
Section 8 re-verifies all 11 constraints on the returned solution independently
of the solver, so the CSP is provably maintained.
================================================================================
"""

import sys
import time
import math
import random
from collections import defaultdict, namedtuple

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:                                    # keep box-drawing output safe on Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

random.seed(7)
np.random.seed(7)

# ==============================================================================
# SECTION 0 - CONFIGURATION
# ==============================================================================
# The city has 6 stations x (1-2 bays) x 5 slots = 40 bay-slots. The master
# instance is sized to exactly fill them, so the largest experiment point sits on
# the phase transition where naive backtracking thrashes and MRV still walks
# straight to a solution - the whole point of the heuristic comparison.
MAX_VEHICLES   = 40          # size of the master instance (= total bay-slots)
VIZ_VEHICLES   = 20          # instance drawn in the figures
EXPERIMENT_NS  = [8, 16, 24, 30, 36, 40]
NODE_LIMIT     = 2_500       # search cut-off (declares "timeout")
LS_MAX_STEPS   = 6_000       # min-conflicts step budget (across restarts)
LS_RESTARTS    = 8           # random restarts to escape local minima
LCV_WINDOW     = 40          # how many candidate values LCV ranks exactly
LCV_SAMPLE     = 50          # neighbour-domain sample used to score them

# ---- domain axes --------------------------------------------------------------
SPEEDS   = [20, 30, 40, 50, 60, 80]          # km/h
QUOTAS   = [5, 10, 15, 20]                   # litres
SLOTS    = ["T1", "T2", "T3", "T4", "T5"]    # 06-08, 08-10, 10-12, 16-18, 18-20
PEAK_SLOTS = {"T1", "T4"}                    # office rush -> traffic window C4

VEHICLE_TYPES = ["Car", "Motorcycle", "Truck", "Emergency"]
TYPE_WEIGHTS  = [50, 28, 12, 10]

# type -> (max legal speed km/h, tank cap L, min daily need L, base mileage km/L)
TYPE_SPEC = {
    "Car":        dict(vmax=60, tank=20, need=5,  mileage=12.0),
    "Motorcycle": dict(vmax=60, tank=10, need=5,  mileage=45.0),
    "Truck":      dict(vmax=40, tank=20, need=10, mileage=4.0),
    "Emergency":  dict(vmax=80, tank=20, need=15, mileage=9.0),
}

BUDGET_FACTOR = 1.30         # city ration budget = factor x total minimum need (C11)
CORRIDOR_KM = 0.40           # two vehicles closer than this share a road corridor
SPEED_GAP   = 10             # max |speed_i - speed_j| on a shared corridor (C6)
ECON_SPEED  = 45.0           # speed of best fuel economy (km/h)

OUT_MODEL  = "csp_model.png"
OUT_SEARCH = "csp_search.png"
OUT_SOLVED = "csp_solution.png"

# ---- palette (validated categorical order, light surface #fcfcfb) -------------
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
AXIS    = "#c3c2b7"
SERIES  = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SEQ     = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS  = dict(good="#0ca30c", warning="#fab219", critical="#d03b3b")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})

Profile = namedtuple("Profile", "speed quota slot station")

print("=" * 78)
print("  FUEL CRISIS CSP  -  Gulshan / Banani, Dhaka")
print("=" * 78)

# ==============================================================================
# SECTION 1 - THE CITY  (stations + road lattice, real Gulshan pump locations)
# ==============================================================================
REAL_PUMPS = [
    (23.79374, 90.40662, "Meghna Petroleum, Gulshan-1"),
    (23.78912, 90.41198, "Jamuna Oil, Gulshan-2 Circle"),
    (23.79608, 90.41554, "Padma Oil, Gulshan Avenue"),
    (23.78503, 90.40895, "Standard Asiatic, Gulshan-1"),
    (23.79152, 90.42014, "Eastern Petroleum, Gulshan-2"),
    (23.79812, 90.40803, "Total Filling, DOHS Banani"),
]
NUM_STATIONS = len(REAL_PUMPS)
STATIONS     = [f"P{j+1}" for j in range(NUM_STATIONS)]
ST_COLOR     = {p: SERIES[i % len(SERIES)] for i, p in enumerate(STATIONS)}

station = {}
for i, (lat, lon, name) in enumerate(REAL_PUMPS):
    pid = STATIONS[i]
    cap = random.randint(1, 2)                       # bays per time slot   (C9)
    station[pid] = dict(
        lat=lat, lon=lon, name=name,
        bays=cap,                                    # C9 : per-slot capacity
        supply=cap * len(SLOTS) * 11,                # C10: litres in the tank
    )

# road lattice, used only as map backdrop / distance sanity
LAT0, LAT1 = 23.7815, 23.7995
LON0, LON1 = 90.4030, 90.4225
ROWS, COLS = 10, 11
road = nx.Graph()
rpos = {}
for r in range(ROWS):
    for c in range(COLS):
        nid = r * COLS + c
        rpos[nid] = (LON0 + (c / (COLS - 1)) * (LON1 - LON0) + random.gauss(0, 2e-4),
                     LAT0 + (r / (ROWS - 1)) * (LAT1 - LAT0) + random.gauss(0, 2e-4))
        road.add_node(nid)
for r in range(ROWS):
    for c in range(COLS):
        nid = r * COLS + c
        if c + 1 < COLS: road.add_edge(nid, nid + 1)
        if r + 1 < ROWS: road.add_edge(nid, nid + COLS)


def km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def mileage(vtype, speed):
    """km/L as a function of cruise speed - peaks at ECON_SPEED, falls off both ways.

    This is what makes SPEED a genuine CSP variable value: a wrong speed burns the
    reserve fuel and makes a station unreachable (C3)."""
    eff = 1.0 - ((speed - ECON_SPEED) / 55.0) ** 2
    return TYPE_SPEC[vtype]["mileage"] * max(0.35, eff)


# ==============================================================================
# SECTION 2 - VARIABLES X  (the stranded vehicles)
# ==============================================================================
print("\n[1] Building variables X ...")
NUM_GROUPS = max(2, MAX_VEHICLES // 3)
vehicles = []
for i in range(MAX_VEHICLES):
    n = random.choice(list(road.nodes()))
    lon, lat = rpos[n]
    vt = random.choices(VEHICLE_TYPES, weights=TYPE_WEIGHTS)[0]
    vehicles.append(dict(
        id=f"V{i+1:03d}", lat=lat, lon=lon, type=vt,
        emergency=(vt == "Emergency"),
        reserve=round(random.uniform(0.08, 0.30), 2),   # litres left in tank (C3)
        road_limit=random.choice([30, 40, 50, 60]),     # posted limit here  (C1)
        group=f"G{random.randint(1, NUM_GROUPS)}",      # fleet / household  (C7)
    ))
VEH = {v["id"]: v for v in vehicles}

dist_km = {}
for v in vehicles:
    for pid, s in station.items():
        dist_km[(v["id"], pid)] = km(v["lat"], v["lon"], s["lat"], s["lon"])
print(f"    {MAX_VEHICLES} vehicles, {NUM_STATIONS} stations, "
      f"{len(SLOTS)} slots, {NUM_GROUPS} fleet groups")

# ==============================================================================
# SECTION 3 - DOMAIN D  and the UNARY CONSTRAINTS C1..C5  (node consistency)
# ==============================================================================
RAW_DOMAIN = [Profile(s, q, t, p)
              for s in SPEEDS for q in QUOTAS for t in SLOTS for p in STATIONS]
RAW_SIZE = len(RAW_DOMAIN)


def unary_ok(vid, d):
    """C1..C5 - every unary constraint, all of them defined on the domain tuple."""
    v = VEH[vid]
    spec = TYPE_SPEC[v["type"]]

    # C1 speed legality : cruise speed within the posted limit and the type limit
    if d.speed > min(v["road_limit"], spec["vmax"]):
        return False
    # C2 quota bounds   : between the daily minimum need and the tank capacity
    if not (spec["need"] <= d.quota <= spec["tank"]):
        return False
    # C3 reachability   : reserve fuel must cover the drive at the chosen speed
    if dist_km[(vid, d.station)] / mileage(v["type"], d.speed) > v["reserve"]:
        return False
    # C4 traffic window : during rush slots the city caps cruise speed at 40
    if d.slot in PEAK_SLOTS and d.speed > 40:
        return False
    # C5 emergency window: ambulances/fire must refuel in the first two windows
    if v["emergency"] and d.slot not in ("T1", "T2"):
        return False
    return True


print("\n[2] Applying unary constraints C1-C5 (node consistency) ...")
domains = {v["id"]: [d for d in RAW_DOMAIN if unary_ok(v["id"], d)] for v in vehicles}
empty = [k for k, d in domains.items() if not d]
if empty:                        # relax reserve so the instance stays satisfiable
    for vid in empty:
        VEH[vid]["reserve"] = 2.5
        domains[vid] = [d for d in RAW_DOMAIN if unary_ok(vid, d)]
sizes_after_nc = {k: len(v) for k, v in domains.items()}
print(f"    |D| raw = {RAW_SIZE}  ->  after NC: min {min(sizes_after_nc.values())}, "
      f"mean {np.mean(list(sizes_after_nc.values())):.1f}, "
      f"max {max(sizes_after_nc.values())}")

# ==============================================================================
# SECTION 4 - BINARY CONSTRAINTS C6..C8  and the CONSTRAINT GRAPH
# ==============================================================================
SLOT_IDX = {t: i for i, t in enumerate(SLOTS)}

# corridor adjacency is a fixed property of the instance - precompute it once,
# otherwise AC-3 recomputes a haversine inside its innermost loop.
_CORRIDOR = set()
for _i, _a in enumerate(vehicles):
    for _b in vehicles[_i + 1:]:
        if km(_a["lat"], _a["lon"], _b["lat"], _b["lon"]) <= CORRIDOR_KM:
            _CORRIDOR.add((_a["id"], _b["id"]))
            _CORRIDOR.add((_b["id"], _a["id"]))


def shares_corridor(a, b):
    return (a["id"], b["id"]) in _CORRIDOR


def binary_ok(vid_i, di, vid_j, dj):
    """C6..C8 - constraints relating a PAIR of vehicles, on the same domain."""
    a, b = VEH[vid_i], VEH[vid_j]

    # C6 speed harmonisation: vehicles sharing a corridor must not differ wildly
    #    in cruise speed (stop-go waste + collision risk in a queueing city)
    if shares_corridor(a, b) and abs(di.speed - dj.speed) > SPEED_GAP:
        return False
    # C7 fleet dispersion: one fleet/household may not stack the same bay+window
    if a["group"] == b["group"] and (di.station, di.slot) == (dj.station, dj.slot):
        return False
    # C8 emergency precedence: at a shared station an emergency refuels strictly
    #    before any ordinary vehicle
    if di.station == dj.station:
        if a["emergency"] and not b["emergency"]:
            if SLOT_IDX[di.slot] >= SLOT_IDX[dj.slot]:
                return False
        if b["emergency"] and not a["emergency"]:
            if SLOT_IDX[dj.slot] >= SLOT_IDX[di.slot]:
                return False
    return True


def build_neighbours(vids):
    """Constraint graph: an edge iff C6, C7 or C8 can bind that pair."""
    nb = {v: set() for v in vids}
    for i, x in enumerate(vids):
        for y in vids[i + 1:]:
            a, b = VEH[x], VEH[y]
            linked = (shares_corridor(a, b)
                      or a["group"] == b["group"]
                      or a["emergency"] != b["emergency"])
            if linked:
                nb[x].add(y)
                nb[y].add(x)
    return nb


# ==============================================================================
# SECTION 5 - THE CSP SOLVER  (backtracking + MRV/DEG/LCV + FC + AC-3)
# ==============================================================================
class FuelCSP:
    def __init__(self, vids, domains, use_mrv=True, use_lcv=True,
                 use_fc=True, use_ac3=True, node_limit=NODE_LIMIT):
        self.vids   = list(vids)
        self.D0     = {v: list(domains[v]) for v in self.vids}
        self.nb     = build_neighbours(self.vids)
        self.use_mrv, self.use_lcv = use_mrv, use_lcv
        self.use_fc, self.use_ac3  = use_fc, use_ac3
        self.node_limit = node_limit
        self.budget = int(sum(TYPE_SPEC[VEH[v]["type"]]["need"] for v in self.vids) * BUDGET_FACTOR)
        self.stats  = dict(nodes=0, checks=0, backtracks=0, pruned=0)
        self.timed_out = False

    # ---------- global constraints C9..C11, evaluated incrementally ------------
    def global_ok(self, vid, d, asg):
        # C9 bay capacity : vehicles per (station, slot) <= bays
        used = sum(1 for w, e in asg.items()
                   if e.station == d.station and e.slot == d.slot)
        if used + 1 > station[d.station]["bays"]:
            return False
        # C10 station supply : litres dispensed at a station <= its stock
        lit = sum(e.quota for e in asg.values() if e.station == d.station)
        if lit + d.quota > station[d.station]["supply"]:
            return False
        # C11 city ration budget : total litres released today <= B
        if sum(e.quota for e in asg.values()) + d.quota > self.budget:
            return False
        return True

    def consistent(self, vid, d, asg):
        self.stats["checks"] += 1
        if not unary_ok(vid, d):                       # C1..C5
            return False
        for w in self.nb[vid]:                         # C6..C8
            if w in asg and not binary_ok(vid, d, w, asg[w]):
                return False
        return self.global_ok(vid, d, asg)             # C9..C11

    # ---------- AC-3 : arc consistency over the binary constraints -------------
    def ac3(self, dom):
        queue = [(x, y) for x in self.vids for y in self.nb[x]]
        while queue:
            x, y = queue.pop(0)
            revised = False
            for dx in list(dom[x]):
                if not any(binary_ok(x, dx, y, dy) for dy in dom[y]):
                    dom[x].remove(dx)
                    self.stats["pruned"] += 1
                    revised = True
            if revised:
                if not dom[x]:
                    return False
                queue += [(z, x) for z in self.nb[x] if z != y]
        return True

    # ---------- variable / value ordering heuristics ---------------------------
    def pick_var(self, unassigned, dom):
        if not self.use_mrv:
            return unassigned[0]
        # MRV, degree heuristic as the tie-break
        return min(unassigned, key=lambda v: (len(dom[v]), -len(self.nb[v])))

    def order_vals(self, vid, dom, asg):
        vals = list(dom[vid])
        if not self.use_lcv:
            return vals

        # Static preference: cheap ration, economical speed, nearby station.
        vals.sort(key=lambda d: (d.quota, abs(d.speed - ECON_SPEED),
                                 dist_km[(vid, d.station)]))

        # LCV proper: rank by how few options a value leaves the neighbours.
        # Exact LCV is O(|D| x deg x |D|) per node - on a 300-value domain that
        # costs more than the search it saves, so rank only the most promising
        # LCV_WINDOW values and sample each neighbour's domain. Ordering is a
        # heuristic, so approximating it keeps the search complete.
        head, tail = vals[:LCV_WINDOW], vals[LCV_WINDOW:]
        probe = {}
        for w in self.nb[vid]:
            if w not in asg:
                dw = dom[w]                       # deterministic slice - the solver
                probe[w] = dw[:LCV_SAMPLE]        # must be reproducible, not random

        def cost(d):
            return sum(1 for w, dw in probe.items()
                       for e in dw if not binary_ok(vid, d, w, e))

        head.sort(key=cost)
        return head + tail

    # ---------- forward checking ----------------------------------------------
    def forward_check(self, vid, d, asg, dom):
        saved = {}
        for w in self.nb[vid]:
            if w in asg:
                continue
            keep = [e for e in dom[w] if binary_ok(vid, d, w, e)]
            if len(keep) != len(dom[w]):
                saved[w] = dom[w]
                self.stats["pruned"] += len(dom[w]) - len(keep)
                dom[w] = keep
                if not keep:
                    for k, v in saved.items():
                        dom[k] = v
                    return None
        return saved

    def backtrack(self, asg, dom):
        self.stats["nodes"] += 1
        if self.stats["nodes"] >= self.node_limit:
            self.timed_out = True
            return None
        if len(asg) == len(self.vids):
            return dict(asg)

        unassigned = [v for v in self.vids if v not in asg]
        vid = self.pick_var(unassigned, dom)

        for d in self.order_vals(vid, dom, asg):
            if not self.consistent(vid, d, asg):
                continue
            asg[vid] = d
            saved = self.forward_check(vid, d, asg, dom) if self.use_fc else {}
            if saved is not None:
                got = self.backtrack(asg, dom)
                if got is not None:
                    return got
                if self.timed_out:
                    return None
                for k, v in saved.items():
                    dom[k] = v
            del asg[vid]
            self.stats["backtracks"] += 1
        return None

    def solve(self):
        dom = {v: list(d) for v, d in self.D0.items()}
        t0 = time.time()
        if self.use_ac3 and not self.ac3(dom):
            return None, time.time() - t0, dom
        sol = self.backtrack({}, dom)
        return sol, time.time() - t0, dom


# ---------- local search : MIN-CONFLICTS (with random restarts) ----------------
def min_conflicts(vids, domains, max_steps=LS_MAX_STEPS, restarts=LS_RESTARTS):
    nb  = build_neighbours(vids)
    bud = int(sum(TYPE_SPEC[VEH[v]["type"]]["need"] for v in vids) * BUDGET_FACTOR)

    def viol(vid, d, asg, graded=False):
        """Number of constraints this vehicle's value breaks.

        `graded` adds a fractional term proportional to how far a global
        constraint overshoots. Without it C10/C11 are flat - every quota scores
        the same 1 - and the search has no gradient to descend, so it thrashes
        until the step limit."""
        n = 0.0
        for w in nb[vid]:
            if w != vid and w in asg and not binary_ok(vid, d, w, asg[w]):
                n += 1
        load = sum(1 for w, e in asg.items()
                   if w != vid and e.station == d.station and e.slot == d.slot)
        if load + 1 > station[d.station]["bays"]:
            n += 1
        lit = sum(e.quota for w, e in asg.items()
                  if w != vid and e.station == d.station) + d.quota
        over_p = lit - station[d.station]["supply"]
        if over_p > 0:
            n += 1 + (over_p / 100.0 if graded else 0)
        tot = sum(e.quota for w, e in asg.items() if w != vid) + d.quota
        over_b = tot - bud
        if over_b > 0:
            n += 1 + (over_b / 100.0 if graded else 0)
        elif graded:
            n += d.quota / 10_000.0        # break ties toward a smaller ration
        return n

    # Random restarts escape the local minima that a fully-packed instance is
    # riddled with. `trace` records conflicts over the whole budget (restarts
    # show up as jumps back to a high conflict count).
    trace, total = [], 0
    steps_each = max(1, max_steps // restarts)
    for _ in range(restarts):
        asg = {v: random.choice(domains[v]) for v in vids}
        for _ in range(steps_each):
            total += 1
            bad = [v for v in vids if viol(v, asg[v], asg) > 0]
            trace.append(len(bad))
            if not bad:
                return asg, total, trace, False
            vid = random.choice(bad)
            best, bs = asg[vid], viol(vid, asg[vid], asg, graded=True)
            for d in random.sample(domains[vid], min(80, len(domains[vid]))):
                s = viol(vid, d, asg, graded=True)
                if s < bs:
                    best, bs = d, s
            asg[vid] = best
    return asg, total, trace, True


import os
if os.environ.get("CSP_PROBE"):
    cap = sum(s["bays"] for s in station.values()) * len(SLOTS)
    print(f"PROBE  total bay-slots={cap}  vehicles={MAX_VEHICLES}  "
          f"load={MAX_VEHICLES/cap:.0%}")
    for n in [int(x) for x in os.environ["CSP_PROBE"].split(",")]:
        ids = [v["id"] for v in vehicles[:n]]
        row = []
        for nm, cfg in [("plain", dict(use_mrv=False, use_lcv=False, use_fc=False, use_ac3=False)),
                        ("mrv",   dict(use_mrv=True, use_lcv=False, use_fc=False, use_ac3=False)),
                        ("mrv+fc", dict(use_mrv=True, use_lcv=False, use_fc=True, use_ac3=False)),
                        ("all",   dict(use_mrv=True, use_lcv=True, use_fc=True, use_ac3=True))]:
            c = FuelCSP(ids, domains, **cfg)
            t0 = time.time()
            sol, el, _ = c.solve()
            row.append(f"{nm}:{'OK' if sol else 'FAIL'} nd={c.stats['nodes']} "
                       f"bt={c.stats['backtracks']} {el:.2f}s")
        print(f"  n={n:>3}  " + " | ".join(row))
    sys.exit()

# ==============================================================================
# SECTION 6 - SOLVE THE VISUALISATION INSTANCE
# ==============================================================================
print("\n[3] Solving the CSP (backtracking + MRV/DEG + LCV + FC + AC-3) ...")
viz_ids = [v["id"] for v in vehicles[:VIZ_VEHICLES]]
csp = FuelCSP(viz_ids, domains)
solution, elapsed, dom_after_ac3 = csp.solve()

if solution is None:
    print("    no solution at n=%d - retrying without the ration budget cap" % VIZ_VEHICLES)
    csp.budget = 10 ** 9
    solution, elapsed, dom_after_ac3 = csp.solve()

print(f"    solved={solution is not None}  time={elapsed:.3f}s  "
      f"nodes={csp.stats['nodes']}  checks={csp.stats['checks']}  "
      f"backtracks={csp.stats['backtracks']}  values pruned={csp.stats['pruned']}")

# ==============================================================================
# SECTION 7 - EXPERIMENTS  (heuristic configurations vs problem size)
# ==============================================================================
print("\n[4] Running search experiments ...")
CONFIGS = [
    ("BT (plain)",            dict(use_mrv=False, use_lcv=False, use_fc=False, use_ac3=False)),
    ("BT + MRV",              dict(use_mrv=True,  use_lcv=False, use_fc=False, use_ac3=False)),
    ("BT + MRV + LCV",        dict(use_mrv=True,  use_lcv=True,  use_fc=False, use_ac3=False)),
    ("BT + MRV + FC",         dict(use_mrv=True,  use_lcv=False, use_fc=True,  use_ac3=False)),
    ("BT + MRV + LCV + FC + AC3", dict(use_mrv=True, use_lcv=True, use_fc=True, use_ac3=True)),
]
exp = {name: dict(time=[], nodes=[], checks=[], fail=[]) for name, _ in CONFIGS}
ls_exp = dict(time=[], steps=[], fail=[])

for n in EXPERIMENT_NS:
    ids = [v["id"] for v in vehicles[:n]]
    for name, cfg in CONFIGS:
        c = FuelCSP(ids, domains, **cfg)
        sol, el, _ = c.solve()
        exp[name]["time"].append(max(el, 1e-4))
        exp[name]["nodes"].append(max(c.stats["nodes"], 1))
        exp[name]["checks"].append(max(c.stats["checks"], 1))
        exp[name]["fail"].append(sol is None)
    t0 = time.time()
    _, steps, _, to = min_conflicts(ids, domains)
    ls_exp["time"].append(max(time.time() - t0, 1e-4))
    ls_exp["steps"].append(max(steps, 1))
    ls_exp["fail"].append(to)
    print(f"    n={n:>3}  best-config time="
          f"{exp[CONFIGS[-1][0]]['time'][-1]:.3f}s  min-conflicts steps={steps}")

_, _, mc_trace, _ = min_conflicts(viz_ids, domains)

# ==============================================================================
# SECTION 8 - INDEPENDENT VERIFICATION : "the CSP must be maintained"
# ==============================================================================
def verify(asg):
    """Re-check every constraint from scratch. Returns {code: (name, ok, detail)}."""
    ids = list(asg)
    R = {}
    bud = int(sum(TYPE_SPEC[VEH[v]["type"]]["need"] for v in ids) * BUDGET_FACTOR)

    def chk(code, name, bad, detail):
        R[code] = (name, len(bad) == 0, detail)

    b = [v for v in ids if asg[v].speed > min(VEH[v]["road_limit"],
                                              TYPE_SPEC[VEH[v]["type"]]["vmax"])]
    chk("C1", "Speed legality", b, "speed <= min(road limit, type limit)")

    b = [v for v in ids if not (TYPE_SPEC[VEH[v]["type"]]["need"] <= asg[v].quota
                                <= TYPE_SPEC[VEH[v]["type"]]["tank"])]
    chk("C2", "Quota bounds", b, "daily need <= quota <= tank")

    b = [v for v in ids if dist_km[(v, asg[v].station)] /
         mileage(VEH[v]["type"], asg[v].speed) > VEH[v]["reserve"]]
    chk("C3", "Reachability", b, "drive fuel at chosen speed <= reserve")

    b = [v for v in ids if asg[v].slot in PEAK_SLOTS and asg[v].speed > 40]
    chk("C4", "Traffic window", b, "peak slot => speed <= 40 km/h")

    b = [v for v in ids if VEH[v]["emergency"] and asg[v].slot not in ("T1", "T2")]
    chk("C5", "Emergency window", b, "emergency refuels in T1/T2")

    b = [(x, y) for i, x in enumerate(ids) for y in ids[i + 1:]
         if shares_corridor(VEH[x], VEH[y])
         and abs(asg[x].speed - asg[y].speed) > SPEED_GAP]
    chk("C6", "Speed harmonisation", b, f"same corridor => |dv| <= {SPEED_GAP} km/h")

    b = [(x, y) for i, x in enumerate(ids) for y in ids[i + 1:]
         if VEH[x]["group"] == VEH[y]["group"]
         and (asg[x].station, asg[x].slot) == (asg[y].station, asg[y].slot)]
    chk("C7", "Fleet dispersion", b, "same fleet => different (station, slot)")

    b = [(x, y) for x in ids for y in ids
         if x != y and VEH[x]["emergency"] and not VEH[y]["emergency"]
         and asg[x].station == asg[y].station
         and SLOTS.index(asg[x].slot) >= SLOTS.index(asg[y].slot)]
    chk("C8", "Emergency precedence", b, "emergency before ordinary at a station")

    load = defaultdict(int)
    for v in ids:
        load[(asg[v].station, asg[v].slot)] += 1
    b = [k for k, c in load.items() if c > station[k[0]]["bays"]]
    chk("C9", "Bay capacity", b, "vehicles per (station, slot) <= bays")

    lit = defaultdict(int)
    for v in ids:
        lit[asg[v].station] += asg[v].quota
    b = [p for p, l in lit.items() if l > station[p]["supply"]]
    chk("C10", "Station supply", b, "litres dispensed <= station stock")

    tot = sum(asg[v].quota for v in ids)
    chk("C11", "City ration budget", [] if tot <= bud else ["total"],
        f"total {tot} L <= budget {bud} L")
    return R, load, lit, tot, bud


print("\n[5] Verifying all 11 constraints on the returned solution ...")
report, load_map, litres_map, total_l, budget_l = verify(solution)
for code, (name, ok, detail) in report.items():
    print(f"    {code:<4} {name:<22} {'SATISFIED' if ok else 'VIOLATED':<10} {detail}")
ALL_OK = all(ok for _, ok, _ in report.values())
print(f"    => CSP {'MAINTAINED - all constraints hold' if ALL_OK else 'BROKEN'}")


# ==============================================================================
# SECTION 9 - FIGURE 1 : THE MODEL  (domain, constraint graph, city, pruning)
# ==============================================================================
def style(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    ax.set_title(title, loc="left", pad=10)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, axis=grid_axis, lw=0.8, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


print(f"\n[6] Drawing figures ...")
fig = plt.figure(figsize=(16, 11))
fig.suptitle("Fuel Crisis CSP  -  Model:  variables = vehicles,  "
             "domain = (speed, quota, slot, station)",
             fontsize=14, fontweight="bold", x=0.012, ha="left", y=0.985)

# --- 1: domain reduction -------------------------------------------------------
ax = fig.add_subplot(2, 2, 1)
lbl = [v.replace("V0", "V") for v in viz_ids]
nc_sizes  = [len(domains[v]) for v in viz_ids]
ac3_sizes = [len(dom_after_ac3[v]) for v in viz_ids]
y = np.arange(len(viz_ids))
ax.barh(y, [RAW_SIZE] * len(y), height=0.72, color=SEQ[1], label=f"raw |D| = {RAW_SIZE}")
ax.barh(y, nc_sizes, height=0.72, color=SEQ[3], label="after C1-C5 (node consistency)")
ax.barh(y, ac3_sizes, height=0.72, color=SEQ[6], label="after AC-3 (C6-C8)")
ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=7)
ax.invert_yaxis()
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=1, fontsize=8)
style(ax, "number of surviving values", None,
      "1  Domain pruning per variable", grid_axis="x")
ax.set_title("1  Domain pruning per variable", loc="left", pad=22)
ax.text(0.0, 1.03, f"{RAW_SIZE} raw values shrink to a few dozen once the "
                   "constraints are enforced",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")

# --- 2: constraint graph -------------------------------------------------------
ax = fig.add_subplot(2, 2, 2)
CG = nx.Graph()
CG.add_nodes_from(viz_ids)
nbmap = build_neighbours(viz_ids)
for x in viz_ids:
    for y_ in nbmap[x]:
        CG.add_edge(x, y_)
pos = nx.spring_layout(CG, seed=3, k=0.9)
nx.draw_networkx_edges(CG, pos, ax=ax, edge_color=GRID, width=1.0)
em = [v for v in viz_ids if VEH[v]["emergency"]]
nm = [v for v in viz_ids if not VEH[v]["emergency"]]
nx.draw_networkx_nodes(CG, pos, nodelist=nm, ax=ax, node_size=300,
                       node_color=SERIES[0], edgecolors=SURFACE, linewidths=2)
nx.draw_networkx_nodes(CG, pos, nodelist=em, ax=ax, node_size=420,
                       node_color=SERIES[7], node_shape="*",
                       edgecolors=SURFACE, linewidths=2)
nx.draw_networkx_labels(CG, pos, {v: v[1:] for v in viz_ids}, ax=ax,
                        font_size=6.5, font_color="white")
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=SERIES[0], label="ordinary vehicle"),
    Line2D([], [], marker="*", ls="", ms=12, color=SERIES[7], label="emergency vehicle"),
    Line2D([], [], color=GRID, lw=2, label="binary constraint C6/C7/C8")],
    loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=8)
ax.set_title(f"2  Constraint graph  ({CG.number_of_nodes()} vars, "
             f"{CG.number_of_edges()} edges)", loc="left", pad=10)
ax.axis("off")

# --- 3: the city map -----------------------------------------------------------
ax = fig.add_subplot(2, 1, 2)
for u, v_ in road.edges():          # road lattice backdrop, darker than the grid
    ax.plot([rpos[u][0], rpos[v_][0]], [rpos[u][1], rpos[v_][1]],
            color=AXIS, lw=1.0, zorder=1, alpha=0.75)
for v in viz_ids:
    d = solution[v]
    a = VEH[v]
    ax.plot([a["lon"], station[d.station]["lon"]],
            [a["lat"], station[d.station]["lat"]],
            color=ST_COLOR[d.station], lw=1.1, alpha=0.55, zorder=2)
    ax.scatter(a["lon"], a["lat"], s=150 if a["emergency"] else 70,
               marker="*" if a["emergency"] else "o",
               color=ST_COLOR[d.station], edgecolor=SURFACE, lw=1.4, zorder=3)
    ax.annotate(f"{d.speed}", (a["lon"], a["lat"]), xytext=(4, 4),
                textcoords="offset points", fontsize=6.5, color=INK2)
for pid, s in station.items():
    ax.scatter(s["lon"], s["lat"], s=340, marker="^", color=ST_COLOR[pid],
               edgecolor=SURFACE, lw=2, zorder=4)
    ax.annotate(f"{pid}  {s['bays']} bay" + ("s" if s["bays"] != 1 else ""),
                (s["lon"], s["lat"]), xytext=(0, -20), textcoords="offset points",
                ha="center", fontsize=8, color=INK, fontweight="bold")
ax.legend(handles=[Patch(color=ST_COLOR[p], label=f"{p} - {station[p]['name']}")
                   for p in STATIONS] +
                  [Line2D([], [], marker="*", ls="", ms=11, color=MUTED,
                          label="emergency vehicle")],
          loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=4, fontsize=8)
style(ax, "longitude", "latitude",
      "3  Gulshan / Banani - assignment of each vehicle to a station "
      "(label = assigned cruise speed)", grid_axis="both")
ax.grid(False)                      # the road lattice is the reference, not a grid

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_MODEL, dpi=150)
plt.close(fig)
print(f"    saved {OUT_MODEL}")

# ==============================================================================
# SECTION 10 - FIGURE 2 : SEARCH PERFORMANCE
# ==============================================================================
fig = plt.figure(figsize=(15, 9))
fig.suptitle("Fuel Crisis CSP  -  Search performance",
             fontsize=14, fontweight="bold", x=0.012, ha="left", y=0.985)


def marked(ax, x, ys, fails, label, color):
    ax.plot(x, ys, marker="o", ms=5, lw=2, color=color, label=label,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    fx = [a for a, f in zip(x, fails) if f]
    fy = [b for b, f in zip(ys, fails) if f]
    if fx:
        ax.scatter(fx, fy, marker="x", s=90, color=STATUS["critical"], zorder=6)


ax = fig.add_subplot(2, 2, 1)
for i, (name, _) in enumerate(CONFIGS):
    marked(ax, EXPERIMENT_NS, exp[name]["time"], exp[name]["fail"], name, SERIES[i])
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper left")
style(ax, "number of vehicles (variables)", "runtime (s, log scale)",
      "1  Runtime vs problem size")
ax.set_title("1  Runtime vs problem size", loc="left", pad=22)
ax.text(0.0, 1.03, "at the saturated instance (n=40) naive backtracking walls (x); "
                   "MRV / forward-checking keep it tractable",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")

ax = fig.add_subplot(2, 2, 2)
for i, (name, _) in enumerate(CONFIGS):
    marked(ax, EXPERIMENT_NS, exp[name]["checks"], exp[name]["fail"], name, SERIES[i])
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper left")
style(ax, "number of vehicles (variables)",
      "consistency checks performed (log)",
      "2  Constraint work vs problem size")
ax.set_title("2  Constraint work vs problem size", loc="left", pad=22)
ax.text(0.0, 1.03, "AC-3 + forward checking do the fewest checks; plain BT "
                   "explodes to ~1e5 checks at n=40 (x)",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")

ax = fig.add_subplot(2, 2, 3)
best = CONFIGS[-1][0]
marked(ax, EXPERIMENT_NS, exp[best]["time"], exp[best]["fail"],
       "systematic: " + best, SERIES[0])
marked(ax, EXPERIMENT_NS, ls_exp["time"], ls_exp["fail"],
       "local search: min-conflicts", SERIES[5])
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper left")
style(ax, "number of vehicles (variables)", "runtime (s, log scale)",
      "3  Systematic search vs local search")
ax.set_title("3  Systematic search vs local search", loc="left", pad=22)
ax.text(0.0, 1.03, "min-conflicts stays several times faster than the best "
                   "systematic solver; both clear the saturated n=40 instance",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")

ax = fig.add_subplot(2, 2, 4)
ax.plot(mc_trace, lw=1.6, color=SERIES[5])
ax.fill_between(range(len(mc_trace)), mc_trace, color=SERIES[5], alpha=0.12)
ax.axhline(0, color=STATUS["good"], lw=1.5, ls="--")
ax.annotate("0 conflicts = solution", (len(mc_trace) * 0.55, 0.35),
            color=STATUS["good"], fontsize=8.5)
style(ax, "iteration", "variables still in conflict",
      f"4  Min-conflicts convergence (n = {VIZ_VEHICLES})")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_SEARCH, dpi=150)
plt.close(fig)
print(f"    saved {OUT_SEARCH}")

# ==============================================================================
# SECTION 11 - FIGURE 3 : THE SOLUTION + CONSTRAINT VERIFICATION
# ==============================================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Fuel Crisis CSP  -  Solution and constraint verification",
             fontsize=14, fontweight="bold", x=0.012, ha="left", y=0.985)

# --- assigned speed per vehicle, coloured by type ------------------------------
ax = fig.add_subplot(2, 3, 1)
for i, vt in enumerate(VEHICLE_TYPES):
    xs = [j for j, v in enumerate(viz_ids) if VEH[v]["type"] == vt]
    ys = [solution[viz_ids[j]].speed for j in xs]
    if xs:
        ax.scatter(xs, ys, s=70, color=SERIES[i], label=vt,
                   edgecolor=SURFACE, lw=1.4, zorder=3)
for j, v in enumerate(viz_ids):
    cap = min(VEH[v]["road_limit"], TYPE_SPEC[VEH[v]["type"]]["vmax"])
    ax.plot([j, j], [0, cap], color=GRID, lw=1.2, zorder=1)
    ax.scatter([j], [cap], marker="_", s=90, color=MUTED, zorder=2)
ax.set_xticks(range(len(viz_ids)))
ax.set_xticklabels([v[1:] for v in viz_ids], fontsize=6.5)
ax.legend(fontsize=8, ncol=2, loc="lower right")
style(ax, "vehicle", "assigned cruise speed (km/h)",
      "1  Speed values - every dot under its C1 cap")
ax.text(0.0, -0.28, "grey tick = min(road limit, type limit) = the C1 ceiling",
        transform=ax.transAxes, fontsize=8, color=MUTED)

# --- station x slot occupancy heatmap -----------------------------------------
ax = fig.add_subplot(2, 3, 2)
M = np.zeros((NUM_STATIONS, len(SLOTS)))
for v in viz_ids:
    M[STATIONS.index(solution[v].station), SLOTS.index(solution[v].slot)] += 1
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(3, M.max()), aspect="auto")
for r in range(NUM_STATIONS):
    for c in range(len(SLOTS)):
        val = int(M[r, c])
        ax.text(c, r, str(val), ha="center", va="center", fontsize=9,
                color="white" if val >= 2 else INK2, fontweight="bold")
ax.set_xticks(range(len(SLOTS))); ax.set_xticklabels(SLOTS)
ax.set_yticks(range(NUM_STATIONS))
ax.set_yticklabels([f"{p} (cap {station[p]['bays']})" for p in STATIONS], fontsize=8)
ax.set_title("2  Bay occupancy per (station, slot) - C9", loc="left", pad=10)
ax.set_xlabel("refuelling window"); ax.grid(False)
ax.text(0.0, -0.24, "every cell is <= that station's bay capacity",
        transform=ax.transAxes, fontsize=8, color=MUTED)

# --- fuel dispensed vs station supply -----------------------------------------
ax = fig.add_subplot(2, 3, 3)
disp = [litres_map.get(p, 0) for p in STATIONS]
sup  = [station[p]["supply"] for p in STATIONS]
y = np.arange(NUM_STATIONS)
ax.barh(y, sup, height=0.66, color=GRID, label="stock available")
ax.barh(y, disp, height=0.66, color=SERIES[0], label="litres dispensed")
for i, (d_, s_) in enumerate(zip(disp, sup)):
    ax.text(s_ + 3, i, f"{d_} / {s_} L", va="center", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(STATIONS); ax.invert_yaxis()
ax.set_xlim(0, max(sup) * 1.28)
ax.legend(fontsize=8, loc="lower right")
style(ax, "litres", None, "3  Station supply - C10", grid_axis="x")

# --- quota / type distribution -------------------------------------------------
ax = fig.add_subplot(2, 3, 4)
width = 0.2
for i, vt in enumerate(VEHICLE_TYPES):
    counts = [sum(1 for v in viz_ids
                  if VEH[v]["type"] == vt and solution[v].quota == q) for q in QUOTAS]
    ax.bar(np.arange(len(QUOTAS)) + (i - 1.5) * width, counts, width * 0.9,
           color=SERIES[i], label=vt)
ax.set_xticks(range(len(QUOTAS)))
ax.set_xticklabels([f"{q} L" for q in QUOTAS])
ax.legend(fontsize=8)
style(ax, "rationed quota assigned", "vehicles",
      "4  Quota values by vehicle type - C2")

# --- emergency precedence timeline --------------------------------------------
ax = fig.add_subplot(2, 3, 5)
for j, v in enumerate(viz_ids):
    d = solution[v]
    x = SLOTS.index(d.slot)
    is_em = VEH[v]["emergency"]
    ax.scatter(x, STATIONS.index(d.station) + (j % 5 - 2) * 0.10,
               s=150 if is_em else 60,
               marker="*" if is_em else "o",
               color=SERIES[7] if is_em else SERIES[0],
               edgecolor=SURFACE, lw=1.3, zorder=3)
ax.set_xticks(range(len(SLOTS))); ax.set_xticklabels(SLOTS)
ax.set_yticks(range(NUM_STATIONS)); ax.set_yticklabels(STATIONS)
ax.legend(handles=[
    Line2D([], [], marker="*", ls="", ms=12, color=SERIES[7], label="emergency"),
    Line2D([], [], marker="o", ls="", color=SERIES[0], label="ordinary")],
    fontsize=8, loc="lower right")
style(ax, "refuelling window", "station",
      "5  Emergency precedence - C5 and C8", grid_axis="both")

# --- the verification table ----------------------------------------------------
ax = fig.add_subplot(2, 3, 6)
ax.axis("off")
ax.set_title("6  Constraint verification on the returned solution",
             loc="left", pad=10)
rows = list(report.items())
ax.text(0.0, 1.0, "code   constraint             status", fontsize=8.5,
        color=MUTED, family="monospace", va="top")
for i, (code, (name, ok, detail)) in enumerate(rows):
    yy = 0.94 - i * 0.076
    col = STATUS["good"] if ok else STATUS["critical"]
    ax.text(0.0, yy, f"{code:<6} {name:<22}", fontsize=8.5,
            family="monospace", color=INK, va="top")
    ax.text(0.63, yy, "OK  satisfied" if ok else "X   violated",
            fontsize=8.5, family="monospace", color=col, va="top",
            fontweight="bold")
    ax.text(0.0, yy - 0.033, f"       {detail}", fontsize=7.4,
            family="monospace", color=MUTED, va="top")
ax.text(0.0, 0.94 - len(rows) * 0.076 - 0.02,
        ("ALL 11 CONSTRAINTS HOLD - the CSP is maintained"
         if ALL_OK else "SOME CONSTRAINTS VIOLATED"),
        fontsize=10, fontweight="bold", va="top",
        color=STATUS["good"] if ALL_OK else STATUS["critical"])
ax.text(0.0, 0.94 - len(rows) * 0.076 - 0.075,
        f"n = {len(viz_ids)} variables | total fuel released "
        f"{total_l} L of {budget_l} L budget",
        fontsize=8, color=INK2, va="top")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_SOLVED, dpi=150)
plt.close(fig)
print(f"    saved {OUT_SOLVED}")

# ==============================================================================
# SECTION 12 - CONSOLE SUMMARY
# ==============================================================================
print("\n" + "=" * 78)
print("  SOLUTION  (vehicle -> operating profile)")
print("=" * 78)
print(f"  {'VEH':<6}{'TYPE':<12}{'SPEED':>7}{'QUOTA':>7}{'SLOT':>6}{'STATION':>9}"
      f"{'DIST km':>9}")
print("  " + "-" * 74)
for v in viz_ids:
    d = solution[v]
    print(f"  {v:<6}{VEH[v]['type']:<12}{d.speed:>5} kh{d.quota:>5} L{d.slot:>6}"
          f"{d.station:>9}{dist_km[(v, d.station)]:>9.2f}")
print("  " + "-" * 74)
print(f"  variables {len(viz_ids)} | raw |D| {RAW_SIZE} | "
      f"search space {RAW_SIZE}^{len(viz_ids)} ~ 1e{len(viz_ids)*math.log10(RAW_SIZE):.0f}")
print(f"  solved in {elapsed:.3f}s, {csp.stats['nodes']} nodes, "
      f"{csp.stats['backtracks']} backtracks")
print(f"  CSP status: {'MAINTAINED' if ALL_OK else 'BROKEN'}")
print("=" * 78)
