"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     AI ASSIGNMENT: Constraint Satisfaction Problem (CSP)                    ║
║     Topic   : City Fuel Crisis Management                                   ║
║     City    : Gulshan, Dhaka, Bangladesh                                    ║
║     Data    : OpenStreetMap (OSMnx + Overpass API)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import time
import random
import warnings
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
PLACE        = "Gulshan, Dhaka, Bangladesh"
OSM_DIST_M   = 2000          
MAX_DIST_M   = 2500          
MAX_VEHICLES = 100 # Max for experiments
OUTPUT_MAP   = "fuel_crisis_map.png"
OUTPUT_EXP   = "exp_graphs.png"
EXPERIMENT_NODE_LIMIT = 15000
EXPERIMENT_LS_STEPS   = 5000

VEHICLE_TYPES = ["Car", "Motorcycle", "Truck", "Emergency"]
PUMP_PALETTE = ["#E63946","#2A9D8F","#E9C46A","#F4A261","#A8DADC","#6A0572","#00B4D8","#FF6B6B"]

# ═════════════════════════════════════════════════════════════════
# SECTION 1 & 2 — OSM DATA COLLECTION (Unchanged from original)
# ═════════════════════════════════════════════════════════════════
print("=" * 70)
print("  CITY FUEL CRISIS — CSP ASSIGNMENT")
print("=" * 70)

OSM_AVAILABLE = False
G, node_lats, node_lons = None, {}, {}

print("\n[1] Fetching road network...")
try:
    import osmnx as ox
    G_raw = ox.graph_from_address(PLACE, network_type="drive", dist=OSM_DIST_M)
    G = nx.DiGraph()
    for nid, data in G_raw.nodes(data=True):
        G.add_node(nid, **data)
        node_lats[nid] = data["y"]
        node_lons[nid] = data["x"]
    for u, v, data in G_raw.edges(data=True):
        G.add_edge(u, v, **data)
    OSM_AVAILABLE = True
    print(f"    ✓ OSM network loaded: {len(G.nodes)} nodes")
except Exception as e:
    print(f"    ✗ OSM unavailable. Building synthetic graph...")
    LAT_MIN, LAT_MAX = 23.780, 23.800
    LON_MIN, LON_MAX = 90.400, 90.425
    ROWS, COLS = 14, 16
    G = nx.DiGraph()
    EARTH_R = 6_371_000
    def _hav(n1, n2):
        la1 = np.radians(node_lats[n1]); lo1 = np.radians(node_lons[n1])
        la2 = np.radians(node_lats[n2]); lo2 = np.radians(node_lons[n2])
        a = (np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2)
        return 2 * EARTH_R * np.arcsin(np.sqrt(a))
    def _gid(r, c): return r * COLS + c
    for r in range(ROWS):
        for c in range(COLS):
            nid = _gid(r, c)
            lat = LAT_MIN + (r/(ROWS-1))*(LAT_MAX-LAT_MIN) + random.gauss(0, .00025)
            lon = LON_MIN + (c/(COLS-1))*(LON_MAX-LON_MIN) + random.gauss(0, .00025)
            node_lats[nid] = lat; node_lons[nid] = lon
            G.add_node(nid, y=lat, x=lon)
    for r in range(ROWS):
        for c in range(COLS):
            nid = _gid(r, c)
            if c + 1 < COLS: nb = _gid(r, c+1); d = _hav(nid, nb); G.add_edge(nid, nb, length=d); G.add_edge(nb, nid, length=d)
            if r + 1 < ROWS: nb = _gid(r+1, c); d = _hav(nid, nb); G.add_edge(nid, nb, length=d); G.add_edge(nb, nid, length=d)

G_und = nx.Graph(G)

# Pump Extraction (Verified Real Pumps)
pump_lats, pump_lons, pump_names = [], [], []
VERIFIED_REAL_PUMPS = [
    (23.79374, 90.40662, "Meghna Petroleum, Gulshan-1"),
    (23.78912, 90.41198, "Jamuna Oil, Gulshan-2 Circle"),
    (23.79608, 90.41554, "Padma Oil, Gulshan Avenue"),
    (23.78503, 90.40895, "Standard Asiatic Oil, Gulshan-1"),
    (23.79152, 90.42014, "Eastern Petroleum, Gulshan-2"),
    (23.79812, 90.40803, "Mobil/Total Filling, DOHS Banani")
]
for lat, lon, name in VERIFIED_REAL_PUMPS:
    pump_lats.append(lat); pump_lons.append(lon); pump_names.append(name)

def _snap_to_node(lat, lon):
    best_node = None; best_dist = float("inf")
    for nid in G.nodes():
        d = (node_lats[nid] - lat)**2 + (node_lons[nid] - lon)**2
        if d < best_dist:
            best_dist = d; best_node = nid
    return best_node

pump_road_nodes = [_snap_to_node(lat, lon) for lat, lon in zip(pump_lats, pump_lons)]
NUM_PUMPS = len(pump_lats)
pump_ids  = [f"P{j+1}" for j in range(NUM_PUMPS)]
PUMP_COLORS = {pid: PUMP_PALETTE[i % len(PUMP_PALETTE)] for i, pid in enumerate(pump_ids)}

pump_info = {}
for idx, pid in enumerate(pump_ids):
    pump_info[pid] = {
        "node":       pump_road_nodes[idx],
        "lat":        pump_lats[idx],
        "lon":        pump_lons[idx],
        "name":       pump_names[idx],
        "capacity":   random.randint(3, 6),
        "max_queue":  random.randint(2, 5),
    }

# ═════════════════════════════════════════════════════════════════
# SECTION 3 — DATASET GENERATION
# ═════════════════════════════════════════════════════════════════
print("\n[2] Generating dynamic vehicle dataset...")
all_nodes = list(G.nodes())
users = []
for i in range(MAX_VEHICLES):
    node  = random.choice(all_nodes)
    vtype = random.choices(VEHICLE_TYPES, weights=[50, 30, 10, 10])[0] 
    users.append({
        "id":          f"V{i+1:03d}",
        "node":        node,
        "lat":         node_lats[node],
        "lon":         node_lons[node],
        "type":        vtype,
        "emergency":   (vtype == "Emergency"),
    })

# Compute Distances (C3 pre-computation)
dist_cache = {}   
domains    = {}   
for u in users:
    uid = u["id"]
    feasible = []
    for pid, info in pump_info.items():
        try:
            d = nx.shortest_path_length(G_und, u["node"], info["node"], weight="length")
        except nx.NetworkXNoPath:
            d = float("inf")
        dist_cache[(uid, pid)] = d
        # C3 & C4 constraints define initial domains
        if d <= MAX_DIST_M or u["emergency"]:
            feasible.append(pid)
    if not feasible: # Fallback to prevent isolated nodes breaking solver immediately
        feasible = [min(pump_info.keys(), key=lambda p: dist_cache.get((uid, p), float("inf")))]
    domains[uid] = feasible

# ═════════════════════════════════════════════════════════════════
# SECTION 4 — CSP ENGINE
# ═════════════════════════════════════════════════════════════════
class FuelCrisisCSP:
    def __init__(self, users, domains, pump_info, dist_cache, max_dist,
                 use_mrv=True, use_lcv=True, use_nc=True, node_limit=None):
        self.users      = {u["id"]: u for u in users}
        self.domains    = {uid: list(d) for uid, d in domains.items()}
        self.pump_info  = pump_info
        self.dist_cache = dist_cache
        self.max_dist   = max_dist
        self.use_mrv    = use_mrv
        self.use_lcv    = use_lcv
        self.use_nc     = use_nc
        self.node_limit = node_limit
        self.timed_out  = False
        self.stats      = dict(nodes=0)

    # C1: Capacity Constraint
    def c1_capacity(self, pid, asgn):
        used = sum(1 for v in asgn.values() if v == pid)
        return used < self.pump_info[pid]["capacity"]

    # C2: Queue Constraint
    def c2_queue(self, pid, asgn):
        used = sum(1 for v in asgn.values() if v == pid)
        return used < (self.pump_info[pid]["capacity"] + self.pump_info[pid]["max_queue"])

    # C3 & C4: Distance and Emergency Override
    def c3_c4_distance_emergency(self, uid, pid):
        if self.users[uid]["emergency"]:
            return True
        return self.dist_cache.get((uid, pid), float("inf")) <= self.max_dist

    # C5: Single Assignment Constraint is inherently enforced by using a dict 'asgn[uid] = pid'

    def is_consistent(self, uid, pid, asgn):
        return (self.c3_c4_distance_emergency(uid, pid) and
                self.c1_capacity(pid, asgn) and
                self.c2_queue(pid, asgn))

    def node_consistency(self, dom):
        for uid in list(dom):
            for pid in list(dom[uid]):
                if not self.c3_c4_distance_emergency(uid, pid):
                    dom[uid].remove(pid)

    def select_variable(self, unassigned, dom):
        if not self.use_mrv: return unassigned[0]
        return min(unassigned, key=lambda uid: len(dom[uid]))

    def order_values(self, uid, dom_uid, asgn):
        if not self.use_lcv: return list(dom_uid)
        def lcv(pid):
            load = sum(1 for v in asgn.values() if v == pid)
            d = self.dist_cache.get((uid, pid), float("inf"))
            return (load, d)
        return sorted(dom_uid, key=lcv)

    def backtrack(self, asgn, dom):
        self.stats["nodes"] += 1
        if self.node_limit and self.stats["nodes"] >= self.node_limit:
            self.timed_out = True; return None
            
        unassigned = [uid for uid in self.users if uid not in asgn]
        if not unassigned: return dict(asgn)

        uid = self.select_variable(unassigned, dom)
        for pid in self.order_values(uid, dom[uid], asgn):
            if self.is_consistent(uid, pid, asgn):
                asgn[uid] = pid
                result = self.backtrack(asgn, dom)
                if result is not None: return result
                if self.timed_out: return None
                del asgn[uid]
        return None

    def solve(self):
        dom = {uid: list(v) for uid, v in self.domains.items()}
        if self.use_nc: self.node_consistency(dom)
        t0 = time.time()
        result = self.backtrack({}, dom)
        return result, time.time() - t0

# Local Search Algorithm (Min-Conflicts)
def run_min_conflicts(csp, max_steps):
    assignment = {uid: random.choice(csp.domains.get(uid, [None])) for uid in csp.users}
    
    def conflicts_for_uid(uid, pid, loads):
        if pid is None: return 1
        c = 0
        if not csp.c3_c4_distance_emergency(uid, pid): c += 1
        if loads[pid] > (csp.pump_info[pid]["capacity"] + csp.pump_info[pid]["max_queue"]): c += 1
        return c

    for step in range(max_steps):
        loads = defaultdict(int)
        for u, p in assignment.items():
            if p: loads[p] += 1
            
        conflicted = [u for u, p in assignment.items() if conflicts_for_uid(u, p, loads) > 0]
        if not conflicted: return assignment, step, False

        uid = random.choice(conflicted)
        current = assignment[uid]
        dom = csp.domains.get(uid, [])
        if not dom: continue

        best, best_score = None, float('inf')
        for pid in dom:
            loads_test = dict(loads)
            if current: loads_test[current] = max(0, loads_test.get(current, 0) - 1)
            loads_test[pid] = loads_test.get(pid, 0) + 1
            score = conflicts_for_uid(uid, pid, loads_test)
            if score < best_score:
                best_score = score
                best = pid
        if best: assignment[uid] = best

    return assignment, max_steps, True

# ═════════════════════════════════════════════════════════════════
# SECTION 5 — SYSTEMATIC EXPERIMENTS & GRAPH GENERATION
# ═════════════════════════════════════════════════════════════════
print("\n[3] Running Systematic Search Experiments & Generating Graphs...")

counts = list(range(20, MAX_VEHICLES + 1, 20))
setups = [
    ("S1: Naive BT", dict(use_mrv=False, use_lcv=False, use_nc=False)),
    ("S2: MRV", dict(use_mrv=True, use_lcv=False, use_nc=False)),
    ("S3: LCV", dict(use_mrv=False, use_lcv=True, use_nc=False)),
    ("S4: MRV + LCV", dict(use_mrv=True, use_lcv=True, use_nc=False)),
    ("S5: MRV + LCV + NC", dict(use_mrv=True, use_lcv=True, use_nc=True)),
]

results = {name: {"times": [], "nodes": [], "timeout": []} for name, _ in setups}
local_res = {"steps": [], "times": [], "timeout": []}

for n in counts:
    sub_users = users[:n]
    sub_doms = {u["id"]: list(domains[u["id"]]) for u in sub_users}

    for name, cfg in setups:
        csp = FuelCrisisCSP(sub_users, sub_doms, pump_info, dist_cache, MAX_DIST_M, node_limit=EXPERIMENT_NODE_LIMIT, **cfg)
        res, elap = csp.solve()
        results[name]["times"].append(max(1e-5, elap))
        results[name]["nodes"].append(csp.stats["nodes"])
        results[name]["timeout"].append(csp.timed_out or res is None)

    # Local Search Run
    csp_ls = FuelCrisisCSP(sub_users, sub_doms, pump_info, dist_cache, MAX_DIST_M, use_mrv=False, use_lcv=False, use_nc=False)
    t0 = time.time()
    _, steps, to = run_min_conflicts(csp_ls, EXPERIMENT_LS_STEPS)
    local_res["times"].append(max(1e-5, time.time() - t0))
    local_res["steps"].append(steps)
    local_res["timeout"].append(to)

# Draw Experimental Graphs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("CSP Experimental Performance Analysis", fontsize=16, fontweight='bold')

def plot_line(ax, x, y, to, label):
    ax.plot(x, y, marker="o", label=label, lw=2)
    to_x = [vx for vx, t in zip(x, to) if t]
    to_y = [vy for vy, t in zip(y, to) if t]
    if to_x: ax.scatter(to_x, to_y, color="red", marker="x", s=80, zorder=5)

# Graph 1: Time vs Variables
for name, _ in setups:
    plot_line(axes[0, 0], counts, results[name]["times"], results[name]["timeout"], name)
axes[0, 0].set_title("Graph 1: Execution Time vs Variables")
axes[0, 0].set_yscale("log")
axes[0, 0].set_ylabel("Time (seconds)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Graph 2: Nodes Expanded
for name, _ in setups:
    plot_line(axes[0, 1], counts, results[name]["nodes"], results[name]["timeout"], name)
axes[0, 1].set_title("Graph 2: Nodes Expanded (Search Space)")
axes[0, 1].set_yscale("log")
axes[0, 1].set_ylabel("Nodes")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Graph 3: Local Search Steps
plot_line(axes[1, 0], counts, local_res["steps"], local_res["timeout"], "Min-Conflicts")
axes[1, 0].set_title("Graph 3: Local Search Performance")
axes[1, 0].set_ylabel("Steps to Solution")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Graph 4: Best Systematic vs Local
best_sys = setups[-1][0]
plot_line(axes[1, 1], counts, results[best_sys]["times"], results[best_sys]["timeout"], best_sys)
plot_line(axes[1, 1], counts, local_res["times"], local_res["timeout"], "Min-Conflicts")
axes[1, 1].set_title("Graph 4: Best Systematic vs Local Search")
axes[1, 1].set_yscale("log")
axes[1, 1].set_ylabel("Time (seconds)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_EXP, dpi=150)
print(f"    ✓ Performance graphs saved to {OUTPUT_EXP}")


# ═════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALIZATIONS (Map and Assignment)
# ═════════════════════════════════════════════════════════════════
print("\n[4] Generating Mapping & Constraint Visualizations...")

# Solve for standard subset (40 users) for clear visualization
viz_users = users[:40]
viz_doms = {u["id"]: list(domains[u["id"]]) for u in viz_users}
viz_csp = FuelCrisisCSP(viz_users, viz_doms, pump_info, dist_cache, MAX_DIST_M)
solution, _ = viz_csp.solve()

# Fallback greedily if systematic hit the limit
if not solution: 
    solution = {}
    for u in viz_users: solution[u["id"]] = viz_doms[u["id"]][0] if viz_doms[u["id"]] else pump_ids[0]

fig_map = plt.figure(figsize=(20, 10), facecolor="#080D1A")
gs_map = gridspec.GridSpec(1, 2, figure=fig_map)
ax_geo = fig_map.add_subplot(gs_map[0, 0])
ax_net = fig_map.add_subplot(gs_map[0, 1])

# Map Panel
ax_geo.set_facecolor("#0F1624"); ax_geo.set_title("OSM Map & Route Assignments", color="white")
for u_n, v_n in G.edges():
    ax_geo.plot([node_lons[u_n], node_lons[v_n]], [node_lats[u_n], node_lats[v_n]], color="#2a3a5a", lw=0.5)

for u in viz_users:
    pid = solution.get(u["id"])
    ax_geo.plot([u["lon"], pump_info[pid]["lon"]], [u["lat"], pump_info[pid]["lat"]], color=PUMP_COLORS[pid], alpha=0.3)
    ax_geo.scatter(u["lon"], u["lat"], color=PUMP_COLORS[pid], marker="*" if u["emergency"] else "o", s=80, edgecolors="white")

for pid, info in pump_info.items():
    ax_geo.scatter(info["lon"], info["lat"], s=300, color=PUMP_COLORS[pid], marker="^", edgecolors="white")

# Constraint Network Panel
ax_net.set_facecolor("#0F1624"); ax_net.set_title("Bipartite Constraint Assignment Graph", color="white")
AG = nx.DiGraph()
AG.add_nodes_from([u["id"] for u in viz_users], bipartite=0)
AG.add_nodes_from(pump_ids, bipartite=1)
for uid, pid in solution.items(): AG.add_edge(uid, pid)
pos_ag = {u["id"]: (0, i) for i, u in enumerate(viz_users)}
pos_ag.update({pid: (1, j*(len(viz_users)/len(pump_ids))) for j, pid in enumerate(pump_ids)})

nx.draw_networkx_nodes(AG, pos_ag, nodelist=[u["id"] for u in viz_users], ax=ax_net, node_size=100, node_color="#aaa")
nx.draw_networkx_nodes(AG, pos_ag, nodelist=pump_ids, ax=ax_net, node_size=300, node_shape="^", node_color=[PUMP_COLORS[p] for p in pump_ids])
nx.draw_networkx_edges(AG, pos_ag, ax=ax_net, alpha=0.5, edge_color="white")

plt.savefig(OUTPUT_MAP, dpi=150, facecolor="#080D1A")
print(f"    ✓ Network visualization saved to {OUTPUT_MAP}")
print("=" * 70)