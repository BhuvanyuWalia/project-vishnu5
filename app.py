"""
Project Vishnu Phase 5 — FastAPI Backend
==========================================
Loads 4 independent zone agents and routes navigation
requests through the ZoneRouter.
"""

import os, math, json, collections, random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────
ZONES = {
    "NW": (  0, 120,   0, 120),
    "NE": (  0, 120,  80, 200),
    "SW": ( 80, 200,   0, 120),
    "SE": ( 80, 200,  80, 200),
}
ACTION_DELTAS   = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
ACTION_OPPOSITE = {0:4,1:5,2:6,3:7,4:0,5:1,6:2,7:3}
device = torch.device("cpu")

# ── Grid ──────────────────────────────────────────────
with open("indian_ocean_200x200.json") as f:
    GRID = np.array(json.load(f), dtype=np.int32)

PORTS = {
    "Mumbai":        ( 75,  85),
    "Visakhapatnam": ( 77, 130),
    "Chennai":       (104,  116),
    "Kochi":         (117,  100),
    "Colombo":       (130, 117),
    "Karachi":       ( 39,  49),
    "Goa":           ( 98,  92),
    "Aden":          ( 80,  4),
    "Singapore":     (121, 198),
}
for name, (r, c) in PORTS.items():
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr, cc = r+dr, c+dc
            if 0 <= rr < 200 and 0 <= cc < 200:
                GRID[rr, cc] = 0

PAD = 5
PADDED = np.ones((210, 210), dtype=np.int32)
PADDED[PAD:PAD+200, PAD:PAD+200] = GRID
WATER_CELLS = list(zip(*np.where(GRID == 0)))
print(f"✅  Grid loaded — water cells: {len(WATER_CELLS):,}")

# ── Network ───────────────────────────────────────────
class DuelingNavDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
        )
        self.coord  = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(192, 256), nn.ReLU())
        self.value_stream     = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(), nn.Linear(128,1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(), nn.Linear(128,8))

    def forward(self, window, coord):
        v  = self.vision(window)
        c  = self.coord(coord)
        sh = self.shared(torch.cat([v, c], dim=1))
        V  = self.value_stream(sh)
        A  = self.advantage_stream(sh)
        return V + (A - A.mean(dim=1, keepdim=True))

# ── Load all 4 zone agents ────────────────────────────
ZONE_NETS    = {}
MODELS_LOADED = {}

for zone_name in ZONES:
    path = Path(f"Vishnu5_{zone_name}_deploy.pth")
    net  = DuelingNavDQN().to(device)
    if path.exists():
        try:
            ckpt = torch.load(path, map_location=device,
                              weights_only=False)
            net.load_state_dict(ckpt["online_state_dict"])
            MODELS_LOADED[zone_name] = True
            ep = ckpt.get("episode", "?")
            print(f"✅  Zone {zone_name} loaded — episode {ep}")
        except Exception as e:
            MODELS_LOADED[zone_name] = False
            print(f"⚠️  Zone {zone_name} failed: {e}")
    else:
        MODELS_LOADED[zone_name] = False
        print(f"⚠️  Zone {zone_name} checkpoint not found: {path}")
    net.eval()
    ZONE_NETS[zone_name] = net

all_loaded = all(MODELS_LOADED.values())
print(f"{'✅' if all_loaded else '⚠️'}  All zones loaded: {all_loaded}")

# ── Zone Router ───────────────────────────────────────
ZONE_GRAPH = {
    "NW": ["NE", "SW"],
    "NE": ["NW", "SE"],
    "SW": ["NW", "SE"],
    "SE": ["NE", "SW"],
}

def get_primary_zone(r, c):
    zones = [n for n,(r0,r1,c0,c1) in ZONES.items()
             if r0 <= r < r1 and c0 <= c < c1]
    if not zones:
        return "NW"
    if len(zones) == 1:
        return zones[0]
    def centre_dist(zn):
        r0,r1,c0,c1 = ZONES[zn]
        return math.sqrt(((r0+r1)/2-r)**2+((c0+c1)/2-c)**2)
    return min(zones, key=centre_dist)

def get_overlap_waypoint(zone_a, zone_b):
    a = ZONES[zone_a]; b = ZONES[zone_b]
    r0 = max(a[0],b[0]); r1 = min(a[1],b[1])
    c0 = max(a[2],b[2]); c1 = min(a[3],b[3])
    candidates = [(r,c) for r in range(r0,r1)
                  for c in range(c0,c1) if GRID[r,c]==0]
    if not candidates:
        return None
    mid_r, mid_c = (r0+r1)//2, (c0+c1)//2
    return min(candidates, key=lambda rc: abs(rc[0]-mid_r)+abs(rc[1]-mid_c))

def plan_route(sr, sc, gr, gc):
    start_zone = get_primary_zone(sr, sc)
    goal_zone  = get_primary_zone(gr, gc)
    if start_zone == goal_zone:
        return [(start_zone, (sr,sc), (gr,gc))]
    from collections import deque
    q = deque([(start_zone, [start_zone])])
    visited = {start_zone}
    path = None
    while q:
        cur, route = q.popleft()
        if cur == goal_zone:
            path = route; break
        for nb in ZONE_GRAPH[cur]:
            if nb not in visited:
                visited.add(nb)
                q.append((nb, route+[nb]))
    if not path:
        return [(start_zone, (sr,sc), (gr,gc))]
    legs = []
    leg_start = (sr, sc)
    for i in range(len(path)-1):
        wp = get_overlap_waypoint(path[i], path[i+1])
        if wp:
            legs.append((path[i], leg_start, wp))
            leg_start = wp
    legs.append((path[-1], leg_start, (gr,gc)))
    return legs

# ── Inference ─────────────────────────────────────────
def run_leg(net, sr, sc, gr, gc, max_steps=3000):
    r, c = sr, sc
    route = [(r, c)]
    bumps = 0
    last_action = None

    with torch.no_grad():
        for _ in range(max_steps):
            window = PADDED[r:r+11, c:c+11].astype(np.float32)
            dx     = (gr-r)/199.0
            dy     = (gc-c)/199.0
            coord  = np.array([r/199.,c/199.,dx,dy], dtype=np.float32)
            w_t    = torch.tensor(window).unsqueeze(0).unsqueeze(0)
            c_t    = torch.tensor(coord).unsqueeze(0)
            q      = net(w_t, c_t).squeeze(0).numpy()
            action = int(np.argmax(q))
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r+dr, c+dc
            oob    = not (0 <= nr < 200 and 0 <= nc < 200)
            land   = (not oob) and (GRID[nr,nc]==1)
            if oob or land:
                bumps += 1
            else:
                r, c = nr, nc
                route.append((r,c))
            last_action = action
            if r == gr and c == gc:
                break
    reached = (r == gr and c == gc)
    return route, reached, bumps

def routed_navigate(sr, sc, gr, gc):
    legs = plan_route(sr, sc, gr, gc)
    full_route = []
    total_bumps = 0
    all_reached = True
    leg_info = []
    for zone_name, leg_start, leg_goal in legs:
        net = ZONE_NETS[zone_name]
        ls_r,ls_c = leg_start
        lg_r,lg_c = leg_goal
        route, reached, bumps = run_leg(net, ls_r, ls_c, lg_r, lg_c)
        if not reached:
            all_reached = False
        total_bumps += bumps
        full_route.extend(route)
        leg_info.append({
            "zone": zone_name,
            "start": list(leg_start),
            "goal": list(leg_goal),
            "steps": len(route)-1,
            "reached": reached,
            "bumps": bumps,
        })
    return full_route, all_reached, total_bumps, legs, leg_info

# ── FastAPI ───────────────────────────────────────────
app = FastAPI(
    title="Project Vishnu Phase 5",
    description="4-Zone Dueling DDQN Naval Navigator",
    version="5.0.0"
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

class NavigateRequest(BaseModel):
    start_row: int
    start_col: int
    goal_row:  int
    goal_col:  int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "all_loaded": all(MODELS_LOADED.values()),
        "grid_shape": [200, 200],
        "water_cells": len(WATER_CELLS),
        "zones": list(ZONES.keys()),
        "phase": "Project Vishnu — Phase 5",
    }

@app.get("/grid")
def get_grid():
    return {
        "grid":  GRID.tolist(),
        "ports": {k: list(v) for k, v in PORTS.items()},
        "zones": {k: list(v) for k, v in ZONES.items()},
        "rows":  200, "cols": 200,
    }

@app.post("/navigate")
def navigate(req: NavigateRequest):
    for val, name in [
        (req.start_row,"start_row"),(req.start_col,"start_col"),
        (req.goal_row,"goal_row"),(req.goal_col,"goal_col")
    ]:
        if not (0 <= val < 200):
            raise HTTPException(400, f"{name}={val} out of range")
    if GRID[req.start_row, req.start_col] == 1:
        raise HTTPException(400, "Start cell is land.")
    if GRID[req.goal_row, req.goal_col] == 1:
        raise HTTPException(400, "Goal cell is land.")
    if req.start_row==req.goal_row and req.start_col==req.goal_col:
        raise HTTPException(400, "Start and goal must differ.")

    route, reached, bumps, legs, leg_info = routed_navigate(
        req.start_row, req.start_col, req.goal_row, req.goal_col
    )
    return {
        "route":        [[r,c] for r,c in route],
        "steps":        len(route)-1,
        "reached_goal": reached,
        "bumps":        bumps,
        "legs":         leg_info,
        "models_loaded": MODELS_LOADED,
    }

@app.get("/")
def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
