#!/usr/bin/env python3
"""
Scenario generator and multi-agent eVTOL routing simulator for a Frankfurt hackathon MVP.

Input:
    outputs_air/frankfurt_air_corridors.graphml

Output:
    outputs_sim/agents.csv
    outputs_sim/vertiports.geojson
    outputs_sim/routes.geojson
    outputs_sim/proposed_routes.geojson
    outputs_sim/baseline_routes.geojson
    outputs_sim/simulation_log.csv
    outputs_sim/simulation_positions.geojson
    outputs_sim/metrics_summary.csv
    outputs_sim/alerts.csv

Core features:
    0. Auto-create initial vertiports from air graph nodes
    1. Weighted shortest path routing
    2. Multi-agent routing for multiple eVTOLs
    3. Reservation table to avoid node and edge conflicts
    4. Emergency priority handling
    5. Dynamic rerouting when a weather zone appears
    6. Baseline vs proposed metrics

Modeling assumptions:
    - This is NOT a full aviation simulator.
    - Nodes are simplified eVTOL waypoints.
    - Edges are simplified directed air corridors.
    - Residential noise and weather are modeled as explicit edge penalties.
    - eVTOLs move in discrete time steps.
    - Conflict avoidance uses reservation tables over nodes and edges.
    - Emergency aircraft get higher priority and are replanned first.
"""

from __future__ import annotations

import ast
import heapq
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CITY_NAME = os.getenv("SKYMESH_CITY_NAME", "Frankfurt")
CITY_SLUG = os.getenv("SKYMESH_CITY_SLUG", "frankfurt")

INPUT_GRAPHML = Path(f"outputs_air/{CITY_SLUG}_air_corridors.graphml")
OUT_DIR = Path("outputs_sim")

OUT_AGENTS = OUT_DIR / "agents.csv"
OUT_VERTIPORTS = OUT_DIR / "vertiports.geojson"
OUT_ROUTES = OUT_DIR / "routes.geojson"
OUT_PROPOSED_ROUTES = OUT_DIR / "proposed_routes.geojson"
OUT_BASELINE_ROUTES = OUT_DIR / "baseline_routes.geojson"
OUT_SIM_LOG = OUT_DIR / "simulation_log.csv"
OUT_SIM_POSITIONS = OUT_DIR / "simulation_positions.geojson"
OUT_METRICS = OUT_DIR / "metrics_summary.csv"
OUT_ALERTS = OUT_DIR / "alerts.csv"
OUT_WEATHER_EVENTS = OUT_DIR / "weather_events.geojson"
OUT_NO_FLY_ZONES = OUT_DIR / "no_flight_zones.geojson"
OUT_FEATURED_MISSIONS = OUT_DIR / "featured_missions.geojson"
OUT_VERTIPORT_DISTRIBUTION = OUT_DIR / "vertiport_distribution.csv"


# ---------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42

N_VERTIPORTS = 14
N_AGENTS = 40
FEATURED_MISSION_IDS = ("A015", "A004", "A038")

MAX_START_TIME = 55
PLANNING_HORIZON = 190

WEATHER_EVENT_TIME = 14
MACHINE_DISORDER_PREFERRED_TIME = 52
MEDICAL_EMERGENCY_PREFERRED_TIME = 28
BIRD_STRIKE_PREFERRED_TIME = 44

TIME_STEP_SECONDS = 60

# Same planning-speed assumption used in the graph builder.
CRUISE_DISTANCE_PER_STEP_M = 1_250

# Cost model. These values are intentionally simple and explainable.
BASE_ENERGY_PER_METER = 0.001
WEATHER_ENERGY_MULTIPLIER = 1.35

NOISE_PENALTY_RESIDENTIAL = 1_200.0
WEATHER_RISK_PENALTY = 2_500.0
WEATHER_AVOIDANCE_PENALTY = 8_000.0

ALPHA_DISTANCE = 1.0
BETA_BATTERY = 500.0
GAMMA_NOISE = 1.0
DELTA_WEATHER = 1.0

WAIT_COST = 80.0
WAIT_BATTERY_COST_PER_STEP = 0.05

DEFAULT_NODE_CAPACITY = 4
VERTIPORT_NODE_CAPACITY = 18
EMERGENCY_NODE_CAPACITY = 8
DEFAULT_EDGE_CAPACITY = 3

# Runtime guards for presentation builds.
# The original time-expanded Dijkstra is accurate but can become very slow
# with dozens of agents, long planning horizons, hard weather blocks, and
# repeated all-fleet rerouting. This fast build uses greedy reservation-aware
# route timing and caps event-triggered replans.
MAX_WAIT_STEPS_PER_CONFLICT = 10
MAX_WEATHER_REROUTES_PER_EVENT = 12
MAX_EMERGENCY_SUPPORT_REROUTES = 8
SUPPRESS_FALLBACK_WARNINGS = True

# Lightweight AI coordination policy.
# The "AI" layer is implemented as a candidate-route utility policy: each
# aircraft proposes several feasible paths, then the coordinator scores them
# under weather, battery, noise, reservation conflict, and priority delay.
AI_COORDINATION_POLICY_NAME = "candidate_utility_reservation_policy"
AI_CANDIDATE_PATHS = 3
AI_WAIT_STEP_PENALTY = 450.0
AI_DELAY_WEIGHT_NORMAL = 45.0
AI_DELAY_WEIGHT_FEATURED = 90.0
AI_DELAY_WEIGHT_EMERGENCY = 900.0

# Presentation-oriented constraints.
# Weather cells can become dynamic no-fly zones by profile/severity. Static
# no-fly boxes are intentionally disabled for the Frankfurt presentation build.
BLOCKED_EDGE_COST = 1e9
NO_FLY_ZONE_PENALTY = 1e8
ADD_AIRPORT_VERTIPORT_IF_COVERED = True
AIRPORT_ATTACH_MAX_DISTANCE_M = 6_000
AIRPORT_NAME = os.getenv("SKYMESH_AIRPORT_NAME", "Frankfurt Airport Gateway")
AIRPORT_LON = float(os.getenv("SKYMESH_AIRPORT_LON", "8.5622"))
AIRPORT_LAT = float(os.getenv("SKYMESH_AIRPORT_LAT", "50.0379"))

# WeatherEvent -> Consequence -> Simulator parameters.
# Multipliers are max-effect values at severity=1.0. `weather_parameters`
# scales them by each event's severity so future real weather feeds can map
# directly into the same cost model.
WEATHER_PROFILES: dict[str, dict[str, Any]] = {
    "wind": {
        "label": "Crosswind",
        "consequence": "unstable approach, higher energy burn",
        "battery_multiplier": 1.65,
        "noise_multiplier": 1.10,
        "risk_multiplier": 1.35,
        "speed_multiplier": 0.92,
        "dynamic_no_fly_threshold": 0.95,
        "color": "orange",
    },
    "thunderstorm": {
        "label": "Lightning Cell",
        "consequence": "lightning strike risk, avionics disruption",
        "battery_multiplier": 1.25,
        "noise_multiplier": 1.20,
        "risk_multiplier": 2.80,
        "speed_multiplier": 0.70,
        "dynamic_no_fly_threshold": 0.70,
        "color": "purple",
    },
    "icing": {
        "label": "Icing Layer",
        "consequence": "rotor efficiency loss, rotor failure risk",
        "battery_multiplier": 1.55,
        "noise_multiplier": 1.35,
        "risk_multiplier": 2.20,
        "speed_multiplier": 0.78,
        "dynamic_no_fly_threshold": 0.78,
        "color": "cyan",
    },
    "heat": {
        "label": "Thermal Stress",
        "consequence": "battery overheat, reduced reserve margin",
        "battery_multiplier": 1.45,
        "noise_multiplier": 1.05,
        "risk_multiplier": 1.45,
        "speed_multiplier": 0.95,
        "dynamic_no_fly_threshold": 0.92,
        "color": "red",
    },
    "heavy_rain": {
        "label": "Heavy Rain",
        "consequence": "sensor degradation, low visibility",
        "battery_multiplier": 1.30,
        "noise_multiplier": 1.15,
        "risk_multiplier": 2.00,
        "speed_multiplier": 0.82,
        "dynamic_no_fly_threshold": 0.82,
        "color": "dodgerblue",
    },
}

INCIDENT_PROFILES: dict[str, dict[str, Any]] = {
    "machine_disorder": {
        "priority": 0,
        "preferred_emergency_site": None,
        "label": "Machine Disorder",
        "reroute_reason": "machine_disorder_priority_reroute",
    },
    "medical_emergency": {
        "priority": 1,
        "preferred_emergency_site": "medical",
        "label": "Medical Emergency",
        "reroute_reason": "medical_emergency_priority_reroute",
    },
    "bird_strike": {
        "priority": 1,
        "preferred_emergency_site": None,
        "label": "Bird Strike",
        "reroute_reason": "bird_strike_priority_reroute",
    },
    "high_priority_mission": {
        "priority": 1,
        "preferred_emergency_site": None,
        "label": "High-Priority Mission",
        "reroute_reason": "high_priority_mission_reroute",
    },
}

EXPORT_CRS = "EPSG:4326"
DEFAULT_PROJECTED_CRS = "EPSG:32632"  # UTM zone 32N, reasonable fallback for Frankfurt.


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class Agent:
    agent_id: str
    origin_vertiport: str
    destination_vertiport: str
    origin_node: str
    final_destination_node: str
    current_destination_node: str
    battery_initial: float
    start_time: int
    priority: int = 3
    status: str = "normal"
    emergency_type: str | None = None
    emergency_start_time: int | None = None
    emergency_destination_node: str | None = None
    route_version: int = 0
    reroute_count: int = 0
    mission_id: str | None = None
    mission_label: str | None = None
    mission_type: str = "point_to_point"
    pickup_node: str | None = None
    dropoff_node: str | None = None
    mission_waypoints: list[str] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WeatherEvent:
    event_id: str
    weather_type: str
    start_time: int
    end_time: int
    severity: float
    hard_block: bool
    description: str
    geometry: BaseGeometry


# ---------------------------------------------------------------------
# Robust parsing helpers
# ---------------------------------------------------------------------

def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        if isinstance(value, str):
            text = value.strip()
            if text == "" or text.lower() in {"nan", "none", "<na>"}:
                return default
            return float(text)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(to_float(value, float(default))))
    except Exception:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()
        return text in {"true", "1", "yes", "y", "t"}

    return False


def parse_listish(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return parse_listish(value.tolist())

    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(parse_listish(item))
        return result

    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"nan", "none", "<na>"}:
            return []

        if (
            (text.startswith("[") and text.endswith("]"))
            or (text.startswith("(") and text.endswith(")"))
        ):
            try:
                parsed = ast.literal_eval(text)
                return parse_listish(parsed)
            except Exception:
                pass

        return [part.strip().lower() for part in text.split(";") if part.strip()]

    return [str(value).strip().lower()]


def json_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)

    if isinstance(value, BaseGeometry):
        return value.wkt

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def get_graph_crs(G: nx.MultiDiGraph) -> str:
    crs = G.graph.get("crs")
    if crs:
        return str(crs)
    return DEFAULT_PROJECTED_CRS


def node_point(G: nx.MultiDiGraph, node: str) -> Point:
    data = G.nodes[node]
    x = to_float(data.get("x"))
    y = to_float(data.get("y"))
    return Point(x, y)


def edge_geometry(G: nx.MultiDiGraph, u: str, v: str, data: dict[str, Any]) -> LineString:
    geom = data.get("geometry")

    if isinstance(geom, LineString):
        return geom

    if isinstance(geom, BaseGeometry):
        if geom.geom_type == "LineString":
            return geom
        return LineString([node_point(G, u), node_point(G, v)])

    if isinstance(geom, str):
        try:
            parsed = wkt.loads(geom)
            if isinstance(parsed, LineString):
                return parsed
        except Exception:
            pass

    return LineString([node_point(G, u), node_point(G, v)])


def interpolate_segment_position(G: nx.MultiDiGraph, segment: dict[str, Any], t: int) -> Point:
    if segment["kind"] == "wait":
        return node_point(G, segment["u"])

    line = segment["geometry"]
    if not isinstance(line, LineString):
        line = edge_geometry(G, segment["u"], segment["v"], segment)

    duration = max(1, segment["t_end"] - segment["t_start"])
    fraction = (t - segment["t_start"]) / duration
    fraction = min(max(fraction, 0.0), 1.0)
    return line.interpolate(fraction, normalized=True)


# ---------------------------------------------------------------------
# Graph preparation
# ---------------------------------------------------------------------

def load_and_prepare_graph(path: Path) -> nx.MultiDiGraph:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run transform_frankfurt_to_air_corridors.py first."
        )

    print(f"[INFO] Loading air corridor graph: {path}")

    # Our generated air graph uses string node IDs like "air_0000".
    # OSMnx defaults to integer OSM IDs, so we override osmid to str.
    try:
        G = ox.io.load_graphml(
            filepath=path,
            node_dtypes={"osmid": str},
        )
    except ValueError as exc:
        print("[WARN] OSMnx load_graphml failed. Falling back to NetworkX read_graphml.")
        print(f"[WARN] Original error: {exc}")

        G = nx.read_graphml(
            path,
            force_multigraph=True,
        )

        if not isinstance(G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(G)

    graph_crs = get_graph_crs(G)
    G.graph["crs"] = graph_crs

    for node, data in G.nodes(data=True):
        node_id = str(node)

        data["x"] = to_float(data.get("x"))
        data["y"] = to_float(data.get("y"))

        data["lon"] = to_float(data.get("lon"), default=np.nan)
        data["lat"] = to_float(data.get("lat"), default=np.nan)

        data["is_emergency_site"] = parse_bool(data.get("is_emergency_site"))
        data["is_vertiport"] = False
        data["vertiport_id"] = None

        if data["is_emergency_site"]:
            data["node_capacity"] = EMERGENCY_NODE_CAPACITY
        else:
            data["node_capacity"] = DEFAULT_NODE_CAPACITY

        data["reservation_node_id"] = data.get("reservation_node_id", node_id)

    for u, v, k, data in G.edges(keys=True, data=True):
        distance = to_float(
            data.get("distance"),
            default=to_float(data.get("length"), default=0.0),
        )

        if distance <= 0:
            distance = node_point(G, u).distance(node_point(G, v))

        data["distance"] = distance
        data["over_residential"] = parse_bool(data.get("over_residential"))
        data["inside_weather_zone"] = parse_bool(data.get("inside_weather_zone"))
        data["altitude_layer"] = to_int(data.get("altitude_layer"), default=1)
        data["capacity"] = max(
            DEFAULT_EDGE_CAPACITY,
            to_int(data.get("capacity"), default=DEFAULT_EDGE_CAPACITY),
        )

        duration = to_int(data.get("travel_time_steps"), default=0)
        if duration <= 0:
            duration = max(
                1,
                int(math.ceil(distance / CRUISE_DISTANCE_PER_STEP_M)),
            )

        data["travel_time_steps"] = duration

        data["battery_cost"] = to_float(
            data.get("battery_cost"),
            default=distance * BASE_ENERGY_PER_METER,
        )

        data["noise_penalty"] = to_float(
            data.get("noise_penalty"),
            default=NOISE_PENALTY_RESIDENTIAL if data["over_residential"] else 0.0,
        )

        data["weather_risk"] = to_float(
            data.get("weather_risk"),
            default=WEATHER_RISK_PENALTY if data["inside_weather_zone"] else 0.0,
        )

    print(f"[INFO] Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} directed edges")
    return G

# ---------------------------------------------------------------------
# Edge cost model
# ---------------------------------------------------------------------

def edge_cost_components(
    edge_data: dict[str, Any],
    active_weather: bool,
    mode: str,
) -> dict[str, float | bool | int | str]:
    """
    Convert edge attributes into route-planning costs.

    Simplified constraint logic:
    - baseline ignores noise, weather, no-flight restrictions and uses distance only.
    - proposed blocks no-flight-zone edges.
    - proposed translates active weather consequences into battery/noise/risk/speed parameters.
    - proposed blocks active dynamic no-fly weather cells.
    """
    distance = to_float(edge_data.get("distance"), default=0.0)
    over_residential = parse_bool(edge_data.get("over_residential"))
    inside_weather = parse_bool(edge_data.get("inside_weather_zone"))
    inside_no_flight = parse_bool(edge_data.get("inside_no_flight_zone"))
    blocked_by_weather = parse_bool(edge_data.get("blocked_by_active_weather"))
    active_weather_type = str(edge_data.get("active_weather_type", "") or "")
    inside_dynamic_no_fly = parse_bool(edge_data.get("inside_dynamic_no_fly_zone"))
    active_battery_multiplier = to_float(edge_data.get("active_battery_multiplier"), 1.0)
    active_noise_multiplier = to_float(edge_data.get("active_noise_multiplier"), 1.0)
    active_risk_multiplier = to_float(edge_data.get("active_risk_multiplier"), 1.0)
    active_speed_multiplier = to_float(edge_data.get("active_speed_multiplier"), 1.0)
    active_weather_consequences = str(edge_data.get("active_weather_consequences", "") or "")
    active_weather_param_summary = str(edge_data.get("active_weather_param_summary", "") or "")
    base_duration = max(1, to_int(edge_data.get("travel_time_steps"), default=1))

    base_battery = distance * BASE_ENERGY_PER_METER

    if mode == "baseline":
        battery_cost = base_battery
        noise_penalty = 0.0
        weather_risk = 0.0
        no_fly_penalty = 0.0
        blocked = False
        travel_time_steps = base_duration
        total_cost = distance
    else:
        blocked = False
        no_fly_penalty = 0.0

        if inside_no_flight:
            blocked = True
            no_fly_penalty = NO_FLY_ZONE_PENALTY

        battery_multiplier = 1.0
        noise_multiplier = 1.0
        risk_multiplier = 1.0
        speed_multiplier = 1.0
        weather_risk = 0.0

        if active_weather and inside_weather:
            battery_multiplier = max(1.0, active_battery_multiplier)
            noise_multiplier = max(1.0, active_noise_multiplier)
            risk_multiplier = max(1.0, active_risk_multiplier)
            speed_multiplier = max(0.25, min(1.0, active_speed_multiplier))
            weather_risk = (WEATHER_RISK_PENALTY + WEATHER_AVOIDANCE_PENALTY) * risk_multiplier

            if blocked_by_weather or inside_dynamic_no_fly:
                blocked = True
                no_fly_penalty = max(no_fly_penalty, NO_FLY_ZONE_PENALTY)

        battery_cost = base_battery * battery_multiplier
        noise_penalty = (NOISE_PENALTY_RESIDENTIAL if over_residential else 0.0) * noise_multiplier
        travel_time_steps = max(1, int(math.ceil(base_duration / max(speed_multiplier, 0.25))))

        total_cost = (
            ALPHA_DISTANCE * distance
            + BETA_BATTERY * battery_cost
            + GAMMA_NOISE * noise_penalty
            + DELTA_WEATHER * weather_risk
            + no_fly_penalty
        )

        if blocked:
            total_cost = BLOCKED_EDGE_COST

    return {
        "distance": float(distance),
        "battery_cost": float(battery_cost),
        "noise_penalty": float(noise_penalty),
        "weather_risk": float(weather_risk),
        "no_fly_penalty": float(no_fly_penalty),
        "total_cost": float(total_cost),
        "blocked": bool(blocked),
        "over_residential": bool(over_residential),
        "inside_weather_zone": bool(inside_weather),
        "inside_no_flight_zone": bool(inside_no_flight),
        "inside_dynamic_no_fly_zone": bool(inside_dynamic_no_fly),
        "active_weather_type": active_weather_type,
        "active_weather_ids": str(edge_data.get("active_weather_ids", "")),
        "active_weather_consequences": active_weather_consequences,
        "active_weather_param_summary": active_weather_param_summary,
        "battery_multiplier": float(active_battery_multiplier if active_weather and inside_weather else 1.0),
        "noise_multiplier": float(active_noise_multiplier if active_weather and inside_weather else 1.0),
        "risk_multiplier": float(active_risk_multiplier if active_weather and inside_weather else 1.0),
        "speed_multiplier": float(active_speed_multiplier if active_weather and inside_weather else 1.0),
        "altitude_layer": int(to_int(edge_data.get("altitude_layer"), default=1)),
        "capacity": int(max(1, to_int(edge_data.get("capacity"), default=DEFAULT_EDGE_CAPACITY))),
        "travel_time_steps": int(travel_time_steps),
    }

def set_temporary_routing_cost(
    G: nx.MultiDiGraph,
    active_weather: bool,
    mode: str,
    attr_name: str = "tmp_routing_cost",
) -> None:
    for _, _, _, data in G.edges(keys=True, data=True):
        data[attr_name] = edge_cost_components(data, active_weather, mode)["total_cost"]


def routing_graph_for_mode(
    G: nx.MultiDiGraph,
    active_weather: bool,
    mode: str,
    attr_name: str = "tmp_routing_cost",
) -> nx.MultiDiGraph:
    set_temporary_routing_cost(G, active_weather=active_weather, mode=mode, attr_name=attr_name)
    if mode != "proposed":
        return G

    H = G.copy()
    blocked_edges = []
    for u, v, key, data in H.edges(keys=True, data=True):
        comp = edge_cost_components(data, active_weather=active_weather, mode=mode)
        if bool(comp.get("blocked")):
            blocked_edges.append((u, v, key))
        else:
            data[attr_name] = float(comp["total_cost"])

    if blocked_edges:
        H.remove_edges_from(blocked_edges)

    return H


def best_edge_between(
    G: nx.MultiDiGraph,
    u: str,
    v: str,
    active_weather: bool,
    mode: str,
) -> tuple[Any, dict[str, Any]] | tuple[None, None]:
    edge_dict = G.get_edge_data(u, v, default={})
    if not edge_dict:
        return None, None

    best_key = None
    best_data = None
    best_cost = float("inf")

    for key, data in edge_dict.items():
        cost = edge_cost_components(data, active_weather, mode)["total_cost"]
        if cost < best_cost:
            best_key = key
            best_data = data
            best_cost = cost

    return best_key, best_data


def compact_routing_graph(
    G: nx.MultiDiGraph,
    active_weather: bool,
    mode: str,
    attr_name: str = "tmp_routing_cost",
) -> nx.DiGraph:
    routing_graph = routing_graph_for_mode(
        G,
        active_weather=active_weather,
        mode=mode,
        attr_name=attr_name,
    )
    H = nx.DiGraph()
    H.add_nodes_from(routing_graph.nodes())

    for u, v, data in routing_graph.edges(data=True):
        cost = to_float(data.get(attr_name), float("inf"))
        if not math.isfinite(cost):
            continue
        if not H.has_edge(u, v) or cost < to_float(H[u][v].get(attr_name), float("inf")):
            H.add_edge(u, v, **{attr_name: cost})

    return H


def candidate_node_paths(
    G: nx.MultiDiGraph,
    origin: str,
    destination: str,
    active_weather: bool,
    mode: str,
    max_candidates: int = AI_CANDIDATE_PATHS,
) -> list[list[str]]:
    H = compact_routing_graph(G, active_weather=active_weather, mode=mode)
    try:
        paths = nx.shortest_simple_paths(H, str(origin), str(destination), weight="tmp_routing_cost")
        return [[str(node) for node in path] for path in islice(paths, max(1, max_candidates))]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def priority_delay_weight(priority: int, reason: str) -> float:
    reason_text = str(reason).lower()
    if priority <= 1 or "emergency" in reason_text or "machine_disorder" in reason_text:
        return AI_DELAY_WEIGHT_EMERGENCY
    if priority == 2:
        return AI_DELAY_WEIGHT_FEATURED
    return AI_DELAY_WEIGHT_NORMAL


def score_route_segments(
    segments: list[dict[str, Any]],
    start_time: int,
    priority: int,
    reason: str,
) -> float:
    route_cost = sum(to_float(seg.get("total_cost"), 0.0) for seg in segments)
    wait_steps = sum(
        max(0, int(seg.get("duration_steps", 0)))
        for seg in segments
        if seg.get("kind") == "wait"
    )
    arrival = max([int(seg.get("t_end", start_time)) for seg in segments], default=int(start_time))
    elapsed = max(0, arrival - int(start_time))
    return (
        route_cost
        + AI_WAIT_STEP_PENALTY * wait_steps
        + priority_delay_weight(priority, reason) * elapsed
    )


# ---------------------------------------------------------------------
# Dynamic weather and no-flight-zone layers
# ---------------------------------------------------------------------

def graph_bounds(G: nx.MultiDiGraph) -> tuple[float, float, float, float]:
    xs = [to_float(data.get("x"), np.nan) for _, data in G.nodes(data=True)]
    ys = [to_float(data.get("y"), np.nan) for _, data in G.nodes(data=True)]
    xs = [x for x in xs if np.isfinite(x)]
    ys = [y for y in ys if np.isfinite(y)]
    if not xs or not ys:
        raise ValueError("Graph nodes do not contain valid x/y coordinates.")
    return min(xs), min(ys), max(xs), max(ys)


def weather_profile(weather_type: str) -> dict[str, Any]:
    return WEATHER_PROFILES.get(str(weather_type).lower(), WEATHER_PROFILES["wind"])


def scaled_multiplier(max_multiplier: float, severity: float) -> float:
    severity = min(max(float(severity), 0.0), 1.0)
    return 1.0 + (float(max_multiplier) - 1.0) * severity


def weather_parameters(weather_type: str, severity: float) -> dict[str, Any]:
    profile = weather_profile(weather_type)
    severity = min(max(float(severity), 0.0), 1.0)
    threshold = float(profile.get("dynamic_no_fly_threshold", 1.01))
    dynamic_no_fly = severity >= threshold
    battery_multiplier = scaled_multiplier(float(profile.get("battery_multiplier", 1.0)), severity)
    noise_multiplier = scaled_multiplier(float(profile.get("noise_multiplier", 1.0)), severity)
    risk_multiplier = scaled_multiplier(float(profile.get("risk_multiplier", 1.0)), severity)
    speed_multiplier = 1.0 - (1.0 - float(profile.get("speed_multiplier", 1.0))) * severity
    speed_multiplier = max(0.25, min(1.0, speed_multiplier))

    param_summary = (
        f"BAT x{battery_multiplier:.2f}, "
        f"NOISE x{noise_multiplier:.2f}, "
        f"RISK x{risk_multiplier:.2f}"
    )
    if dynamic_no_fly:
        param_summary += ", DYNAMIC NFZ"

    return {
        "label": str(profile.get("label", weather_type)),
        "consequence": str(profile.get("consequence", "")),
        "battery_multiplier": battery_multiplier,
        "noise_multiplier": noise_multiplier,
        "risk_multiplier": risk_multiplier,
        "speed_multiplier": speed_multiplier,
        "dynamic_no_fly": dynamic_no_fly,
        "dynamic_no_fly_threshold": threshold,
        "param_summary": param_summary,
    }


def incident_profile(incident_type: str) -> dict[str, Any]:
    return INCIDENT_PROFILES.get(str(incident_type), {
        "priority": 2,
        "preferred_emergency_site": None,
        "label": str(incident_type).replace("_", " ").title(),
        "reroute_reason": f"{incident_type}_priority_reroute",
    })


def make_zone_box(bounds: tuple[float, float, float, float], cx_ratio: float, cy_ratio: float, w_ratio: float, h_ratio: float) -> BaseGeometry:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    cx = minx + cx_ratio * width
    cy = miny + cy_ratio * height
    return box(cx - 0.5 * w_ratio * width, cy - 0.5 * h_ratio * height, cx + 0.5 * w_ratio * width, cy + 0.5 * h_ratio * height)


def make_zone_circle(bounds: tuple[float, float, float, float], cx_ratio: float, cy_ratio: float, radius_ratio: float) -> BaseGeometry:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    cx = minx + cx_ratio * width
    cy = miny + cy_ratio * height
    radius = min(width, height) * float(radius_ratio)
    return Point(cx, cy).buffer(radius, resolution=72)


def make_synthetic_no_flight_zones(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Static box-shaped no-fly zones are disabled for the presentation build."""
    crs = get_graph_crs(G)
    return gpd.GeoDataFrame(
        {
            "zone_id": [],
            "name": [],
            "rule": [],
            "geometry": [],
        },
        geometry="geometry",
        crs=crs,
    )


def make_dynamic_weather_events(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Create circular weather cells with consequence-to-parameter mappings."""
    bounds = graph_bounds(G)
    crs = get_graph_crs(G)

    def event(
        event_id: str,
        weather_type: str,
        start_time: int,
        end_time: int,
        severity: float,
        cx_ratio: float,
        cy_ratio: float,
        radius_ratio: float,
    ) -> dict[str, Any]:
        params = weather_parameters(weather_type, severity)
        return {
            "event_id": event_id,
            "weather_type": weather_type,
            "label": params["label"],
            "start_time": int(start_time),
            "end_time": int(end_time),
            "severity": float(severity),
            "consequence": params["consequence"],
            "battery_multiplier": params["battery_multiplier"],
            "noise_multiplier": params["noise_multiplier"],
            "risk_multiplier": params["risk_multiplier"],
            "speed_multiplier": params["speed_multiplier"],
            "dynamic_no_fly": params["dynamic_no_fly"],
            "hard_block": params["dynamic_no_fly"],
            "dynamic_no_fly_threshold": params["dynamic_no_fly_threshold"],
            "param_summary": params["param_summary"],
            "description": f"{params['label']}: {params['consequence']} -> {params['param_summary']}",
            "geometry": make_zone_circle(bounds, cx_ratio, cy_ratio, radius_ratio),
        }

    rows = [
        event("WX_HEAT_AIRPORT", "heat", 6, 58, 0.58, 0.43, 0.24, 0.125),
        event("WX_WIND_WEST", "wind", 10, 36, 0.68, 0.30, 0.52, 0.145),
        event("WX_LIGHTNING_CORE", "thunderstorm", 22, 48, 0.90, 0.59, 0.42, 0.100),
        event("WX_ICING_NORTH", "icing", 35, 64, 0.82, 0.66, 0.70, 0.110),
        event("WX_HEAVY_RAIN_EAST", "heavy_rain", 44, 68, 0.78, 0.74, 0.48, 0.120),
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def apply_no_flight_zones_to_graph(G: nx.MultiDiGraph, no_flight_zones: gpd.GeoDataFrame) -> None:
    if no_flight_zones.empty:
        for _, _, _, data in G.edges(keys=True, data=True):
            data["inside_no_flight_zone"] = False
        return

    try:
        no_fly_union = no_flight_zones.geometry.union_all()
    except AttributeError:
        no_fly_union = no_flight_zones.geometry.unary_union

    blocked_edges = 0
    for u, v, _, data in G.edges(keys=True, data=True):
        line = edge_geometry(G, str(u), str(v), data)
        inside = bool(line.intersects(no_fly_union))
        data["inside_no_flight_zone"] = inside
        if inside:
            blocked_edges += 1
    print(f"[INFO] No-flight zones mark {blocked_edges:,} directed edges as hard-avoid.")


def apply_weather_state_to_graph(G: nx.MultiDiGraph, weather_events: gpd.GeoDataFrame, time_step: int) -> list[str]:
    if weather_events.empty:
        active = weather_events
    else:
        active = weather_events[
            (weather_events["start_time"].astype(int) <= int(time_step))
            & (weather_events["end_time"].astype(int) > int(time_step))
        ].copy()

    active_records = []
    for _, row in active.iterrows():
        severity = to_float(row.get("severity"), default=0.0)
        params = weather_parameters(str(row["weather_type"]).lower(), severity)
        dynamic_no_fly = parse_bool(row.get("dynamic_no_fly")) or bool(params["dynamic_no_fly"])
        active_records.append(
            {
                "event_id": str(row["event_id"]),
                "weather_type": str(row["weather_type"]).lower(),
                "label": str(row.get("label", params["label"])),
                "consequence": str(row.get("consequence", params["consequence"])),
                "battery_multiplier": to_float(row.get("battery_multiplier"), params["battery_multiplier"]),
                "noise_multiplier": to_float(row.get("noise_multiplier"), params["noise_multiplier"]),
                "risk_multiplier": to_float(row.get("risk_multiplier"), params["risk_multiplier"]),
                "speed_multiplier": to_float(row.get("speed_multiplier"), params["speed_multiplier"]),
                "dynamic_no_fly": dynamic_no_fly,
                "hard_block": parse_bool(row.get("hard_block")) or dynamic_no_fly,
                "param_summary": str(row.get("param_summary", params["param_summary"])),
                "geometry": row.geometry,
            }
        )

    for u, v, _, data in G.edges(keys=True, data=True):
        line = edge_geometry(G, str(u), str(v), data)
        hit_ids: list[str] = []
        hit_types: list[str] = []
        hit_labels: list[str] = []
        hit_consequences: list[str] = []
        hit_summaries: list[str] = []
        hard_block = False
        dynamic_no_fly = False
        battery_multiplier = 1.0
        noise_multiplier = 1.0
        risk_multiplier = 1.0
        speed_multiplier = 1.0

        for rec in active_records:
            if line.intersects(rec["geometry"]):
                hit_ids.append(rec["event_id"])
                hit_types.append(rec["weather_type"])
                hit_labels.append(rec["label"])
                hit_consequences.append(rec["consequence"])
                hit_summaries.append(rec["param_summary"])
                battery_multiplier = max(battery_multiplier, float(rec["battery_multiplier"]))
                noise_multiplier = max(noise_multiplier, float(rec["noise_multiplier"]))
                risk_multiplier = max(risk_multiplier, float(rec["risk_multiplier"]))
                speed_multiplier = min(speed_multiplier, float(rec["speed_multiplier"]))
                if rec["hard_block"]:
                    hard_block = True
                if rec["dynamic_no_fly"]:
                    dynamic_no_fly = True

        data["inside_weather_zone"] = bool(hit_ids)
        data["active_weather_ids"] = ";".join(hit_ids)
        data["active_weather_type"] = ";".join(sorted(set(hit_types)))
        data["active_weather_labels"] = ";".join(sorted(set(hit_labels)))
        data["active_weather_consequences"] = ";".join(sorted(set(hit_consequences)))
        data["active_weather_param_summary"] = " | ".join(dict.fromkeys(hit_summaries))
        data["active_battery_multiplier"] = float(battery_multiplier)
        data["active_noise_multiplier"] = float(noise_multiplier)
        data["active_risk_multiplier"] = float(risk_multiplier)
        data["active_speed_multiplier"] = float(speed_multiplier)
        data["inside_dynamic_no_fly_zone"] = bool(dynamic_no_fly)
        data["blocked_by_active_weather"] = bool(hard_block)

    return [rec["event_id"] for rec in active_records]


def has_active_weather(weather_events: gpd.GeoDataFrame, time_step: int) -> bool:
    if weather_events.empty:
        return False
    active = weather_events[
        (weather_events["start_time"].astype(int) <= int(time_step))
        & (weather_events["end_time"].astype(int) > int(time_step))
    ]
    return not active.empty


def weather_timeline_times(weather_events: gpd.GeoDataFrame) -> list[int]:
    if weather_events.empty:
        return []
    times = sorted(set(weather_events["start_time"].astype(int).tolist() + weather_events["end_time"].astype(int).tolist()))
    return times



# ---------------------------------------------------------------------
# Segment creation
# ---------------------------------------------------------------------

def make_move_segment(
    G: nx.MultiDiGraph,
    agent_id: str,
    u: str,
    v: str,
    key: Any,
    data: dict[str, Any],
    t_start: int,
    active_weather: bool,
    mode: str,
    route_version: int,
    reason: str,
) -> dict[str, Any]:
    comp = edge_cost_components(data, active_weather=active_weather, mode=mode)
    duration = int(comp["travel_time_steps"])
    t_end = t_start + duration

    return {
        "mode": mode,
        "agent_id": agent_id,
        "route_version": route_version,
        "reason": reason,
        "kind": "move",
        "u": str(u),
        "v": str(v),
        "edge_key": str(key),
        "t_start": int(t_start),
        "t_end": int(t_end),
        "duration_steps": int(duration),
        "distance": comp["distance"],
        "battery_cost": comp["battery_cost"],
        "noise_penalty": comp["noise_penalty"],
        "weather_risk": comp["weather_risk"],
        "no_fly_penalty": comp.get("no_fly_penalty", 0.0),
        "total_cost": comp["total_cost"],
        "blocked": comp.get("blocked", False),
        "over_residential": comp["over_residential"],
        "inside_weather_zone": comp["inside_weather_zone"],
        "inside_no_flight_zone": comp.get("inside_no_flight_zone", False),
        "inside_dynamic_no_fly_zone": comp.get("inside_dynamic_no_fly_zone", False),
        "active_weather_type": comp.get("active_weather_type", ""),
        "active_weather_ids": comp.get("active_weather_ids", ""),
        "active_weather_consequences": comp.get("active_weather_consequences", ""),
        "active_weather_param_summary": comp.get("active_weather_param_summary", ""),
        "battery_multiplier": comp.get("battery_multiplier", 1.0),
        "noise_multiplier": comp.get("noise_multiplier", 1.0),
        "risk_multiplier": comp.get("risk_multiplier", 1.0),
        "speed_multiplier": comp.get("speed_multiplier", 1.0),
        "altitude_layer": comp["altitude_layer"],
        "capacity": comp["capacity"],
        "geometry": edge_geometry(G, u, v, data),
    }


def make_wait_segment(
    G: nx.MultiDiGraph,
    agent_id: str,
    node: str,
    t_start: int,
    route_version: int,
    reason: str,
    mode: str = "proposed",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "agent_id": agent_id,
        "route_version": route_version,
        "reason": reason,
        "kind": "wait",
        "u": str(node),
        "v": str(node),
        "edge_key": None,
        "t_start": int(t_start),
        "t_end": int(t_start + 1),
        "duration_steps": 1,
        "distance": 0.0,
        "battery_cost": WAIT_BATTERY_COST_PER_STEP,
        "noise_penalty": 0.0,
        "weather_risk": 0.0,
        "total_cost": WAIT_COST,
        "over_residential": False,
        "inside_weather_zone": False,
        "inside_dynamic_no_fly_zone": False,
        "altitude_layer": 0,
        "capacity": G.nodes[node].get("node_capacity", DEFAULT_NODE_CAPACITY),
        "geometry": Point(node_point(G, node)),
    }


# ---------------------------------------------------------------------
# Vertiport generation
# ---------------------------------------------------------------------

def auto_create_vertiports(G: nx.MultiDiGraph, n_vertiports: int) -> list[dict[str, Any]]:
    nodes = list(G.nodes)

    valid_nodes = [
        n for n in nodes
        if np.isfinite(to_float(G.nodes[n].get("x"), np.nan))
        and np.isfinite(to_float(G.nodes[n].get("y"), np.nan))
        and G.degree(n) > 0
    ]

    if len(valid_nodes) < 2:
        raise ValueError("Not enough valid graph nodes to create vertiports.")

    coords = np.array([[to_float(G.nodes[n]["x"]), to_float(G.nodes[n]["y"])] for n in valid_nodes])
    centroid = coords.mean(axis=0)

    non_emergency = [n for n in valid_nodes if not parse_bool(G.nodes[n].get("is_emergency_site"))]
    if not non_emergency:
        non_emergency = valid_nodes

    def dist_to_centroid(n: str) -> float:
        p = np.array([to_float(G.nodes[n]["x"]), to_float(G.nodes[n]["y"])])
        return float(np.linalg.norm(p - centroid))

    selected: list[str] = []

    # 1. First vertiport: central city hub.
    central = min(non_emergency, key=dist_to_centroid)
    selected.append(central)

    # 2. Add one or two medical / helipad-like emergency-compatible hubs.
    emergency_like = []
    for n in valid_nodes:
        if parse_bool(G.nodes[n].get("is_emergency_site")):
            etype = str(G.nodes[n].get("emergency_type", "")).lower()
            if "medical" in etype or "helipad" in etype or "heliport" in etype:
                emergency_like.append(n)

    emergency_like = sorted(emergency_like, key=dist_to_centroid)

    for n in emergency_like:
        if len(selected) >= min(n_vertiports, 3):
            break
        if n not in selected:
            selected.append(n)

    # 3. Fill the remaining vertiports using farthest-point sampling.
    candidate_pool = non_emergency + emergency_like
    candidate_pool = list(dict.fromkeys(candidate_pool))

    while len(selected) < min(n_vertiports, len(candidate_pool)):
        best_node = None
        best_score = -float("inf")

        selected_coords = np.array(
            [[to_float(G.nodes[n]["x"]), to_float(G.nodes[n]["y"])] for n in selected]
        )

        for n in candidate_pool:
            if n in selected:
                continue

            p = np.array([to_float(G.nodes[n]["x"]), to_float(G.nodes[n]["y"])])
            min_distance_to_selected = float(np.min(np.linalg.norm(selected_coords - p, axis=1)))
            degree_bonus = 50.0 * math.log1p(G.degree(n))
            score = min_distance_to_selected + degree_bonus

            if score > best_score:
                best_score = score
                best_node = n

        if best_node is None:
            break

        selected.append(best_node)

    vertiports: list[dict[str, Any]] = []

    for i, node in enumerate(selected):
        is_emergency = parse_bool(G.nodes[node].get("is_emergency_site"))
        emergency_type = str(G.nodes[node].get("emergency_type", "") or "")

        if i == 0:
            vtype = "central_city_hub"
            name = "Central City Hub"
            demand_weight = 2.0
        elif is_emergency and ("medical" in emergency_type.lower() or "helipad" in emergency_type.lower()):
            vtype = "medical_or_helipad_hub"
            name = f"Medical Hub {i}"
            demand_weight = 1.4
        else:
            vtype = "urban_vertiport"
            name = f"Urban Vertiport {i}"
            demand_weight = 1.0

        vertiport_id = f"V{i:02d}"

        G.nodes[node]["is_vertiport"] = True
        G.nodes[node]["vertiport_id"] = vertiport_id
        G.nodes[node]["node_capacity"] = VERTIPORT_NODE_CAPACITY

        vertiports.append(
            {
                "vertiport_id": vertiport_id,
                "node_id": str(node),
                "name": name,
                "type": vtype,
                "capacity": VERTIPORT_NODE_CAPACITY,
                "demand_weight": demand_weight,
                "is_emergency_compatible": bool(is_emergency),
                "emergency_type": emergency_type,
                "geometry": node_point(G, node),
            }
        )

    print(f"[INFO] Auto-created {len(vertiports)} vertiports")
    return vertiports


def export_vertiports(vertiports: list[dict[str, Any]], graph_crs: str, path: Path) -> None:
    gdf = gpd.GeoDataFrame(vertiports, geometry="geometry", crs=graph_crs)
    export_geojson(gdf, path)


def maybe_add_airport_vertiport(G: nx.MultiDiGraph, vertiports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add the configured airport gateway only if the current graph actually covers it.
    A municipal extract may not include the airport, so this safely skips it.
    """
    if not ADD_AIRPORT_VERTIPORT_IF_COVERED:
        return vertiports

    graph_crs = get_graph_crs(G)
    try:
        transformer = Transformer.from_crs(EXPORT_CRS, graph_crs, always_xy=True)
        airport_x, airport_y = transformer.transform(AIRPORT_LON, AIRPORT_LAT)
    except Exception:
        print(f"[WARN] Could not transform {AIRPORT_NAME} coordinate. Skipping airport vertiport.")
        return vertiports

    valid_nodes = [n for n, data in G.nodes(data=True) if np.isfinite(to_float(data.get("x"), np.nan)) and np.isfinite(to_float(data.get("y"), np.nan))]
    if not valid_nodes:
        return vertiports

    airport_pt = Point(airport_x, airport_y)
    nearest = min(valid_nodes, key=lambda n: airport_pt.distance(node_point(G, n)))
    distance = airport_pt.distance(node_point(G, nearest))

    if distance > AIRPORT_ATTACH_MAX_DISTANCE_M:
        print(f"[INFO] {AIRPORT_NAME} is outside this graph extent; nearest node is {distance/1000:.1f} km away. Skipping airport vertiport.")
        return vertiports

    vertiport_id = f"V{len(vertiports):02d}"
    G.nodes[nearest]["is_vertiport"] = True
    G.nodes[nearest]["vertiport_id"] = vertiport_id
    G.nodes[nearest]["node_capacity"] = max(VERTIPORT_NODE_CAPACITY * 2, 24)

    vertiports.append(
        {
            "vertiport_id": vertiport_id,
            "node_id": str(nearest),
            "name": AIRPORT_NAME,
            "type": "airport_gateway",
            "capacity": G.nodes[nearest]["node_capacity"],
            "demand_weight": 3.0,
            "is_emergency_compatible": True,
            "emergency_type": "airport_gateway",
            "geometry": node_point(G, nearest),
        }
    )
    print(f"[INFO] Added {AIRPORT_NAME} vertiport snapped to {nearest} ({distance:.0f} m from airport coordinate).")
    return vertiports


# ---------------------------------------------------------------------
# Fleet generation
# ---------------------------------------------------------------------

def weighted_choice(rng: np.random.Generator, items: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.array([float(item.get("demand_weight", 1.0)) for item in items])
    weights = weights / weights.sum()
    idx = int(rng.choice(len(items), p=weights))
    return items[idx]


def generate_fleet(
    vertiports: list[dict[str, Any]],
    n_agents: int,
    seed: int,
) -> list[Agent]:
    rng = np.random.default_rng(seed)
    agents: list[Agent] = []

    od_pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for i in range(n_agents):
        origin = weighted_choice(rng, vertiports)
        destination = weighted_choice(rng, vertiports)

        # Encourage route diversity: avoid overusing the same OD pair and prefer
        # destinations that are not trivially close to the origin.
        guard = 0
        while guard < 200:
            pair = (origin["vertiport_id"], destination["vertiport_id"])
            same = destination["vertiport_id"] == origin["vertiport_id"]
            repeated = od_pair_counts.get(pair, 0) >= 2
            if not same and not repeated:
                break
            origin = weighted_choice(rng, vertiports)
            destination = weighted_choice(rng, vertiports)
            guard += 1
        od_pair_counts[(origin["vertiport_id"], destination["vertiport_id"])] += 1

        start_time = int(rng.integers(0, MAX_START_TIME + 1))
        battery = float(round(rng.uniform(65.0, 100.0), 1))

        agent = Agent(
            agent_id=f"A{i:03d}",
            origin_vertiport=origin["vertiport_id"],
            destination_vertiport=destination["vertiport_id"],
            origin_node=str(origin["node_id"]),
            final_destination_node=str(destination["node_id"]),
            current_destination_node=str(destination["node_id"]),
            battery_initial=battery,
            start_time=start_time,
            priority=3,
            status="normal",
        )

        agents.append(agent)

    print(f"[INFO] Generated {len(agents)} eVTOL agents")
    return agents


def non_terminal_air_nodes(G: nx.MultiDiGraph) -> list[str]:
    return [
        str(node)
        for node, data in G.nodes(data=True)
        if not parse_bool(data.get("is_vertiport"))
        and not parse_bool(data.get("is_emergency_site"))
        and G.out_degree(node) > 0
        and G.in_degree(node) > 0
    ]


def node_near_ratio(
    G: nx.MultiDiGraph,
    x_ratio: float,
    y_ratio: float,
    banned: set[str],
    from_node: str | None = None,
    to_node: str | None = None,
) -> str:
    minx, miny, maxx, maxy = graph_bounds(G)
    target = Point(minx + (maxx - minx) * x_ratio, miny + (maxy - miny) * y_ratio)
    candidates = sorted(
        [node for node in non_terminal_air_nodes(G) if node not in banned],
        key=lambda n: node_point(G, n).distance(target),
    )

    for node in candidates:
        try:
            if from_node is not None and not nx.has_path(G, str(from_node), str(node)):
                continue
            if to_node is not None and not nx.has_path(G, str(node), str(to_node)):
                continue
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        return str(node)

    raise ValueError("Could not find a reachable featured mission vertistop node.")


def assign_featured_missions(
    G: nx.MultiDiGraph,
    agents: list[Agent],
) -> gpd.GeoDataFrame:
    specs = [
        {
            "mission_id": "FM01",
            "label": "Ride pickup",
            "type": "passenger",
            "agent_id": "A015",
            "start_time": 0,
            "pickup_ratio": (0.37, 0.35),
            "dropoff_ratio": (0.58, 0.54),
            "color": "#ff4d6d",
        },
        {
            "mission_id": "FM02",
            "label": "Express delivery",
            "type": "delivery",
            "agent_id": "A004",
            "start_time": 10,
            "pickup_ratio": (0.62, 0.30),
            "dropoff_ratio": (0.45, 0.68),
            "color": "#4dd8ff",
        },
        {
            "mission_id": "FM03",
            "label": "Airport transfer",
            "type": "passenger",
            "agent_id": "A038",
            "start_time": 12,
            "pickup_ratio": (0.30, 0.58),
            "dropoff_ratio": (0.76, 0.43),
            "color": "#ffd34d",
        },
    ]
    by_id = {a.agent_id: a for a in agents}
    rows: list[dict[str, Any]] = []
    banned: set[str] = set()

    for spec in specs:
        agent = by_id.get(spec["agent_id"])
        if agent is None:
            continue

        pickup = node_near_ratio(
            G,
            spec["pickup_ratio"][0],
            spec["pickup_ratio"][1],
            banned=banned,
            from_node=agent.origin_node,
        )
        banned.add(pickup)
        dropoff = node_near_ratio(
            G,
            spec["dropoff_ratio"][0],
            spec["dropoff_ratio"][1],
            banned=banned,
            from_node=pickup,
            to_node=agent.final_destination_node,
        )
        banned.add(dropoff)

        agent.start_time = int(spec["start_time"])
        agent.priority = min(agent.priority, 2)
        agent.mission_id = str(spec["mission_id"])
        agent.mission_label = str(spec["label"])
        agent.mission_type = str(spec["type"])
        agent.pickup_node = pickup
        agent.dropoff_node = dropoff
        agent.mission_waypoints = [pickup, dropoff, agent.final_destination_node]

        for sequence, (role, node) in enumerate((("pickup", pickup), ("dropoff", dropoff)), start=1):
            point = node_point(G, node)
            rows.append(
                {
                    "mission_id": agent.mission_id,
                    "agent_id": agent.agent_id,
                    "mission_label": agent.mission_label,
                    "mission_type": agent.mission_type,
                    "stop_role": role,
                    "sequence": sequence,
                    "node_id": node,
                    "color": spec["color"],
                    "geometry": point,
                }
            )

    print(f"[INFO] Assigned {len(rows) // 2} featured pickup/dropoff missions")
    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=get_graph_crs(G))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=get_graph_crs(G))


# ---------------------------------------------------------------------
# Reservation table
# ---------------------------------------------------------------------

class ReservationTable:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self.node_reservations: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.edge_reservations: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)
        self.physical_edge_reservations: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)

    def node_capacity(self, node: str) -> int:
        return max(1, to_int(self.G.nodes[node].get("node_capacity"), DEFAULT_NODE_CAPACITY))

    @staticmethod
    def _physical_key(u: str, v: str, layer: int, t: int) -> tuple[str, str, int, int]:
        a, b = sorted([str(u), str(v)])
        return a, b, int(layer), int(t)

    @staticmethod
    def _add_unique(store: dict, key: tuple, agent_id: str) -> None:
        if agent_id not in store[key]:
            store[key].append(agent_id)

    def node_available(self, node: str, t: int, agent_id: str | None = None) -> bool:
        occupants = set(self.node_reservations.get((str(node), int(t)), []))
        if agent_id is not None:
            occupants.discard(agent_id)
        return len(occupants) < self.node_capacity(str(node))

    def edge_available(
        self,
        u: str,
        v: str,
        layer: int,
        t: int,
        capacity: int,
        agent_id: str | None = None,
    ) -> bool:
        edge_key = (str(u), str(v), int(layer), int(t))
        physical_key = self._physical_key(str(u), str(v), int(layer), int(t))

        edge_occupants = set(self.edge_reservations.get(edge_key, []))
        physical_occupants = set(self.physical_edge_reservations.get(physical_key, []))

        if agent_id is not None:
            edge_occupants.discard(agent_id)
            physical_occupants.discard(agent_id)

        return len(edge_occupants) < capacity and len(physical_occupants) < capacity

    def reserve_node(self, node: str, t: int, agent_id: str) -> None:
        self._add_unique(self.node_reservations, (str(node), int(t)), agent_id)

    def reserve_edge(
        self,
        u: str,
        v: str,
        layer: int,
        t: int,
        capacity: int,
        agent_id: str,
    ) -> None:
        edge_key = (str(u), str(v), int(layer), int(t))
        physical_key = self._physical_key(str(u), str(v), int(layer), int(t))

        self._add_unique(self.edge_reservations, edge_key, agent_id)
        self._add_unique(self.physical_edge_reservations, physical_key, agent_id)

    def reserve_segments(self, segments: list[dict[str, Any]], agent_id: str) -> None:
        for seg in segments:
            t0 = int(seg["t_start"])
            t1 = int(seg["t_end"])

            if seg["kind"] == "wait":
                for t in range(t0, t1 + 1):
                    self.reserve_node(seg["u"], t, agent_id)
                continue

            self.reserve_node(seg["u"], t0, agent_id)

            layer = int(seg.get("altitude_layer", 1))
            capacity = max(1, to_int(seg.get("capacity"), DEFAULT_EDGE_CAPACITY))

            for t in range(t0, t1):
                self.reserve_edge(seg["u"], seg["v"], layer, t, capacity, agent_id)

            self.reserve_node(seg["v"], t1, agent_id)

    def release_agent_from_time(self, agent_id: str, from_time: int) -> None:
        from_time = int(from_time)

        def clean_store(store: dict[tuple, list[str]]) -> None:
            keys_to_delete = []

            for key, occupants in store.items():
                t = int(key[-1])
                if t >= from_time and agent_id in occupants:
                    occupants[:] = [a for a in occupants if a != agent_id]

                if not occupants:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del store[key]

        clean_store(self.node_reservations)
        clean_store(self.edge_reservations)
        clean_store(self.physical_edge_reservations)


# ---------------------------------------------------------------------
# Routing algorithms
# ---------------------------------------------------------------------

def reconstruct_route(
    parent: dict[tuple[str, int], tuple[tuple[str, int], dict[str, Any]]],
    final_state: tuple[str, int],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    state = final_state

    while state in parent:
        prev_state, action = parent[state]
        segments.append(action)
        state = prev_state

    segments.reverse()
    return segments


def timed_segments_for_path(
    G: nx.MultiDiGraph,
    reservation: ReservationTable,
    agent_id: str,
    path: list[str],
    start_time: int,
    active_weather: bool,
    route_version: int,
    reason: str,
    max_time: int = PLANNING_HORIZON,
) -> list[dict[str, Any]] | None:
    if len(path) <= 1:
        return []

    segments: list[dict[str, Any]] = []
    t = int(start_time)

    for u, v in zip(path[:-1], path[1:]):
        key, data = best_edge_between(G, str(u), str(v), active_weather=active_weather, mode="proposed")
        if data is None:
            return None

        comp = edge_cost_components(data, active_weather=active_weather, mode="proposed")
        if bool(comp.get("blocked")):
            # If the shortest path is forced through a blocked edge, signal failure.
            # The caller can fall back or another event can skip this route.
            return None

        duration = int(comp["travel_time_steps"])
        layer = int(comp["altitude_layer"])
        capacity = int(comp["capacity"])

        waited = 0
        while True:
            arrival_t = t + duration
            if arrival_t > max_time:
                return segments if segments else None

            node_ok = reservation.node_available(str(u), t, agent_id) and reservation.node_available(str(v), arrival_t, agent_id)
            edge_ok = True
            for occupied_t in range(t, arrival_t):
                if not reservation.edge_available(str(u), str(v), layer, occupied_t, capacity, agent_id):
                    edge_ok = False
                    break

            if node_ok and edge_ok:
                break

            if waited >= MAX_WAIT_STEPS_PER_CONFLICT:
                return None

            wait_segment = make_wait_segment(
                G=G,
                agent_id=agent_id,
                node=str(u),
                t_start=t,
                route_version=route_version,
                reason=f"{reason}_yield_wait",
                mode="proposed",
            )
            segments.append(wait_segment)
            t += 1
            waited += 1

        segment = make_move_segment(
            G=G,
            agent_id=agent_id,
            u=str(u),
            v=str(v),
            key=key,
            data=data,
            t_start=t,
            active_weather=active_weather,
            mode="proposed",
            route_version=route_version,
            reason=reason,
        )
        segments.append(segment)
        t = int(segment["t_end"])

    return segments


def time_aware_reserved_route(
    G: nx.MultiDiGraph,
    reservation: ReservationTable,
    agent_id: str,
    origin: str,
    destination: str,
    start_time: int,
    active_weather: bool,
    route_version: int,
    reason: str,
    priority: int = 3,
    max_time: int = PLANNING_HORIZON,
) -> list[dict[str, Any]] | None:
    """
    AI coordination policy for presentation-scale multi-agent routing.

    The coordinator first generates several feasible spatial path candidates
    with blocked dynamic-NFZ edges removed. It then simulates reservation-table
    timing for each candidate and chooses the route with the best utility score:
    route cost + wait/conflict cost + priority-sensitive delay cost.
    """
    origin = str(origin)
    destination = str(destination)

    if origin == destination:
        return []

    paths = candidate_node_paths(
        G=G,
        origin=origin,
        destination=destination,
        active_weather=active_weather,
        mode="proposed",
        max_candidates=AI_CANDIDATE_PATHS,
    )
    if not paths:
        return None

    evaluated: list[tuple[float, int, list[dict[str, Any]]]] = []
    for candidate_index, path in enumerate(paths, start=1):
        segments = timed_segments_for_path(
            G=G,
            reservation=reservation,
            agent_id=agent_id,
            path=path,
            start_time=start_time,
            active_weather=active_weather,
            route_version=route_version,
            reason=reason,
            max_time=max_time,
        )
        if segments is None:
            continue
        score = score_route_segments(
            segments=segments,
            start_time=start_time,
            priority=priority,
            reason=reason,
        )
        evaluated.append((score, candidate_index, segments))

    if not evaluated:
        return None

    evaluated.sort(key=lambda item: (item[0], item[1]))
    selected_score, selected_index, selected_segments = evaluated[0]
    for seg in selected_segments:
        seg["ai_policy"] = AI_COORDINATION_POLICY_NAME
        seg["ai_candidate_count"] = len(paths)
        seg["ai_feasible_candidate_count"] = len(evaluated)
        seg["ai_selected_candidate_index"] = selected_index
        seg["ai_route_score"] = round(float(selected_score), 3)

    return selected_segments

def simple_shortest_path_segments(
    G: nx.MultiDiGraph,
    agent_id: str,
    origin: str,
    destination: str,
    start_time: int,
    active_weather: bool,
    mode: str,
    route_version: int,
    reason: str,
) -> list[dict[str, Any]]:
    """
    Simple non-time-aware shortest path.

    Used for:
    - baseline routing
    - fallback if time-expanded reservation planning fails
    """
    if origin == destination:
        return []

    routing_graph = routing_graph_for_mode(G, active_weather=active_weather, mode=mode)

    try:
        path = nx.shortest_path(routing_graph, origin, destination, weight="tmp_routing_cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print(f"[WARN] No path found for {agent_id}: {origin} -> {destination}")
        return []

    segments: list[dict[str, Any]] = []
    t = int(start_time)

    for u, v in zip(path[:-1], path[1:]):
        key, data = best_edge_between(G, str(u), str(v), active_weather=active_weather, mode=mode)
        if data is None:
            continue

        comp = edge_cost_components(data, active_weather=active_weather, mode=mode)
        if mode == "proposed" and bool(comp.get("blocked")):
            print(f"[WARN] Blocked path rejected for {agent_id}: {origin} -> {destination}")
            return []

        segment = make_move_segment(
            G=G,
            agent_id=agent_id,
            u=str(u),
            v=str(v),
            key=key,
            data=data,
            t_start=t,
            active_weather=active_weather,
            mode=mode,
            route_version=route_version,
            reason=reason,
        )

        segments.append(segment)
        t = int(segment["t_end"])

    return segments


def tag_mission_segments(
    segments: list[dict[str, Any]],
    agent: Agent,
    leg_index: int,
    stage: str,
) -> None:
    if not agent.mission_id:
        return
    for seg in segments:
        seg["mission_featured"] = True
        seg["mission_id"] = agent.mission_id
        seg["mission_label"] = agent.mission_label or ""
        seg["mission_type"] = agent.mission_type
        seg["mission_leg_index"] = int(leg_index)
        seg["mission_stage"] = stage


def mission_stage_name(index: int) -> str:
    labels = ["to_pickup", "to_dropoff", "return_to_vertiport"]
    return labels[index] if index < len(labels) else f"mission_leg_{index + 1}"


def remaining_destinations_for_replan(
    agent: Agent,
    current_node: str,
    kept_segments: list[dict[str, Any]],
) -> list[str]:
    if (
        not agent.mission_waypoints
        or agent.emergency_type is not None
        or agent.current_destination_node == agent.emergency_destination_node
    ):
        return [agent.current_destination_node]

    reached_idx = -1
    waypoints = [str(node) for node in agent.mission_waypoints]
    current_node = str(current_node)

    for idx, waypoint in enumerate(waypoints):
        if current_node == waypoint:
            reached_idx = max(reached_idx, idx)

    for seg in kept_segments:
        if seg.get("kind") == "move":
            try:
                idx = waypoints.index(str(seg.get("v")))
                reached_idx = max(reached_idx, idx)
            except ValueError:
                pass

    remaining = waypoints[reached_idx + 1 :]
    return remaining if remaining else [agent.current_destination_node]


def route_destinations_for_agent(
    G: nx.MultiDiGraph,
    agent: Agent,
    reservation: ReservationTable,
    origin: str,
    destinations: list[str],
    start_time: int,
    active_weather: bool,
    route_version: int,
    reason: str,
) -> list[dict[str, Any]]:
    all_segments: list[dict[str, Any]] = []
    route_origin = str(origin)
    route_start = int(start_time)

    for destination in destinations:
        destination = str(destination)
        mission_idx = -1
        stage = ""
        leg_reason = reason
        if agent.mission_waypoints and destination in [str(node) for node in agent.mission_waypoints]:
            mission_idx = [str(node) for node in agent.mission_waypoints].index(destination)
            stage = mission_stage_name(mission_idx)
            leg_reason = f"{reason}_{stage}" if not reason.endswith(stage) else reason

        segments = time_aware_reserved_route(
            G=G,
            reservation=reservation,
            agent_id=agent.agent_id,
            origin=route_origin,
            destination=destination,
            start_time=route_start,
            active_weather=active_weather,
            route_version=route_version,
            reason=leg_reason,
            priority=agent.priority,
            max_time=PLANNING_HORIZON,
        )

        if segments is None:
            if not SUPPRESS_FALLBACK_WARNINGS:
                print(f"[WARN] Reserved route failed for {agent.agent_id}. Falling back to simple path.")
            segments = simple_shortest_path_segments(
                G=G,
                agent_id=agent.agent_id,
                origin=route_origin,
                destination=destination,
                start_time=route_start,
                active_weather=active_weather,
                mode="proposed",
                route_version=route_version,
                reason=f"fallback_{leg_reason}",
            )

        if mission_idx >= 0:
            tag_mission_segments(segments, agent, mission_idx + 1, stage)

        if not segments and route_origin != destination:
            break

        all_segments.extend(segments)
        if segments:
            route_start = int(segments[-1]["t_end"])
        route_origin = destination

    return all_segments


def initial_route_for_agent(
    G: nx.MultiDiGraph,
    agent: Agent,
    reservation: ReservationTable,
) -> list[dict[str, Any]]:
    waypoints = list(agent.mission_waypoints) if agent.mission_waypoints else [agent.current_destination_node]
    reason = "featured_mission" if agent.mission_id else "initial_constraint_aware_plan"
    return route_destinations_for_agent(
        G=G,
        agent=agent,
        reservation=reservation,
        origin=agent.origin_node,
        destinations=waypoints,
        start_time=agent.start_time,
        active_weather=False,
        route_version=agent.route_version,
        reason=reason,
    )


def plan_initial_proposed_routes(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    reservation: ReservationTable,
) -> None:
    ordered_agents = sorted(agents, key=lambda a: (a.priority, a.start_time, a.agent_id))

    for agent in ordered_agents:
        segments = initial_route_for_agent(G, agent, reservation)
        agent.segments = segments
        reservation.reserve_segments(segments, agent.agent_id)


def compute_baseline_routes(G: nx.MultiDiGraph, agents: list[Agent]) -> dict[str, list[dict[str, Any]]]:
    baseline: dict[str, list[dict[str, Any]]] = {}

    for agent in agents:
        segments = simple_shortest_path_segments(
            G=G,
            agent_id=agent.agent_id,
            origin=agent.origin_node,
            destination=agent.final_destination_node,
            start_time=agent.start_time,
            active_weather=False,
            mode="baseline",
            route_version=0,
            reason="baseline_distance_only_no_reservation",
        )
        baseline[agent.agent_id] = segments

    return baseline


# ---------------------------------------------------------------------
# Dynamic rerouting and emergency handling
# ---------------------------------------------------------------------

def arrival_time_from_segments(segments: list[dict[str, Any]]) -> int | None:
    if not segments:
        return None
    return max(int(seg["t_end"]) for seg in segments)


def agent_arrival_time(agent: Agent) -> int | None:
    if not agent.segments:
        return None

    target = str(agent.current_destination_node)
    last_node = str(agent.segments[-1].get("v"))
    if last_node != target:
        return None

    return arrival_time_from_segments(agent.segments)


def agent_not_arrived_at(agent: Agent, t: int) -> bool:
    arrival = agent_arrival_time(agent)
    if arrival is None:
        return True
    return arrival > t


def future_route_crosses_weather(agent: Agent, event_time: int) -> bool:
    for seg in agent.segments:
        if seg["kind"] == "move" and int(seg["t_end"]) > event_time:
            if bool(seg.get("inside_weather_zone")):
                return True
    return False


def current_state_for_replan(
    agent: Agent,
    event_time: int,
    blocked_geoms: list[BaseGeometry] | None = None,
) -> tuple[str, int, list[dict[str, Any]]] | None:
    """
    Returns:
        current_node, replan_start_time, kept_segments

    Simplification:
        - If an event happens while the aircraft is in an edge,
          it first finishes that edge, then replans from the next node.
        - If the event happens while waiting at a node, it replans immediately.
    """
    if event_time < agent.start_time:
        return agent.origin_node, agent.start_time, []

    if not agent.segments:
        return agent.origin_node, max(event_time, agent.start_time), []

    kept: list[dict[str, Any]] = []

    for seg in agent.segments:
        t0 = int(seg["t_start"])
        t1 = int(seg["t_end"])

        if event_time < t0:
            current_node = kept[-1]["v"] if kept else agent.origin_node
            return str(current_node), event_time, kept

        if event_time == t0:
            return str(seg["u"]), event_time, kept

        if t0 < event_time < t1:
            if seg["kind"] == "wait":
                shortened = dict(seg)
                shortened["t_end"] = event_time
                shortened["duration_steps"] = max(0, event_time - t0)
                if shortened["duration_steps"] > 0:
                    kept.append(shortened)
                return str(seg["u"]), event_time, kept

            geom = seg.get("geometry")
            if blocked_geoms and geom is not None and any(geom.intersects(g) for g in blocked_geoms):
                return str(seg["u"]), event_time, kept

            # In flight: finish the current edge, then replan.
            kept.append(seg)
            return str(seg["v"]), t1, kept

        kept.append(seg)

    # Already arrived.
    return None


def reroute_agents(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    reservation: ReservationTable,
    affected_agent_ids: set[str],
    event_time: int,
    active_weather: bool,
    reason: str,
    alerts: list[dict[str, Any]],
    blocked_geoms: list[BaseGeometry] | None = None,
) -> None:
    id_to_agent = {a.agent_id: a for a in agents}

    replan_info: dict[str, tuple[str, int, list[dict[str, Any]]]] = {}

    for agent_id in affected_agent_ids:
        agent = id_to_agent[agent_id]
        state = current_state_for_replan(agent, event_time, blocked_geoms=blocked_geoms)

        if state is None:
            continue

        current_node, replan_start, kept_segments = state
        replan_info[agent_id] = (current_node, replan_start, kept_segments)

    # First release all future reservations for affected agents.
    for agent_id, (_, replan_start, _) in replan_info.items():
        reservation.release_agent_from_time(agent_id, replan_start)

    # Then replan in priority order.
    ordered = sorted(
        replan_info.keys(),
        key=lambda aid: (
            id_to_agent[aid].priority,
            replan_info[aid][1],
            id_to_agent[aid].start_time,
            aid,
        ),
    )

    for agent_id in ordered:
        agent = id_to_agent[agent_id]
        current_node, replan_start, kept_segments = replan_info[agent_id]

        agent.route_version += 1
        destinations = remaining_destinations_for_replan(agent, current_node, kept_segments)

        new_segments = route_destinations_for_agent(
            G=G,
            agent=agent,
            reservation=reservation,
            origin=current_node,
            destinations=destinations,
            start_time=replan_start,
            active_weather=active_weather,
            route_version=agent.route_version,
            reason=reason,
        )

        agent.segments = kept_segments + new_segments
        agent.reroute_count += 1

        reservation.reserve_segments(new_segments, agent.agent_id)

        alerts.append(
            {
                "time_step": event_time,
                "agent_id": agent_id,
                "event_type": "reroute",
                "reason": reason,
                "current_node": current_node,
                "destination_node": destinations[-1] if destinations else agent.current_destination_node,
                "next_destination_node": destinations[0] if destinations else agent.current_destination_node,
                "replan_start_time": replan_start,
                "route_version": agent.route_version,
            }
        )


def process_weather_timeline_event(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    reservation: ReservationTable,
    weather_events: gpd.GeoDataFrame,
    event_time: int,
    alerts: list[dict[str, Any]],
) -> None:
    active_ids = apply_weather_state_to_graph(G, weather_events, event_time)
    active_agent_ids = {
        a.agent_id for a in agents
        if agent_not_arrived_at(a, event_time)
    }

    active_rows = weather_events[
        (weather_events["start_time"].astype(int) <= int(event_time))
        & (weather_events["end_time"].astype(int) > int(event_time))
    ]
    active_types = ",".join(sorted(set(active_rows["weather_type"].astype(str).str.lower().tolist()))) if not active_rows.empty else "none"
    active_consequences = " | ".join(
        dict.fromkeys(active_rows["consequence"].astype(str).tolist())
    ) if "consequence" in active_rows.columns and not active_rows.empty else ""
    active_parameter_summary = " | ".join(
        dict.fromkeys(active_rows["param_summary"].astype(str).tolist())
    ) if "param_summary" in active_rows.columns and not active_rows.empty else ""
    active_dynamic_no_fly = int(active_rows["dynamic_no_fly"].map(parse_bool).sum()) if "dynamic_no_fly" in active_rows.columns and not active_rows.empty else 0

    # Soft weather changes replan aircraft whose future route intersects the
    # cells. Dynamic NFZ activation is stricter: every active aircraft replans
    # against a graph where blocked weather edges are removed.
    affected: set[str] = set()
    active_geoms = [row.geometry for _, row in active_rows.iterrows()] if not active_rows.empty else []
    active_dynamic_geoms = [
        row.geometry
        for _, row in active_rows.iterrows()
        if parse_bool(row.get("dynamic_no_fly"))
    ] if not active_rows.empty else []

    if active_dynamic_geoms:
        affected = set(active_agent_ids)
    else:
        for agent in agents:
            if agent.agent_id not in active_agent_ids:
                continue
            for seg in agent.segments:
                if seg.get("kind") != "move" or int(seg.get("t_end", 0)) <= int(event_time):
                    continue
                geom = seg.get("geometry")
                if geom is not None and any(geom.intersects(g) for g in active_geoms):
                    affected.add(agent.agent_id)
                    break

    # If weather dissipates, no need to replan everyone. If weather appears and no
    # current route intersects it, replan a small deterministic sample for demo motion.
    reroute_ids = sorted(affected)
    if active_ids and not reroute_ids:
        reroute_ids = sorted(active_agent_ids)[: min(4, len(active_agent_ids))]

    if not active_dynamic_geoms and len(reroute_ids) > MAX_WEATHER_REROUTES_PER_EVENT:
        reroute_ids = reroute_ids[:MAX_WEATHER_REROUTES_PER_EVENT]

    alerts.append(
        {
            "time_step": event_time,
            "agent_id": None,
            "event_type": "weather_update",
            "reason": "dynamic_weather_state_changed_fast_replan",
            "active_weather_ids": ";".join(active_ids),
            "active_weather_types": active_types,
            "active_weather_consequences": active_consequences,
            "active_weather_parameters": active_parameter_summary,
            "active_dynamic_no_fly_cells": active_dynamic_no_fly,
            "active_aircraft": len(active_agent_ids),
            "weather_affected_aircraft": len(affected),
            "active_aircraft_replanned": len(reroute_ids),
        }
    )

    if reroute_ids:
        reroute_agents(
            G=G,
            agents=agents,
            reservation=reservation,
            affected_agent_ids=set(reroute_ids),
            event_time=event_time,
            active_weather=bool(active_ids),
        reason=f"weather_update_{active_types}",
        alerts=alerts,
        blocked_geoms=active_dynamic_geoms,
    )

def process_weather_event(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    reservation: ReservationTable,
    event_time: int,
    alerts: list[dict[str, Any]],
) -> None:
    active_agent_ids = {
        a.agent_id for a in agents
        if agent_not_arrived_at(a, event_time)
    }

    affected_by_weather = {
        a.agent_id for a in agents
        if a.agent_id in active_agent_ids and future_route_crosses_weather(a, event_time)
    }

    # For a stronger and clearer demo, replan all active aircraft.
    # The alert still records how many originally crossed weather-risk edges.
    reroute_ids = active_agent_ids

    alerts.append(
        {
            "time_step": event_time,
            "agent_id": None,
            "event_type": "weather_shift",
            "reason": "synthetic_weather_zone_becomes_active",
            "active_aircraft": len(active_agent_ids),
            "aircraft_with_weather_exposure_on_old_route": len(affected_by_weather),
            "rerouted_aircraft": len(reroute_ids),
        }
    )

    reroute_agents(
        G=G,
        agents=agents,
        reservation=reservation,
        affected_agent_ids=reroute_ids,
        event_time=event_time,
        active_weather=True,
        reason="weather_shift_reroute",
        alerts=alerts,
    )


def emergency_candidate_nodes(
    G: nx.MultiDiGraph,
    preferred: str | None = None,
) -> list[str]:
    candidates = []

    for node, data in G.nodes(data=True):
        if parse_bool(data.get("is_emergency_site")):
            etype = str(data.get("emergency_type", "") or "").lower()

            if preferred == "medical":
                if "medical" in etype or "hospital" in etype or "helipad" in etype or "heliport" in etype:
                    candidates.append(str(node))
            else:
                candidates.append(str(node))

    if candidates:
        return candidates

    # Fallback: use vertiports.
    return [
        str(node)
        for node, data in G.nodes(data=True)
        if parse_bool(data.get("is_vertiport"))
    ]


def nearest_emergency_node(
    G: nx.MultiDiGraph,
    from_node: str,
    active_weather: bool,
    preferred: str | None = None,
) -> str:
    candidates = emergency_candidate_nodes(G, preferred=preferred)
    if not candidates:
        raise ValueError("No emergency nodes or vertiports available.")

    candidates_excluding_current = [node for node in candidates if str(node) != str(from_node)]
    if candidates_excluding_current:
        candidates = candidates_excluding_current

    routing_graph = routing_graph_for_mode(G, active_weather=active_weather, mode="proposed")

    best_node = None
    best_cost = float("inf")

    for candidate in candidates:
        try:
            cost = nx.shortest_path_length(routing_graph, from_node, candidate, weight="tmp_routing_cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if cost < best_cost:
            best_cost = float(cost)
            best_node = candidate

    if best_node is not None:
        return best_node

    # Geometric fallback if graph path is unavailable.
    from_pt = node_point(G, from_node)
    return min(candidates, key=lambda n: from_pt.distance(node_point(G, n)))


def select_emergency_agent_and_time(
    agents: list[Agent],
    preferred_time: int,
    excluded_agent_ids: set[str],
) -> tuple[Agent | None, int]:
    active = [
        a for a in agents
        if a.agent_id not in excluded_agent_ids
        and a.status != "emergency"
        and a.start_time <= preferred_time
        and agent_not_arrived_at(a, preferred_time)
    ]

    non_featured_active = [a for a in active if not a.mission_id]
    if non_featured_active:
        active = non_featured_active

    if active:
        # Choose the aircraft with the longest remaining trip.
        def remaining(a: Agent) -> int:
            arrival = agent_arrival_time(a)
            if arrival is None:
                return 0
            return arrival - preferred_time

        return max(active, key=remaining), preferred_time

    # Fallback: choose the longest-trip non-emergency aircraft and put the event near its midpoint.
    candidates = [
        a for a in agents
        if a.agent_id not in excluded_agent_ids
        and a.status != "emergency"
        and agent_arrival_time(a) is not None
    ]

    non_featured_candidates = [a for a in candidates if not a.mission_id]
    if non_featured_candidates:
        candidates = non_featured_candidates

    if not candidates:
        return None, preferred_time

    def trip_duration(a: Agent) -> int:
        arrival = agent_arrival_time(a)
        return max(0, int(arrival or a.start_time) - a.start_time)

    target = max(candidates, key=trip_duration)
    arrival = agent_arrival_time(target) or (target.start_time + 1)
    midpoint = target.start_time + max(1, (arrival - target.start_time) // 2)

    return target, midpoint


def process_emergency_event(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    reservation: ReservationTable,
    preferred_time: int,
    emergency_type: str,
    active_weather: bool,
    excluded_agent_ids: set[str],
    alerts: list[dict[str, Any]],
) -> tuple[int | None, str | None]:
    target, event_time = select_emergency_agent_and_time(
        agents=agents,
        preferred_time=preferred_time,
        excluded_agent_ids=excluded_agent_ids,
    )

    if target is None:
        alerts.append(
            {
                "time_step": preferred_time,
                "agent_id": None,
                "event_type": emergency_type,
                "reason": "no_available_aircraft_for_emergency",
            }
        )
        return None, None

    state = current_state_for_replan(target, event_time)

    if state is None:
        alerts.append(
            {
                "time_step": event_time,
                "agent_id": target.agent_id,
                "event_type": emergency_type,
                "reason": "selected_aircraft_already_arrived",
            }
        )
        return event_time, target.agent_id

    current_node, _, _ = state

    profile = incident_profile(emergency_type)
    preferred_emergency_site = profile.get("preferred_emergency_site")
    emergency_destination = nearest_emergency_node(
        G=G,
        from_node=current_node,
        active_weather=active_weather,
        preferred=preferred_emergency_site,
    )

    target.status = "emergency"
    target.emergency_type = emergency_type
    target.emergency_start_time = event_time
    target.emergency_destination_node = emergency_destination
    target.current_destination_node = emergency_destination
    target.priority = int(profile.get("priority", 2))

    active_agent_ids = {
        a.agent_id for a in agents
        if agent_not_arrived_at(a, event_time)
    }

    # Fast demo policy: always replan the emergency aircraft first, then a small
    # deterministic support set. This still shows priority/yield behavior without
    # replanning the entire fleet at every emergency event.
    support_ids = sorted(active_agent_ids - {target.agent_id})[:MAX_EMERGENCY_SUPPORT_REROUTES]
    reroute_ids = {target.agent_id, *support_ids}

    alerts.append(
        {
            "time_step": event_time,
            "agent_id": target.agent_id,
            "event_type": emergency_type,
            "reason": "emergency_priority_reroute_fast",
            "event_label": profile.get("label", emergency_type),
            "current_node": current_node,
            "emergency_destination_node": emergency_destination,
            "active_aircraft": len(active_agent_ids),
            "active_aircraft_replanned": len(reroute_ids),
        }
    )

    reroute_agents(
        G=G,
        agents=agents,
        reservation=reservation,
        affected_agent_ids=reroute_ids,
        event_time=event_time,
        active_weather=active_weather,
        reason=str(profile.get("reroute_reason", f"{emergency_type}_priority_reroute")),
        alerts=alerts,
    )

    return event_time, target.agent_id


# ---------------------------------------------------------------------
# Conflict counting and metrics
# ---------------------------------------------------------------------

def collect_segments_from_agents(agents: list[Agent]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for agent in agents:
        segments.extend(agent.segments)
    return segments


def count_conflicts(
    G: nx.MultiDiGraph,
    segments_by_agent: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    node_occ: dict[tuple[str, int], set[str]] = defaultdict(set)
    edge_occ: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    physical_occ: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    edge_caps: dict[tuple[str, str, int, int], int] = {}

    for agent_id, segments in segments_by_agent.items():
        for seg in segments:
            t0 = int(seg["t_start"])
            t1 = int(seg["t_end"])

            if seg["kind"] == "wait":
                for t in range(t0, t1 + 1):
                    node_occ[(seg["u"], t)].add(agent_id)
                continue

            node_occ[(seg["u"], t0)].add(agent_id)
            node_occ[(seg["v"], t1)].add(agent_id)

            layer = int(seg.get("altitude_layer", 1))
            cap = max(1, to_int(seg.get("capacity"), DEFAULT_EDGE_CAPACITY))

            a, b = sorted([seg["u"], seg["v"]])

            for t in range(t0, t1):
                edge_key = (seg["u"], seg["v"], layer, t)
                physical_key = (a, b, layer, t)

                edge_occ[edge_key].add(agent_id)
                physical_occ[physical_key].add(agent_id)
                edge_caps[edge_key] = cap

    node_conflicts = 0
    edge_conflicts = 0
    physical_conflicts = 0

    for (node, _), occupants in node_occ.items():
        cap = max(1, to_int(G.nodes[node].get("node_capacity"), DEFAULT_NODE_CAPACITY))
        if len(occupants) > cap:
            node_conflicts += len(occupants) - cap

    for key, occupants in edge_occ.items():
        cap = edge_caps.get(key, DEFAULT_EDGE_CAPACITY)
        if len(occupants) > cap:
            edge_conflicts += len(occupants) - cap

    for key, occupants in physical_occ.items():
        cap = DEFAULT_EDGE_CAPACITY
        if len(occupants) > cap:
            physical_conflicts += len(occupants) - cap

    return {
        "node_conflicts": node_conflicts,
        "edge_conflicts": edge_conflicts,
        "physical_edge_conflicts": physical_conflicts,
        "total_conflicts": node_conflicts + edge_conflicts + physical_conflicts,
    }


def metrics_for_mode(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    segments_by_agent: dict[str, list[dict[str, Any]]],
    mode_name: str,
    weather_event_time: int,
) -> dict[str, Any]:
    all_segments = [seg for segs in segments_by_agent.values() for seg in segs]

    move_segments = [seg for seg in all_segments if seg["kind"] == "move"]
    wait_segments = [seg for seg in all_segments if seg["kind"] == "wait"]

    total_distance = sum(float(seg["distance"]) for seg in move_segments)
    total_battery = sum(float(seg["battery_cost"]) for seg in all_segments)
    total_noise_penalty = sum(float(seg["noise_penalty"]) for seg in move_segments)
    total_weather_risk = sum(float(seg["weather_risk"]) for seg in move_segments)
    wait_steps = sum(int(seg["duration_steps"]) for seg in wait_segments)

    residential_exposure = sum(
        float(seg["distance"]) for seg in move_segments
        if bool(seg.get("over_residential"))
    )

    weather_exposure_after_event = sum(
        float(seg["distance"]) for seg in move_segments
        if bool(seg.get("inside_weather_zone")) and int(seg["t_end"]) >= weather_event_time
    )

    no_flight_exposure = sum(
        float(seg["distance"]) for seg in move_segments
        if bool(seg.get("inside_no_flight_zone"))
    )

    dynamic_no_fly_exposure = sum(
        float(seg["distance"]) for seg in move_segments
        if bool(seg.get("inside_dynamic_no_fly_zone"))
    )

    weather_parameterized_distance = sum(
        float(seg["distance"]) for seg in move_segments
        if bool(seg.get("inside_weather_zone"))
    )

    battery_overhead = sum(
        max(0.0, float(seg.get("battery_cost", 0.0)) - float(seg.get("distance", 0.0)) * BASE_ENERGY_PER_METER)
        for seg in move_segments
    )

    max_battery_multiplier = max(
        [float(seg.get("battery_multiplier", 1.0)) for seg in move_segments],
        default=1.0,
    )
    max_noise_multiplier = max(
        [float(seg.get("noise_multiplier", 1.0)) for seg in move_segments],
        default=1.0,
    )

    if mode_name == "proposed":
        arrivals = {
            agent.agent_id: agent_arrival_time(agent)
            for agent in agents
        }
    else:
        arrivals = {
            agent_id: arrival_time_from_segments(segs)
            for agent_id, segs in segments_by_agent.items()
        }

    completed = sum(1 for a in agents if arrivals.get(a.agent_id) is not None)
    trip_times = [
        arrivals[a.agent_id] - a.start_time
        for a in agents
        if arrivals.get(a.agent_id) is not None
    ]

    avg_trip_time = float(np.mean(trip_times)) if trip_times else np.nan
    max_arrival_time = max([t for t in arrivals.values() if t is not None], default=np.nan)

    conflicts = count_conflicts(G, segments_by_agent)

    emergency_agents = [a for a in agents if a.emergency_type is not None]
    emergency_success = 0
    emergency_landing_times: list[int] = []
    emergency_types_handled = sorted(
        {str(a.emergency_type) for a in emergency_agents if a.emergency_type}
    )

    for agent in emergency_agents:
        arrival = arrivals.get(agent.agent_id)
        if arrival is None:
            continue

        last_node = segments_by_agent[agent.agent_id][-1]["v"] if segments_by_agent[agent.agent_id] else None

        if last_node == agent.emergency_destination_node:
            emergency_success += 1
            if agent.emergency_start_time is not None:
                emergency_landing_times.append(arrival - agent.emergency_start_time)

    avg_emergency_landing_time = (
        float(np.mean(emergency_landing_times))
        if emergency_landing_times
        else np.nan
    )

    ai_segments = [
        seg for seg in move_segments
        if str(seg.get("ai_policy", "")) == AI_COORDINATION_POLICY_NAME
    ]
    ai_candidate_counts = [
        to_float(seg.get("ai_candidate_count"), np.nan)
        for seg in ai_segments
        if not np.isnan(to_float(seg.get("ai_candidate_count"), np.nan))
    ]
    ai_feasible_counts = [
        to_float(seg.get("ai_feasible_candidate_count"), np.nan)
        for seg in ai_segments
        if not np.isnan(to_float(seg.get("ai_feasible_candidate_count"), np.nan))
    ]

    return {
        "mode": mode_name,
        "agents": len(agents),
        "completed_agents": completed,
        "total_distance_m": round(total_distance, 2),
        "total_battery_cost": round(total_battery, 3),
        "total_wait_steps": int(wait_steps),
        "total_noise_penalty": round(total_noise_penalty, 2),
        "residential_exposure_m": round(residential_exposure, 2),
        "weather_exposure_after_event_m": round(weather_exposure_after_event, 2),
        "no_flight_exposure_m": round(no_flight_exposure, 2),
        "dynamic_weather_nfz_exposure_m": round(dynamic_no_fly_exposure, 2),
        "weather_parameterized_distance_m": round(weather_parameterized_distance, 2),
        "weather_battery_overhead": round(battery_overhead, 3),
        "max_weather_battery_multiplier": round(max_battery_multiplier, 3),
        "max_weather_noise_multiplier": round(max_noise_multiplier, 3),
        "total_weather_risk": round(total_weather_risk, 2),
        "avg_trip_time_steps": round(avg_trip_time, 2) if np.isfinite(avg_trip_time) else np.nan,
        "max_arrival_time_step": max_arrival_time,
        "node_conflicts": conflicts["node_conflicts"],
        "edge_conflicts": conflicts["edge_conflicts"],
        "physical_edge_conflicts": conflicts["physical_edge_conflicts"],
        "total_conflicts": conflicts["total_conflicts"],
        "emergency_agents": len(emergency_agents) if mode_name == "proposed" else 0,
        "emergency_success": emergency_success if mode_name == "proposed" else 0,
        "emergency_types_handled": ";".join(emergency_types_handled) if mode_name == "proposed" else "",
        "avg_emergency_landing_time_steps": (
            round(avg_emergency_landing_time, 2)
            if np.isfinite(avg_emergency_landing_time)
            else np.nan
        ),
        "ai_policy": AI_COORDINATION_POLICY_NAME if mode_name == "proposed" else "",
        "ai_routed_segments": len(ai_segments) if mode_name == "proposed" else 0,
        "avg_ai_candidate_paths": (
            round(float(np.mean(ai_candidate_counts)), 2)
            if ai_candidate_counts and mode_name == "proposed"
            else 0.0
        ),
        "avg_ai_feasible_paths": (
            round(float(np.mean(ai_feasible_counts)), 2)
            if ai_feasible_counts and mode_name == "proposed"
            else 0.0
        ),
        "total_reroutes": sum(a.reroute_count for a in agents) if mode_name == "proposed" else 0,
    }


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------

def export_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}),
            encoding="utf-8",
        )
        print(f"[INFO] Wrote empty GeoJSON: {path}")
        return

    out = gdf.copy()

    for col in out.columns:
        if col == "geometry":
            continue
        if out[col].dtype == "object":
            out[col] = out[col].map(json_safe)

    out = out.to_crs(EXPORT_CRS)
    out.to_file(path, driver="GeoJSON")
    print(f"[INFO] Wrote GeoJSON: {path}")


def segments_to_gdf(
    segments: list[dict[str, Any]],
    graph_crs: str,
) -> gpd.GeoDataFrame:
    rows = []

    for seg in segments:
        if seg["kind"] != "move":
            continue

        row = {k: v for k, v in seg.items() if k != "geometry"}
        row["geometry"] = seg["geometry"]
        rows.append(row)

    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=graph_crs)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=graph_crs)


def export_routes(
    baseline: dict[str, list[dict[str, Any]]],
    agents: list[Agent],
    graph_crs: str,
) -> None:
    baseline_segments = [seg for segs in baseline.values() for seg in segs]
    proposed_segments = collect_segments_from_agents(agents)

    baseline_gdf = segments_to_gdf(baseline_segments, graph_crs)
    proposed_gdf = segments_to_gdf(proposed_segments, graph_crs)

    combined = pd.concat([baseline_gdf, proposed_gdf], ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=graph_crs)

    export_geojson(baseline_gdf, OUT_BASELINE_ROUTES)
    export_geojson(proposed_gdf, OUT_PROPOSED_ROUTES)
    export_geojson(combined_gdf, OUT_ROUTES)


def export_agents_csv(agents: list[Agent]) -> None:
    rows = []

    for agent in agents:
        arrival = agent_arrival_time(agent)

        final_status = agent.status
        if arrival is not None:
            if agent.emergency_type:
                final_status = "emergency_landed"
            else:
                final_status = "arrived"

        rows.append(
            {
                "agent_id": agent.agent_id,
                "origin_vertiport": agent.origin_vertiport,
                "destination_vertiport": agent.destination_vertiport,
                "origin_node": agent.origin_node,
                "final_destination_node": agent.final_destination_node,
                "current_destination_node": agent.current_destination_node,
                "battery_initial": agent.battery_initial,
                "start_time": agent.start_time,
                "arrival_time": arrival,
                "priority": agent.priority,
                "status": final_status,
                "emergency_type": agent.emergency_type,
                "emergency_start_time": agent.emergency_start_time,
                "emergency_destination_node": agent.emergency_destination_node,
                "mission_id": agent.mission_id,
                "mission_label": agent.mission_label,
                "mission_type": agent.mission_type,
                "pickup_node": agent.pickup_node,
                "dropoff_node": agent.dropoff_node,
                "mission_waypoints": json_safe(agent.mission_waypoints),
                "route_version": agent.route_version,
                "reroute_count": agent.reroute_count,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_AGENTS, index=False)
    print(f"[INFO] Wrote agents CSV: {OUT_AGENTS}")


def export_vertiport_distribution(
    vertiports: list[dict[str, Any]],
    agents: list[Agent],
) -> None:
    rows = []
    for vp in vertiports:
        vertiport_id = str(vp["vertiport_id"])
        departures = sum(1 for a in agents if a.origin_vertiport == vertiport_id)
        planned_returns = sum(1 for a in agents if a.destination_vertiport == vertiport_id)
        emergency_diversions = sum(
            1
            for a in agents
            if a.emergency_type
            and str(a.emergency_destination_node) == str(vp["node_id"])
        )
        final_arrivals = planned_returns + emergency_diversions
        net_balance = final_arrivals - departures
        rows.append(
            {
                "vertiport_id": vertiport_id,
                "node_id": vp["node_id"],
                "departing_aircraft": departures,
                "planned_arrivals": planned_returns,
                "emergency_diversions": emergency_diversions,
                "final_arrivals_est": final_arrivals,
                "net_balance_est": net_balance,
                "rebalance_need": "send_aircraft" if net_balance < -1 else ("receive_aircraft" if net_balance > 1 else "balanced"),
                "reserve_aircraft_recommended": max(0, -net_balance),
            }
        )

    df = pd.DataFrame(rows).sort_values(["reserve_aircraft_recommended", "vertiport_id"], ascending=[False, True])
    df.to_csv(OUT_VERTIPORT_DISTRIBUTION, index=False)
    print(f"[INFO] Wrote vertiport distribution CSV: {OUT_VERTIPORT_DISTRIBUTION}")


# ---------------------------------------------------------------------
# Simulation log
# ---------------------------------------------------------------------

def battery_at_time(agent: Agent, t: int) -> float:
    used = 0.0

    for seg in agent.segments:
        t0 = int(seg["t_start"])
        t1 = int(seg["t_end"])

        if t <= t0:
            continue

        elapsed = min(t, t1) - t0
        if elapsed <= 0:
            continue

        duration = max(1, t1 - t0)
        used += float(seg["battery_cost"]) * (elapsed / duration)

    return max(0.0, agent.battery_initial - used)


def position_at_time(
    G: nx.MultiDiGraph,
    agent: Agent,
    t: int,
) -> dict[str, Any]:
    if t < agent.start_time:
        point = node_point(G, agent.origin_node)
        return {
            "phase": "waiting_to_start",
            "node_id": agent.origin_node,
            "from_node": None,
            "to_node": None,
            "route_version": 0,
            "geometry": point,
        }

    last_node = agent.origin_node

    for seg in agent.segments:
        t0 = int(seg["t_start"])
        t1 = int(seg["t_end"])

        if t < t0:
            point = node_point(G, last_node)
            return {
                "phase": "holding_between_segments",
                "node_id": last_node,
                "from_node": None,
                "to_node": None,
                "route_version": seg["route_version"],
                "geometry": point,
            }

        if t0 <= t < t1:
            point = interpolate_segment_position(G, seg, t)

            if seg["kind"] == "wait":
                phase = "holding"
                node_id = seg["u"]
            else:
                phase = "enroute"
                node_id = None

            return {
                "phase": phase,
                "node_id": node_id,
                "from_node": seg["u"],
                "to_node": seg["v"],
                "route_version": seg["route_version"],
                "geometry": point,
            }

        if t == t1:
            last_node = seg["v"]

        if t > t1:
            last_node = seg["v"]

    point = node_point(G, last_node)
    return {
        "phase": "arrived",
        "node_id": last_node,
        "from_node": None,
        "to_node": None,
        "route_version": agent.route_version,
        "geometry": point,
    }


def status_at_time(agent: Agent, t: int) -> str:
    arrival = agent_arrival_time(agent)

    if t < agent.start_time:
        return "waiting"

    if arrival is not None and t >= arrival:
        if agent.emergency_type:
            return "emergency_landed"
        return "arrived"

    if agent.emergency_start_time is not None and t >= agent.emergency_start_time:
        return str(agent.emergency_type)

    return "enroute"


def export_simulation_log(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    graph_crs: str,
) -> None:
    max_arrival = max(
        [agent_arrival_time(a) for a in agents if agent_arrival_time(a) is not None],
        default=PLANNING_HORIZON,
    )
    horizon = min(PLANNING_HORIZON, int(max_arrival) + 5)

    rows = []

    for t in range(0, horizon + 1):
        for agent in agents:
            pos = position_at_time(G, agent, t)
            point = pos["geometry"]

            rows.append(
                {
                    "time_step": t,
                    "time_seconds": t * TIME_STEP_SECONDS,
                    "agent_id": agent.agent_id,
                    "status": status_at_time(agent, t),
                    "phase": pos["phase"],
                    "node_id": pos["node_id"],
                    "from_node": pos["from_node"],
                    "to_node": pos["to_node"],
                    "route_version": pos["route_version"],
                    "battery_remaining": round(battery_at_time(agent, t), 3),
                    "origin_vertiport": agent.origin_vertiport,
                    "destination_vertiport": agent.destination_vertiport,
                    "current_destination_node": agent.current_destination_node,
                    "emergency_type": agent.emergency_type,
                    "mission_id": agent.mission_id,
                    "mission_label": agent.mission_label,
                    "mission_type": agent.mission_type,
                    "geometry": point,
                }
            )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=graph_crs)
    gdf_wgs = gdf.to_crs(EXPORT_CRS)

    gdf_wgs["lon"] = gdf_wgs.geometry.x
    gdf_wgs["lat"] = gdf_wgs.geometry.y

    csv_df = pd.DataFrame(gdf_wgs.drop(columns=["geometry"]))
    csv_df.to_csv(OUT_SIM_LOG, index=False)
    print(f"[INFO] Wrote simulation log CSV: {OUT_SIM_LOG}")

    export_geojson(gdf, OUT_SIM_POSITIONS)


# ---------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------

def export_metrics(
    G: nx.MultiDiGraph,
    agents: list[Agent],
    baseline: dict[str, list[dict[str, Any]]],
) -> None:
    proposed = {a.agent_id: a.segments for a in agents}

    baseline_metrics = metrics_for_mode(
        G=G,
        agents=agents,
        segments_by_agent=baseline,
        mode_name="baseline",
        weather_event_time=WEATHER_EVENT_TIME,
    )

    proposed_metrics = metrics_for_mode(
        G=G,
        agents=agents,
        segments_by_agent=proposed,
        mode_name="proposed",
        weather_event_time=WEATHER_EVENT_TIME,
    )

    baseline_arrivals = {
        aid: arrival_time_from_segments(segs)
        for aid, segs in baseline.items()
    }
    proposed_arrivals = {
        a.agent_id: agent_arrival_time(a)
        for a in agents
    }

    delays = []
    for agent in agents:
        b = baseline_arrivals.get(agent.agent_id)
        p = proposed_arrivals.get(agent.agent_id)
        if b is not None and p is not None:
            delays.append(p - b)

    proposed_metrics["avg_delay_vs_baseline_steps"] = round(float(np.mean(delays)), 2) if delays else np.nan
    baseline_metrics["avg_delay_vs_baseline_steps"] = 0.0

    df = pd.DataFrame([baseline_metrics, proposed_metrics])
    df.to_csv(OUT_METRICS, index=False)
    print(f"[INFO] Wrote metrics summary: {OUT_METRICS}")


def export_alerts(alerts: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(alerts)
    df.to_csv(OUT_ALERTS, index=False)
    print(f"[INFO] Wrote alerts CSV: {OUT_ALERTS}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("[INFO] SkyMesh FAST simulator: greedy reservation planner + capped event replans")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    G = load_and_prepare_graph(INPUT_GRAPHML)
    graph_crs = get_graph_crs(G)

    # Static hard-avoid no-flight zones.
    no_flight_zones = make_synthetic_no_flight_zones(G)
    apply_no_flight_zones_to_graph(G, no_flight_zones)
    export_geojson(no_flight_zones, OUT_NO_FLY_ZONES)

    # Dynamic weather cells: wind = higher energy; rain/snow = hard avoid.
    weather_events = make_dynamic_weather_events(G)
    apply_weather_state_to_graph(G, weather_events, 0)
    export_geojson(weather_events, OUT_WEATHER_EVENTS)

    # 0. Create vertiports.
    vertiports = auto_create_vertiports(G, N_VERTIPORTS)
    vertiports = maybe_add_airport_vertiport(G, vertiports)
    export_vertiports(vertiports, graph_crs, OUT_VERTIPORTS)

    # 1. Generate fleet and missions.
    agents = generate_fleet(vertiports, N_AGENTS, RANDOM_SEED)
    featured_missions = assign_featured_missions(G, agents)
    export_geojson(featured_missions, OUT_FEATURED_MISSIONS)

    # 2. Baseline: shortest path only, no reservation table, no emergency priority.
    baseline_routes = compute_baseline_routes(G, agents)

    # 3. Proposed: constraint-aware routing with reservation table.
    reservation = ReservationTable(G)
    plan_initial_proposed_routes(G, agents, reservation)

    alerts: list[dict[str, Any]] = []

    # 4. Weather changes and emergencies are handled on one timeline.
    emergency_agent_ids: set[str] = set()

    timeline: list[tuple[int, str]] = []
    for t in weather_timeline_times(weather_events):
        timeline.append((int(t), "weather"))
    timeline.append((MACHINE_DISORDER_PREFERRED_TIME, "machine_disorder"))
    timeline.append((MEDICAL_EMERGENCY_PREFERRED_TIME, "medical_emergency"))
    timeline.append((BIRD_STRIKE_PREFERRED_TIME, "bird_strike"))
    timeline = sorted(timeline, key=lambda x: (x[0], x[1]))

    machine_time = None
    machine_agent = None
    medical_time = None
    medical_agent = None
    bird_time = None
    bird_agent = None

    for event_time, event_type in timeline:
        apply_weather_state_to_graph(G, weather_events, event_time)
        active_weather_now = has_active_weather(weather_events, event_time)

        if event_type == "weather":
            process_weather_timeline_event(
                G=G,
                agents=agents,
                reservation=reservation,
                weather_events=weather_events,
                event_time=event_time,
                alerts=alerts,
            )
        elif event_type == "machine_disorder":
            machine_time, machine_agent = process_emergency_event(
                G=G,
                agents=agents,
                reservation=reservation,
                preferred_time=event_time,
                emergency_type="machine_disorder",
                active_weather=active_weather_now,
                excluded_agent_ids=emergency_agent_ids,
                alerts=alerts,
            )
            if machine_agent:
                emergency_agent_ids.add(machine_agent)
        elif event_type == "medical_emergency":
            medical_time, medical_agent = process_emergency_event(
                G=G,
                agents=agents,
                reservation=reservation,
                preferred_time=event_time,
                emergency_type="medical_emergency",
                active_weather=active_weather_now,
                excluded_agent_ids=emergency_agent_ids,
                alerts=alerts,
            )
            if medical_agent:
                emergency_agent_ids.add(medical_agent)
        elif event_type == "bird_strike":
            bird_time, bird_agent = process_emergency_event(
                G=G,
                agents=agents,
                reservation=reservation,
                preferred_time=event_time,
                emergency_type="bird_strike",
                active_weather=active_weather_now,
                excluded_agent_ids=emergency_agent_ids,
                alerts=alerts,
            )
            if bird_agent:
                emergency_agent_ids.add(bird_agent)

    # 5. Export dashboard-ready artifacts.
    export_agents_csv(agents)
    export_vertiport_distribution(vertiports, agents)
    export_routes(baseline_routes, agents, graph_crs)
    export_simulation_log(G, agents, graph_crs)
    export_metrics(G, agents, baseline_routes)
    export_alerts(alerts)

    print("\n[SUMMARY]")
    print(f"Agents: {len(agents)}")
    print(f"Vertiports: {len(vertiports)}")
    print(f"Weather events: {len(weather_events)}")
    print(f"No-flight zones: {len(no_flight_zones)}")
    print(f"Machine disorder event: time={machine_time}, agent={machine_agent}")
    print(f"Medical emergency event: time={medical_time}, agent={medical_agent}")
    print(f"Bird strike event: time={bird_time}, agent={bird_agent}")
    print(f"Output folder: {OUT_DIR.resolve()}")

    print("\n[NEXT]")
    print("Use simulation_log.csv for aircraft positions, battery levels, and animation.")
    print("Use proposed_routes.geojson for planned routes.")
    print("Use weather_events.geojson and no_flight_zones.geojson for dynamic overlays.")
    print("Use alerts.csv for warning panels and emergency highlights.")
    print("Use metrics_summary.csv for baseline vs proposed comparison.")


if __name__ == "__main__":
    main()
