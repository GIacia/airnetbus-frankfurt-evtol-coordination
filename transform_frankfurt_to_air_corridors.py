#!/usr/bin/env python3
"""
Transform a Frankfurt OSMnx road graph into a simplified eVTOL air corridor graph.

Input:
    outputs/frankfurt_evtol.graphml
    outputs/frankfurt_residential_zones.geojson
    outputs/frankfurt_emergency_landing_candidates.geojson
    outputs/frankfurt_weather_zone.geojson

Output:
    outputs_air/frankfurt_air_corridors.graphml
    outputs_air/frankfurt_air_nodes.geojson
    outputs_air/frankfurt_air_edges.geojson
    outputs_air/frankfurt_air_emergency_sites.geojson
    outputs_air/frankfurt_air_residential_zones.geojson
    outputs_air/frankfurt_air_weather_zone.geojson

Core idea:
    - Use the OSMnx graph as a city geometry source.
    - Keep sparse waypoint candidates from:
        major road corridors
        sampled graph nodes
        optional rivers / waterway corridors
        optional parks / open spaces
        emergency landing candidates
    - Build a directed k-nearest-neighbor air corridor graph.
    - Assign explicit edge costs:
        distance
        battery_cost
        noise_penalty
        weather_risk
        total_cost
    - Add multi-agent planning attributes:
        travel_time_steps
        capacity
        altitude_layer
        reservation_edge_id
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PLACE = os.getenv("SKYMESH_PLACE", "Frankfurt am Main, Hesse, Germany")
CITY_NAME = os.getenv("SKYMESH_CITY_NAME", "Frankfurt")
CITY_SLUG = os.getenv("SKYMESH_CITY_SLUG", "frankfurt")

INPUT_GRAPHML = Path(f"outputs/{CITY_SLUG}_evtol.graphml")

INPUT_RESIDENTIAL = Path(f"outputs/{CITY_SLUG}_residential_zones.geojson")
INPUT_EMERGENCY = Path(f"outputs/{CITY_SLUG}_emergency_landing_candidates.geojson")
INPUT_WEATHER = Path(f"outputs/{CITY_SLUG}_weather_zone.geojson")

OUT_DIR = Path("outputs_air")

OUT_GRAPHML = OUT_DIR / f"{CITY_SLUG}_air_corridors.graphml"
OUT_NODES = OUT_DIR / f"{CITY_SLUG}_air_nodes.geojson"
OUT_EDGES = OUT_DIR / f"{CITY_SLUG}_air_edges.geojson"
OUT_EMERGENCY = OUT_DIR / f"{CITY_SLUG}_air_emergency_sites.geojson"
OUT_RESIDENTIAL = OUT_DIR / f"{CITY_SLUG}_air_residential_zones.geojson"
OUT_WEATHER = OUT_DIR / f"{CITY_SLUG}_air_weather_zone.geojson"


# ---------------------------------------------------------------------
# Hackathon model configuration
# ---------------------------------------------------------------------

# Set this to False if Overpass / OSM feature downloads are too slow.
FETCH_OPTIONAL_RIVERS_AND_PARKS = True

# Waypoint simplification
GRID_CELL_M = 850
MAX_WAYPOINTS = 450
MAX_EMERGENCY_SITES = 80

# Sampling density
MAJOR_ROAD_SAMPLE_EVERY_M = 1_200
RIVER_SAMPLE_EVERY_M = 1_500

# Corridor graph construction
K_NEAREST = 6
MAX_CORRIDOR_LENGTH_M = 3_800
FORCE_WEAK_CONNECTIVITY = True

# Simulation time model
TIME_STEP_SECONDS = 60
CRUISE_DISTANCE_PER_STEP_M = 1_800
# 1,800 m/min = 108 km/h. This is a simplified planning speed.

# Altitude layering
NUM_ALTITUDE_LAYERS = 2
DEFAULT_CAPACITY = 1

# Cost model
BASE_ENERGY_PER_METER = 0.001
WEATHER_ENERGY_MULTIPLIER = 1.35

NOISE_PENALTY_RESIDENTIAL = 1_200.0
WEATHER_RISK_PENALTY = 2_500.0

ALPHA_DISTANCE = 1.0
BETA_BATTERY = 500.0
GAMMA_NOISE = 1.0
DELTA_WEATHER = 1.0

# Residential policy
# False = soft constraint with penalty.
# True = remove residential-crossing air corridors unless they are emergency connectors.
DROP_RESIDENTIAL_EDGES = False

EXPORT_CRS = "EPSG:4326"


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def setup_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.requests_timeout = 180


def empty_gdf(crs: str | Any = EXPORT_CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def safe_union(gdf: gpd.GeoDataFrame) -> BaseGeometry | None:
    if gdf.empty:
        return None
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


def read_layer_or_empty(path: Path, crs: str | Any = EXPORT_CRS) -> gpd.GeoDataFrame:
    if not path.exists():
        print(f"[WARN] Missing layer: {path}")
        return empty_gdf(crs)

    gdf = gpd.read_file(path)
    if gdf.empty:
        return empty_gdf(crs)

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)

    return gdf


def project_layer(gdf: gpd.GeoDataFrame, target_crs: Any) -> gpd.GeoDataFrame:
    if gdf.empty:
        return empty_gdf(target_crs)

    if gdf.crs is None:
        gdf = gdf.set_crs(EXPORT_CRS)

    return gdf.to_crs(target_crs)


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

    if out.crs is None:
        out = out.set_crs(EXPORT_CRS)

    out = out.to_crs(EXPORT_CRS).reset_index()

    for col in out.columns:
        if col == "geometry":
            continue
        if out[col].dtype == "object":
            out[col] = out[col].map(json_safe)

    out.to_file(path, driver="GeoJSON")
    print(f"[INFO] Wrote GeoJSON: {path}")


def json_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    return value


# ---------------------------------------------------------------------
# Graph loading and projection
# ---------------------------------------------------------------------

def load_projected_graph(graphml_path: Path) -> nx.MultiDiGraph:
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"Input graph not found: {graphml_path}. "
            "Run the city extraction script first."
        )

    print(f"[INFO] Loading graph: {graphml_path}")
    G = ox.io.load_graphml(graphml_path)

    nodes, _ = ox.convert.graph_to_gdfs(G, nodes=True, edges=True)

    if nodes.crs is not None and nodes.crs.is_projected:
        print(f"[INFO] Graph already projected: {nodes.crs}")
        return G

    print("[INFO] Projecting graph to local metric CRS...")
    return ox.projection.project_graph(G)


# ---------------------------------------------------------------------
# OSM feature fetching
# ---------------------------------------------------------------------

def fetch_osm_features(place: str, tags: dict[str, Any], label: str) -> gpd.GeoDataFrame:
    print(f"[INFO] Fetching optional OSM features: {label}")
    try:
        gdf = ox.features.features_from_place(place, tags=tags)

        if gdf.empty:
            print(f"[WARN] No features found for {label}")
            return empty_gdf()

        if gdf.crs is None:
            gdf = gdf.set_crs(EXPORT_CRS)

        print(f"[INFO] Fetched {len(gdf):,} {label} features")
        return gdf

    except Exception as exc:
        print(f"[WARN] Failed to fetch {label}: {exc}")
        return empty_gdf()


# ---------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------

MAJOR_HIGHWAY_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}

def parse_osm_tag_values(value: Any) -> list[str]:
    """
    Robustly parse OSM tag values.

    OSMnx / GraphML can store tags like highway as:
    - "primary"
    - ["primary", "secondary"]
    - np.array(["primary", "secondary"])
    - "['primary', 'secondary']"
    - None / NaN / pd.NA
    """

    if value is None:
        return []

    # Important: handle array/list-like values BEFORE calling pd.isna().
    if isinstance(value, np.ndarray):
        return parse_osm_tag_values(value.tolist())

    if isinstance(value, (list, tuple, set)):
        parsed_values: list[str] = []
        for item in value:
            parsed_values.extend(parse_osm_tag_values(item))
        return parsed_values

    if isinstance(value, dict):
        parsed_values: list[str] = []
        for item in value.values():
            parsed_values.extend(parse_osm_tag_values(item))
        return parsed_values

    # Now it is safe to check scalar missing values.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return []
    except Exception:
        pass

    if isinstance(value, str):
        text = value.strip()

        if text == "" or text.lower() in {"nan", "none", "<na>"}:
            return []

        # GraphML often stores Python lists as strings:
        # "['primary', 'secondary']"
        if (
            (text.startswith("[") and text.endswith("]"))
            or (text.startswith("(") and text.endswith(")"))
        ):
            try:
                parsed = ast.literal_eval(text)
                return parse_osm_tag_values(parsed)
            except Exception:
                pass

        # Some OSM tags are semicolon-separated.
        return [
            part.strip().lower()
            for part in text.split(";")
            if part.strip()
        ]

    return [str(value).strip().lower()]


def is_major_highway(value: Any) -> bool:
    parsed_values = parse_osm_tag_values(value)
    return any(v in MAJOR_HIGHWAY_TYPES for v in parsed_values)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def iter_line_parts(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            yield part
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_line_parts(part)


def sample_line_points(geom: BaseGeometry, every_m: float) -> list[Point]:
    points: list[Point] = []

    for line in iter_line_parts(geom):
        if line.length <= 0:
            continue

        distances = list(np.arange(0, line.length, every_m))
        distances.append(line.length)

        for distance in distances:
            points.append(line.interpolate(float(distance)))

    return points


def representative_point_from_geometry(geom: BaseGeometry) -> Point | None:
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Point":
        return geom

    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)[0]

    # Works for polygons, lines, and geometry collections.
    return geom.representative_point()


def make_synthetic_weather_zone(nodes_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = nodes_proj.total_bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    width = maxx - minx
    height = maxy - miny

    weather_geom = box(
        cx - 0.12 * width,
        cy - 0.08 * height,
        cx + 0.16 * width,
        cy + 0.10 * height,
    )

    return gpd.GeoDataFrame(
        {
            "name": ["synthetic_weather_cell"],
            "weather_type": ["storm_or_high_wind"],
            "weather_risk_penalty": [WEATHER_RISK_PENALTY],
            "energy_multiplier": [WEATHER_ENERGY_MULTIPLIER],
        },
        geometry=[weather_geom],
        crs=nodes_proj.crs,
    )


# ---------------------------------------------------------------------
# Emergency-site classification
# ---------------------------------------------------------------------

def classify_emergency_candidate(row: pd.Series) -> str:
    def norm(value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        return str(value).lower()

    aeroway = norm(row.get("aeroway"))
    amenity = norm(row.get("amenity"))
    leisure = norm(row.get("leisure"))
    landuse = norm(row.get("landuse"))
    natural = norm(row.get("natural"))
    existing_type = norm(row.get("candidate_type"))

    if existing_type:
        return existing_type
    if aeroway in {"helipad", "heliport"}:
        return "helipad_or_heliport"
    if amenity in {"hospital", "clinic"}:
        return "medical_facility"
    if leisure in {"park", "pitch", "sports_centre", "recreation_ground"}:
        return "park_or_recreation_area"
    if landuse in {"grass", "meadow", "recreation_ground", "village_green"}:
        return "open_space"
    if natural in {"grassland", "heath", "scrub"}:
        return "natural_open_area"

    return "other_candidate"


def emergency_priority(candidate_type: str) -> int:
    order = {
        "helipad_or_heliport": 0,
        "medical_facility": 1,
        "park_or_recreation_area": 2,
        "open_space": 3,
        "natural_open_area": 4,
        "other_candidate": 5,
    }
    return order.get(candidate_type, 5)


# ---------------------------------------------------------------------
# Waypoint candidate generation
# ---------------------------------------------------------------------

def make_major_road_waypoints(edges_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "highway" not in edges_proj.columns:
        print("[WARN] Edge column 'highway' missing. Using all edges as corridor skeleton.")
        major_edges = edges_proj.copy()
    else:
        mask = edges_proj["highway"].apply(is_major_highway).fillna(False).astype(bool)
        major_edges = edges_proj.loc[mask].copy()

        if major_edges.empty:
            print("[WARN] No major highway edges found. Falling back to all edges.")
            major_edges = edges_proj.copy()

    rows: list[dict[str, Any]] = []

    for idx, row in major_edges.iterrows():
        points = sample_line_points(row.geometry, MAJOR_ROAD_SAMPLE_EVERY_M)
        for point in points:
            rows.append(
                {
                    "source": "major_road_corridor",
                    "node_type": "waypoint",
                    "priority": 2,
                    "is_emergency_site": False,
                    "emergency_type": None,
                    "source_id": str(idx),
                    "geometry": point,
                }
            )

    print(f"[INFO] Major-road waypoint candidates: {len(rows):,}")

    if not rows:
        return empty_gdf(edges_proj.crs)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=edges_proj.crs)

def make_sampled_node_waypoints(nodes_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []

    for idx, row in nodes_proj.iterrows():
        point = representative_point_from_geometry(row.geometry)
        if point is None:
            continue

        rows.append(
            {
                "source": "sampled_osm_node",
                "node_type": "waypoint",
                "priority": 5,
                "is_emergency_site": False,
                "emergency_type": None,
                "source_id": str(idx),
                "geometry": point,
            }
        )

    print(f"[INFO] Raw sampled-node waypoint candidates: {len(rows):,}")

    if not rows:
        return empty_gdf(nodes_proj.crs)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=nodes_proj.crs)


def make_river_waypoints(rivers_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if rivers_proj.empty:
        return empty_gdf(rivers_proj.crs)

    rows: list[dict[str, Any]] = []

    for idx, row in rivers_proj.iterrows():
        points = sample_line_points(row.geometry, RIVER_SAMPLE_EVERY_M)
        for point in points:
            rows.append(
                {
                    "source": "river_or_waterway_corridor",
                    "node_type": "waypoint",
                    "priority": 1,
                    "is_emergency_site": False,
                    "emergency_type": None,
                    "source_id": str(idx),
                    "geometry": point,
                }
            )

    print(f"[INFO] River/waterway waypoint candidates: {len(rows):,}")

    if not rows:
        return empty_gdf(rivers_proj.crs)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=rivers_proj.crs)


def make_park_waypoints(parks_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if parks_proj.empty:
        return empty_gdf(parks_proj.crs)

    rows: list[dict[str, Any]] = []

    for idx, row in parks_proj.iterrows():
        point = representative_point_from_geometry(row.geometry)
        if point is None:
            continue

        rows.append(
            {
                "source": "park_or_open_space",
                "node_type": "waypoint",
                "priority": 1,
                "is_emergency_site": False,
                "emergency_type": None,
                "source_id": str(idx),
                "geometry": point,
            }
        )

    print(f"[INFO] Park/open-space waypoint candidates: {len(rows):,}")

    if not rows:
        return empty_gdf(parks_proj.crs)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=parks_proj.crs)


def make_emergency_waypoints(emergency_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if emergency_proj.empty:
        return empty_gdf(emergency_proj.crs)

    rows: list[dict[str, Any]] = []

    for idx, row in emergency_proj.iterrows():
        point = representative_point_from_geometry(row.geometry)
        if point is None:
            continue

        candidate_type = classify_emergency_candidate(row)

        rows.append(
            {
                "source": "emergency_landing_candidate",
                "node_type": "emergency_site",
                "priority": 0,
                "is_emergency_site": True,
                "emergency_type": candidate_type,
                "emergency_rank": emergency_priority(candidate_type),
                "source_id": str(idx),
                "geometry": point,
            }
        )

    if not rows:
        return empty_gdf(emergency_proj.crs)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=emergency_proj.crs)
    gdf = gdf.sort_values(["emergency_rank", "source_id"]).head(MAX_EMERGENCY_SITES)

    print(f"[INFO] Emergency waypoint candidates kept: {len(gdf):,}")
    return gdf


def grid_reduce_points(
    gdf: gpd.GeoDataFrame,
    cell_size_m: float,
    max_points: int,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    reduced = gdf.copy()
    reduced = reduced[reduced.geometry.notna() & ~reduced.geometry.is_empty].copy()

    reduced["x_tmp"] = reduced.geometry.x
    reduced["y_tmp"] = reduced.geometry.y

    reduced["grid_x"] = np.floor(reduced["x_tmp"] / cell_size_m).astype("int64")
    reduced["grid_y"] = np.floor(reduced["y_tmp"] / cell_size_m).astype("int64")

    # Emergency sites and high-quality corridor sources survive first.
    if "emergency_rank" not in reduced.columns:
        reduced["emergency_rank"] = 999

    reduced = reduced.sort_values(
        ["priority", "emergency_rank", "source", "source_id"],
        ascending=True,
    )

    reduced = reduced.drop_duplicates(["grid_x", "grid_y"], keep="first")

    if len(reduced) > max_points:
        must_keep = reduced[reduced["is_emergency_site"] == True].copy()
        rest = reduced[reduced["is_emergency_site"] != True].copy()

        remaining = max(0, max_points - len(must_keep))
        rest = rest.sort_values(["priority", "source_id"]).head(remaining)

        reduced = pd.concat([must_keep, rest], ignore_index=True)
        reduced = gpd.GeoDataFrame(reduced, geometry="geometry", crs=gdf.crs)

    reduced = reduced.drop(columns=["x_tmp", "y_tmp", "grid_x", "grid_y"], errors="ignore")

    print(f"[INFO] Waypoints after grid reduction: {len(reduced):,}")
    return reduced


def finalize_air_nodes(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    nodes = candidates.copy().reset_index(drop=True)

    nodes["node_id"] = [f"air_{i:04d}" for i in range(len(nodes))]
    nodes["x"] = nodes.geometry.x
    nodes["y"] = nodes.geometry.y

    nodes_wgs = nodes.to_crs(EXPORT_CRS)
    nodes["lon"] = nodes_wgs.geometry.x
    nodes["lat"] = nodes_wgs.geometry.y

    nodes["preferred_altitude_layer"] = 1
    nodes["reservation_node_id"] = nodes["node_id"]

    keep_cols = [
        "node_id",
        "x",
        "y",
        "lon",
        "lat",
        "node_type",
        "source",
        "source_id",
        "priority",
        "is_emergency_site",
        "emergency_type",
        "preferred_altitude_layer",
        "reservation_node_id",
        "geometry",
    ]

    for col in keep_cols:
        if col not in nodes.columns:
            nodes[col] = None

    nodes = nodes[keep_cols].copy()
    nodes = nodes.set_index("node_id")
    nodes.index.name = "osmid"

    return nodes


# ---------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------

def altitude_layer_for_direction(p1: Point, p2: Point) -> int:
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    sector_width = 360.0 / NUM_ALTITUDE_LAYERS

    return int(angle // sector_width) + 1


def make_edge_record(
    u: str,
    v: str,
    nodes_air: gpd.GeoDataFrame,
    residential_geom: BaseGeometry | None,
    weather_geom: BaseGeometry | None,
    edge_kind: str = "air_corridor",
) -> dict[str, Any] | None:
    p1 = nodes_air.loc[u].geometry
    p2 = nodes_air.loc[v].geometry

    if p1.equals(p2):
        return None

    line = LineString([p1, p2])
    distance = float(line.length)

    if distance <= 0:
        return None

    over_residential = False
    if residential_geom is not None and not residential_geom.is_empty:
        over_residential = bool(line.intersects(residential_geom))

    inside_weather_zone = False
    if weather_geom is not None and not weather_geom.is_empty:
        inside_weather_zone = bool(line.intersects(weather_geom))

    is_emergency_connector = bool(
        nodes_air.loc[u]["is_emergency_site"] or nodes_air.loc[v]["is_emergency_site"]
    )

    if DROP_RESIDENTIAL_EDGES and over_residential and not is_emergency_connector:
        return None

    altitude_layer = altitude_layer_for_direction(p1, p2)

    weather_multiplier = WEATHER_ENERGY_MULTIPLIER if inside_weather_zone else 1.0

    battery_cost = distance * BASE_ENERGY_PER_METER * weather_multiplier
    noise_penalty = NOISE_PENALTY_RESIDENTIAL if over_residential else 0.0
    weather_risk = WEATHER_RISK_PENALTY if inside_weather_zone else 0.0

    total_cost = (
        ALPHA_DISTANCE * distance
        + BETA_BATTERY * battery_cost
        + GAMMA_NOISE * noise_penalty
        + DELTA_WEATHER * weather_risk
    )

    travel_time_steps = max(1, int(math.ceil(distance / CRUISE_DISTANCE_PER_STEP_M)))

    reservation_edge_id = f"{u}->{v}@L{altitude_layer}"

    return {
        "u": u,
        "v": v,
        "key": 0,
        "edge_kind": edge_kind,
        "distance": distance,
        "battery_cost": float(battery_cost),
        "noise_penalty": float(noise_penalty),
        "weather_risk": float(weather_risk),
        "total_cost": float(total_cost),
        "over_residential": bool(over_residential),
        "inside_weather_zone": bool(inside_weather_zone),
        "altitude_layer": int(altitude_layer),
        "capacity": int(DEFAULT_CAPACITY),
        "travel_time_steps": int(travel_time_steps),
        "reservation_edge_id": reservation_edge_id,
        "geometry": line,
    }


def build_knn_directed_edges(
    nodes_air: gpd.GeoDataFrame,
    residential_geom: BaseGeometry | None,
    weather_geom: BaseGeometry | None,
) -> list[dict[str, Any]]:
    ids = list(nodes_air.index)
    coords = nodes_air[["x", "y"]].to_numpy(dtype=float)

    pair_seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []

    for i, u in enumerate(ids):
        delta = coords - coords[i]
        dists = np.sqrt((delta ** 2).sum(axis=1))

        order = np.argsort(dists)

        selected = 0

        for j in order:
            if i == j:
                continue

            distance = dists[j]
            if distance > MAX_CORRIDOR_LENGTH_M:
                continue

            v = ids[j]
            pair = tuple(sorted((u, v)))

            if pair in pair_seen:
                continue

            pair_seen.add(pair)

            # Add both directed versions. Direction still matters because altitude_layer differs.
            rec_uv = make_edge_record(
                u,
                v,
                nodes_air,
                residential_geom,
                weather_geom,
                edge_kind="air_corridor",
            )
            rec_vu = make_edge_record(
                v,
                u,
                nodes_air,
                residential_geom,
                weather_geom,
                edge_kind="air_corridor",
            )

            if rec_uv is not None:
                records.append(rec_uv)
            if rec_vu is not None:
                records.append(rec_vu)

            selected += 1

            if selected >= K_NEAREST:
                break

    print(f"[INFO] Initial directed air corridor edges: {len(records):,}")
    return records


def weak_components_from_records(
    nodes_air: gpd.GeoDataFrame,
    records: list[dict[str, Any]],
) -> list[set[str]]:
    G_tmp = nx.MultiDiGraph()
    G_tmp.add_nodes_from(nodes_air.index)

    for rec in records:
        G_tmp.add_edge(rec["u"], rec["v"])

    return [set(c) for c in nx.weakly_connected_components(G_tmp)]


def nearest_pair_between_components(
    nodes_air: gpd.GeoDataFrame,
    comp_a: set[str],
    comp_b: set[str],
) -> tuple[str, str]:
    ids_a = list(comp_a)
    ids_b = list(comp_b)

    coords_a = nodes_air.loc[ids_a, ["x", "y"]].to_numpy(dtype=float)
    coords_b = nodes_air.loc[ids_b, ["x", "y"]].to_numpy(dtype=float)

    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dist2 = (diff ** 2).sum(axis=2)

    idx_a, idx_b = np.unravel_index(np.argmin(dist2), dist2.shape)

    return ids_a[idx_a], ids_b[idx_b]


def ensure_weak_connectivity(
    nodes_air: gpd.GeoDataFrame,
    records: list[dict[str, Any]],
    residential_geom: BaseGeometry | None,
    weather_geom: BaseGeometry | None,
) -> list[dict[str, Any]]:
    if not FORCE_WEAK_CONNECTIVITY:
        return records

    max_iterations = len(nodes_air)

    for _ in range(max_iterations):
        components = weak_components_from_records(nodes_air, records)
        components = sorted(components, key=len, reverse=True)

        if len(components) <= 1:
            print("[INFO] Air corridor graph is weakly connected.")
            return records

        main_component = components[0]
        next_component = components[1]

        u, v = nearest_pair_between_components(nodes_air, main_component, next_component)

        rec_uv = make_edge_record(
            u,
            v,
            nodes_air,
            residential_geom,
            weather_geom,
            edge_kind="connectivity_bridge",
        )
        rec_vu = make_edge_record(
            v,
            u,
            nodes_air,
            residential_geom,
            weather_geom,
            edge_kind="connectivity_bridge",
        )

        if rec_uv is not None:
            records.append(rec_uv)
        if rec_vu is not None:
            records.append(rec_vu)

        print(f"[INFO] Added connectivity bridge between {u} and {v}")

    print("[WARN] Could not fully connect graph within iteration limit.")
    return records


def assign_edge_keys(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}

    for rec in records:
        pair = (rec["u"], rec["v"])
        key = counts.get(pair, 0)
        rec["key"] = key
        counts[pair] = key + 1

    return records


def edge_records_to_gdf(
    records: list[dict[str, Any]],
    crs: Any,
) -> gpd.GeoDataFrame:
    if not records:
        raise ValueError("No edge records were created. Increase MAX_CORRIDOR_LENGTH_M or K_NEAREST.")

    records = assign_edge_keys(records)

    edges = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    edges = edges.set_index(["u", "v", "key"])
    return edges


# ---------------------------------------------------------------------
# Optional feature layers: rivers and parks
# ---------------------------------------------------------------------

def load_optional_rivers_and_parks(target_crs: Any) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not FETCH_OPTIONAL_RIVERS_AND_PARKS:
        return empty_gdf(target_crs), empty_gdf(target_crs)

    river_tags = {
        "waterway": ["river", "canal"],
        "water": ["river", "canal"],
    }

    park_tags = {
        "leisure": ["park", "recreation_ground"],
        "landuse": ["grass", "meadow", "recreation_ground", "village_green"],
        "natural": ["grassland"],
    }

    rivers = fetch_osm_features(PLACE, river_tags, "rivers and waterways")
    parks = fetch_osm_features(PLACE, park_tags, "parks and open spaces")

    rivers_proj = project_layer(rivers, target_crs)
    parks_proj = project_layer(parks, target_crs)

    return rivers_proj, parks_proj


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    setup_osmnx()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load original city OSMnx graph.
    G_original = load_projected_graph(INPUT_GRAPHML)
    nodes_proj, edges_proj = ox.convert.graph_to_gdfs(G_original, nodes=True, edges=True)

    print(f"[INFO] Original projected graph: {len(nodes_proj):,} nodes, {len(edges_proj):,} edges")
    print(f"[INFO] Working CRS: {nodes_proj.crs}")

    # 2. Load residential, emergency, and weather layers from previous script.
    residential = read_layer_or_empty(INPUT_RESIDENTIAL)
    emergency = read_layer_or_empty(INPUT_EMERGENCY)
    weather = read_layer_or_empty(INPUT_WEATHER)

    residential_proj = project_layer(residential, nodes_proj.crs)
    emergency_proj = project_layer(emergency, nodes_proj.crs)

    if weather.empty:
        print("[WARN] Weather layer missing. Creating synthetic weather zone.")
        weather_proj = make_synthetic_weather_zone(nodes_proj)
    else:
        weather_proj = project_layer(weather, nodes_proj.crs)

    residential_geom = safe_union(residential_proj)
    weather_geom = safe_union(weather_proj)

    # 3. Optional river and park features.
    rivers_proj, parks_proj = load_optional_rivers_and_parks(nodes_proj.crs)

    # 4. Build waypoint candidates.
    major_road_points = make_major_road_waypoints(edges_proj)
    sampled_node_points = make_sampled_node_waypoints(nodes_proj)
    river_points = make_river_waypoints(rivers_proj)
    park_points = make_park_waypoints(parks_proj)
    emergency_points = make_emergency_waypoints(emergency_proj)

    candidate_layers = [
        major_road_points,
        sampled_node_points,
        river_points,
        park_points,
        emergency_points,
    ]

    non_empty_layers = [gdf for gdf in candidate_layers if not gdf.empty]

    if not non_empty_layers:
        raise ValueError("No waypoint candidates created.")

    candidates = pd.concat(non_empty_layers, ignore_index=True)
    candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs=nodes_proj.crs)

    candidates = grid_reduce_points(
        candidates,
        cell_size_m=GRID_CELL_M,
        max_points=MAX_WAYPOINTS,
    )

    nodes_air = finalize_air_nodes(candidates)

    print(f"[INFO] Final air nodes: {len(nodes_air):,}")
    print(f"[INFO] Emergency nodes: {int(nodes_air['is_emergency_site'].sum()):,}")

    # 5. Build directed air corridor edges.
    edge_records = build_knn_directed_edges(
        nodes_air,
        residential_geom=residential_geom,
        weather_geom=weather_geom,
    )

    edge_records = ensure_weak_connectivity(
        nodes_air,
        edge_records,
        residential_geom=residential_geom,
        weather_geom=weather_geom,
    )

    edges_air = edge_records_to_gdf(edge_records, nodes_air.crs)

    print(f"[INFO] Final air edges: {len(edges_air):,}")

    # 6. Build NetworkX MultiDiGraph.
    graph_attrs = {
        "crs": nodes_air.crs.to_string(),
        "name": f"{CITY_NAME} simplified eVTOL air corridor graph",
        "place": PLACE,
        "model_note": (
            "Simplified hackathon model. Uses sparse waypoints and directed "
            "straight-line air corridors. Not a physical aviation simulator."
        ),
        "time_step_seconds": TIME_STEP_SECONDS,
        "cruise_distance_per_step_m": CRUISE_DISTANCE_PER_STEP_M,
        "num_altitude_layers": NUM_ALTITUDE_LAYERS,
    }

    G_air = ox.convert.graph_from_gdfs(
        nodes_air,
        edges_air,
        graph_attrs=graph_attrs,
    )

    # 7. Quick routing sanity check.
    node_ids = list(G_air.nodes)
    if len(node_ids) >= 2:
        origin = node_ids[0]
        destination = node_ids[-1]

        try:
            route = nx.shortest_path(
                G_air,
                origin,
                destination,
                weight="total_cost",
            )
            print(
                f"[INFO] Example weighted route: "
                f"{origin} -> {destination}, nodes={len(route)}"
            )
        except nx.NetworkXNoPath:
            print("[WARN] No route found in sanity check.")

    # 8. Export.
    print(f"[INFO] Saving air corridor GraphML: {OUT_GRAPHML}")
    ox.io.save_graphml(G_air, filepath=OUT_GRAPHML)

    export_geojson(nodes_air, OUT_NODES)
    export_geojson(edges_air, OUT_EDGES)
    export_geojson(nodes_air[nodes_air["is_emergency_site"] == True], OUT_EMERGENCY)
    export_geojson(residential_proj, OUT_RESIDENTIAL)
    export_geojson(weather_proj, OUT_WEATHER)

    # 9. Summary.
    print("\n[SUMMARY]")
    print(f"Air nodes: {len(nodes_air):,}")
    print(f"Air edges: {len(edges_air):,}")
    print(f"Emergency nodes: {int(nodes_air['is_emergency_site'].sum()):,}")
    print(f"Residential-crossing edges: {int(edges_air['over_residential'].sum()):,}")
    print(f"Weather-risk edges: {int(edges_air['inside_weather_zone'].sum()):,}")
    print(f"Mean edge distance: {edges_air['distance'].mean():.1f} m")
    print(f"Median edge distance: {edges_air['distance'].median():.1f} m")
    print(f"Output folder: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
