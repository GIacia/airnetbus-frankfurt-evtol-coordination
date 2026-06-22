#!/usr/bin/env python3
"""
Build a simplified Frankfurt eVTOL routing graph for a science hackathon MVP.

Modeling assumption:
- We do NOT build a realistic aviation simulator.
- We use Frankfurt's street network as a simplified "air corridor skeleton".
- Each graph edge is a possible eVTOL corridor.
- Constraints are explicit edge costs:
    distance
    battery_cost
    noise_penalty
    weather_risk
    total_cost
- Residential zones increase noise_penalty.
- A synthetic weather polygon increases weather_risk and battery_cost.
- Hospitals, parks, open spaces, and helipads are exported as emergency landing candidates.

Outputs:
    outputs/frankfurt_evtol.graphml
    outputs/frankfurt_nodes.geojson
    outputs/frankfurt_edges.geojson
    outputs/frankfurt_residential_zones.geojson
    outputs/frankfurt_emergency_landing_candidates.geojson
    outputs/frankfurt_weather_zone.geojson
    outputs/frankfurt_city_boundary.geojson
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PLACE = os.getenv("SKYMESH_PLACE", "Frankfurt am Main, Hesse, Germany")
CITY_NAME = os.getenv("SKYMESH_CITY_NAME", "Frankfurt")
CITY_SLUG = os.getenv("SKYMESH_CITY_SLUG", "frankfurt")

# "drive" is smaller and usually enough for a hackathon MVP.
# Use "all" if you want a denser corridor skeleton.
NETWORK_TYPE = "drive"

OUT_DIR = Path("outputs")

# For an air-corridor skeleton, car one-way rules are not meaningful.
# This asks OSMnx to create a fully bidirectional graph for this network type.
MAKE_BIDIRECTIONAL_AIR_CORRIDORS = True

# Cost model: arbitrary but explainable.
# Think of total_cost as "meters-equivalent generalized routing cost".
BASE_ENERGY_PER_METER = 0.001
WEATHER_ENERGY_MULTIPLIER = 1.35

NOISE_PENALTY_RESIDENTIAL = 1500.0   # meters-equivalent penalty
WEATHER_RISK_PENALTY = 3000.0        # meters-equivalent penalty

ALPHA_DISTANCE = 1.0
BETA_BATTERY = 500.0
GAMMA_NOISE = 1.0
DELTA_WEATHER = 1.0

# For the first MVP, keep weather as a soft constraint.
# Set True if you want weather-affected edges to become nearly unusable.
APPLY_HARD_WEATHER_BLOCK = False
BLOCKED_EDGE_COST = 1e9

EXPORT_CRS = "EPSG:4326"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def setup_osmnx() -> None:
    """Configure OSMnx for repeatable hackathon work."""
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.requests_timeout = 180

    if MAKE_BIDIRECTIONAL_AIR_CORRIDORS:
        if NETWORK_TYPE not in ox.settings.bidirectional_network_types:
            ox.settings.bidirectional_network_types.append(NETWORK_TYPE)


def empty_gdf(crs: str = EXPORT_CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def fetch_osm_features(place: str, tags: dict[str, Any], label: str) -> gpd.GeoDataFrame:
    """
    Fetch OSM features. If Overpass fails or returns nothing, return an empty GeoDataFrame.
    """
    print(f"[INFO] Fetching {label} from OSM...")
    try:
        gdf = ox.features.features_from_place(place, tags=tags)
        if gdf.empty:
            print(f"[WARN] No {label} features found.")
            return empty_gdf()

        if gdf.crs is None:
            gdf = gdf.set_crs(EXPORT_CRS)

        print(f"[INFO] Fetched {len(gdf):,} {label} features.")
        return gdf

    except Exception as exc:
        print(f"[WARN] Failed to fetch {label}: {exc}")
        return empty_gdf()


def keep_polygon_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only Polygon / MultiPolygon geometries."""
    if gdf.empty:
        return gdf

    valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    valid = valid[valid.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return valid


def keep_landing_candidate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep geometries that are reasonable as landing candidate features."""
    if gdf.empty:
        return gdf

    valid_types = ["Point", "MultiPoint", "Polygon", "MultiPolygon"]
    valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    valid = valid[valid.geometry.geom_type.isin(valid_types)].copy()
    return valid


def union_geometries(gdf: gpd.GeoDataFrame) -> BaseGeometry | None:
    """Return a single union geometry. Compatible with older/newer GeoPandas."""
    if gdf.empty:
        return None

    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


def classify_emergency_candidate(row: pd.Series) -> str:
    """Assign a simple class label for emergency landing candidates."""
    def norm(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).lower()

    aeroway = norm(row.get("aeroway"))
    amenity = norm(row.get("amenity"))
    leisure = norm(row.get("leisure"))
    landuse = norm(row.get("landuse"))
    natural = norm(row.get("natural"))

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


def make_synthetic_weather_zone(city_boundary_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a synthetic weather cell near the city center.

    This is intentionally simple:
    - It is a rectangular polygon.
    - It intersects the city boundary.
    - Edges crossing it become more expensive.
    """
    city_geom = union_geometries(city_boundary_proj)
    if city_geom is None or city_geom.is_empty:
        raise ValueError("City boundary geometry is empty; cannot create weather zone.")

    center = city_geom.centroid

    # A roughly central storm cell, in projected meters.
    raw_weather = box(
        center.x - 2500,
        center.y - 1500,
        center.x + 3500,
        center.y + 1800,
    )

    weather_geom = raw_weather.intersection(city_geom)

    return gpd.GeoDataFrame(
        {
            "name": ["synthetic_weather_cell"],
            "weather_type": ["storm_or_high_wind"],
            "weather_risk_penalty": [WEATHER_RISK_PENALTY],
            "energy_multiplier": [WEATHER_ENERGY_MULTIPLIER],
        },
        geometry=[weather_geom],
        crs=city_boundary_proj.crs,
    )


def json_safe(value: Any) -> Any:
    """Convert list/dict-like values into strings so GeoJSON export is robust."""
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    return value


def export_geojson(gdf: gpd.GeoDataFrame, path: Path, target_crs: str = EXPORT_CRS) -> None:
    """Export a GeoDataFrame to GeoJSON with simple JSON-safe attributes."""
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
        out = out.set_crs(target_crs)

    out = out.to_crs(target_crs).reset_index()

    for col in out.columns:
        if col != "geometry" and out[col].dtype == "object":
            out[col] = out[col].map(json_safe)

    out.to_file(path, driver="GeoJSON")
    print(f"[INFO] Wrote GeoJSON: {path}")


def add_evtol_edge_costs(
    edges_proj: gpd.GeoDataFrame,
    residential_geom: BaseGeometry | None,
    weather_geom: BaseGeometry | None,
) -> gpd.GeoDataFrame:
    """
    Add simplified eVTOL routing attributes to graph edges.

    All costs are simple and explainable:
    - distance: projected geometry length in meters
    - battery_cost: energy proxy, distance-based and increased by weather
    - noise_penalty: added if edge crosses residential areas
    - weather_risk: added if edge crosses weather zone
    - total_cost: weighted sum for shortest path routing
    """
    edges = edges_proj.copy()

    # Distance in projected meters. Fallback to OSMnx length if needed.
    edges["distance"] = edges.geometry.length
    if "length" in edges.columns:
        invalid_distance = edges["distance"].isna() | (edges["distance"] <= 0)
        edges.loc[invalid_distance, "distance"] = edges.loc[invalid_distance, "length"]

    # Spatial constraint flags.
    if residential_geom is not None and not residential_geom.is_empty:
        edges["over_residential"] = edges.geometry.intersects(residential_geom)
    else:
        edges["over_residential"] = False

    if weather_geom is not None and not weather_geom.is_empty:
        edges["inside_weather_zone"] = edges.geometry.intersects(weather_geom)
    else:
        edges["inside_weather_zone"] = False

    # Soft constraints as penalties.
    edges["noise_penalty"] = edges["over_residential"].astype(float) * NOISE_PENALTY_RESIDENTIAL
    edges["weather_risk"] = edges["inside_weather_zone"].astype(float) * WEATHER_RISK_PENALTY

    # Battery proxy. Weather makes flight less efficient.
    weather_multiplier = edges["inside_weather_zone"].map(
        {True: WEATHER_ENERGY_MULTIPLIER, False: 1.0}
    )
    edges["battery_cost"] = edges["distance"] * BASE_ENERGY_PER_METER * weather_multiplier

    # Optional hard constraint.
    edges["blocked_by_weather"] = False
    if APPLY_HARD_WEATHER_BLOCK:
        edges.loc[edges["inside_weather_zone"], "blocked_by_weather"] = True

    edges["total_cost"] = (
        ALPHA_DISTANCE * edges["distance"]
        + BETA_BATTERY * edges["battery_cost"]
        + GAMMA_NOISE * edges["noise_penalty"]
        + DELTA_WEATHER * edges["weather_risk"]
    )

    if APPLY_HARD_WEATHER_BLOCK:
        edges.loc[edges["blocked_by_weather"], "total_cost"] = BLOCKED_EDGE_COST

    # Useful for later multi-agent reservation-table planning.
    edges["capacity"] = 1
    edges["altitude_layer"] = 1

    return edges


def print_example_route(G: nx.MultiDiGraph) -> None:
    """Run a quick weighted shortest path sanity check."""
    nodes = list(G.nodes)
    if len(nodes) < 2:
        print("[WARN] Not enough nodes for example routing.")
        return

    origin = nodes[0]
    destination = nodes[len(nodes) // 2]

    try:
        route = nx.shortest_path(G, origin, destination, weight="total_cost")
        print(f"[INFO] Example route computed.")
        print(f"       origin={origin}")
        print(f"       destination={destination}")
        print(f"       route_nodes={len(route):,}")
    except nx.NetworkXNoPath:
        print("[WARN] No example route found. Try NETWORK_TYPE='all' or bidirectional corridors.")


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    setup_osmnx()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {CITY_NAME} street network...")
    G = ox.graph.graph_from_place(
        PLACE,
        network_type=NETWORK_TYPE,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    print(f"[INFO] Raw graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    print("[INFO] Projecting graph to local CRS...")
    G_proj = ox.projection.project_graph(G)

    nodes_proj, edges_proj = ox.convert.graph_to_gdfs(G_proj, nodes=True, edges=True)
    graph_crs = nodes_proj.crs
    print(f"[INFO] Projected CRS: {graph_crs}")

    print(f"[INFO] Fetching {CITY_NAME} city boundary...")
    city_boundary = ox.geocoder.geocode_to_gdf(PLACE)
    if city_boundary.crs is None:
        city_boundary = city_boundary.set_crs(EXPORT_CRS)
    city_boundary_proj = city_boundary.to_crs(graph_crs)

    # -----------------------------------------------------------------
    # Residential zones for noise constraints
    # -----------------------------------------------------------------

    residential_tags = {
        "landuse": ["residential"],
    }

    residential = fetch_osm_features(
        PLACE,
        residential_tags,
        label="residential zones",
    )
    residential = keep_polygon_features(residential)

    if not residential.empty:
        residential_proj = residential.to_crs(graph_crs)
    else:
        residential_proj = empty_gdf(crs=graph_crs)

    residential_geom = union_geometries(residential_proj)

    # -----------------------------------------------------------------
    # Emergency landing candidates
    # -----------------------------------------------------------------

    emergency_tags = {
        "aeroway": ["helipad", "heliport"],
        "amenity": ["hospital", "clinic"],
        "leisure": ["park", "pitch", "sports_centre", "recreation_ground"],
        "landuse": ["grass", "meadow", "recreation_ground", "village_green"],
        "natural": ["grassland", "heath", "scrub"],
    }

    emergency_candidates = fetch_osm_features(
        PLACE,
        emergency_tags,
        label="emergency landing candidates",
    )
    emergency_candidates = keep_landing_candidate_geometries(emergency_candidates)

    if not emergency_candidates.empty:
        emergency_candidates["candidate_type"] = emergency_candidates.apply(
            classify_emergency_candidate,
            axis=1,
        )

        emergency_candidates_proj = emergency_candidates.to_crs(graph_crs)

        # Add a representative landing point for polygon candidates.
        landing_points_proj = emergency_candidates_proj.geometry.representative_point()
        landing_points_wgs = gpd.GeoSeries(landing_points_proj, crs=graph_crs).to_crs(EXPORT_CRS)

        emergency_candidates_proj["landing_x"] = landing_points_proj.x
        emergency_candidates_proj["landing_y"] = landing_points_proj.y
        emergency_candidates_proj["landing_lon"] = landing_points_wgs.x
        emergency_candidates_proj["landing_lat"] = landing_points_wgs.y
    else:
        emergency_candidates_proj = empty_gdf(crs=graph_crs)

    # -----------------------------------------------------------------
    # Synthetic weather zone
    # -----------------------------------------------------------------

    weather_zone_proj = make_synthetic_weather_zone(city_boundary_proj)
    weather_geom = union_geometries(weather_zone_proj)

    # -----------------------------------------------------------------
    # Add eVTOL edge costs
    # -----------------------------------------------------------------

    print("[INFO] Adding eVTOL routing costs to edges...")
    edges_evtol = add_evtol_edge_costs(
        edges_proj,
        residential_geom=residential_geom,
        weather_geom=weather_geom,
    )

    # Rebuild NetworkX graph with the updated edge attributes.
    graph_attrs = dict(G_proj.graph)
    graph_attrs["name"] = f"{CITY_NAME} simplified eVTOL routing graph"
    graph_attrs["place"] = PLACE
    graph_attrs["network_type"] = NETWORK_TYPE
    graph_attrs["model_note"] = (
        "Street network used as simplified eVTOL air-corridor skeleton; "
        "not a realistic aviation simulator."
    )

    G_evtol = ox.convert.graph_from_gdfs(
        nodes_proj,
        edges_evtol,
        graph_attrs=graph_attrs,
    )

    print(f"[INFO] eVTOL graph: {len(G_evtol.nodes):,} nodes, {len(G_evtol.edges):,} edges")

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    graphml_path = OUT_DIR / f"{CITY_SLUG}_evtol.graphml"
    print(f"[INFO] Saving GraphML: {graphml_path}")
    ox.io.save_graphml(G_evtol, filepath=graphml_path)

    export_geojson(nodes_proj, OUT_DIR / f"{CITY_SLUG}_nodes.geojson")
    export_geojson(edges_evtol, OUT_DIR / f"{CITY_SLUG}_edges.geojson")
    export_geojson(residential_proj, OUT_DIR / f"{CITY_SLUG}_residential_zones.geojson")
    export_geojson(emergency_candidates_proj, OUT_DIR / f"{CITY_SLUG}_emergency_landing_candidates.geojson")
    export_geojson(weather_zone_proj, OUT_DIR / f"{CITY_SLUG}_weather_zone.geojson")
    export_geojson(city_boundary_proj, OUT_DIR / f"{CITY_SLUG}_city_boundary.geojson")

    # -----------------------------------------------------------------
    # Quick sanity check
    # -----------------------------------------------------------------

    print_example_route(G_evtol)

    # Simple summary metrics.
    n_res_edges = int(edges_evtol["over_residential"].sum())
    n_weather_edges = int(edges_evtol["inside_weather_zone"].sum())

    print("\n[SUMMARY]")
    print(f"Nodes: {len(G_evtol.nodes):,}")
    print(f"Edges: {len(G_evtol.edges):,}")
    print(f"Residential zones: {len(residential_proj):,}")
    print(f"Emergency landing candidates: {len(emergency_candidates_proj):,}")
    print(f"Edges over residential zones: {n_res_edges:,}")
    print(f"Edges inside synthetic weather zone: {n_weather_edges:,}")
    print(f"Output folder: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
