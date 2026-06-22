#!/usr/bin/env python3
"""
SkyMesh presentation visualizer - enhanced version.

Opens an interactive preview by default, or exports a high-resolution animation
when --output is provided:
- many red moving eVTOL agents
- dotted air-corridor graph
- dynamic, low-opacity active planned route segments
- transparent no-flight zones
- dynamic weather cells with wind/rain/snow styling
- emergency agent highlighting with a warning marker and alert panel

Expected inputs:
    outputs_sim/simulation_log.csv
    outputs_sim/vertiports.geojson
    outputs_sim/proposed_routes.geojson
    outputs_sim/alerts.csv
    outputs_sim/weather_events.geojson       optional, produced by skymesh_simulator_plus.py
    outputs_sim/no_flight_zones.geojson      optional, produced by skymesh_simulator_plus.py
    outputs_air/frankfurt_air_edges.geojson     optional
    outputs_air/frankfurt_air_nodes.geojson     optional

Examples:
    python skymesh_visualizer.py
    python skymesh_visualizer.py --project-root . --basemap satellite --duration 30 --fps 20 --output outputs_viz/skymesh_animation_plus.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

WEB_MERCATOR = "EPSG:3857"
WGS84 = "EPSG:4326"

STATUS_INACTIVE = {"arrived", "emergency_landed"}
EMERGENCY_TYPES = {"machine_disorder", "medical_emergency", "bird_strike", "high_priority_mission"}

WEATHER_STYLE = {
    "wind": {"facecolor": "#ff9d2e", "edgecolor": "#ffd08a", "alpha": 0.24, "label": "crosswind"},
    "thunderstorm": {"facecolor": "#8a3ffc", "edgecolor": "#f3dcff", "alpha": 0.28, "label": "lightning risk"},
    "icing": {"facecolor": "#56d6ff", "edgecolor": "#d7fbff", "alpha": 0.27, "label": "icing layer"},
    "heat": {"facecolor": "#ff4d3d", "edgecolor": "#ffd0ca", "alpha": 0.22, "label": "battery heat"},
    "heavy_rain": {"facecolor": "#2089ff", "edgecolor": "#cce7ff", "alpha": 0.25, "label": "heavy rain"},
}

EMERGENCY_LABELS = {
    "machine_disorder": "MACHINE DISORDER",
    "medical_emergency": "MEDICAL EMERGENCY",
    "bird_strike": "BIRD STRIKE",
    "high_priority_mission": "HIGH-PRIORITY MISSION",
}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def is_emergency_reason(reason: str) -> bool:
    text = str(reason).lower()
    return any(token in text for token in EMERGENCY_TYPES) or "emergency_priority" in text


def emergency_destination_label(event_type: str) -> str:
    if event_type == "medical_emergency":
        return "hospital / helipad"
    if event_type == "machine_disorder":
        return "nearest safe vertiport"
    if event_type == "bird_strike":
        return "nearest safe vertiport"
    if event_type == "high_priority_mission":
        return "priority destination"
    return "priority landing site"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview or export the enhanced SkyMesh eVTOL animation")
    parser.add_argument("--project-root", default=".", help="Folder containing outputs_sim and outputs_air")
    parser.add_argument("--city-name", default="Frankfurt", help="City name shown in animation overlays")
    parser.add_argument("--city-slug", default="frankfurt", help="Filename prefix for air graph layers")
    parser.add_argument("--output", default=None, help="Optional output .mp4 or .gif path. If omitted, open a live preview window.")
    parser.add_argument("--duration", type=float, default=30.0, help="Target animation duration in seconds")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI")
    parser.add_argument("--fig-width", type=float, default=16.0, help="Figure width in inches")
    parser.add_argument("--fig-height", type=float, default=10.0, help="Figure height in inches")
    parser.add_argument("--basemap", choices=["satellite", "osm", "none"], default="satellite", help="Online basemap mode")
    parser.add_argument("--zoom", default="auto", help="Contextily basemap zoom, or auto")
    parser.add_argument("--background-png", default=None, help="Optional local background PNG")
    parser.add_argument("--background-extent-json", default=None, help="Lon/lat extent JSON for local PNG")
    parser.add_argument("--trail", type=float, default=8.0, help="Show previous N simulation minutes as faint agent trail")
    parser.add_argument("--route-lookahead", type=float, default=12.0, help="Show route segments starting within this many future simulation minutes")
    parser.add_argument("--route-tail", type=float, default=4.0, help="Keep route segments visible this many minutes after use")
    parser.add_argument("--show-all-planned-routes", action="store_true", help="Draw all proposed routes statically, very faint")
    parser.add_argument("--show-nodes", action="store_true", help="Show waypoint nodes")
    parser.add_argument("--max-edges", type=int, default=3500, help="Max air edges to draw for speed")
    parser.add_argument("--emergency-window", type=float, default=16.0, help="Highlight emergency agent for N simulation minutes")
    parser.add_argument("--poster", action="store_true", help="Also export a final poster PNG when --output is used")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def read_geojson(path: Path, target_crs: str = WEB_MERCATOR) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=target_crs)
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=target_crs)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    return gdf.to_crs(target_crs)


def build_positions_gdf(sim_log_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(sim_log_path)
    required = {"time_step", "agent_id", "lon", "lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"simulation_log.csv is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["lon", "lat"]).copy()
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84).to_crs(WEB_MERCATOR)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    gdf["time_step"] = gdf["time_step"].astype(float)
    return gdf


def prepare_agent_tracks(positions: gpd.GeoDataFrame) -> dict[str, dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    for agent_id, grp in positions.sort_values(["agent_id", "time_step"]).groupby("agent_id"):
        times = grp["time_step"].to_numpy(dtype=float)
        tracks[str(agent_id)] = {
            "time": times,
            "x": grp["x"].to_numpy(dtype=float),
            "y": grp["y"].to_numpy(dtype=float),
            "status": grp["status"].astype(str).to_numpy() if "status" in grp.columns else np.array(["enroute"] * len(grp)),
            "battery": grp["battery_remaining"].to_numpy(dtype=float) if "battery_remaining" in grp.columns else np.full(len(grp), np.nan),
        }
    return tracks


def status_at_track_time(track: dict[str, Any], t: float) -> str:
    times = track["time"]
    idx = int(np.searchsorted(times, t, side="right") - 1)
    idx = max(0, min(idx, len(times) - 1))
    return str(track["status"][idx])


def interpolate_agents(tracks: dict[str, dict[str, Any]], t: float, include_inactive: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for agent_id, tr in tracks.items():
        times = tr["time"]
        if len(times) == 0:
            continue
        if t < times[0] or t > times[-1]:
            continue
        status = status_at_track_time(tr, t)
        if not include_inactive and status in STATUS_INACTIVE:
            continue
        x = float(np.interp(t, times, tr["x"]))
        y = float(np.interp(t, times, tr["y"]))
        battery = float(np.interp(t, times, tr["battery"])) if np.isfinite(tr["battery"]).any() else np.nan
        rows.append({"agent_id": agent_id, "x": x, "y": y, "status": status, "battery": battery})
    return pd.DataFrame(rows)


def compute_bounds(layers: list[gpd.GeoDataFrame], margin_ratio: float = 0.08) -> tuple[float, float, float, float]:
    bounds = []
    for gdf in layers:
        if gdf is not None and not gdf.empty:
            bounds.append(gdf.total_bounds)
    if not bounds:
        raise ValueError("No non-empty spatial layers for bounds")
    arr = np.array(bounds)
    minx = float(np.nanmin(arr[:, 0]))
    miny = float(np.nanmin(arr[:, 1]))
    maxx = float(np.nanmax(arr[:, 2]))
    maxy = float(np.nanmax(arr[:, 3]))
    dx = maxx - minx or 1000
    dy = maxy - miny or 1000
    return (
        minx - dx * margin_ratio,
        miny - dy * margin_ratio,
        maxx + dx * margin_ratio,
        maxy + dy * margin_ratio,
    )


def draw_local_png_background(ax: plt.Axes, png_path: Path, extent_json_path: Path) -> bool:
    if not png_path.exists() or not extent_json_path.exists():
        return False
    with open(extent_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    west = float(meta["west"])
    south = float(meta["south"])
    east = float(meta["east"])
    north = float(meta["north"])
    crs = str(meta.get("crs", WGS84))
    if crs.upper() != WGS84:
        raise ValueError("PNG extent JSON must use EPSG:4326 lon/lat")
    transformer = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)
    x0, y0 = transformer.transform(west, south)
    x1, y1 = transformer.transform(east, north)
    img = Image.open(png_path)
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="upper", zorder=0)
    return True


def draw_online_basemap(ax: plt.Axes, bounds: tuple[float, float, float, float], mode: str, zoom: str) -> None:
    if mode == "none":
        ax.set_facecolor("#08111f")
        return
    try:
        import contextily as ctx
    except Exception as exc:
        print(f"[WARN] contextily not installed or unavailable: {exc}")
        ax.set_facecolor("#08111f")
        return
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    try:
        source = ctx.providers.Esri.WorldImagery if mode == "satellite" else ctx.providers.CartoDB.Positron
        zoom_arg: Any = "auto" if str(zoom).lower() == "auto" else int(zoom)
        ctx.add_basemap(ax, source=source, crs=WEB_MERCATOR, zoom=zoom_arg, attribution=False)
    except Exception as exc:
        print(f"[WARN] Failed to draw {mode} basemap: {exc}")
        ax.set_facecolor("#08111f")


def downsample_edges(edges: gpd.GeoDataFrame, max_edges: int) -> gpd.GeoDataFrame:
    if edges.empty or len(edges) <= max_edges:
        return edges
    return edges.sample(max_edges, random_state=42)


def route_segments_from_gdf(routes: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    if routes.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in routes.iterrows():
        geom = row.geometry
        line_parts: list[LineString] = []
        if isinstance(geom, LineString):
            line_parts = [geom]
        elif isinstance(geom, MultiLineString):
            line_parts = list(geom.geoms)
        else:
            continue
        for line in line_parts:
            coords = np.asarray(line.coords, dtype=float)
            if len(coords) >= 2:
                records.append(
                    {
                        "coords": coords,
                        "t_start": float(row.get("t_start", 0.0)),
                        "t_end": float(row.get("t_end", row.get("t_start", 0.0) + 1.0)),
                        "agent_id": str(row.get("agent_id", "")),
                        "reason": str(row.get("reason", "")),
                        "is_emergency_route": is_emergency_reason(str(row.get("reason", ""))),
                        "mission_featured": truthy(row.get("mission_featured", False)),
                        "mission_id": str(row.get("mission_id", "")),
                        "mission_label": str(row.get("mission_label", "")),
                        "mission_stage": str(row.get("mission_stage", "")),
                    }
                )
    return records


def draw_static_layers(
    ax: plt.Axes,
    air_edges: gpd.GeoDataFrame,
    air_nodes: gpd.GeoDataFrame,
    vertiports: gpd.GeoDataFrame,
    no_fly_zones: gpd.GeoDataFrame,
    proposed_routes: gpd.GeoDataFrame,
    show_all_planned_routes: bool,
    show_nodes: bool,
) -> None:
    if not no_fly_zones.empty:
        no_fly_zones.plot(ax=ax, facecolor="#ff2b2b", edgecolor="#ffdddd", alpha=0.23, linewidth=1.1, hatch="///", zorder=2)
        for _, row in no_fly_zones.iterrows():
            c = row.geometry.representative_point()
            ax.text(c.x, c.y, "NO-FLY", fontsize=8, color="white", ha="center", va="center", weight="bold", zorder=10)

    if not air_edges.empty:
        air_edges.plot(ax=ax, color="white", linewidth=0.42, alpha=0.22, linestyle="dotted", zorder=3)

    if show_all_planned_routes and not proposed_routes.empty:
        proposed_routes.plot(ax=ax, color="#a8f3ff", linewidth=0.45, alpha=0.09, zorder=4)

    if show_nodes and not air_nodes.empty:
        air_nodes.plot(ax=ax, color="white", markersize=3, alpha=0.30, zorder=5)

    if not vertiports.empty:
        vertiports.plot(ax=ax, color="#ffd42a", edgecolor="black", linewidth=0.9, markersize=110, alpha=0.96, zorder=8)
        for _, row in vertiports.iterrows():
            label = str(row.get("vertiport_id", "V"))
            ax.text(row.geometry.x, row.geometry.y, label, fontsize=7.2, color="black", ha="center", va="center", weight="bold", zorder=9)


def draw_featured_mission_stops(ax: plt.Axes, mission_stops: gpd.GeoDataFrame) -> None:
    if mission_stops.empty:
        return

    styles = {
        "pickup": {"color": "#fff200", "size": 72},
        "dropoff": {"color": "#39ff14", "size": 72},
    }
    for role, style in styles.items():
        subset = mission_stops[mission_stops["stop_role"].astype(str) == role] if "stop_role" in mission_stops.columns else mission_stops.iloc[0:0]
        if subset.empty:
            continue
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=style["size"],
            marker="*",
            c=style["color"],
            edgecolors="#111111",
            linewidths=0.55,
            alpha=0.98,
            zorder=17,
        )


def build_weather_artists(ax: plt.Axes, weather_events: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    artists: list[dict[str, Any]] = []
    if weather_events.empty:
        return artists
    for _, row in weather_events.iterrows():
        weather_type = str(row.get("weather_type", "wind")).lower()
        style = WEATHER_STYLE.get(weather_type, WEATHER_STYLE["wind"])
        dynamic_no_fly = str(row.get("dynamic_no_fly", "false")).lower() in {"true", "1", "yes"}
        before = len(ax.collections)
        gpd.GeoDataFrame([row], geometry="geometry", crs=weather_events.crs).plot(
            ax=ax,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            alpha=style["alpha"],
            linewidth=2.0 if dynamic_no_fly else 1.4,
            linestyle="--" if dynamic_no_fly else "-",
            zorder=6,
        )
        new_artists = ax.collections[before:]
        for art in new_artists:
            art.set_visible(False)
        c = row.geometry.representative_point()
        label = str(row.get("label", style["label"]))
        severity = float(row.get("severity", 0.0))
        label_text = f"{label.upper()}\nsev {severity:.2f}"
        if dynamic_no_fly:
            label_text += "\nDYNAMIC NFZ"
        text_artist = ax.text(
            c.x,
            c.y,
            label_text,
            fontsize=8.3,
            color="white",
            ha="center",
            va="center",
            weight="bold",
            bbox=dict(facecolor="black", edgecolor=style["edgecolor"], alpha=0.54, pad=3),
            zorder=12,
        )
        text_artist.set_visible(False)
        artists.append(
            {
                "event_id": str(row.get("event_id", "WX")),
                "weather_type": weather_type,
                "label": str(row.get("label", style["label"])),
                "consequence": str(row.get("consequence", "")),
                "param_summary": str(row.get("param_summary", "")),
                "dynamic_no_fly": dynamic_no_fly,
                "start_time": float(row.get("start_time", 0)),
                "end_time": float(row.get("end_time", 0)),
                "artists": [*new_artists, text_artist],
            }
        )
    return artists


def active_weather_label(weather_events: gpd.GeoDataFrame, t: float) -> str:
    if weather_events.empty:
        return "WX: clear"
    active = weather_events[(weather_events["start_time"].astype(float) <= t) & (weather_events["end_time"].astype(float) > t)]
    if active.empty:
        return "WX: clear"
    parts = []
    for _, row in active.iterrows():
        label = str(row.get("label", row.get("weather_type", "wx")))
        params = str(row.get("param_summary", ""))
        params = params.replace(", ", " ")
        parts.append(f"{label} {int(row.get('end_time', 0) - t)}m | {params}")
    return "WX -> consequence -> params\n" + "\n".join(parts[:3])


def short_text(text: str, max_chars: int = 68) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def read_alerts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "time_step" in df.columns:
        df["time_step"] = pd.to_numeric(df["time_step"], errors="coerce")
    return df


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def outcome_summary(metrics: pd.DataFrame) -> str:
    if metrics.empty or "mode" not in metrics.columns:
        return "Outcome\nSimulation complete"
    by_mode = {str(row["mode"]): row for _, row in metrics.iterrows()}
    baseline = by_mode.get("baseline")
    proposed = by_mode.get("proposed")
    if baseline is None or proposed is None:
        return "Outcome\nSimulation complete"

    def val(row: pd.Series, key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except Exception:
            return default

    return "\n".join(
        [
            "Final System Check",
            f"All aircraft completed: {int(val(proposed, 'completed_agents'))}/{int(val(proposed, 'agents'))}",
            f"Airspace conflicts: {int(val(proposed, 'total_conflicts'))}",
            f"Emergency landings: {int(val(proposed, 'emergency_success'))}/{int(val(proposed, 'emergency_agents'))}",
            f"Dynamic no-fly violations: {val(proposed, 'dynamic_weather_nfz_exposure_m')/1000:.1f} km",
            f"AI route choices: {int(val(proposed, 'ai_routed_segments'))} segments",
        ]
    )


def emergency_events(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty or "event_type" not in alerts.columns or "agent_id" not in alerts.columns:
        return pd.DataFrame()
    events = alerts[alerts["event_type"].astype(str).isin(EMERGENCY_TYPES)].copy()
    events = events.dropna(subset=["agent_id", "time_step"])
    return events


def emergency_landing_targets(positions: gpd.GeoDataFrame, emergencies: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or emergencies.empty or "status" not in positions.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, ev in emergencies.iterrows():
        agent_id = str(ev.get("agent_id"))
        event_type = str(ev.get("event_type", "emergency"))
        event_time = float(ev.get("time_step", 0.0))

        agent_rows = positions[positions["agent_id"].astype(str) == agent_id].sort_values("time_step")
        after_event = agent_rows[agent_rows["time_step"].astype(float) >= event_time]
        if after_event.empty:
            continue

        landed = after_event[after_event["status"].astype(str) == "emergency_landed"]
        target_row = landed.iloc[0] if not landed.empty else after_event.iloc[-1]
        rows.append(
            {
                "agent_id": agent_id,
                "event_type": event_type,
                "time_step": event_time,
                "target_time": float(target_row["time_step"]),
                "x": float(target_row["x"]),
                "y": float(target_row["y"]),
                "label": emergency_destination_label(event_type),
            }
        )

    return pd.DataFrame(rows)


def status_counts(frame_df: pd.DataFrame) -> str:
    if frame_df.empty or "status" not in frame_df.columns:
        return ""
    counts = frame_df["status"].value_counts().to_dict()
    keys = ["enroute", "holding", "machine_disorder", "medical_emergency", "waiting", "arrived", "emergency_landed"]
    parts = [f"{k}: {counts[k]}" for k in keys if k in counts]
    return " | ".join(parts)


def add_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markeredgecolor="white", markersize=7, label="eVTOL agents"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="yellow", markeredgecolor="red", markersize=13, label="emergency aircraft"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#ffeb3b", markeredgecolor="black", markersize=9, label="emergency target"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffd42a", markeredgecolor="black", markersize=9, label="vertiport"),
        Line2D([0], [0], color="white", linestyle="dotted", linewidth=1.2, alpha=0.55, label="air corridor"),
        Line2D([0], [0], color="#a8f3ff", linewidth=2.0, alpha=0.55, label="active planned route"),
        Line2D([0], [0], color="#ffec6e", linewidth=3.2, alpha=0.95, label="priority emergency route"),
        Line2D([0], [0], color="#ff9d2e", linewidth=8, alpha=0.35, label="weather cell"),
        Line2D([0], [0], color="#f3dcff", linestyle="--", linewidth=2.0, alpha=0.85, label="dynamic no-fly weather"),
        Line2D([0], [0], color="#56d6ff", linewidth=8, alpha=0.35, label="icing / rotor risk"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=True)
    leg.get_frame().set_alpha(0.72)


def create_animation(args: argparse.Namespace) -> Path | None:
    root = Path(args.project_root)
    out_path: Path | None = None
    export_mode = bool(args.output)
    if export_mode:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

    sim_log_path = require_file(root / "outputs_sim" / "simulation_log.csv", "simulation log")
    vertiports_path = require_file(root / "outputs_sim" / "vertiports.geojson", "vertiports GeoJSON")

    positions = build_positions_gdf(sim_log_path)
    tracks = prepare_agent_tracks(positions)
    vertiports = read_geojson(vertiports_path)
    proposed_routes = read_geojson(root / "outputs_sim" / "proposed_routes.geojson")
    air_edges = downsample_edges(read_geojson(root / "outputs_air" / f"{args.city_slug}_air_edges.geojson"), args.max_edges)
    air_nodes = read_geojson(root / "outputs_air" / f"{args.city_slug}_air_nodes.geojson")
    weather_events = read_geojson(root / "outputs_sim" / "weather_events.geojson")
    no_fly_zones = read_geojson(root / "outputs_sim" / "no_flight_zones.geojson")
    featured_missions = read_geojson(root / "outputs_sim" / "featured_missions.geojson")
    alerts = read_alerts(root / "outputs_sim" / "alerts.csv")
    metrics = read_metrics(root / "outputs_sim" / "metrics_summary.csv")
    outcome_text = outcome_summary(metrics)
    emergencies = emergency_events(alerts)
    emergency_targets = emergency_landing_targets(positions, emergencies)

    bounds = compute_bounds([positions, vertiports, proposed_routes, air_edges, no_fly_zones, weather_events, featured_missions], margin_ratio=0.08)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    fig.patch.set_facecolor("#08111f")

    local_background_used = False
    if args.background_png and args.background_extent_json:
        local_background_used = draw_local_png_background(ax, Path(args.background_png), Path(args.background_extent_json))
    if not local_background_used:
        draw_online_basemap(ax, bounds, args.basemap, args.zoom)

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal", adjustable="box")

    draw_static_layers(
        ax=ax,
        air_edges=air_edges,
        air_nodes=air_nodes,
        vertiports=vertiports,
        no_fly_zones=no_fly_zones,
        proposed_routes=proposed_routes,
        show_all_planned_routes=args.show_all_planned_routes,
        show_nodes=args.show_nodes,
    )
    draw_featured_mission_stops(ax, featured_missions)

    weather_artists = build_weather_artists(ax, weather_events)
    route_records = route_segments_from_gdf(proposed_routes)
    active_route_collection = LineCollection([], colors="#a8f3ff", linewidths=1.25, alpha=0.40, zorder=7)
    ax.add_collection(active_route_collection)
    emergency_route_collection = LineCollection([], colors="#ffec6e", linewidths=3.0, alpha=0.92, zorder=18)
    ax.add_collection(emergency_route_collection)

    trail_scatter = ax.scatter([], [], s=12, c="red", alpha=0.18, edgecolors="none", zorder=19)
    agent_scatter = ax.scatter([], [], s=42, c="red", edgecolors="white", linewidths=0.65, zorder=20)
    emergency_scatter = ax.scatter([], [], s=220, marker="*", c="yellow", edgecolors="red", linewidths=1.2, zorder=25)
    emergency_target_scatter = ax.scatter([], [], s=120, marker="X", c="#ffeb3b", edgecolors="black", linewidths=0.9, zorder=24)

    title = ax.text(
        0.012,
        0.985,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.66, pad=7),
        zorder=30,
    )
    alert_box = ax.text(
        0.988,
        0.985,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.7,
        color="white",
        linespacing=1.08,
        bbox=dict(facecolor="#7a0000", edgecolor="yellow", alpha=0.70, pad=4.5),
        zorder=31,
    )
    outcome_box = ax.text(
        0.988,
        0.235,
        outcome_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color="white",
        bbox=dict(facecolor="#061826", edgecolor="#a8f3ff", alpha=0.82, pad=7),
        zorder=31,
    )
    outcome_box.set_visible(False)
    footer = ax.text(
        0.99,
        0.02,
        f"SkyMesh | Constraint-aware multi-agent eVTOL routing over {args.city_name}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.50, pad=4),
        zorder=30,
    )
    add_legend(ax)
    ax.axis("off")

    min_t = float(positions["time_step"].min())
    max_t = float(positions["time_step"].max())
    n_frames = max(2, int(args.duration * args.fps))
    frames = np.linspace(min_t, max_t, n_frames)

    def active_emergency_rows(t: float) -> pd.DataFrame:
        if emergencies.empty:
            return pd.DataFrame()
        return emergencies[(emergencies["time_step"].astype(float) <= t) & (emergencies["time_step"].astype(float) + args.emergency_window >= t)]

    def update(t: float):
        frame_df = interpolate_agents(tracks, float(t), include_inactive=False)
        if len(frame_df):
            agent_scatter.set_offsets(frame_df[["x", "y"]].to_numpy())
        else:
            agent_scatter.set_offsets(np.empty((0, 2)))

        # Trail points, sampled over the last few simulation minutes.
        trail_offsets: list[list[float]] = []
        if args.trail > 0:
            trail_times = np.linspace(max(min_t, t - args.trail), t, 5)
            for tt in trail_times[:-1]:
                trail_df = interpolate_agents(tracks, float(tt), include_inactive=False)
                if len(trail_df):
                    trail_offsets.extend(trail_df[["x", "y"]].to_numpy().tolist())
        trail_scatter.set_offsets(np.asarray(trail_offsets) if trail_offsets else np.empty((0, 2)))

        # Dynamic planned routes: only near-current segments are visible.
        route_lines = [
            rec["coords"]
            for rec in route_records
            if rec["t_start"] <= t + args.route_lookahead and rec["t_end"] >= t - args.route_tail
        ]
        active_route_collection.set_segments(route_lines)

        # Dynamic weather visibility.
        active_weather_msgs = []
        for rec in weather_artists:
            active = rec["start_time"] <= t < rec["end_time"]
            for art in rec["artists"]:
                art.set_visible(active)
            if active:
                active_weather_msgs.append(str(rec.get("label", rec["weather_type"])))

        # Emergency highlighting.
        active_emergencies = active_emergency_rows(float(t))
        emergency_offsets: list[list[float]] = []
        emergency_target_offsets: list[list[float]] = []
        emergency_route_lines: list[np.ndarray] = []
        emergency_msgs: list[str] = []
        pulse = 0.75 + 0.35 * (0.5 + 0.5 * np.sin(float(t) * np.pi * 1.8))
        if not active_emergencies.empty:
            all_agents_now = interpolate_agents(tracks, float(t), include_inactive=True)
            active_event_times = {
                str(ev.get("agent_id")): float(ev.get("time_step", t))
                for _, ev in active_emergencies.iterrows()
            }
            emergency_route_lines = [
                rec["coords"]
                for rec in route_records
                if rec.get("is_emergency_route")
                and rec["agent_id"] in active_event_times
                and rec["t_end"] >= active_event_times[rec["agent_id"]]
                and rec["t_start"] <= t + max(args.route_lookahead, 18.0)
                and rec["t_end"] >= t - max(args.route_tail, 8.0)
            ]
            for _, ev in active_emergencies.iterrows():
                agent_id = str(ev.get("agent_id"))
                event_type = str(ev.get("event_type", "emergency"))
                match = all_agents_now[all_agents_now["agent_id"] == agent_id]
                if match.empty:
                    continue
                x = float(match.iloc[0]["x"])
                y = float(match.iloc[0]["y"])
                emergency_offsets.append([x, y])
                label = EMERGENCY_LABELS.get(event_type, event_type.replace("_", " ").upper())
                dest_label = emergency_destination_label(event_type)
                emergency_msgs.append(f"{label} -> {dest_label}")

                if not emergency_targets.empty:
                    target_match = emergency_targets[
                        (emergency_targets["agent_id"].astype(str) == agent_id)
                        & (emergency_targets["event_type"].astype(str) == event_type)
                    ]
                    if not target_match.empty:
                        target = target_match.iloc[0]
                        tx = float(target["x"])
                        ty = float(target["y"])
                        emergency_target_offsets.append([tx, ty])
        emergency_scatter.set_offsets(np.asarray(emergency_offsets) if emergency_offsets else np.empty((0, 2)))
        emergency_scatter.set_sizes(
            np.full(len(emergency_offsets), 210.0 * pulse) if emergency_offsets else np.asarray([220.0])
        )
        emergency_target_scatter.set_offsets(
            np.asarray(emergency_target_offsets) if emergency_target_offsets else np.empty((0, 2))
        )
        emergency_target_scatter.set_sizes(
            np.full(len(emergency_target_offsets), 130.0 * pulse) if emergency_target_offsets else np.asarray([120.0])
        )
        emergency_route_collection.set_segments(emergency_route_lines)

        status_text = status_counts(frame_df)
        title.set_text(f"SkyMesh {args.city_name} eVTOL Routing | t = {int(round(t)):03d} min\n{status_text}")

        panel_lines: list[str] = []
        if active_weather_msgs:
            panel_lines.append("Weather: " + ", ".join(active_weather_msgs[:2]))
            dynamic_count = sum(1 for rec in weather_artists if rec["start_time"] <= t < rec["end_time"] and rec.get("dynamic_no_fly"))
            if dynamic_count:
                panel_lines.append(f"Dynamic NFZ: {dynamic_count}")
        if emergency_msgs:
            active_types = []
            for msg in emergency_msgs:
                label = msg.split(" -> ", 1)[0].strip()
                if label and label not in active_types:
                    active_types.append(label)
            panel_lines.append("Emergency: " + ", ".join(active_types[:2]))
        alert_box.set_text("\n".join(panel_lines))
        alert_box.set_visible(bool(active_weather_msgs or emergency_msgs))
        outcome_box.set_visible(t >= max_t - max(8.0, 0.14 * (max_t - min_t)))

        artists = [
            agent_scatter,
            trail_scatter,
            emergency_scatter,
            emergency_target_scatter,
            active_route_collection,
            emergency_route_collection,
            title,
            alert_box,
            outcome_box,
            footer,
        ]
        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=int(1000 / max(1, args.fps)), blit=False)

    if not export_mode:
        try:
            fig.canvas.manager.set_window_title(f"{args.city_name} eVTOL Coordination Preview")
        except Exception:
            pass
        print("[INFO] Opening interactive preview. Use --output path.mp4 or --output path.gif to export.")
        if args.poster:
            print("[WARN] --poster is ignored in preview mode. Use --output together with --poster.")
        plt.show()
        return None

    assert out_path is not None
    if out_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
        anim.save(out_path, writer=writer, dpi=args.dpi)
    else:
        if shutil.which("ffmpeg"):
            writer = FFMpegWriter(fps=args.fps, bitrate=10000)
            anim.save(out_path, writer=writer, dpi=args.dpi)
        else:
            fallback = out_path.with_suffix(".gif")
            print("[WARN] ffmpeg not found. Saving GIF instead:", fallback)
            writer = PillowWriter(fps=args.fps)
            anim.save(fallback, writer=writer, dpi=args.dpi)
            out_path = fallback

    if args.poster:
        poster_path = out_path.with_name(out_path.stem + "_poster.png")
        update(float(frames[-1]))
        fig.savefig(poster_path, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print("[INFO] Wrote poster:", poster_path)

    plt.close(fig)
    print("[INFO] Wrote animation:", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    create_animation(args)


if __name__ == "__main__":
    main()
