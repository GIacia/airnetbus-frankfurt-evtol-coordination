# AirNetBus Frankfurt eVTOL Coordination

City-scale multi-agent eVTOL routing and control-view prototype for the TUM Science Hackathon 2026 Airbus challenge.

The project demonstrates a centralized Frankfurt airspace view where autonomous eVTOL agents receive missions, react to dynamic weather, avoid temporary no-fly zones, and prioritize emergency landings.

## What This Prototype Does

- Builds a Frankfurt city graph from OpenStreetMap using OSMnx.
- Converts the city graph into a simplified directed eVTOL air-corridor network.
- Treats nodes as vertistops, selected nodes as vertiports, and edges as flyable air corridors.
- Assigns explainable edge costs from distance, battery use, wind/weather risk, residential noise exposure, and operational constraints.
- Simulates a fleet of autonomous agents with route planning, battery state, priorities, and reservation-table conflict avoidance.
- Converts weather events into dynamic risk regions or temporary no-fly zones.
- Handles emergency cases including medical emergency, bird strike, machine disorder, and high-priority mission routing.
- Exports scenario data and an optional presentation animation.

## Pipeline

1. `build_frankfurt_evtol_graph.py`
   - Downloads and preprocesses Frankfurt OSM data.
   - Exports the base graph, residential zones, weather zone, and emergency landing candidates.

2. `transform_frankfurt_to_air_corridors.py`
   - Samples city waypoints from major roads, graph nodes, open spaces, waterways, and emergency sites.
   - Builds a sparse directed k-nearest-neighbor air-corridor graph.
   - Adds altitude layers, travel-time steps, capacity, reservation IDs, and generalized route costs.

3. `skymesh_simulator.py`
   - Generates vertiports, eVTOL agents, missions, weather events, emergency events, and fleet routes.
   - Uses an explainable multi-agent coordination policy: candidate routing, cost scoring, priority handling, and reservation-table conflict checks.
   - Exports logs, routes, metrics, alerts, and dynamic weather/no-fly-zone layers.

4. `skymesh_visualizer.py`
   - Renders the city-scale control view from simulator outputs.
   - Visualizes agents, corridors, weather regions, emergency priority routing, and final scenario metrics.

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full data and simulation pipeline:

```powershell
python build_frankfurt_evtol_graph.py
python transform_frankfurt_to_air_corridors.py
python skymesh_simulator.py
```

Generate a presentation video only when needed:

```powershell
python skymesh_visualizer.py --project-root . --city-name Frankfurt --city-slug frankfurt --duration 30 --fps 20 --output outputs_viz/airnetbus_frankfurt_demo.mp4 --poster
```

## Outputs

Generated artifacts are intentionally ignored by Git because they can be large and reproducible:

- `outputs/`: base Frankfurt graph and OSM-derived layers.
- `outputs_air/`: eVTOL air-corridor graph and waypoint layers.
- `outputs_sim/`: agents, routes, simulation logs, alerts, metrics, weather events, and no-fly zones.
- `outputs_viz/`: rendered MP4/GIF/PNG presentation assets.

## Modeling Scope

This is a hackathon decision-support and visualization prototype, not a certified aviation simulator. The model is intentionally explainable: every route decision is based on explicit graph costs, weather/noise/battery constraints, emergency priority rules, and reservation conflicts. The structure is designed so real weather feeds, stronger traffic-management rules, or learned prediction models can be added later without replacing the whole pipeline.

## Repository Name

Suggested GitHub repository name:

```text
airnetbus-frankfurt-evtol-coordination
```
