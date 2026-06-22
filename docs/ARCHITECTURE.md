# Architecture Notes

This document explains the code at a level useful for future maintainers, presenters, or teammates who want to extend the prototype after the hackathon.

## System Goal

The prototype models a simplified city-scale eVTOL traffic-management layer for Frankfurt. It does not simulate certified flight dynamics. Instead, it shows how a fleet-level coordinator can combine:

- city graph data,
- flyable air corridors,
- weather risk,
- residential noise exposure,
- battery cost,
- emergency priority,
- and reservation-based conflict avoidance.

The main idea is explainability: every route decision is produced from visible costs and constraints rather than a black-box model.

## File-Level Flow

```text
OpenStreetMap / OSMnx
        |
        v
build_frankfurt_evtol_graph.py
        |
        v
outputs/frankfurt_*.geojson + outputs/frankfurt_evtol.graphml
        |
        v
transform_frankfurt_to_air_corridors.py
        |
        v
outputs_air/frankfurt_air_*.geojson + outputs_air/frankfurt_air_corridors.graphml
        |
        v
skymesh_simulator.py
        |
        v
outputs_sim/*.csv + outputs_sim/*.geojson
        |
        v
skymesh_visualizer.py
        |
        v
outputs_viz/*.mp4 / *.png
```

## 1. Base City Graph

Entry point: `build_frankfurt_evtol_graph.py`

Main responsibilities:

- Download Frankfurt street-network geometry with OSMnx.
- Export city boundary, graph nodes, graph edges, residential polygons, emergency landing candidates, and a synthetic weather zone.
- Add early generalized eVTOL edge costs.

Important concepts:

- The street network is used as a geometric skeleton, not as a claim that eVTOLs fly along roads.
- Residential polygons increase the `noise_penalty`.
- Weather polygons increase `weather_risk` and `battery_cost`.
- The total route cost is a weighted sum of distance, battery, noise, and weather risk.

Representative cost model:

```text
total_cost =
    ALPHA_DISTANCE * distance_m
  + BETA_BATTERY * battery_cost
  + GAMMA_NOISE * noise_penalty
  + DELTA_WEATHER * weather_risk
```

## 2. Air-Corridor Graph Conversion

Entry point: `transform_frankfurt_to_air_corridors.py`

Main responsibilities:

- Load the base Frankfurt graph and OSM-derived layers.
- Create waypoint candidates from major roads, sampled graph nodes, parks/open spaces, waterways, and emergency sites.
- Reduce dense points into a manageable city-scale network.
- Build a directed k-nearest-neighbor air-corridor graph.
- Add simulation fields such as travel time, capacity, altitude layer, and reservation edge ID.

Core algorithm:

- Candidate points are sampled from city geometry.
- Nearby points are connected using kNN.
- Overlong edges are filtered.
- Weak connectivity is repaired so isolated components remain reachable.
- Every directed edge receives explicit cost attributes.

Important limitation:

- kNN is a geometric graph-construction method, not the AI coordination policy. It is used to create a sparse, usable corridor network from city-scale map data.

## 3. Multi-Agent Simulator

Entry point: `skymesh_simulator.py`

Main responsibilities:

- Load the air-corridor graph.
- Create vertiports and eVTOL agents.
- Generate missions and optional featured mission stops.
- Create dynamic weather events and emergency events.
- Plan and replan routes under constraints.
- Export routes, simulation log, alert log, metrics, weather layers, no-fly zones, and vertiport distribution.

Important data structures:

- `Agent`: one eVTOL with mission, battery, route, priority, status, and emergency state.
- `WeatherEvent`: active time window, geometry, severity, and derived routing parameters.
- `ReservationTable`: time-indexed node and edge reservations for conflict avoidance.

## 4. Explainable AI Coordination Policy

The coordinator is not a machine-learning model. It is closer to an explainable AI / rule-based multi-agent decision policy.

The policy combines:

- graph shortest-path search,
- multiple candidate route generation,
- explicit utility scoring,
- weather and no-fly-zone penalties,
- reservation-table conflict checks,
- and priority-aware emergency handling.

In simple terms:

```text
For each agent:
    generate candidate routes
    estimate route cost, delay, battery use, risk, and reservation conflicts
    adjust score by mission priority
    select the safest feasible route
    reserve its future nodes and edges in time
```

Why this can be described as AI in the hackathon context:

- Each aircraft is represented as an autonomous agent with state and goals.
- The coordinator reasons over changing constraints and competing agents.
- It chooses actions from alternatives using an explainable scoring policy.
- Decisions are adaptive: weather and emergency events can trigger replanning.

What it is not:

- It is not neural-network-based.
- It does not learn from data.
- It does not predict real traffic/weather from historical observations.

## 5. Reservation Table

The reservation table is the main conflict-avoidance layer.

It stores future occupancy of:

- nodes at time steps,
- directed edges at time steps,
- and reverse-edge conflicts where needed.

When a route is evaluated, the simulator checks whether another aircraft has already reserved the same space-time slot. Candidate routes with fewer conflicts or shorter waits score better. Emergency aircraft receive higher priority, so they are planned earlier and tolerate less delay.

The AI policy does not "create" the reservation table from nothing. The table is a shared coordination memory. The policy reads existing reservations, scores candidate plans against them, then writes the selected aircraft plan back into the table.

## 6. Weather and No-Fly-Zone Logic

Weather events are converted into simulator parameters:

- battery drain multiplier,
- noise multiplier,
- weather risk penalty,
- speed multiplier,
- and temporary no-fly-zone behavior when severity is high enough.

Current weather types include:

- wind,
- thunderstorm / lightning risk,
- icing,
- heat,
- heavy rain.

Dynamic events can change route costs during the scenario. If the event is severe enough, affected edges become effectively blocked or extremely expensive, forcing replanning around the region.

To add a new weather type:

1. Add or modify `weather_profile()`.
2. Add derived parameters in `weather_parameters()`.
3. Add an event geometry/time window in `make_dynamic_weather_events()`.
4. Add visual styling in `WEATHER_STYLE` inside `skymesh_visualizer.py`.

## 7. Emergency Logic

Current emergency cases include:

- `medical_emergency`,
- `bird_strike`,
- `machine_disorder`,
- `high_priority_mission`.

Different emergency types can use different target logic:

- medical emergency prefers medical landing sites such as hospitals or helipads,
- bird strike and machine disorder prefer the nearest safe vertiport,
- high-priority mission prefers a priority destination.

To add a new emergency type:

1. Add behavior in `incident_profile()`.
2. Add destination logic in `process_emergency_event()`.
3. Add visual label/target handling in `skymesh_visualizer.py`.
4. Add any metric or alert text needed for presentation.

## 8. Visualizer

Entry point: `skymesh_visualizer.py`

Main responsibilities:

- Read simulator outputs.
- Draw Frankfurt, air corridors, agents, weather cells, no-fly zones, emergency signals, target markers, and summary overlays.
- Export MP4/GIF/PNG assets for demos.

The visualizer is intentionally separated from the simulator. This lets contributors change route logic without touching presentation rendering, or redesign the animation without changing simulation state.

## 9. Generated Files

The repo ignores generated outputs:

- `outputs/`
- `outputs_air/`
- `outputs_sim/`
- `outputs_viz/`

This keeps GitHub lightweight. To reproduce a full scenario, rerun the pipeline from `README.md`.

## 10. Practical Extension Points

Likely future improvements:

- Replace synthetic weather events with live or recorded weather data.
- Replace synthetic missions with real demand generation.
- Add charging time and vertiport queue management.
- Add learned demand prediction while keeping the routing policy explainable.
- Add stricter airspace separation constraints.
- Add a config file so scenario parameters can be changed without editing Python constants.
