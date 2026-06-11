# GeoReasoner — Roadmap

This document outlines planned improvements and research directions.

---

## v0.5 — Multi-Hazard Extension

- **Cyclone risk layer**: integrate IBTrACS track data, compute distance-to-track proximity raster
- **Drought risk layer**: SPEI index from CHIRPS rainfall rasters
- **Composite hazard index**: combine flood, cyclone, and drought sub-indices with user-configurable weights
- **Time-series support**: accept a date range in the query; pipeline processes each year and returns trend data

---

## v0.6 — Real-Time Data Integration

- **Flood extent from Sentinel-1 SAR**: SNAP/eo-learn pipeline for near-real-time surface water classification
- **OpenStreetMap live fetch**: replace static OSMnx cache with incremental updates via Overpass API
- **BWDB gauge integration**: pull river stage data from Bangladesh Water Development Board for calibrating the FSI proximity layer

---

## v0.7 — Scalability

- **Async agent execution**: run GIS Analyst and Remote Sensing nodes concurrently using LangGraph's parallel branching
- **Tile server**: serve FSI raster as XYZ tiles via `titiler` instead of GeoJSON payloads for large areas
- **Distributed compute**: offload raster operations to Dask or Ray for national-scale analysis

---

## v0.8 — Multimodal Queries

- **Image input**: accept a satellite scene URL; Remote Sensing agent classifies land cover via the LLM's vision capability (Gemma 3 12B multimodal)
- **Sketch-to-query**: draw a bounding box on the map, convert to a structured spatial filter
- **Voice input**: Web Speech API in the Angular frontend, transcript sent to `/query`

---

## v1.0 — Production Hardening

- **Authentication**: JWT-based auth on `/query` and `/reports`; rate limiting via `slowapi`
- **Persistent run history**: store `GeoReasonerState` snapshots in DuckDB; `/runs` list endpoint
- **Streaming responses**: LangGraph `astream_events` with Server-Sent Events so the frontend shows per-agent progress in real time rather than waiting for the full pipeline
- **Angular PWA**: service worker + offline cache for the Leaflet basemap tiles
- **Kubernetes helm chart**: production deployment manifests

---

## Research Directions

- **Uncertainty quantification**: propagate raster uncertainty through the FSI formula using Monte Carlo sampling
- **Causal reasoning**: encode hydraulic causality (elevation → drainage → flood depth) as constraints in the LLM prompt to improve answer quality
- **LLM-free mode**: entirely tool-driven pipeline (no LM Studio dependency) for resource-constrained deployments
- **Comparative evaluation**: benchmark GeoReasoner FSI rankings against BWDB historical flood records for model validation
