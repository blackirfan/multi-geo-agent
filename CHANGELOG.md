# Changelog

All notable changes to GeoReasoner are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_Changes not yet assigned a version._

---

## [0.4.0] — 2026-06-11

### Added
- **Phase 5: Angular 20+ Dashboard** — full single-page app with live agent workflow view, FSI choropleth map, ranking table, and log panel
  - `WorkflowViewComponent` — SVG pipeline graph with per-node status colours and pulse-ring animation for running agents
  - `MapViewComponent` — Leaflet choropleth initialised via `afterNextRender()`, updated reactively via Angular signals `effect()`
  - `QueryPanelComponent` — natural-language query input, dual PDF/HTML export buttons
  - `RankingTableComponent` — FSI bar chart per upazila
  - `LogPanelComponent` — scrollable agent trace viewer
  - `AppStateService` — reactive state hub using `signal()`, `computed()`, `effect()`
  - `GeoReasonerService` — typed `HttpClient` wrapper for all backend endpoints
  - Tailwind CSS 3 + PrimeNG 21 + Angular 21 + TypeScript 5.9
  - Dev proxy (`proxy.conf.json`) forwards `/query`, `/reports`, `/layers`, `/health` to FastAPI `:8000`

### Changed
- Angular TypeScript pinned to `~5.9.3` (required by `@angular-devkit/build-angular ~21.2.14`)

### Fixed
- CSS `@import` rules moved before `@tailwind` directives in `styles.css` to satisfy PostCSS ordering constraint
- Leaflet type errors resolved by using local `GeoFeature` interface and `Parameters<typeof L.geoJSON>[0]` cast

---

## [0.3.0] — 2026-06-10

### Added
- **Phase 4: Report Writer + REST API expansion**
  - `georeasoner/report_writer.py` — `build_folium_map()`, `render_html_report()`, `generate_pdf()`, `generate_report()`
  - `georeasoner/templates/report.html.j2` — styled Jinja2 template: cover page, FSI ranking table with bars, Folium map embed, agent trace
  - `georeasoner/static/index.html` — minimal Leaflet fallback page
  - `GET /` — serves Leaflet HTML frontend
  - `GET /layers/admin` — admin boundary GeoJSON
  - `GET /layers/rivers` — waterways GeoJSON
  - `POST /layers/fsi` — FSI-joined admin boundaries GeoJSON
  - `POST /reports` — generates PDF + HTML report, returns run URL
  - `GET /reports/{run_id}?format=pdf|html` — serves requested format with fallback to the other
- Dual export (PDF + HTML): `generate_report()` always writes both files; API serves whichever format is requested

### Fixed
- WeasyPrint `OSError: cannot load library 'libpango-1.0-0'` — fixed by `brew install pango` on macOS
- `test_full_pipeline_then_report` 503 error — `TestClient` now used as context manager to trigger `@app.on_event("startup")`

---

## [0.2.0] — 2026-06-09

### Added
- **Phase 3: Reasoner Agent + full pipeline integration**
  - `georeasoner/agents/reasoner.py` — synthesises FSI ranking into a natural-language flood risk assessment
  - End-to-end `GeoReasonerState` pipeline: Planner → GIS Analyst → Remote Sensing → Hydrology → Reasoner
  - `tests/test_phase3.py` — 16 integration tests, full pipeline patched without LM Studio
- **Real GADM data support**
  - `scripts/fetch_data.py` — downloads GADM Level-3 Sylhet boundaries and saves to `data/vector/`
  - `data/vector/sylhet_upazilas_shp/` — 12 real upazilas: Balaganj, BeaniBazar, Bishwanath, Companiganj, DakshinSurma, Fenchuganj, Golabganj, Gowainghat, Jaintiapur, Kanaighat, SylhetSadar, Zakiganj
  - `data_utils.py` — `ensure_admin_boundaries()` priority: GADM GeoPackage → GADM Shapefile → synthetic GeoPackage → generate synthetic

### Changed
- `data_utils.py` — `ensure_admin_boundaries()` updated to check for real GADM data before synthetic fallback

---

## [0.1.0] — 2026-06-08

### Added
- **Phase 1: Core infrastructure**
  - `georeasoner/` Python package with `__version__ = "0.1.0"`
  - `GeoReasonerState` TypedDict with `Annotated[list[dict], operator.add]` reducer
  - `LangGraph StateGraph` with five nodes wired in sequence
  - `get_llm()` — OpenAI-compatible client pointed at LM Studio (`http://localhost:1234/v1`)
  - `georeasoner/db.py` — DuckDB with spatial extension
  - `georeasoner/config.py` — Pydantic `Settings` for env-var configuration
  - `pyproject.toml` — Python 3.12 package, optional `[dev]` extras
  - `Dockerfile` + `docker-compose.yml`
  - `.github/workflows/ci.yml` — lint + pytest on push/PR
- **Phase 2: Geospatial tools + first three agents**
  - `georeasoner/tools/vector_ops.py` — buffer, spatial join, clip, overlay, proximity raster
  - `georeasoner/tools/raster_ops.py` — NDWI, slope, reclassify, zonal stats, write raster
  - `georeasoner/tools/hydrology_ops.py` — FSI computation with weighted elevation/slope/proximity/LULC layers
  - `georeasoner/agents/planner.py` — task decomposition agent
  - `georeasoner/agents/gis_analyst.py` — vector data acquisition and processing
  - `georeasoner/agents/remote_sensing.py` — raster data acquisition (DEM, LULC)
  - `georeasoner/agents/hydrology.py` — FSI raster + upazila ranking
  - `georeasoner/data_utils.py` — `ensure_*()` functions for all four data types
  - `tests/test_phase1.py`, `tests/test_phase2.py` — 58 tests

[Unreleased]: https://github.com/your-org/geo-multi-agent/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/your-org/geo-multi-agent/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/your-org/geo-multi-agent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/your-org/geo-multi-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/your-org/geo-multi-agent/releases/tag/v0.1.0
