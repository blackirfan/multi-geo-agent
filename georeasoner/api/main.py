"""FastAPI application — Phase 4: /query, /reports, /layers, + Leaflet frontend."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from georeasoner import __version__
from georeasoner.config import settings
from georeasoner.db import init_db
from georeasoner.graph import assemble_graph
from georeasoner.report_writer import (
    admin_geojson,
    fsi_geojson,
    generate_report,
    rivers_geojson,
)
from georeasoner.state import empty_state

app = FastAPI(
    title="GeoReasoner",
    description="LLM-Orchestrated Multi-Agent Geospatial Reasoning System",
    version=__version__,
)

_STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_graph = None


@app.on_event("startup")
async def _startup() -> None:
    global _graph
    init_db()
    _graph = assemble_graph()


# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    run_id: str | None = None


class QueryResponse(BaseModel):
    run_id: str
    answer: str | None
    fsi_ranking: list[dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    error: str | None


class ReportRequest(BaseModel):
    run_id: str
    query: str
    answer: str | None = None
    fsi_ranking: list[dict[str, Any]] = []
    agent_trace: list[dict[str, Any]] = []


class ReportResponse(BaseModel):
    run_id: str
    report_url: str
    format: str


class FsiLayerRequest(BaseModel):
    fsi_ranking: list[dict[str, Any]]


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> FileResponse:
    """Serve the Leaflet map frontend."""
    html_path = _STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(html_path), media_type="text/html")


# ── Ops ────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": __version__,
        "model": settings.lm_studio_model,
        "lm_studio": settings.lm_studio_base_url,
    }


# ── Analysis ───────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["analysis"])
async def run_query(request: QueryRequest) -> QueryResponse:
    """
    Run the full GeoReasoner multi-agent pipeline on a natural-language query.

    Pipeline: Planner → GIS Analyst → Remote Sensing → Hydrology → Reasoner.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialised")

    run_id = request.run_id or str(uuid.uuid4())
    state = empty_state(request.query, run_id)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _graph.invoke, state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        run_id=run_id,
        answer=result.get("answer"),
        fsi_ranking=result.get("fsi_ranking", []),
        agent_trace=result.get("agent_trace", []),
        error=result.get("error"),
    )


# ── Reports ────────────────────────────────────────────────────────────────────

@app.post("/reports", response_model=ReportResponse, tags=["reports"])
async def create_report(request: ReportRequest) -> ReportResponse:
    """
    Generate a PDF flood vulnerability report from a completed analysis run.

    If the report already exists it is regenerated with the supplied data.
    """
    payload = {
        "query": request.query,
        "answer": request.answer,
        "fsi_ranking": request.fsi_ranking,
        "agent_trace": request.agent_trace,
    }

    try:
        report_path = await asyncio.get_event_loop().run_in_executor(
            None, generate_report, request.run_id, payload
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    suffix = report_path.suffix.lstrip(".")
    return ReportResponse(
        run_id=request.run_id,
        report_url=f"/reports/{request.run_id}",
        format=suffix,
    )


@app.get("/reports/{run_id}", tags=["reports"])
async def get_report(run_id: str, format: str = "pdf") -> FileResponse:
    """
    Download the generated report for a run.

    Query param ``format`` selects the output format:
    - ``pdf``  — PDF document (default)
    - ``html`` — standalone HTML page (opens in browser)

    If the requested format is not yet generated, falls back to the other format.
    Both files are created by ``POST /reports``.
    """
    reports_dir = Path(settings.reports_dir)

    fmt = format.lower().strip()
    if fmt not in {"pdf", "html"}:
        raise HTTPException(status_code=400, detail="format must be 'pdf' or 'html'")

    # Try the requested format first, then fall back to the other
    order = [".pdf", ".html"] if fmt == "pdf" else [".html", ".pdf"]
    for suffix in order:
        path = reports_dir / f"{run_id}{suffix}"
        if path.exists():
            media_type = "application/pdf" if suffix == ".pdf" else "text/html"
            return FileResponse(
                str(path),
                media_type=media_type,
                filename=f"georeasoner_report_{run_id}{suffix}",
            )

    raise HTTPException(
        status_code=404,
        detail=f"No report found for run_id={run_id}. POST /reports first.",
    )


# ── Layers ─────────────────────────────────────────────────────────────────────

@app.get("/layers/admin", tags=["layers"])
async def get_admin_layer() -> dict:
    """Return admin boundaries (upazila polygons) as GeoJSON FeatureCollection."""
    try:
        return await asyncio.get_event_loop().run_in_executor(None, admin_geojson)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/layers/rivers", tags=["layers"])
async def get_rivers_layer() -> dict:
    """Return waterways (river polylines) as GeoJSON FeatureCollection."""
    try:
        return await asyncio.get_event_loop().run_in_executor(None, rivers_geojson)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/layers/fsi", tags=["layers"])
async def get_fsi_layer(request: FsiLayerRequest) -> dict:
    """
    Return admin boundaries joined with FSI ranking data as GeoJSON.

    Suitable for choropleth rendering; features carry ``mean_fsi`` and
    ``fsi_rank`` properties.
    """
    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, fsi_geojson, request.fsi_ranking
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
