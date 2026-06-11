"""Shared state for the GeoReasoner LangGraph workflow."""

import operator
import uuid
from typing import Annotated, TypedDict

from pydantic import BaseModel


class TaskStep(BaseModel):
    agent: str  # planner | gis_analyst | remote_sensing | hydrology | reasoner
    description: str


class TaskPlan(BaseModel):
    rationale: str
    steps: list[TaskStep]


class GeoReasonerState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    run_id: str

    # ── Planner output ────────────────────────────────────────────────────────
    task_plan: list[dict]  # list[TaskStep.model_dump()]

    # ── Data paths (populated by GIS Analyst and Remote Sensing) ─────────────
    admin_gdf_path: str | None
    waterways_gdf_path: str | None
    dem_path: str | None
    lulc_path: str | None
    fsi_raster_path: str | None

    # ── Computed results (populated by Hydrology) ─────────────────────────────
    # [{"rank": 1, "name": "...", "mean_fsi": 0.85, "max_fsi": 0.95}, ...]
    fsi_ranking: list[dict]

    # ── Accumulated execution trace ───────────────────────────────────────────
    # operator.add merges lists: each node returns {"agent_trace": [new_steps]}
    agent_trace: Annotated[list[dict], operator.add]

    # ── Final outputs ─────────────────────────────────────────────────────────
    reasoning: str | None
    answer: str | None
    error: str | None


def empty_state(query: str, run_id: str | None = None) -> GeoReasonerState:
    """Return an initialised state dict for a new query."""
    return GeoReasonerState(
        query=query,
        run_id=run_id or str(uuid.uuid4()),
        task_plan=[],
        admin_gdf_path=None,
        waterways_gdf_path=None,
        dem_path=None,
        lulc_path=None,
        fsi_raster_path=None,
        fsi_ranking=[],
        agent_trace=[],
        reasoning=None,
        answer=None,
        error=None,
    )
