"""GIS Analyst agent — loads vector data for the study area."""

import json

import geopandas as gpd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from georeasoner.agents._utils import trace_entry
from georeasoner.data_utils import ensure_admin_boundaries, ensure_waterways
from georeasoner.llm import get_llm
from georeasoner.state import GeoReasonerState

_SYSTEM_PROMPT = """You are a GIS Analyst in a flood risk assessment pipeline for Sylhet, Bangladesh.

Your task: load the vector datasets needed for flood susceptibility analysis.
1. Administrative boundaries (upazilas) — used for zonal statistics and reporting
2. Waterways (rivers, streams, canals) — used for river-proximity computation

Call 'load_admin_boundaries' and 'load_waterways' with district_name="Sylhet"."""


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def load_admin_boundaries(district_name: str = "Sylhet") -> str:
    """
    Load administrative unit (upazila) boundaries for a given district.
    Falls back to synthetic data if the real GeoPackage is not present.
    Returns JSON: {"status": "ok", "path": "...", "feature_count": N}
    """
    path = ensure_admin_boundaries()
    gdf = gpd.read_file(str(path))
    return json.dumps({"status": "ok", "path": str(path), "feature_count": len(gdf)})


@tool
def load_waterways(district_name: str = "Sylhet") -> str:
    """
    Load waterway features (rivers, streams, canals) for the study area.
    Falls back to synthetic data if the real GeoPackage is not present.
    Returns JSON: {"status": "ok", "path": "...", "feature_count": N}
    """
    path = ensure_waterways()
    gdf = gpd.read_file(str(path))
    return json.dumps({"status": "ok", "path": str(path), "feature_count": len(gdf)})


_TOOLS = [load_admin_boundaries, load_waterways]
_TOOLS_MAP = {t.name: t for t in _TOOLS}


# ── Node ───────────────────────────────────────────────────────────────────────

def gis_analyst_node(state: GeoReasonerState) -> dict:
    """Load administrative boundaries and waterways via LLM tool-calling."""
    trace_steps: list[dict] = []
    result: dict = {}

    try:
        llm = get_llm().bind_tools(_TOOLS)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {state['query']}\n\nLoad the required geodata for Sylhet."),
        ])
        for tc in response.tool_calls or []:
            fn = _TOOLS_MAP.get(tc["name"])
            if fn:
                raw = fn.invoke(tc["args"])
                data = json.loads(raw)
                trace_steps.append(trace_entry("gis_analyst", tc["name"], tc["args"], raw))
                if tc["name"] == "load_admin_boundaries":
                    result["admin_gdf_path"] = data.get("path")
                elif tc["name"] == "load_waterways":
                    result["waterways_gdf_path"] = data.get("path")
    except Exception as exc:
        trace_steps.append(trace_entry("gis_analyst", "llm_call", {}, f"ERROR: {exc}"))

    # Fallback: call tools directly if LLM did not populate required fields
    if not result.get("admin_gdf_path"):
        raw = load_admin_boundaries.invoke({"district_name": "Sylhet"})
        data = json.loads(raw)
        result["admin_gdf_path"] = data["path"]
        trace_steps.append(trace_entry("gis_analyst", "load_admin_boundaries[fallback]", {}, raw))

    if not result.get("waterways_gdf_path"):
        raw = load_waterways.invoke({"district_name": "Sylhet"})
        data = json.loads(raw)
        result["waterways_gdf_path"] = data["path"]
        trace_steps.append(trace_entry("gis_analyst", "load_waterways[fallback]", {}, raw))

    return {"agent_trace": trace_steps, **result}
