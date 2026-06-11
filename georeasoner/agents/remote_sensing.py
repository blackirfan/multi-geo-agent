"""Remote Sensing agent — loads terrain and land-cover rasters; computes slope."""

import json
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from georeasoner.agents._utils import trace_entry
from georeasoner.data_utils import ensure_dem, ensure_lulc
from georeasoner.llm import get_llm
from georeasoner.state import GeoReasonerState
from georeasoner.tools.raster_ops import compute_slope, write_raster

_SYSTEM_PROMPT = """You are a Remote Sensing analyst in a flood risk pipeline for Sylhet, Bangladesh.

Your task:
1. Load the SRTM Digital Elevation Model (DEM) — call 'load_dem'
2. Load the ESA WorldCover LULC raster — call 'load_lulc'
3. Compute terrain slope from the DEM — call 'compute_slope_raster' with the DEM path

Call all three tools in sequence."""


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def load_dem() -> str:
    """
    Load the SRTM Digital Elevation Model for Sylhet.
    Returns JSON: {"status": "ok", "path": "..."}
    """
    return json.dumps({"status": "ok", "path": str(ensure_dem())})


@tool
def load_lulc() -> str:
    """
    Load the ESA WorldCover 2021 Land Use / Land Cover raster for Sylhet.
    Returns JSON: {"status": "ok", "path": "..."}
    """
    return json.dumps({"status": "ok", "path": str(ensure_lulc())})


@tool
def compute_slope_raster(dem_path: str) -> str:
    """
    Compute terrain slope in degrees from a DEM GeoTIFF.
    Returns JSON: {"status": "ok", "path": "...", "mean_slope_deg": N, "max_slope_deg": N}
    """
    slope_arr, meta = compute_slope(Path(dem_path))
    out = Path(dem_path).parent / "slope.tif"
    write_raster(slope_arr, meta, out)
    valid = slope_arr[~np.isnan(slope_arr)]
    return json.dumps({
        "status": "ok",
        "path": str(out),
        "mean_slope_deg": round(float(np.nanmean(valid)), 2),
        "max_slope_deg": round(float(np.nanmax(valid)), 2),
    })


_TOOLS = [load_dem, load_lulc, compute_slope_raster]
_TOOLS_MAP = {t.name: t for t in _TOOLS}


# ── Node ───────────────────────────────────────────────────────────────────────

def remote_sensing_node(state: GeoReasonerState) -> dict:
    """Load DEM + LULC and compute slope via LLM tool-calling."""
    trace_steps: list[dict] = []
    result: dict = {}

    try:
        llm = get_llm().bind_tools(_TOOLS)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {state['query']}\n\nLoad terrain data and compute slope."),
        ])
        for tc in response.tool_calls or []:
            fn = _TOOLS_MAP.get(tc["name"])
            if fn:
                raw = fn.invoke(tc["args"])
                data = json.loads(raw)
                trace_steps.append(trace_entry("remote_sensing", tc["name"], tc["args"], raw))
                if tc["name"] == "load_dem":
                    result["dem_path"] = data.get("path")
                elif tc["name"] == "load_lulc":
                    result["lulc_path"] = data.get("path")
    except Exception as exc:
        trace_steps.append(trace_entry("remote_sensing", "llm_call", {}, f"ERROR: {exc}"))

    # Fallback
    if not result.get("dem_path"):
        raw = load_dem.invoke({})
        data = json.loads(raw)
        result["dem_path"] = data["path"]
        trace_steps.append(trace_entry("remote_sensing", "load_dem[fallback]", {}, raw))

    if not result.get("lulc_path"):
        raw = load_lulc.invoke({})
        data = json.loads(raw)
        result["lulc_path"] = data["path"]
        trace_steps.append(trace_entry("remote_sensing", "load_lulc[fallback]", {}, raw))

    return {"agent_trace": trace_steps, **result}
