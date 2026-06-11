"""Hydrology agent — computes Flood Susceptibility Index and ranks admin units."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from georeasoner.agents._utils import trace_entry
from georeasoner.llm import get_llm
from georeasoner.state import GeoReasonerState
from georeasoner.tools.hydrology_ops import flood_susceptibility_index
from georeasoner.tools.raster_ops import zonal_stats

_SYSTEM_PROMPT = """You are a Hydrology analyst in a flood risk pipeline for Sylhet, Bangladesh.

Your task: compute the Flood Susceptibility Index (FSI) and rank all upazilas.

Call 'run_flood_susceptibility' with:
- dem_path: path to the DEM (provided in the context)
- lulc_path: path to the LULC raster (provided in the context)
- waterways_path: path to waterways GeoPackage (provided in the context)
- admin_path: path to admin boundaries GeoPackage (provided in the context)"""


# ── Tool ───────────────────────────────────────────────────────────────────────

@tool
def run_flood_susceptibility(
    dem_path: str,
    lulc_path: str,
    waterways_path: str,
    admin_path: str,
) -> str:
    """
    Compute the Flood Susceptibility Index (FSI) and rank administrative units.

    Combines four normalised layers with weights:
      elevation 35% + slope 25% + river proximity 25% + land cover 15%

    Returns JSON:
    {
      "status": "ok",
      "fsi_raster_path": "...",
      "global_mean_fsi": 0.XX,
      "ranking": [{"rank": 1, "name": "...", "mean_fsi": 0.XX, "max_fsi": 0.XX}, ...]
    }
    """
    out = Path("reports/sylhet_flood_susceptibility.tif")
    out.parent.mkdir(parents=True, exist_ok=True)

    river_gdf = gpd.read_file(waterways_path)
    fsi, meta = flood_susceptibility_index(
        dem_path=Path(dem_path),
        lulc_path=Path(lulc_path),
        river_gdf=river_gdf,
        output_path=out,
    )

    admin_gdf = gpd.read_file(admin_path)
    stats_gdf = zonal_stats(admin_gdf, out, stats=["mean", "max"])
    stats_gdf = stats_gdf.sort_values("mean", ascending=False).reset_index(drop=True)

    name_col = next(
        (c for c in ["upazila_name", "NAME_3", "NAME_2", "name"] if c in stats_gdf.columns),
        None,
    )
    ranking = [
        {
            "rank": i + 1,
            "name": str(row[name_col]) if name_col else f"Unit_{i + 1}",
            "mean_fsi": round(float(row.get("mean", 0) or 0), 4),
            "max_fsi": round(float(row.get("max", 0) or 0), 4),
        }
        for i, (_, row) in enumerate(stats_gdf.iterrows())
    ]

    valid = fsi[~np.isnan(fsi)]
    return json.dumps({
        "status": "ok",
        "fsi_raster_path": str(out),
        "global_mean_fsi": round(float(np.nanmean(valid)), 4),
        "ranking": ranking,
    })


_TOOLS = [run_flood_susceptibility]
_TOOLS_MAP = {t.name: t for t in _TOOLS}


# ── Node ───────────────────────────────────────────────────────────────────────

def hydrology_node(state: GeoReasonerState) -> dict:
    """Compute FSI and upazila ranking via LLM tool-calling."""
    trace_steps: list[dict] = []
    result: dict = {}

    # Context for the LLM (includes actual data paths)
    context = (
        f"Query: {state['query']}\n\n"
        f"Available data:\n"
        f"  DEM          : {state.get('dem_path', 'unknown')}\n"
        f"  LULC         : {state.get('lulc_path', 'unknown')}\n"
        f"  Waterways    : {state.get('waterways_gdf_path', 'unknown')}\n"
        f"  Admin units  : {state.get('admin_gdf_path', 'unknown')}\n\n"
        "Call run_flood_susceptibility with the paths above."
    )

    try:
        llm = get_llm().bind_tools(_TOOLS)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        for tc in response.tool_calls or []:
            fn = _TOOLS_MAP.get(tc["name"])
            if fn:
                raw = fn.invoke(tc["args"])
                data = json.loads(raw)
                trace_steps.append(trace_entry("hydrology", tc["name"], tc["args"], raw))
                if tc["name"] == "run_flood_susceptibility":
                    result["fsi_raster_path"] = data.get("fsi_raster_path")
                    result["fsi_ranking"] = data.get("ranking", [])
    except Exception as exc:
        trace_steps.append(trace_entry("hydrology", "llm_call", {}, f"ERROR: {exc}"))

    # Fallback — call tool directly with paths from state
    if not result.get("fsi_raster_path"):
        args = {
            "dem_path": state.get("dem_path") or "",
            "lulc_path": state.get("lulc_path") or "",
            "waterways_path": state.get("waterways_gdf_path") or "",
            "admin_path": state.get("admin_gdf_path") or "",
        }
        # Ensure data is available before calling
        from georeasoner.data_utils import (
            ensure_admin_boundaries,
            ensure_dem,
            ensure_lulc,
            ensure_waterways,
        )
        args = {
            "dem_path": args["dem_path"] or str(ensure_dem()),
            "lulc_path": args["lulc_path"] or str(ensure_lulc()),
            "waterways_path": args["waterways_path"] or str(ensure_waterways()),
            "admin_path": args["admin_path"] or str(ensure_admin_boundaries()),
        }
        raw = run_flood_susceptibility.invoke(args)
        data = json.loads(raw)
        result["fsi_raster_path"] = data.get("fsi_raster_path")
        result["fsi_ranking"] = data.get("ranking", [])
        trace_steps.append(trace_entry("hydrology", "run_flood_susceptibility[fallback]", args, raw))

    return {"agent_trace": trace_steps, **result}
