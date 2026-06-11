"""Planner agent — decomposes the user query into a typed task graph."""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from georeasoner.agents._utils import trace_entry
from georeasoner.llm import get_llm
from georeasoner.state import GeoReasonerState, TaskPlan, TaskStep

_SYSTEM_PROMPT = """You are a GeoAI planning agent for flood risk analysis.

Given a geospatial query about flood risk or vulnerability, output a JSON task plan.

Available agents:
- gis_analyst      : loads administrative boundaries and waterway vectors
- remote_sensing   : loads DEM and LULC rasters, computes slope
- hydrology        : computes Flood Susceptibility Index, ranks admin units
- reasoner         : interprets numeric results, provides scientific explanation

Respond with ONLY a JSON object (no markdown, no extra text):
{
  "rationale": "<one sentence explaining the plan>",
  "steps": [
    {"agent": "<agent>", "description": "<what this agent should do>"},
    ...
  ]
}"""


def planner_node(state: GeoReasonerState) -> dict:
    """Decompose the query into a four-step task plan."""
    query = state["query"]

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ])
        plan = _parse_plan(response.content)
    except Exception:
        plan = _default_plan()

    entry = trace_entry(
        "planner", "plan_query",
        {"query": query},
        json.dumps([s.model_dump() for s in plan.steps]),
    )
    return {
        "task_plan": [s.model_dump() for s in plan.steps],
        "agent_trace": [entry],
    }


def _parse_plan(text: str) -> TaskPlan:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in planner response")
    data = json.loads(match.group())
    return TaskPlan(
        rationale=data.get("rationale", ""),
        steps=[TaskStep(**s) for s in data.get("steps", [])],
    )


def _default_plan() -> TaskPlan:
    return TaskPlan(
        rationale="Standard flood risk assessment: load data → compute terrain → FSI → interpret",
        steps=[
            TaskStep(agent="gis_analyst", description="Load Sylhet upazila boundaries and OSM waterways"),
            TaskStep(agent="remote_sensing", description="Load SRTM DEM and ESA WorldCover LULC; compute slope"),
            TaskStep(agent="hydrology", description="Compute Flood Susceptibility Index; rank upazilas"),
            TaskStep(agent="reasoner", description="Interpret FSI results; produce ranked scientific explanation"),
        ],
    )
