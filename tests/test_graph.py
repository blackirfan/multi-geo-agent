"""
Phase 3 integration tests.

All tests run without LM Studio — every agent has a hard fallback that calls
tools directly when get_llm() raises or returns no tool calls.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from georeasoner.graph import assemble_graph
from georeasoner.state import GeoReasonerState, TaskPlan, TaskStep, empty_state

# ── Helpers ────────────────────────────────────────────────────────────────────

def _no_llm_patch(module_path: str):
    """Patch get_llm in *module_path* to raise ConnectionError."""
    return patch(module_path, side_effect=ConnectionError("LM Studio not running"))


# ── State / schema tests ───────────────────────────────────────────────────────

def test_empty_state_has_required_keys() -> None:
    s = empty_state("test query")
    assert s["query"] == "test query"
    assert s["agent_trace"] == []
    assert s["fsi_ranking"] == []
    assert s["answer"] is None


def test_task_plan_schema() -> None:
    plan = TaskPlan(
        rationale="Flood risk assessment",
        steps=[
            TaskStep(agent="gis_analyst", description="Load boundaries"),
            TaskStep(agent="hydrology", description="Compute FSI"),
        ],
    )
    assert len(plan.steps) == 2
    assert plan.steps[0].agent == "gis_analyst"


# ── Graph structure tests ──────────────────────────────────────────────────────

def test_graph_compiles() -> None:
    graph = assemble_graph()
    assert graph is not None


def test_graph_has_expected_nodes() -> None:
    graph = assemble_graph()
    node_names = set(graph.nodes.keys())
    for expected in {"planner", "gis_analyst", "remote_sensing", "hydrology", "reasoner"}:
        assert expected in node_names


# ── Tool unit tests ────────────────────────────────────────────────────────────

def test_gis_load_admin_returns_valid_json() -> None:
    from georeasoner.agents.gis_analyst import load_admin_boundaries

    raw = load_admin_boundaries.invoke({"district_name": "Sylhet"})
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert Path(data["path"]).exists()
    assert data["feature_count"] > 0


def test_gis_load_waterways_returns_valid_json() -> None:
    from georeasoner.agents.gis_analyst import load_waterways

    raw = load_waterways.invoke({"district_name": "Sylhet"})
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert Path(data["path"]).exists()


def test_rs_load_dem_returns_valid_json() -> None:
    from georeasoner.agents.remote_sensing import load_dem

    raw = load_dem.invoke({})
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert Path(data["path"]).exists()


def test_rs_load_lulc_returns_valid_json() -> None:
    from georeasoner.agents.remote_sensing import load_lulc

    raw = load_lulc.invoke({})
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert Path(data["path"]).exists()


def test_hydro_tool_returns_ranking() -> None:
    from georeasoner.agents.gis_analyst import load_admin_boundaries, load_waterways
    from georeasoner.agents.hydrology import run_flood_susceptibility
    from georeasoner.agents.remote_sensing import load_dem, load_lulc

    dem = json.loads(load_dem.invoke({}))["path"]
    lulc = json.loads(load_lulc.invoke({}))["path"]
    rivers = json.loads(load_waterways.invoke({}))["path"]
    admin = json.loads(load_admin_boundaries.invoke({}))["path"]

    raw = run_flood_susceptibility.invoke({
        "dem_path": dem, "lulc_path": lulc,
        "waterways_path": rivers, "admin_path": admin,
    })
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert len(data["ranking"]) > 0
    assert all("mean_fsi" in r for r in data["ranking"])


# ── Full-pipeline integration tests ───────────────────────────────────────────

@pytest.mark.parametrize("query", [
    "Which unions in Sylhet have the highest flood risk?",
    "Identify the most flood-vulnerable upazilas in Sylhet District.",
])
def test_full_pipeline_with_fallback(query: str) -> None:
    """Graph runs end-to-end without LM Studio using agent fallbacks."""
    patches = [
        _no_llm_patch("georeasoner.agents.planner.get_llm"),
        _no_llm_patch("georeasoner.agents.gis_analyst.get_llm"),
        _no_llm_patch("georeasoner.agents.remote_sensing.get_llm"),
        _no_llm_patch("georeasoner.agents.hydrology.get_llm"),
        _no_llm_patch("georeasoner.agents.reasoner.get_llm"),
    ]

    graph = assemble_graph()
    state = empty_state(query)

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result: GeoReasonerState = graph.invoke(state)

    assert result["answer"] is not None, "answer must be populated"
    assert len(result["fsi_ranking"]) > 0, "FSI ranking must be non-empty"
    assert len(result["agent_trace"]) >= 5, "at least one trace entry per agent"


def test_pipeline_trace_has_all_agents() -> None:
    """Every agent must appear at least once in the trace."""
    patches = [
        _no_llm_patch("georeasoner.agents.planner.get_llm"),
        _no_llm_patch("georeasoner.agents.gis_analyst.get_llm"),
        _no_llm_patch("georeasoner.agents.remote_sensing.get_llm"),
        _no_llm_patch("georeasoner.agents.hydrology.get_llm"),
        _no_llm_patch("georeasoner.agents.reasoner.get_llm"),
    ]

    graph = assemble_graph()
    state = empty_state("Which unions in Sylhet have the highest flood risk?")

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result: GeoReasonerState = graph.invoke(state)

    agents_in_trace = {e["agent"] for e in result["agent_trace"]}
    for expected_agent in {"planner", "gis_analyst", "remote_sensing", "hydrology", "reasoner"}:
        assert expected_agent in agents_in_trace, f"{expected_agent} missing from trace"


def test_pipeline_ranking_is_sorted() -> None:
    """FSI ranking must be in descending order of mean FSI."""
    patches = [
        _no_llm_patch("georeasoner.agents.planner.get_llm"),
        _no_llm_patch("georeasoner.agents.gis_analyst.get_llm"),
        _no_llm_patch("georeasoner.agents.remote_sensing.get_llm"),
        _no_llm_patch("georeasoner.agents.hydrology.get_llm"),
        _no_llm_patch("georeasoner.agents.reasoner.get_llm"),
    ]

    graph = assemble_graph()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = graph.invoke(empty_state("Which unions in Sylhet have the highest flood risk?"))

    ranking = result["fsi_ranking"]
    fsi_values = [r["mean_fsi"] for r in ranking]
    assert fsi_values == sorted(fsi_values, reverse=True), "ranking must be descending"


def test_pipeline_answer_mentions_top_unit() -> None:
    """The answer string must reference the top-ranked unit's name."""
    patches = [
        _no_llm_patch("georeasoner.agents.planner.get_llm"),
        _no_llm_patch("georeasoner.agents.gis_analyst.get_llm"),
        _no_llm_patch("georeasoner.agents.remote_sensing.get_llm"),
        _no_llm_patch("georeasoner.agents.hydrology.get_llm"),
        _no_llm_patch("georeasoner.agents.reasoner.get_llm"),
    ]

    graph = assemble_graph()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = graph.invoke(empty_state("Which unions in Sylhet have the highest flood risk?"))

    top_name = result["fsi_ranking"][0]["name"]
    assert top_name in (result["answer"] or ""), \
        f"Top unit '{top_name}' not found in answer"
