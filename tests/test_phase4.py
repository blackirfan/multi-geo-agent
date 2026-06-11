"""
Phase 4 tests: report writer + /layers + /reports + Leaflet frontend endpoint.

All tests run offline (no LM Studio, no WeasyPrint binary required).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from georeasoner.api.main import app
from georeasoner.report_writer import (
    admin_geojson,
    build_folium_map,
    fsi_geojson,
    render_html_report,
    rivers_geojson,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_ranking() -> list[dict]:
    return [
        {"rank": 1, "name": "Unit_1", "mean_fsi": 0.82, "max_fsi": 0.95},
        {"rank": 2, "name": "Unit_2", "mean_fsi": 0.74, "max_fsi": 0.88},
        {"rank": 3, "name": "Unit_3", "mean_fsi": 0.61, "max_fsi": 0.79},
    ]


@pytest.fixture(scope="module")
def sample_trace() -> list[dict]:
    return [
        {"agent": "planner", "tool": "plan", "result": "ok", "timestamp": "2026-01-01T00:00:00"},
        {"agent": "hydrology", "tool": "run_fsi", "result": "ok", "timestamp": "2026-01-01T00:01:00"},
    ]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Report writer unit tests ───────────────────────────────────────────────────

def test_render_html_contains_ranking(sample_ranking, sample_trace) -> None:
    html = render_html_report(
        run_id="test-run-001",
        query="Which unions are most flood prone?",
        answer="Unit_1 has the highest FSI.",
        ranking=sample_ranking,
        agent_trace=sample_trace,
        map_html="",
    )
    assert "Unit_1" in html
    assert "0.8200" in html
    assert "test-run-001" in html


def test_render_html_contains_methods(sample_ranking, sample_trace) -> None:
    html = render_html_report(
        run_id="test-run-002",
        query="Flood risk?",
        answer=None,
        ranking=sample_ranking,
        agent_trace=sample_trace,
        map_html="",
    )
    assert "35%" in html  # elevation weight
    assert "Methodology" in html


def test_build_folium_map_returns_html(sample_ranking) -> None:
    import geopandas as gpd

    admin_gdf = gpd.read_file(__import__("georeasoner.data_utils", fromlist=["ensure_admin_boundaries"]).ensure_admin_boundaries())
    river_gdf = gpd.read_file(__import__("georeasoner.data_utils", fromlist=["ensure_waterways"]).ensure_waterways())

    html = build_folium_map(admin_gdf, river_gdf, sample_ranking)
    assert "<div" in html
    assert len(html) > 100


def test_generate_report_creates_html(tmp_path, sample_ranking, sample_trace) -> None:
    """generate_report must always produce at least an HTML file."""
    from georeasoner.report_writer import generate_report

    result = {
        "query": "Flood risk assessment",
        "answer": "Unit_1 is highest risk.",
        "fsi_ranking": sample_ranking,
        "agent_trace": sample_trace,
    }

    # Patch WeasyPrint to simulate unavailability
    with patch("georeasoner.report_writer.generate_pdf", side_effect=RuntimeError("no weasyprint")):
        output = generate_report("test-gen-001", result, output_dir=tmp_path)

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Unit_1" in content


def test_generate_report_creates_pdf(tmp_path, sample_ranking, sample_trace) -> None:
    """When WeasyPrint is available, generate_report returns a .pdf path."""
    from georeasoner.report_writer import generate_report

    result = {
        "query": "Flood risk",
        "answer": "Unit_1.",
        "fsi_ranking": sample_ranking,
        "agent_trace": sample_trace,
    }

    # Stub WeasyPrint to write a 1-byte file
    def _fake_pdf(html: str, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"%PDF-stub")
        return out

    with patch("georeasoner.report_writer.generate_pdf", side_effect=_fake_pdf):
        output = generate_report("test-gen-002", result, output_dir=tmp_path)

    assert output.suffix == ".pdf"
    assert output.exists()


# ── GeoJSON helpers ────────────────────────────────────────────────────────────

def test_admin_geojson_is_feature_collection() -> None:
    geo = admin_geojson()
    assert geo["type"] == "FeatureCollection"
    assert isinstance(geo["features"], list)
    assert len(geo["features"]) > 0


def test_rivers_geojson_is_feature_collection() -> None:
    geo = rivers_geojson()
    assert geo["type"] == "FeatureCollection"
    assert isinstance(geo["features"], list)


def test_fsi_geojson_carries_mean_fsi() -> None:
    import geopandas as gpd

    from georeasoner.data_utils import ensure_admin_boundaries

    gdf = gpd.read_file(ensure_admin_boundaries())
    name_col = next(
        (c for c in ["upazila_name", "NAME_3", "NAME_2", "name"] if c in gdf.columns), None
    )
    names = gdf[name_col].tolist()[:3] if name_col else []
    ranking = [
        {"rank": i + 1, "name": n, "mean_fsi": round(0.8 - i * 0.1, 2), "max_fsi": 0.9}
        for i, n in enumerate(names)
    ]

    geo = fsi_geojson(ranking)
    assert geo["type"] == "FeatureCollection"
    props = [f["properties"] for f in geo["features"]]
    fsi_values = [p.get("mean_fsi") for p in props if p.get("mean_fsi") is not None]
    assert len(fsi_values) > 0


# ── API endpoint tests ─────────────────────────────────────────────────────────

def test_health_endpoint(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_root_returns_html(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "GeoReasoner" in res.text


def test_layers_admin_endpoint(client: TestClient) -> None:
    res = client.get("/layers/admin")
    assert res.status_code == 200
    body = res.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) > 0


def test_layers_rivers_endpoint(client: TestClient) -> None:
    res = client.get("/layers/rivers")
    assert res.status_code == 200
    body = res.json()
    assert body["type"] == "FeatureCollection"


def test_layers_fsi_endpoint(client: TestClient, sample_ranking) -> None:
    res = client.post("/layers/fsi", json={"fsi_ranking": sample_ranking})
    assert res.status_code == 200
    body = res.json()
    assert body["type"] == "FeatureCollection"


def test_reports_post_creates_file(client: TestClient, tmp_path, sample_ranking, sample_trace) -> None:
    """POST /reports should generate a file and return a URL."""
    with (
        patch("georeasoner.api.main.generate_report") as mock_gen,
    ):
        fake_path = tmp_path / "abc-123.html"
        fake_path.write_text("<html>stub</html>")
        mock_gen.return_value = fake_path

        res = client.post("/reports", json={
            "run_id": "abc-123",
            "query": "Flood risk?",
            "answer": "Unit_1 is highest.",
            "fsi_ranking": sample_ranking,
            "agent_trace": sample_trace,
        })

    assert res.status_code == 200
    body = res.json()
    assert body["run_id"] == "abc-123"
    assert "/reports/abc-123" in body["report_url"]


def test_reports_get_not_found(client: TestClient) -> None:
    res = client.get("/reports/nonexistent-run-xyz")
    assert res.status_code == 404


def test_full_pipeline_then_report(client: TestClient, tmp_path) -> None:
    """Run /query with mocked agents, then POST /reports and verify response."""
    from unittest.mock import patch

    patches = [
        patch("georeasoner.agents.planner.get_llm", side_effect=ConnectionError("no llm")),
        patch("georeasoner.agents.gis_analyst.get_llm", side_effect=ConnectionError("no llm")),
        patch("georeasoner.agents.remote_sensing.get_llm", side_effect=ConnectionError("no llm")),
        patch("georeasoner.agents.hydrology.get_llm", side_effect=ConnectionError("no llm")),
        patch("georeasoner.agents.reasoner.get_llm", side_effect=ConnectionError("no llm")),
    ]

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        qres = client.post("/query", json={"query": "Which unions have highest flood risk?"})

    assert qres.status_code == 200
    qdata = qres.json()
    run_id = qdata["run_id"]
    assert qdata["fsi_ranking"]

    # Now generate report
    fake_path = tmp_path / f"{run_id}.html"
    fake_path.write_text("<html>stub</html>")

    with patch("georeasoner.api.main.generate_report", return_value=fake_path):
        rres = client.post("/reports", json={
            "run_id": run_id,
            "query": "Which unions have highest flood risk?",
            "answer": qdata.get("answer"),
            "fsi_ranking": qdata["fsi_ranking"],
            "agent_trace": qdata["agent_trace"],
        })

    assert rres.status_code == 200
    assert rres.json()["run_id"] == run_id
