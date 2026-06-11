"""Phase 1 smoke tests — config, DB bootstrap, and fixture validation.

No network I/O, no real geospatial data required.
"""

import importlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Package / config
# ---------------------------------------------------------------------------

def test_package_importable() -> None:
    mod = importlib.import_module("georeasoner")
    assert hasattr(mod, "__version__")


def test_config_loads() -> None:
    from georeasoner.config import settings

    assert settings.lm_studio_base_url.startswith("http")
    assert settings.lm_studio_model
    assert settings.lm_studio_api_key


def test_config_bbox_valid() -> None:
    from georeasoner.config import settings

    assert settings.study_bbox_west < settings.study_bbox_east
    assert settings.study_bbox_south < settings.study_bbox_north


def test_config_bbox_covers_sylhet_city() -> None:
    """Sylhet city is at ~91.87 E, 24.89 N — must fall inside the study bbox."""
    from georeasoner.config import settings

    assert settings.study_bbox_west <= 91.87 <= settings.study_bbox_east
    assert settings.study_bbox_south <= 24.89 <= settings.study_bbox_north


def test_config_study_bbox_property() -> None:
    from georeasoner.config import settings

    bbox = settings.study_bbox
    assert len(bbox) == 4
    assert bbox[0] == settings.study_bbox_west
    assert bbox[1] == settings.study_bbox_south
    assert bbox[2] == settings.study_bbox_east
    assert bbox[3] == settings.study_bbox_north


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def test_init_db_creates_tables(tmp_path: Path) -> None:
    from georeasoner.db import get_connection, init_db

    db = str(tmp_path / "test.duckdb")
    init_db(db)

    conn = get_connection(db)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    conn.close()

    assert "analysis_runs" in tables
    assert "agent_traces" in tables


def test_db_insert_and_query(tmp_path: Path) -> None:
    from georeasoner.db import get_connection, init_db

    db = str(tmp_path / "insert_test.duckdb")
    init_db(db)
    conn = get_connection(db)
    conn.execute(
        "INSERT INTO analysis_runs (id, query, status) VALUES (?, ?, ?)",
        ["run-001", "Which upazilas have highest flood risk?", "pending"],
    )
    row = conn.execute(
        "SELECT query, status FROM analysis_runs WHERE id = 'run-001'"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "Which upazilas have highest flood risk?"
    assert row[1] == "pending"


# ---------------------------------------------------------------------------
# Geospatial fixtures
# ---------------------------------------------------------------------------

def test_admin_fixture_geometry(sample_admin_gdf: gpd.GeoDataFrame) -> None:
    assert len(sample_admin_gdf) == 5
    assert sample_admin_gdf.crs.to_epsg() == 4326
    assert not sample_admin_gdf.geometry.is_empty.any()
    assert (sample_admin_gdf["district"] == "Sylhet").all()


def test_river_fixture_geometry(sample_river_gdf: gpd.GeoDataFrame) -> None:
    assert len(sample_river_gdf) == 1
    assert sample_river_gdf.crs.to_epsg() == 4326
    geom = sample_river_gdf.geometry.iloc[0]
    assert geom.geom_type == "LineString"


def test_dem_fixture_valid(sample_dem_file: Path) -> None:
    with rasterio.open(sample_dem_file) as src:
        assert src.count == 1
        assert src.crs.to_epsg() == 4326
        assert src.dtypes[0] == "int16"
        data = src.read(1)
    assert data.shape == (20, 20)
    assert data.min() >= 0
    assert data.max() < 200


def test_lulc_fixture_valid_classes(sample_lulc_file: Path) -> None:
    valid_classes = {10, 20, 30, 40, 60, 80, 90}
    with rasterio.open(sample_lulc_file) as src:
        assert src.count == 1
        assert src.crs.to_epsg() == 4326
        data = src.read(1)
    unique_vals = set(np.unique(data).tolist())
    assert unique_vals.issubset(valid_classes)


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

def test_data_directories_exist() -> None:
    assert Path("data/vector").is_dir()
    assert Path("data/raster").is_dir()
    assert Path("reports").is_dir()
