"""Unit tests for georeasoner.tools.hydrology_ops."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from georeasoner.tools.hydrology_ops import (
    DEFAULT_WEIGHTS,
    ESA_VULNERABILITY,
    _norm,
    _norm_invert,
    flood_susceptibility_index,
)

# ── ESA_VULNERABILITY constant ────────────────────────────────────────────────

def test_esa_vulnerability_all_in_range() -> None:
    for cls, score in ESA_VULNERABILITY.items():
        assert 0.0 <= score <= 1.0, f"Class {cls} score {score} out of [0,1]"


def test_esa_water_is_max() -> None:
    assert ESA_VULNERABILITY[80] == 1.0


def test_esa_tree_cover_low() -> None:
    assert ESA_VULNERABILITY[10] < 0.4


# ── DEFAULT_WEIGHTS ───────────────────────────────────────────────────────────

def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)


def test_default_weights_all_positive() -> None:
    assert all(v > 0 for v in DEFAULT_WEIGHTS.values())


# ── helper normalisation functions ────────────────────────────────────────────

def test_norm_output_range() -> None:
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    out = _norm(arr)
    assert float(out.min()) == pytest.approx(0.0)
    assert float(out.max()) == pytest.approx(1.0)


def test_norm_invert_is_complement() -> None:
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    assert _norm_invert(arr)[0] == pytest.approx(1.0)
    assert _norm_invert(arr)[-1] == pytest.approx(0.0)


def test_norm_constant_array_returns_zeros() -> None:
    arr = np.full((5, 5), 42.0, dtype=np.float32)
    out = _norm(arr)
    np.testing.assert_array_equal(out, 0.0)


# ── flood_susceptibility_index ────────────────────────────────────────────────

def test_fsi_shape(
    sample_dem_file: Path,
    sample_lulc_file: Path,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    fsi, _ = flood_susceptibility_index(sample_dem_file, sample_lulc_file, sample_river_gdf)
    assert fsi.shape == (20, 20)


def test_fsi_dtype(
    sample_dem_file: Path,
    sample_lulc_file: Path,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    fsi, _ = flood_susceptibility_index(sample_dem_file, sample_lulc_file, sample_river_gdf)
    assert fsi.dtype == np.float32


def test_fsi_range_zero_to_one(
    sample_dem_file: Path,
    sample_lulc_file: Path,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    fsi, _ = flood_susceptibility_index(sample_dem_file, sample_lulc_file, sample_river_gdf)
    valid = fsi[~np.isnan(fsi)]
    assert valid.size > 0
    assert float(valid.min()) >= -1e-5
    assert float(valid.max()) <= 1.0 + 1e-5


def test_fsi_meta_dtype(
    sample_dem_file: Path,
    sample_lulc_file: Path,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    _, meta = flood_susceptibility_index(sample_dem_file, sample_lulc_file, sample_river_gdf)
    assert meta["dtype"] == "float32"
    assert meta["count"] == 1


def test_fsi_custom_weights(
    sample_dem_file: Path,
    sample_lulc_file: Path,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    custom = {"elevation": 0.5, "slope": 0.2, "river_proximity": 0.2, "land_cover": 0.1}
    fsi, _ = flood_susceptibility_index(
        sample_dem_file, sample_lulc_file, sample_river_gdf, weights=custom
    )
    assert fsi.shape == (20, 20)


def test_fsi_invalid_weights_raise(
    sample_dem_file: Path,
    sample_lulc_file: Path,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    bad = {"elevation": 0.5, "slope": 0.5, "river_proximity": 0.5, "land_cover": 0.5}
    with pytest.raises(ValueError, match="sum to 1.0"):
        flood_susceptibility_index(sample_dem_file, sample_lulc_file, sample_river_gdf, weights=bad)


def test_fsi_writes_output_file(
    tmp_path: Path,
    sample_dem_file: Path,
    sample_lulc_file: Path,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    out = tmp_path / "fsi.tif"
    flood_susceptibility_index(
        sample_dem_file, sample_lulc_file, sample_river_gdf, output_path=out
    )
    assert out.exists()
    assert out.stat().st_size > 0
