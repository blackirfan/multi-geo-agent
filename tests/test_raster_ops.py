"""Unit tests for georeasoner.tools.raster_ops."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from georeasoner.tools.raster_ops import (
    compute_ndwi,
    compute_slope,
    reclassify_raster,
    write_raster,
    zonal_stats,
)

# ── compute_ndwi ──────────────────────────────────────────────────────────────

def test_ndwi_shape(sample_multispectral_file: Path) -> None:
    ndwi, _ = compute_ndwi(sample_multispectral_file, green_band=1, nir_band=2)
    assert ndwi.shape == (20, 20)


def test_ndwi_range(sample_multispectral_file: Path) -> None:
    ndwi, _ = compute_ndwi(sample_multispectral_file, green_band=1, nir_band=2)
    assert float(np.nanmin(ndwi)) >= -1.0
    assert float(np.nanmax(ndwi)) <= 1.0


def test_ndwi_positive_when_green_dominates(sample_multispectral_file: Path) -> None:
    """Fixture has green > NIR for most pixels → mean NDWI should be positive."""
    ndwi, _ = compute_ndwi(sample_multispectral_file, green_band=1, nir_band=2)
    assert float(np.nanmean(ndwi)) > 0.0


def test_ndwi_meta_dtype(sample_multispectral_file: Path) -> None:
    _, meta = compute_ndwi(sample_multispectral_file)
    assert meta["dtype"] == "float32"
    assert meta["count"] == 1


# ── compute_slope ─────────────────────────────────────────────────────────────

def test_slope_shape(sample_dem_file: Path) -> None:
    slope, _ = compute_slope(sample_dem_file)
    assert slope.shape == (20, 20)


def test_slope_range(sample_dem_file: Path) -> None:
    slope, _ = compute_slope(sample_dem_file)
    assert float(np.nanmin(slope)) >= 0.0
    assert float(np.nanmax(slope)) < 90.0


def test_slope_dtype(sample_dem_file: Path) -> None:
    slope, meta = compute_slope(sample_dem_file)
    assert slope.dtype == np.float32
    assert meta["dtype"] == "float32"


def test_slope_flat_dem_is_zero(tmp_path: Path, small_bbox: tuple) -> None:
    """A perfectly flat DEM must yield zero slope everywhere."""
    from rasterio.transform import from_bounds

    west, south, east, north = small_bbox
    transform = from_bounds(west, south, east, north, 10, 10)
    flat = np.full((1, 10, 10), 50, dtype=np.int16)
    path = tmp_path / "flat_dem.tif"
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10,
                       count=1, dtype="int16", crs="EPSG:4326",
                       transform=transform, nodata=-32768) as dst:
        dst.write(flat)

    slope, _ = compute_slope(path)
    assert float(np.nanmax(np.abs(slope))) < 1e-4


# ── reclassify_raster ─────────────────────────────────────────────────────────

def test_reclassify_maps_values() -> None:
    data = np.array([[10, 20, 30], [40, 80, 90]], dtype=np.uint8)
    class_map = {10: 0.2, 40: 0.6, 80: 1.0}
    out = reclassify_raster(data, class_map)
    assert out[0, 0] == pytest.approx(0.2)
    assert out[1, 0] == pytest.approx(0.6)
    assert out[1, 1] == pytest.approx(1.0)


def test_reclassify_unknown_class_is_nan() -> None:
    data = np.array([[99]], dtype=np.uint8)
    out = reclassify_raster(data, {10: 0.5})
    assert np.isnan(out[0, 0])


def test_reclassify_output_dtype() -> None:
    data = np.array([[10, 20]], dtype=np.uint8)
    out = reclassify_raster(data, {10: 0.1, 20: 0.2})
    assert out.dtype == np.float32


# ── zonal_stats ───────────────────────────────────────────────────────────────

def test_zonal_stats_returns_stat_columns(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    result = zonal_stats(sample_admin_gdf, sample_dem_file, stats=["mean", "min", "max"])
    for col in ["mean", "min", "max"]:
        assert col in result.columns


def test_zonal_stats_row_count(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    result = zonal_stats(sample_admin_gdf, sample_dem_file)
    assert len(result) == len(sample_admin_gdf)


def test_zonal_stats_mean_in_range(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    result = zonal_stats(sample_admin_gdf, sample_dem_file, stats=["mean"])
    valid_means = result["mean"].dropna()
    assert len(valid_means) > 0
    assert (valid_means >= 0).all()
    assert (valid_means <= 200).all()


def test_zonal_stats_count(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    result = zonal_stats(sample_admin_gdf, sample_dem_file, stats=["count"])
    assert "count" in result.columns
    valid_counts = result["count"].dropna()
    assert (valid_counts >= 0).all()


# ── write_raster ──────────────────────────────────────────────────────────────

def test_write_raster_creates_file(tmp_path: Path, sample_dem_file: Path) -> None:
    with rasterio.open(sample_dem_file) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    out = tmp_path / "output.tif"
    result = write_raster(data, meta, out)
    assert result == out
    assert out.exists()


def test_write_raster_readable(tmp_path: Path, sample_dem_file: Path) -> None:
    with rasterio.open(sample_dem_file) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    out = tmp_path / "output.tif"
    write_raster(data, meta, out)
    with rasterio.open(out) as dst:
        assert dst.count == 1
        assert dst.width == data.shape[1]
        assert dst.height == data.shape[0]
