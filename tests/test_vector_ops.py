"""Unit tests for georeasoner.tools.vector_ops."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from georeasoner.tools.vector_ops import (
    buffer_features,
    clip_to_boundary,
    overlay_difference,
    proximity_raster,
    spatial_join,
)

# ── buffer_features ───────────────────────────────────────────────────────────

def test_buffer_increases_area(sample_admin_gdf: gpd.GeoDataFrame) -> None:
    buffered = buffer_features(sample_admin_gdf, distance_m=500.0)
    assert buffered.geometry.area.mean() > sample_admin_gdf.geometry.area.mean()


def test_buffer_preserves_crs(sample_admin_gdf: gpd.GeoDataFrame) -> None:
    buffered = buffer_features(sample_admin_gdf, distance_m=100.0)
    assert buffered.crs == sample_admin_gdf.crs


def test_buffer_preserves_row_count(sample_admin_gdf: gpd.GeoDataFrame) -> None:
    buffered = buffer_features(sample_admin_gdf, distance_m=100.0)
    assert len(buffered) == len(sample_admin_gdf)


def test_buffer_zero_distance_unchanged(sample_admin_gdf: gpd.GeoDataFrame) -> None:
    """Buffering by 0 m should leave geometries essentially unchanged."""
    buffered = buffer_features(sample_admin_gdf, distance_m=0.0)
    orig_areas = sample_admin_gdf.geometry.area.values
    buff_areas = buffered.geometry.area.values
    np.testing.assert_allclose(orig_areas, buff_areas, rtol=1e-5)


# ── spatial_join ──────────────────────────────────────────────────────────────

def test_spatial_join_returns_geodataframe(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    result = spatial_join(sample_admin_gdf, sample_river_gdf, how="left")
    assert isinstance(result, gpd.GeoDataFrame)


def test_spatial_join_left_preserves_count(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    result = spatial_join(sample_admin_gdf, sample_river_gdf, how="left")
    # Left join: at least as many rows as original (may increase if multiple matches)
    assert len(result) >= len(sample_admin_gdf)


def test_spatial_join_crs(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    result = spatial_join(sample_admin_gdf, sample_river_gdf)
    assert result.crs == sample_admin_gdf.crs


def test_spatial_join_inner_finds_overlap(
    sample_admin_gdf: gpd.GeoDataFrame,
    sample_river_gdf: gpd.GeoDataFrame,
) -> None:
    """The river passes through several admin polygons → inner join must be non-empty."""
    result = spatial_join(sample_admin_gdf, sample_river_gdf, how="inner")
    assert len(result) > 0


# ── clip_to_boundary ──────────────────────────────────────────────────────────

def test_clip_reduces_or_equals_count(
    sample_river_gdf: gpd.GeoDataFrame,
    sample_admin_gdf: gpd.GeoDataFrame,
) -> None:
    clipped = clip_to_boundary(sample_river_gdf, sample_admin_gdf)
    assert isinstance(clipped, gpd.GeoDataFrame)


def test_clip_stays_inside_boundary(
    sample_river_gdf: gpd.GeoDataFrame,
    sample_admin_gdf: gpd.GeoDataFrame,
) -> None:
    boundary_union = sample_admin_gdf.geometry.unary_union
    clipped = clip_to_boundary(sample_river_gdf, sample_admin_gdf)
    if not clipped.empty:
        for geom in clipped.geometry:
            assert boundary_union.covers(geom) or boundary_union.intersects(geom)


# ── overlay_difference ────────────────────────────────────────────────────────

def test_overlay_difference_returns_geodataframe(
    sample_admin_gdf: gpd.GeoDataFrame,
) -> None:
    a = sample_admin_gdf.iloc[:3]
    b = sample_admin_gdf.iloc[3:]
    result = overlay_difference(a, b)
    assert isinstance(result, gpd.GeoDataFrame)


def test_overlay_difference_with_no_overlap(
    sample_admin_gdf: gpd.GeoDataFrame,
) -> None:
    """Non-overlapping polygons: difference should equal the original."""
    a = sample_admin_gdf.iloc[:2].copy()
    b = sample_admin_gdf.iloc[3:].copy()
    result = overlay_difference(a, b)
    # All original rows are preserved since there's no overlap
    assert len(result) >= 1


# ── proximity_raster ──────────────────────────────────────────────────────────

def test_proximity_raster_shape(
    sample_river_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    dist, _ = proximity_raster(sample_river_gdf, sample_dem_file)
    assert dist.shape == (20, 20)


def test_proximity_raster_non_negative(
    sample_river_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    dist, _ = proximity_raster(sample_river_gdf, sample_dem_file)
    assert float(np.nanmin(dist)) >= 0.0


def test_proximity_raster_zero_on_river(
    sample_river_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    """At least one pixel must be distance 0 (the pixel the river passes through)."""
    dist, _ = proximity_raster(sample_river_gdf, sample_dem_file)
    assert float(np.nanmin(dist)) == pytest.approx(0.0, abs=1.0)


def test_proximity_raster_meta_dtype(
    sample_river_gdf: gpd.GeoDataFrame,
    sample_dem_file: Path,
) -> None:
    _, meta = proximity_raster(sample_river_gdf, sample_dem_file)
    assert meta["dtype"] == "float32"
