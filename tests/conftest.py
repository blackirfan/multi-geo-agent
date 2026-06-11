"""Shared pytest fixtures — all synthetic, no network I/O required."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box


@pytest.fixture
def study_bbox() -> tuple[float, float, float, float]:
    """Full Sylhet study area (W, S, E, N)."""
    return (91.5, 24.0, 92.5, 25.5)


@pytest.fixture
def small_bbox() -> tuple[float, float, float, float]:
    """Tiny sub-area for fast unit tests (W, S, E, N)."""
    return (91.70, 24.30, 91.90, 24.50)


@pytest.fixture
def sample_admin_gdf() -> gpd.GeoDataFrame:
    """Five synthetic upazila polygons inside the study area."""
    polygons = [
        box(91.70 + i * 0.04, 24.30, 91.74 + i * 0.04, 24.34)
        for i in range(5)
    ]
    return gpd.GeoDataFrame(
        {
            "upazila_id": [f"U{i:02d}" for i in range(5)],
            "upazila_name": [f"Upazila_{i}" for i in range(5)],
            "district": ["Sylhet"] * 5,
        },
        geometry=polygons,
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_river_gdf() -> gpd.GeoDataFrame:
    """Simplified Surma river segment."""
    line = LineString([(91.75, 24.30), (91.80, 24.35), (91.85, 24.32)])
    return gpd.GeoDataFrame(
        {"name": ["Surma"], "waterway": ["river"]},
        geometry=[line],
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_dem_file(tmp_path: Path, small_bbox: tuple) -> Path:
    """20×20 synthetic DEM GeoTIFF (int16, EPSG:4326)."""
    west, south, east, north = small_bbox
    width, height = 20, 20
    transform = from_bounds(west, south, east, north, width, height)
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 200, (1, height, width), dtype=np.int16)
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width,
        count=1, dtype="int16", crs="EPSG:4326",
        transform=transform, nodata=-32768,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def sample_lulc_file(tmp_path: Path, small_bbox: tuple) -> Path:
    """20×20 synthetic LULC GeoTIFF using ESA WorldCover class codes (uint8)."""
    west, south, east, north = small_bbox
    width, height = 20, 20
    transform = from_bounds(west, south, east, north, width, height)
    # ESA WorldCover: 10=tree, 20=shrub, 30=grass, 40=crop, 60=bare, 80=water, 90=wetland
    esa_classes = np.array([10, 20, 30, 40, 60, 80, 90], dtype=np.uint8)
    rng = np.random.default_rng(seed=7)
    data = rng.choice(esa_classes, (1, height, width)).astype(np.uint8)
    path = tmp_path / "lulc.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width,
        count=1, dtype="uint8", crs="EPSG:4326",
        transform=transform, nodata=0,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def sample_multispectral_file(tmp_path: Path, small_bbox: tuple) -> Path:
    """20×20 2-band GeoTIFF (band 1 = green, band 2 = NIR) for NDWI tests."""
    west, south, east, north = small_bbox
    width, height = 20, 20
    transform = from_bounds(west, south, east, north, width, height)
    rng = np.random.default_rng(seed=13)
    # Green > NIR → positive NDWI (water-like pixels)
    green = rng.integers(500, 1000, (height, width), dtype=np.int16)
    nir = rng.integers(200, 600, (height, width), dtype=np.int16)
    path = tmp_path / "multispectral.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width,
        count=2, dtype="int16", crs="EPSG:4326",
        transform=transform, nodata=-9999,
    ) as dst:
        dst.write(green, 1)
        dst.write(nir, 2)
    return path
