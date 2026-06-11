"""
Resolve study-area data paths.

Each ``ensure_*`` function returns the path to a ready-to-use file:
  - Real data (downloaded by scripts/fetch_data.py) is preferred.
  - If absent a synthetic representation is created and cached.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box

VECTOR_DIR = Path("data/vector")
RASTER_DIR = Path("data/raster")

# Sylhet District bounding box (W, S, E, N)
BBOX = (91.5, 24.0, 92.5, 25.5)


# ── Public helpers ─────────────────────────────────────────────────────────────

def ensure_admin_boundaries() -> Path:
    """Return path to Sylhet upazila boundaries (real GADM data preferred over synthetic).

    Priority: GADM GeoPackage → GADM Shapefile → synthetic GeoPackage → generate synthetic.
    """
    for candidate in [
        VECTOR_DIR / "sylhet_upazilas.gpkg",
        VECTOR_DIR / "sylhet_upazilas_shp" / "sylhet_upazilas.shp",
        VECTOR_DIR / "sylhet_upazilas_synthetic.gpkg",
    ]:
        if candidate.exists():
            return candidate
    return _make_synthetic_admin(VECTOR_DIR / "sylhet_upazilas_synthetic.gpkg")


def ensure_waterways() -> Path:
    """Return path to Sylhet waterway LineString GeoPackage (real or synthetic)."""
    for candidate in [
        VECTOR_DIR / "sylhet_waterways.gpkg",
        VECTOR_DIR / "sylhet_waterways_synthetic.gpkg",
    ]:
        if candidate.exists():
            return candidate
    return _make_synthetic_waterways(VECTOR_DIR / "sylhet_waterways_synthetic.gpkg")


def ensure_dem() -> Path:
    """Return path to Sylhet DEM GeoTIFF (real SRTM or synthetic)."""
    for candidate in [
        RASTER_DIR / "srtm_sylhet.tif",
        RASTER_DIR / "srtm_sylhet_synthetic.tif",
    ]:
        if candidate.exists():
            return candidate
    return _make_synthetic_dem(RASTER_DIR / "srtm_sylhet_synthetic.tif")


def ensure_lulc() -> Path:
    """Return path to Sylhet LULC GeoTIFF (real ESA WorldCover or synthetic)."""
    for candidate in [
        RASTER_DIR / "esa_worldcover_sylhet.tif",
        RASTER_DIR / "esa_worldcover_sylhet_synthetic.tif",
    ]:
        if candidate.exists():
            return candidate
    return _make_synthetic_lulc(RASTER_DIR / "esa_worldcover_sylhet_synthetic.tif")


# ── Synthetic data generators ─────────────────────────────────────────────────

def _make_synthetic_admin(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = BBOX
    lon_edges = np.linspace(west, east, 4)
    lat_edges = np.linspace(south, north, 5)
    polys, names = [], []
    for r in range(4):
        for c in range(3):
            polys.append(box(lon_edges[c], lat_edges[r], lon_edges[c + 1], lat_edges[r + 1]))
            names.append(f"Upazila_{r}_{c}")
    gdf = gpd.GeoDataFrame(
        {"upazila_name": names, "district": ["Sylhet"] * 12},
        geometry=polys, crs="EPSG:4326",
    )
    gdf.to_file(str(out), driver="GPKG")
    return out


def _make_synthetic_waterways(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    surma = LineString([(91.6, 24.3), (91.75, 24.5), (91.9, 24.7), (92.1, 24.9), (92.3, 25.1)])
    kushiyara = LineString([(92.0, 24.1), (92.1, 24.4), (92.2, 24.7), (92.25, 25.0)])
    gdf = gpd.GeoDataFrame(
        {"name": ["Surma", "Kushiyara"], "waterway": ["river", "river"]},
        geometry=[surma, kushiyara], crs="EPSG:4326",
    )
    gdf.to_file(str(out), driver="GPKG")
    return out


def _make_synthetic_dem(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = BBOX
    rows, cols = 150, 100
    transform = from_bounds(west, south, east, north, cols, rows)
    rng = np.random.default_rng(seed=2024)
    xx, yy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    base = (xx * 80 + yy * 40).astype(np.float32)
    noise = rng.normal(0, 8, (rows, cols)).astype(np.float32)
    river_band = np.abs(yy - 0.5 - 0.3 * xx) < 0.08
    data = np.clip(base + noise - river_band.astype(np.float32) * 30, 0, 200).astype(np.int16)
    with rasterio.open(out, "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="int16", crs="EPSG:4326",
                       transform=transform, nodata=-32768) as dst:
        dst.write(data, 1)
    return out


def _make_synthetic_lulc(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = BBOX
    rows, cols = 150, 100
    transform = from_bounds(west, south, east, north, cols, rows)
    xx, yy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    base = (xx * 80 + yy * 40).astype(np.float32)
    river_band = np.abs(yy - 0.5 - 0.3 * xx) < 0.08
    wetland_fringe = (np.abs(yy - 0.5 - 0.3 * xx) < 0.14) & (~river_band)
    city = (xx > 0.4) & (xx < 0.6) & (yy > 0.55) & (yy < 0.75)
    data = np.full((rows, cols), 40, dtype=np.uint8)
    data[base > 90] = 10
    data[wetland_fringe] = 90
    data[river_band] = 80
    data[city] = 50
    with rasterio.open(out, "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="uint8", crs="EPSG:4326",
                       transform=transform, nodata=0) as dst:
        dst.write(data, 1)
    return out
