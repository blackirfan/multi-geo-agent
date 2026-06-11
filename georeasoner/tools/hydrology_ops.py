"""Hydrology tools — flood susceptibility index via weighted raster overlay."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from georeasoner.tools.raster_ops import compute_slope, reclassify_raster, write_raster
from georeasoner.tools.vector_ops import proximity_raster

# ESA WorldCover 2021 class → flood vulnerability score [0, 1]
# Higher score = more susceptible to flooding
ESA_VULNERABILITY: dict[int, float] = {
    10: 0.20,   # Tree cover          — canopy interception reduces runoff
    20: 0.30,   # Shrubland
    30: 0.50,   # Grassland
    40: 0.60,   # Cropland            — flat, often saturated in monsoon
    50: 0.70,   # Built-up            — impervious, poor drainage
    60: 0.50,   # Bare / sparse veg
    70: 0.10,   # Snow and ice        — negligible in Sylhet
    80: 1.00,   # Permanent water     — always inundated
    90: 0.95,   # Herbaceous wetland  — seasonally flooded
    95: 0.40,   # Mangroves
    100: 0.30,  # Moss and lichen
}

# Weights must sum to 1.0
DEFAULT_WEIGHTS: dict[str, float] = {
    "elevation": 0.35,
    "slope": 0.25,
    "river_proximity": 0.25,
    "land_cover": 0.15,
}


def flood_susceptibility_index(
    dem_path: Path,
    lulc_path: Path,
    river_gdf: gpd.GeoDataFrame,
    weights: dict[str, float] | None = None,
    output_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Compute a normalised Flood Susceptibility Index (FSI) in [0, 1].

    FSI = Σ wᵢ × normalised_layer_i

    Four factor layers (all normalised 0–1, higher = more susceptible):
      * **elevation**       – inverted; low elevation → high susceptibility
      * **slope**           – inverted; flat terrain → high susceptibility
      * **river_proximity** – inverted; short distance to river → high susceptibility
      * **land_cover**      – ESA WorldCover classes remapped via ESA_VULNERABILITY

    All input rasters are resampled to the DEM grid before computation.

    Parameters
    ----------
    dem_path:     Path to single-band DEM GeoTIFF.
    lulc_path:    Path to ESA WorldCover (or equivalent) LULC GeoTIFF.
    river_gdf:    GeoDataFrame of river/waterway line features.
    weights:      Override default weights (must sum to 1.0).
    output_path:  If given, the FSI raster is written here as a GeoTIFF.

    Returns
    -------
    (fsi_array float32, metadata dict)
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(w.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

    # ── 1. Elevation layer ────────────────────────────────────────────────
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float32)
        nodata = src.nodata
        meta = src.meta.copy()

    if nodata is not None:
        elev[elev == nodata] = np.nan

    # ── 2. Slope layer ────────────────────────────────────────────────────
    slope, _ = compute_slope(dem_path)

    # ── 3. River proximity layer ──────────────────────────────────────────
    river_dist, _ = proximity_raster(river_gdf, dem_path)

    # ── 4. Land-cover vulnerability layer ────────────────────────────────
    lulc_raw = _resample_to_match(lulc_path, dem_path, Resampling.nearest)
    lulc_vuln = reclassify_raster(lulc_raw.astype(np.int16), ESA_VULNERABILITY)

    # ── 5. Normalise all layers to [0, 1] ─────────────────────────────────
    e_norm = _norm_invert(elev)    # low elevation → 1
    s_norm = _norm_invert(slope)   # flat slope   → 1
    r_norm = _norm_invert(river_dist)  # near river → 1
    lc_norm = _norm(lulc_vuln)         # already 0-1 vulnerability

    # ── 6. Weighted sum ───────────────────────────────────────────────────
    fsi = (
        w["elevation"] * e_norm
        + w["slope"] * s_norm
        + w["river_proximity"] * r_norm
        + w["land_cover"] * lc_norm
    ).astype(np.float32)

    # Propagate NaN from DEM nodata mask
    nan_mask = np.isnan(elev) | np.isnan(slope)
    fsi[nan_mask] = np.nan

    meta.update(dtype="float32", count=1, nodata=np.nan)

    if output_path is not None:
        write_raster(fsi, meta, Path(output_path))

    return fsi, meta


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resample_to_match(
    src_path: Path,
    ref_path: Path,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject/resample *src_path* band-1 to exactly match the grid of *ref_path*."""
    with rasterio.open(ref_path) as ref:
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs

    dst = np.zeros(dst_shape, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_crs=src.crs,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            resampling=resampling,
        )
    return dst


def _norm(arr: np.ndarray) -> np.ndarray:
    """Min–max normalise to [0, 1]; returns zeros if range is zero."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    mn, mx = float(valid.min()), float(valid.max())
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _norm_invert(arr: np.ndarray) -> np.ndarray:
    """Normalise then invert so that *low* original values map to *high* output."""
    return 1.0 - _norm(arr)
