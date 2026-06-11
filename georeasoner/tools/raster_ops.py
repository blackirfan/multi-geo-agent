"""Pure raster operations — no LLM, no side effects unless output_path is given."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


def compute_ndwi(
    src_path: Path,
    green_band: int = 1,
    nir_band: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    Compute the Normalised Difference Water Index.

    NDWI = (Green − NIR) / (Green + NIR)

    Returns a float32 array in [-1, 1] and an updated metadata dict.
    Pixels where Green + NIR == 0 are set to 0.0.
    """
    with rasterio.open(src_path) as src:
        green = src.read(green_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)
        nodata = src.nodata
        meta = src.meta.copy()

    if nodata is not None:
        green[green == nodata] = np.nan
        nir[nir == nodata] = np.nan

    denom = green + nir
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(denom != 0, (green - nir) / denom, 0.0).astype(np.float32)

    meta.update(dtype="float32", count=1, nodata=np.nan)
    return ndwi, meta


def compute_slope(dem_path: Path) -> tuple[np.ndarray, dict]:
    """
    Compute slope in degrees from a single-band DEM GeoTIFF.

    Uses NumPy's central-difference gradient. Geographic CRS pixel sizes
    are converted to approximate metres using the standard haversine scaling.
    Returns a float32 array (degrees) and updated metadata.
    """
    with rasterio.open(dem_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        meta = src.meta.copy()

    if nodata is not None:
        data[data == nodata] = np.nan

    if crs.is_geographic:
        lat_c = (bounds.top + bounds.bottom) / 2.0
        cell_x_m = abs(transform.a) * 111_320.0 * np.cos(np.radians(lat_c))
        cell_y_m = abs(transform.e) * 111_320.0
    else:
        cell_x_m = abs(transform.a)
        cell_y_m = abs(transform.e)

    # np.gradient returns [dy_gradient, dx_gradient] for 2-D arrays
    grad_y, grad_x = np.gradient(data, cell_y_m, cell_x_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2))).astype(np.float32)

    if nodata is not None:
        slope_deg[np.isnan(data)] = np.nan

    meta.update(dtype="float32", count=1, nodata=np.nan)
    return slope_deg, meta


def reclassify_raster(
    data: np.ndarray,
    class_map: dict[int, float],
    nodata_val: float = np.nan,
) -> np.ndarray:
    """
    Remap integer class values to floats via *class_map*.

    Any pixel whose value is not in *class_map* receives *nodata_val*.
    Returns a float32 array.
    """
    out = np.full(data.shape, nodata_val, dtype=np.float32)
    for src_class, dst_val in class_map.items():
        out[data == src_class] = dst_val
    return out


def zonal_stats(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    band: int = 1,
    stats: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute raster statistics for each polygon in *gdf*.

    Supported stat names: ``mean``, ``min``, ``max``, ``std``, ``count``.
    Returns a copy of *gdf* with one extra column per requested stat.
    Polygons that don't intersect the raster receive NaN.
    """
    if stats is None:
        stats = ["mean", "min", "max", "std"]

    records: list[dict] = []

    with rasterio.open(raster_path) as src:
        src_nodata = src.nodata

        for geom in gdf.geometry:
            try:
                windowed, _ = mask(src, [geom.__geo_interface__], crop=True, all_touched=True)
                arr = windowed[band - 1].astype(np.float32)
                if src_nodata is not None:
                    arr[arr == src_nodata] = np.nan
                valid = arr[~np.isnan(arr)]
            except Exception:
                valid = np.array([], dtype=np.float32)

            rec: dict = {}
            if "mean" in stats:
                rec["mean"] = float(np.nanmean(valid)) if valid.size > 0 else np.nan
            if "min" in stats:
                rec["min"] = float(np.nanmin(valid)) if valid.size > 0 else np.nan
            if "max" in stats:
                rec["max"] = float(np.nanmax(valid)) if valid.size > 0 else np.nan
            if "std" in stats:
                rec["std"] = float(np.nanstd(valid)) if valid.size > 0 else np.nan
            if "count" in stats:
                rec["count"] = int(valid.size)
            records.append(rec)

    result = gdf.reset_index(drop=True).copy()
    for col in stats:
        result[col] = [r.get(col, np.nan) for r in records]
    return result


def write_raster(data: np.ndarray, meta: dict, output_path: Path) -> Path:
    """Write a 2-D or 3-D numpy array to a LZW-compressed GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta = meta.copy()
    out_meta.update(driver="GTiff", dtype=str(data.dtype), compress="lzw")

    if data.ndim == 2:
        out_meta["count"] = 1
        data = data[np.newaxis, ...]
    else:
        out_meta["count"] = data.shape[0]

    out_meta["height"] = data.shape[1]
    out_meta["width"] = data.shape[2]

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(data)
    return output_path
