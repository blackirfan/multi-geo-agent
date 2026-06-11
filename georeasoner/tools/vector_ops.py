"""Pure vector operations — no LLM, no side effects."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt


def buffer_features(gdf: gpd.GeoDataFrame, distance_m: float) -> gpd.GeoDataFrame:
    """
    Buffer every feature by *distance_m* metres.

    Projects to the dataset's UTM zone, buffers, then reprojects back to
    the original CRS.  Returns a new GeoDataFrame.
    """
    utm_crs = gdf.estimate_utm_crs()
    projected = gdf.to_crs(utm_crs)
    buffered = projected.copy()
    buffered.geometry = projected.geometry.buffer(distance_m)
    return buffered.to_crs(gdf.crs)


def spatial_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    how: str = "left",
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between *left* and *right*.

    Both inputs are projected to *left*'s CRS before joining.
    Returns a new GeoDataFrame with the combined attributes.
    """
    right_aligned = right.to_crs(left.crs)
    return gpd.sjoin(left, right_aligned, how=how, predicate=predicate)


def clip_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Clip *gdf* to the union of *boundary* geometries.

    Returns only the portions of features that fall inside the boundary.
    """
    boundary_aligned = boundary.to_crs(gdf.crs)
    return gpd.clip(gdf, boundary_aligned)


def overlay_difference(
    gdf: gpd.GeoDataFrame,
    clip_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return the parts of *gdf* that do NOT overlap with *clip_gdf*."""
    clip_aligned = clip_gdf.to_crs(gdf.crs)
    return gpd.overlay(gdf, clip_aligned, how="difference", keep_geom_type=True)


def proximity_raster(
    reference_gdf: gpd.GeoDataFrame,
    like_raster_path: Path,
) -> tuple[np.ndarray, dict]:
    """
    Compute a raster of Euclidean distances (metres) from each pixel to the
    nearest feature in *reference_gdf*, matching the grid of *like_raster_path*.

    Steps:
      1. Reproject *reference_gdf* to the raster's CRS.
      2. Rasterize the features (burned = 1, background = 0).
      3. Apply ``scipy.ndimage.distance_transform_edt`` to get pixel-distances.
      4. Convert pixel-distances to metres using approximate cell size.

    Returns ``(distance_array_float32, updated_meta)``.
    """
    with rasterio.open(like_raster_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        meta = src.meta.copy()

    ref = reference_gdf.to_crs(crs)

    # Rasterize: pixels that touch a reference feature get value 1
    shapes = (
        (geom.__geo_interface__, 1)
        for geom in ref.geometry
        if geom is not None and not geom.is_empty
    )
    burned = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Distance transform (in pixels) → convert to metres
    dist_px = distance_transform_edt(burned == 0).astype(np.float32)

    if crs.is_geographic:
        lat_c = (bounds.top + bounds.bottom) / 2.0
        px_x_m = abs(transform.a) * 111_320.0 * np.cos(np.radians(lat_c))
        px_y_m = abs(transform.e) * 111_320.0
        px_m = (px_x_m + px_y_m) / 2.0
    else:
        px_m = abs(transform.a)

    dist_m = (dist_px * px_m).astype(np.float32)

    meta.update(dtype="float32", count=1, nodata=np.nan)
    return dist_m, meta
