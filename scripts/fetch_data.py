#!/usr/bin/env python3
"""
Download all geospatial datasets required for the GeoReasoner Sylhet flood analysis.

Sources (all public, no authentication required):
  - GADM 4.1  → admin boundaries for Bangladesh (level 3 = upazilas)
  - OSMnx     → waterways (rivers, streams, canals) via Overpass API
  - AWS Terrain Tiles (Tilezen / elevation-tiles-prod)
              → SRTM 1-arc-second (~30 m) HGT tiles, merged + clipped
  - ESA WorldCover 2021 v200 (10 m LULC)
              → public S3 bucket, 3×3-degree COG tile, clipped to study area

Usage:
    python scripts/fetch_data.py [--bbox W S E N]

All outputs land in data/vector/ and data/raster/.
"""

import gzip
import shutil
import zipfile
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import rasterio
import requests
import typer
from rasterio.mask import mask
from rasterio.merge import merge
from rich.console import Console
from shapely.geometry import box
from tqdm import tqdm

console = Console()
app = typer.Typer(help=__doc__, add_completion=False)

VECTOR_DIR = Path("data/vector")
RASTER_DIR = Path("data/raster")

# Default study-area bbox: Sylhet District, Bangladesh (W, S, E, N)
DEFAULT_BBOX = (91.5, 24.0, 92.5, 25.5)


class Bbox(NamedTuple):
    west: float
    south: float
    east: float
    north: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, label: str) -> bool:
    """Stream-download *url* to *dest* with a progress bar. Returns success."""
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=label, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as exc:
        console.print(f"  [red]✗ {label}: {exc}[/red]")
        return False


def _clip_raster(src_path: Path, dest_path: Path, bbox: Bbox) -> None:
    """Clip *src_path* to *bbox*, write LZW-compressed GeoTIFF to *dest_path*."""
    clip_geom = [box(bbox.west, bbox.south, bbox.east, bbox.north).__geo_interface__]
    with rasterio.open(src_path) as src:
        out_image, out_transform = mask(src, clip_geom, crop=True)
        meta = src.meta.copy()
        meta.update(
            driver="GTiff",
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="lzw",
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest_path, "w", **meta) as dst:
        dst.write(out_image)


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def fetch_gadm_admin(bbox: Bbox) -> None:
    """Download GADM 4.1 BGD level-3 (upazilas) and filter to Sylhet District."""
    console.print("[bold cyan]→ GADM 4.1 Bangladesh Admin Level 3[/bold cyan]")
    out = VECTOR_DIR / "sylhet_upazilas.gpkg"
    if out.exists():
        console.print(f"  [green]✓ already present:[/green] {out}")
        return

    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_3.json.zip"
    tmp_zip = VECTOR_DIR / "gadm41_BGD_3.json.zip"

    if not _download(url, tmp_zip, "GADM BGD L3"):
        return

    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(VECTOR_DIR)
    tmp_zip.unlink(missing_ok=True)

    raw = VECTOR_DIR / "gadm41_BGD_3.json"
    gdf: gpd.GeoDataFrame = gpd.read_file(raw)
    # NAME_2 == "Sylhet" is the district column in GADM level-3
    sylhet = gdf[gdf["NAME_2"].str.strip().str.lower() == "sylhet"].copy()
    sylhet.to_file(out, driver="GPKG")
    raw.unlink(missing_ok=True)
    console.print(f"  [green]✓ {len(sylhet)} upazila polygons →[/green] {out}")


def fetch_osm_rivers(bbox: Bbox) -> None:
    """Fetch OSM waterways (rivers, streams, canals) via OSMnx."""
    console.print("[bold cyan]→ OSM Waterways (rivers / streams / canals)[/bold cyan]")
    out = VECTOR_DIR / "sylhet_waterways.gpkg"
    if out.exists():
        console.print(f"  [green]✓ already present:[/green] {out}")
        return

    try:
        import osmnx as ox  # imported here; not a hard dep for CI smoke tests

        tags = {"waterway": ["river", "stream", "canal"]}
        # osmnx 2.x bbox convention: (left, bottom, right, top) = (W, S, E, N)
        gdf: gpd.GeoDataFrame = ox.features_from_bbox(
            bbox=(bbox.west, bbox.south, bbox.east, bbox.north),
            tags=tags,
        )
        if gdf.empty:
            console.print("  [yellow]⚠ No waterway features returned[/yellow]")
            return
        lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        lines = lines.reset_index(drop=True)
        lines.to_file(out, driver="GPKG")
        console.print(f"  [green]✓ {len(lines)} waterway segments →[/green] {out}")
    except Exception as exc:
        console.print(f"  [red]✗ OSMnx error: {exc}[/red]")


def fetch_srtm_dem(bbox: Bbox) -> None:
    """
    Download SRTM 1-arc-second (~30 m) HGT tiles from the public AWS Terrain
    Tiles dataset (s3://elevation-tiles-prod/skadi/…), merge, clip, and save.

    Falls back to CGIAR SRTM 90 m if the AWS tiles are unavailable.
    """
    console.print("[bold cyan]→ SRTM DEM (30 m, AWS Terrain Tiles)[/bold cyan]")
    out = RASTER_DIR / "srtm_sylhet.tif"
    if out.exists():
        console.print(f"  [green]✓ already present:[/green] {out}")
        return

    # Determine which 1°×1° HGT tiles overlap our bbox
    lat_range = range(int(bbox.south), int(bbox.north) + 1)
    lon_range = range(int(bbox.west), int(bbox.east) + 1)
    tiles: list[tuple[str, str]] = []
    for lat in lat_range:
        lat_dir = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
        for lon in lon_range:
            lon_tag = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tile = f"{lat_dir}{lon_tag}"
            tiles.append((lat_dir, tile))

    base = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
    tmp_files: list[Path] = []

    for lat_dir, tile in tiles:
        url = f"{base}/{lat_dir}/{tile}.hgt.gz"
        gz_path = RASTER_DIR / f"{tile}.hgt.gz"
        hgt_path = RASTER_DIR / f"{tile}.hgt"

        if not _download(url, gz_path, f"SRTM tile {tile}"):
            console.print(f"  [yellow]⚠ Skipping tile {tile}[/yellow]")
            continue

        with gzip.open(gz_path, "rb") as gz_in, open(hgt_path, "wb") as hgt_out:
            shutil.copyfileobj(gz_in, hgt_out)
        gz_path.unlink(missing_ok=True)
        tmp_files.append(hgt_path)

    if not tmp_files:
        console.print("  [yellow]↩ Falling back to CGIAR SRTM 90 m[/yellow]")
        _fetch_srtm_cgiar(bbox, out)
        return

    # Merge tiles and clip to study area
    datasets = [rasterio.open(f) for f in tmp_files]
    merged, transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update(driver="GTiff", height=merged.shape[1], width=merged.shape[2],
                transform=transform, compress="lzw")
    for ds in datasets:
        ds.close()

    tmp_merged = RASTER_DIR / "_srtm_merged.tif"
    with rasterio.open(tmp_merged, "w", **meta) as dst:
        dst.write(merged)

    _clip_raster(tmp_merged, out, bbox)
    tmp_merged.unlink(missing_ok=True)
    for hgt in tmp_files:
        hgt.unlink(missing_ok=True)

    console.print(f"  [green]✓ SRTM merged + clipped →[/green] {out}")


def _fetch_srtm_cgiar(bbox: Bbox, out: Path) -> None:
    """Fallback: CGIAR SRTM v4.1 90 m tile covering Bangladesh (srtm_55_08)."""
    # Tile covering 90–95 E, 20–25 N
    url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/tiff/srtm_55_08.zip"
    tmp_zip = RASTER_DIR / "srtm_55_08.zip"
    if not _download(url, tmp_zip, "CGIAR srtm_55_08"):
        console.print(
            "  [red]✗ SRTM download failed. Manually download srtm_55_08.zip from[/red]\n"
            "  [red]  https://srtm.csi.cgiar.org → data/raster/ then re-run.[/red]"
        )
        return
    with zipfile.ZipFile(tmp_zip) as zf:
        tif_name = next(n for n in zf.namelist() if n.lower().endswith(".tif"))
        zf.extract(tif_name, RASTER_DIR)
    tmp_zip.unlink(missing_ok=True)
    raw = RASTER_DIR / tif_name
    _clip_raster(raw, out, bbox)
    raw.unlink(missing_ok=True)
    console.print(f"  [green]✓ CGIAR SRTM clipped →[/green] {out}")


def fetch_esa_worldcover(bbox: Bbox) -> None:
    """
    Download ESA WorldCover 2021 v200 (10 m LULC) COG tile N24E090 which covers
    24–27 N, 90–93 E — fully enclosing the Sylhet study area.
    """
    console.print("[bold cyan]→ ESA WorldCover 2021 v200 (10 m LULC)[/bold cyan]")
    out = RASTER_DIR / "esa_worldcover_sylhet.tif"
    if out.exists():
        console.print(f"  [green]✓ already present:[/green] {out}")
        return

    tile = "ESA_WorldCover_10m_2021_v200_N24E090_Map.tif"
    url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{tile}"
    tmp = RASTER_DIR / tile

    if not _download(url, tmp, "ESA WorldCover N24E090"):
        console.print(
            "  [red]✗ ESA WorldCover download failed.[/red]\n"
            "  [red]  Manual download: https://worldcover2021.esa.int/downloader[/red]\n"
            "  [red]  Select tile N24E090 → place in data/raster/ → re-run.[/red]"
        )
        return

    _clip_raster(tmp, out, bbox)
    tmp.unlink(missing_ok=True)
    console.print(f"  [green]✓ ESA WorldCover clipped →[/green] {out}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

@app.command()
def main(
    west: float = typer.Option(DEFAULT_BBOX[0], help="Bounding box west longitude"),
    south: float = typer.Option(DEFAULT_BBOX[1], help="Bounding box south latitude"),
    east: float = typer.Option(DEFAULT_BBOX[2], help="Bounding box east longitude"),
    north: float = typer.Option(DEFAULT_BBOX[3], help="Bounding box north latitude"),
) -> None:
    """Download all geospatial datasets for GeoReasoner."""
    bbox = Bbox(west, south, east, north)

    console.print(
        f"\n[bold]GeoReasoner — Data Fetch[/bold]\n"
        f"Bounding box: W={bbox.west} S={bbox.south} E={bbox.east} N={bbox.north}\n"
    )

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_DIR.mkdir(parents=True, exist_ok=True)

    fetch_gadm_admin(bbox)
    fetch_osm_rivers(bbox)
    fetch_srtm_dem(bbox)
    fetch_esa_worldcover(bbox)

    console.print("\n[bold green]✓ Data fetch complete.[/bold green]")
    console.print(f"  Vector → {VECTOR_DIR.resolve()}")
    console.print(f"  Raster → {RASTER_DIR.resolve()}")


if __name__ == "__main__":
    app()
