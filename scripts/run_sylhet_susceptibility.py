#!/usr/bin/env python3
"""
End-to-end Sylhet flood susceptibility pipeline — no LLM required.

Usage:
    python scripts/run_sylhet_susceptibility.py [--synthetic]

If real data exists in data/vector/ and data/raster/ the script uses it;
otherwise (or when --synthetic is passed) it generates representative
synthetic data that mimics the Sylhet area for demonstration purposes.

Outputs (all written to reports/):
  sylhet_flood_susceptibility.tif   — FSI raster [0, 1]
  sylhet_fsi_ranking.csv            — admin units ranked by mean FSI
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import typer
from rasterio.transform import from_bounds
from rich.console import Console
from rich.table import Table
from shapely.geometry import LineString, box

from georeasoner.tools.hydrology_ops import flood_susceptibility_index
from georeasoner.tools.raster_ops import zonal_stats

console = Console()
app = typer.Typer(add_completion=False)

DATA_VECTOR = Path("data/vector")
DATA_RASTER = Path("data/raster")
REPORTS_DIR = Path("reports")

# Sylhet bbox (W, S, E, N)
BBOX = (91.5, 24.0, 92.5, 25.5)


@app.command()
def main(synthetic: bool = typer.Option(False, help="Force synthetic data even if real data exists")) -> None:
    """Run the Sylhet flood susceptibility index pipeline."""
    console.print("\n[bold cyan]══ GeoReasoner — Sylhet Flood Susceptibility Index ══[/bold cyan]\n")

    dem_path, lulc_path, river_gdf, admin_gdf = _resolve_inputs(synthetic)

    # ── Compute FSI ──────────────────────────────────────────────────────────
    console.print("[bold]▶ Computing Flood Susceptibility Index (4 layers)...[/bold]")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_raster = REPORTS_DIR / "sylhet_flood_susceptibility.tif"

    fsi, meta = flood_susceptibility_index(
        dem_path=dem_path,
        lulc_path=lulc_path,
        river_gdf=river_gdf,
        output_path=out_raster,
    )
    valid = fsi[~np.isnan(fsi)]
    console.print(f"  [green]✓[/green] FSI raster → {out_raster}")
    console.print(f"  Shape: {fsi.shape}  |  Range: [{valid.min():.3f}, {valid.max():.3f}]  |  Mean: {valid.mean():.3f}")

    # ── Zonal stats per admin unit ────────────────────────────────────────────
    if admin_gdf is not None and not admin_gdf.empty:
        console.print("\n[bold]▶ Zonal statistics per administrative unit...[/bold]")
        stats_gdf = zonal_stats(admin_gdf, out_raster, stats=["mean", "min", "max", "std"])
        stats_gdf = stats_gdf.sort_values("mean", ascending=False).reset_index(drop=True)

        # Print ranked table
        name_col = next(
            (c for c in ["upazila_name", "NAME_3", "NAME_2", "name"] if c in stats_gdf.columns),
            None,
        )
        table = Table(title="Flood Susceptibility Ranking (top 10)", show_lines=False)
        table.add_column("Rank", style="dim", width=5)
        table.add_column("Unit", style="cyan")
        table.add_column("Mean FSI", justify="right")
        table.add_column("Max FSI", justify="right")

        for rank, (_, row) in enumerate(stats_gdf.head(10).iterrows(), 1):
            label = str(row[name_col]) if name_col else f"Unit {rank}"
            mean_v = row.get("mean", np.nan)
            max_v = row.get("max", np.nan)
            mean_str = f"{mean_v:.3f}" if not np.isnan(mean_v) else "—"
            max_str = f"{max_v:.3f}" if not np.isnan(max_v) else "—"
            table.add_row(str(rank), label, mean_str, max_str)

        console.print(table)

        csv_path = REPORTS_DIR / "sylhet_fsi_ranking.csv"
        stats_gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
        console.print(f"  [green]✓[/green] Full ranking → {csv_path}")

    console.print("\n[bold green]Pipeline complete.[/bold green]")
    console.print(f"  Raster : {out_raster.resolve()}")


# ── Input resolution ──────────────────────────────────────────────────────────

def _resolve_inputs(
    force_synthetic: bool,
) -> tuple[Path, Path, gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    dem_path = DATA_RASTER / "srtm_sylhet.tif"
    lulc_path = DATA_RASTER / "esa_worldcover_sylhet.tif"
    river_file = DATA_VECTOR / "sylhet_waterways.gpkg"
    admin_file = DATA_VECTOR / "sylhet_upazilas.gpkg"

    real_data = (
        not force_synthetic
        and dem_path.exists()
        and lulc_path.exists()
        and river_file.exists()
    )

    if real_data:
        console.print("[green]✓ Using real data from data/[/green]")
        river_gdf = gpd.read_file(river_file)
        admin_gdf = gpd.read_file(admin_file) if admin_file.exists() else None
        return dem_path, lulc_path, river_gdf, admin_gdf

    console.print("[yellow]⚠ Real data not found — generating synthetic demonstration data[/yellow]")
    return _make_synthetic_inputs()


# ── Synthetic data generator ──────────────────────────────────────────────────

def _make_synthetic_inputs() -> tuple[Path, Path, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build synthetic DEM, LULC, rivers, and admin polygons representative of
    the Sylhet study area.  All files are written to data/raster/ and data/vector/.
    """
    DATA_RASTER.mkdir(parents=True, exist_ok=True)
    DATA_VECTOR.mkdir(parents=True, exist_ok=True)

    west, south, east, north = BBOX
    rows, cols = 150, 100  # ~1 km resolution

    transform = from_bounds(west, south, east, north, cols, rows)

    # ── Synthetic DEM ────────────────────────────────────────────────────────
    # Elevation increases from SW (floodplain) to NE (hills)
    rng = np.random.default_rng(seed=2024)
    xx, yy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    base_elev = (xx * 80 + yy * 40).astype(np.float32)  # 0–120 m gradient
    noise = rng.normal(0, 8, (rows, cols)).astype(np.float32)
    # River valleys: low elevation band running SW–NE (Surma river corridor)
    river_mask = np.abs(yy - 0.5 - 0.3 * xx) < 0.08
    dem_data = np.clip(base_elev + noise - river_mask.astype(np.float32) * 30, 0, 200).astype(np.int16)

    dem_path = DATA_RASTER / "srtm_sylhet_synthetic.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=rows, width=cols,
        count=1, dtype="int16", crs="EPSG:4326", transform=transform, nodata=-32768,
    ) as dst:
        dst.write(dem_data, 1)
    console.print(f"  [dim]→ Synthetic DEM: {dem_path}[/dim]")

    # ── Synthetic LULC ───────────────────────────────────────────────────────
    # Cropland dominates; water along the river corridor; trees on hills
    lulc_data = np.full((rows, cols), 40, dtype=np.uint8)          # cropland default
    lulc_data[base_elev > 90] = 10                                  # tree cover on hills
    lulc_data[river_mask] = 80                                      # water in river band
    lulc_data[np.abs(yy - 0.5 - 0.3 * xx) < 0.14] |= 0            # wetland fringe
    wetland_fringe = (np.abs(yy - 0.5 - 0.3 * xx) < 0.14) & (~river_mask)
    lulc_data[wetland_fringe] = 90                                  # herbaceous wetland
    # Add some built-up area near the city centre
    city_mask = (xx > 0.4) & (xx < 0.6) & (yy > 0.55) & (yy < 0.75)
    lulc_data[city_mask] = 50

    lulc_path = DATA_RASTER / "esa_worldcover_sylhet_synthetic.tif"
    with rasterio.open(
        lulc_path, "w", driver="GTiff", height=rows, width=cols,
        count=1, dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(lulc_data, 1)
    console.print(f"  [dim]→ Synthetic LULC: {lulc_path}[/dim]")

    # ── Synthetic rivers ─────────────────────────────────────────────────────
    # Main river: Surma — diagonal SW to NE
    surma = LineString([
        (91.6, 24.3), (91.75, 24.5), (91.9, 24.7), (92.1, 24.9), (92.3, 25.1),
    ])
    # Tributary: Kushiyara — eastern branch
    kushiyara = LineString([
        (92.0, 24.1), (92.1, 24.4), (92.2, 24.7), (92.25, 25.0),
    ])
    river_gdf = gpd.GeoDataFrame(
        {"name": ["Surma", "Kushiyara"], "waterway": ["river", "river"]},
        geometry=[surma, kushiyara],
        crs="EPSG:4326",
    )
    console.print("  [dim]→ Synthetic rivers: Surma + Kushiyara[/dim]")

    # ── Synthetic admin units (upazilas) ─────────────────────────────────────
    # 3×4 grid of upazila polygons
    lon_edges = np.linspace(west, east, 4)
    lat_edges = np.linspace(south, north, 5)
    upazilas = []
    names = []
    for r in range(4):
        for c in range(3):
            geom = box(lon_edges[c], lat_edges[r], lon_edges[c + 1], lat_edges[r + 1])
            upazilas.append(geom)
            names.append(f"Upazila_{r}_{c}")

    admin_gdf = gpd.GeoDataFrame(
        {"upazila_name": names, "district": ["Sylhet"] * len(names)},
        geometry=upazilas,
        crs="EPSG:4326",
    )
    console.print(f"  [dim]→ Synthetic admin units: {len(admin_gdf)} upazilas[/dim]")

    return dem_path, lulc_path, river_gdf, admin_gdf


if __name__ == "__main__":
    app()
