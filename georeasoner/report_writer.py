"""Report Writer — generates HTML + PDF flood vulnerability reports."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from georeasoner.config import settings
from georeasoner.data_utils import ensure_admin_boundaries, ensure_waterways

log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_FSI_COLORMAP = [
    (0.0, "#2ecc71"),
    (0.25, "#f1c40f"),
    (0.5, "#e67e22"),
    (0.75, "#e74c3c"),
    (1.0, "#8e44ad"),
]


def _fsi_to_color(fsi: float) -> str:
    """Interpolate FSI value to a hex colour."""
    fsi = max(0.0, min(1.0, fsi))
    for i in range(len(_FSI_COLORMAP) - 1):
        lo_v, lo_c = _FSI_COLORMAP[i]
        hi_v, hi_c = _FSI_COLORMAP[i + 1]
        if lo_v <= fsi <= hi_v:
            t = (fsi - lo_v) / (hi_v - lo_v) if hi_v > lo_v else 0.0

            def _hex_to_rgb(h: str) -> tuple[int, int, int]:
                h = h.lstrip("#")
                return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

            r1, g1, b1 = _hex_to_rgb(lo_c)
            r2, g2, b2 = _hex_to_rgb(hi_c)
            r = int(r1 + t * (r2 - r1))
            g = int(g1 + t * (g2 - g1))
            b = int(b1 + t * (b2 - b1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#8e44ad"


def build_folium_map(
    admin_gdf: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame | None,
    fsi_ranking: list[dict[str, Any]],
    *,
    width: int = 700,
    height: int = 420,
) -> str:
    """Build a Folium choropleth map and return standalone HTML fragment."""
    ranking_map = {r["name"]: r for r in fsi_ranking}

    # Centre on Sylhet
    centre = [
        (settings.study_bbox_north + settings.study_bbox_south) / 2,
        (settings.study_bbox_east + settings.study_bbox_west) / 2,
    ]

    m = folium.Map(
        location=centre,
        zoom_start=9,
        tiles="CartoDB positron",
        width=width,
        height=height,
    )

    name_col = next(
        (c for c in ["upazila_name", "NAME_3", "NAME_2", "name"] if c in admin_gdf.columns),
        None,
    )

    for _, row in admin_gdf.iterrows():
        unit_name = str(row[name_col]) if name_col else "Unknown"
        info = ranking_map.get(unit_name, {})
        mean_fsi = info.get("mean_fsi", 0.0)
        rank = info.get("rank", "?")
        color = _fsi_to_color(float(mean_fsi))

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _f, c=color: {
                "fillColor": c,
                "color": "#ffffff",
                "weight": 1,
                "fillOpacity": 0.65,
            },
            tooltip=folium.Tooltip(
                f"<b>{unit_name}</b><br>Rank: {rank}<br>Mean FSI: {mean_fsi:.4f}"
            ),
        ).add_to(m)

    if river_gdf is not None and not river_gdf.empty:
        try:
            folium.GeoJson(
                json.loads(river_gdf.to_json()),
                style_function=lambda _: {"color": "#1a6ca8", "weight": 1.5, "opacity": 0.7},
                name="Waterways",
            ).add_to(m)
        except Exception:
            pass

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
                 padding:10px 14px;border-radius:6px;border:1px solid #ccc;font-size:11px;">
      <b>Flood Susceptibility Index</b><br>
      <span style="color:#2ecc71">■</span> Low (0.00–0.25)<br>
      <span style="color:#f1c40f">■</span> Moderate (0.25–0.50)<br>
      <span style="color:#e67e22">■</span> High (0.50–0.75)<br>
      <span style="color:#8e44ad">■</span> Very High (0.75–1.00)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()


def render_html_report(
    run_id: str,
    query: str,
    answer: str | None,
    ranking: list[dict[str, Any]],
    agent_trace: list[dict[str, Any]],
    map_html: str,
) -> str:
    """Render Jinja2 HTML report template to a string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")
    return template.render(
        run_id=run_id,
        query=query,
        answer=answer or "",
        ranking=ranking,
        agent_trace=agent_trace,
        map_html=map_html,
        area_name=settings.study_area_name,
        model_name=settings.lm_studio_model,
        generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def generate_pdf(html_content: str, output_path: Path) -> Path:
    """Convert HTML string to PDF via WeasyPrint."""
    from weasyprint import HTML  # lazy import — heavy dependency

    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_content).write_pdf(str(output_path))
    return output_path


def generate_report(
    run_id: str,
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    """
    Full pipeline: load geo layers → build Folium map → render HTML → write PDF.

    Returns the Path to the generated PDF.
    """
    output_dir = output_dir or settings.reports_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ranking: list[dict] = result.get("fsi_ranking") or []
    answer: str | None = result.get("answer")
    agent_trace: list[dict] = result.get("agent_trace") or []
    query: str = result.get("query", "Flood vulnerability assessment")

    # Load geo layers (use cached or synthetic)
    try:
        admin_gdf = gpd.read_file(ensure_admin_boundaries())
        river_gdf = gpd.read_file(ensure_waterways())
    except Exception:
        log.warning("Could not load geo layers for map; using empty GeoDataFrames")
        admin_gdf = gpd.GeoDataFrame()
        river_gdf = None

    # Build map
    try:
        map_html = build_folium_map(admin_gdf, river_gdf, ranking)
    except Exception as exc:
        log.warning("Folium map generation failed: %s", exc)
        map_html = ""

    html_content = render_html_report(
        run_id=run_id,
        query=query,
        answer=answer,
        ranking=ranking,
        agent_trace=agent_trace,
        map_html=map_html,
    )

    # Write HTML alongside PDF for debugging
    html_path = Path(output_dir) / f"{run_id}.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_path = Path(output_dir) / f"{run_id}.pdf"
    try:
        generate_pdf(html_content, pdf_path)
    except Exception as exc:
        log.warning("WeasyPrint PDF generation failed: %s — returning HTML path", exc)
        return html_path

    return pdf_path


def admin_geojson() -> dict:
    """Return admin boundaries as a GeoJSON-serialisable dict."""
    try:
        gdf = gpd.read_file(ensure_admin_boundaries())
        gdf = gdf.to_crs(epsg=4326)
        keep_cols = [c for c in gdf.columns if c != "geometry"]
        gdf = gdf[keep_cols + ["geometry"]]
        return json.loads(gdf.to_json())
    except Exception as exc:
        log.warning("admin_geojson failed: %s", exc)
        return {"type": "FeatureCollection", "features": []}


def rivers_geojson() -> dict:
    """Return waterways as a GeoJSON-serialisable dict."""
    try:
        gdf = gpd.read_file(ensure_waterways())
        gdf = gdf.to_crs(epsg=4326)
        keep_cols = [c for c in gdf.columns if c != "geometry"]
        gdf = gdf[keep_cols + ["geometry"]]
        return json.loads(gdf.to_json())
    except Exception as exc:
        log.warning("rivers_geojson failed: %s", exc)
        return {"type": "FeatureCollection", "features": []}


def fsi_geojson(fsi_ranking: list[dict[str, Any]]) -> dict:
    """
    Join FSI ranking back onto admin boundaries and return as GeoJSON.

    Properties include mean_fsi and rank for choropleth rendering.
    """
    try:
        gdf = gpd.read_file(ensure_admin_boundaries())
        gdf = gdf.to_crs(epsg=4326)
        name_col = next(
            (c for c in ["upazila_name", "NAME_3", "NAME_2", "name"] if c in gdf.columns),
            None,
        )
        ranking_map = {r["name"]: r for r in fsi_ranking}
        if name_col:
            gdf["mean_fsi"] = gdf[name_col].apply(
                lambda n: ranking_map.get(str(n), {}).get("mean_fsi", np.nan)
            )
            gdf["fsi_rank"] = gdf[name_col].apply(
                lambda n: ranking_map.get(str(n), {}).get("rank", None)
            )
        keep_cols = [c for c in gdf.columns if c != "geometry"]
        gdf = gdf[keep_cols + ["geometry"]]
        return json.loads(gdf.to_json())
    except Exception as exc:
        log.warning("fsi_geojson failed: %s", exc)
        return {"type": "FeatureCollection", "features": []}
