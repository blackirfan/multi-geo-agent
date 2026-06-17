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

# Fixed fallback colour stops (used when FSI values span < 0.02)
_FSI_COLORMAP_FIXED = [
    (0.00, "#2ecc71"),
    (0.25, "#f1c40f"),
    (0.50, "#e67e22"),
    (0.75, "#e74c3c"),
    (1.00, "#8e44ad"),
]

_CATEGORY_LABELS = ["Low", "Moderate", "High", "Very High"]


def _build_dynamic_colormap(
    fsi_ranking: list[dict[str, Any]],
) -> tuple[list[tuple[float, str]], bool]:
    """
    Return (colormap, is_dynamic).

    When FSI values span ≥ 0.02, the scale is stretched from the actual
    data min→max so that closely-clustered values produce visually distinct
    colours.  Otherwise the fixed 0–1 fallback is returned.
    """
    if len(fsi_ranking) < 2:
        return _FSI_COLORMAP_FIXED, False

    values = [
        float(r["mean_fsi"])
        for r in fsi_ranking
        if r.get("mean_fsi") is not None
    ]
    if not values:
        return _FSI_COLORMAP_FIXED, False

    data_min, data_max = min(values), max(values)
    data_range = data_max - data_min
    if data_range < 0.02:
        return _FSI_COLORMAP_FIXED, False

    step = data_range / 4
    colormap: list[tuple[float, str]] = [
        (data_min,              "#2ecc71"),
        (data_min + step,       "#f1c40f"),
        (data_min + step * 2,   "#e67e22"),
        (data_min + step * 3,   "#e74c3c"),
        (data_max,              "#8e44ad"),
    ]
    return colormap, True


def _fsi_to_color(fsi: float, colormap: list[tuple[float, str]] | None = None) -> str:
    """Interpolate an FSI value to a hex colour using the given (or fixed) colormap."""
    cmap = colormap or _FSI_COLORMAP_FIXED
    fsi = max(cmap[0][0], min(cmap[-1][0], fsi))

    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    for i in range(len(cmap) - 1):
        lo_v, lo_c = cmap[i]
        hi_v, hi_c = cmap[i + 1]
        if lo_v <= fsi <= hi_v:
            t = (fsi - lo_v) / (hi_v - lo_v) if hi_v > lo_v else 0.0
            r1, g1, b1 = _hex_to_rgb(lo_c)
            r2, g2, b2 = _hex_to_rgb(hi_c)
            r = int(r1 + t * (r2 - r1))
            g = int(g1 + t * (g2 - g1))
            b = int(b1 + t * (b2 - b1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return cmap[-1][1]


def generate_static_map_png(
    admin_gdf: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame | None,
    fsi_ranking: list[dict[str, Any]],
    colormap: list[tuple[float, str]] | None = None,
    figsize: tuple[int, int] = (10, 7),
    dpi: int = 150,
) -> str | None:
    """
    Render a static choropleth PNG using matplotlib.

    Returns a base64-encoded PNG string suitable for embedding in HTML as
    ``data:image/png;base64,...``.  Returns None on any failure so the
    caller can fall back gracefully.
    """
    try:
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe in server context
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — static map PNG skipped")
        return None

    cmap = colormap or _FSI_COLORMAP_FIXED
    ranking_map = {r["name"]: r for r in fsi_ranking}
    band_labels = ["Low", "Moderate", "High", "Very High"]

    try:
        gdf = admin_gdf.copy().to_crs(epsg=4326)
        name_col = next(
            (c for c in ["upazila_name", "NAME_3", "NAME_2", "name"] if c in gdf.columns),
            None,
        )

        if name_col:
            gdf["_mean_fsi"] = gdf[name_col].apply(
                lambda n: ranking_map.get(str(n), {}).get("mean_fsi")
            )
            gdf["_color"] = gdf["_mean_fsi"].apply(
                lambda v: _fsi_to_color(float(v), cmap) if v is not None else "#2d3748"
            )
            gdf["_label"] = gdf[name_col].apply(str)
        else:
            gdf["_mean_fsi"] = None
            gdf["_color"] = "#2d3748"
            gdf["_label"] = ""

        bg = "#1e293b"
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)

        # Upazila polygons
        gdf.plot(
            ax=ax,
            color=gdf["_color"].tolist(),
            edgecolor="#ffffff",
            linewidth=0.7,
            alpha=0.88,
        )

        # River network
        if river_gdf is not None and not river_gdf.empty:
            try:
                river_gdf.to_crs(epsg=4326).plot(
                    ax=ax, color="#3b82f6", linewidth=1.5, alpha=0.75, zorder=3
                )
            except Exception as exc:
                log.warning("River overlay failed: %s", exc)

        # Upazila name + FSI value labels at polygon centroids
        stroke = [pe.withStroke(linewidth=2.5, foreground="black")]
        if name_col:
            for _, row in gdf.iterrows():
                centroid = row.geometry.centroid
                fsi_val  = row["_mean_fsi"]
                line1    = row["_label"]
                line2    = f"{fsi_val:.3f}" if fsi_val is not None else ""

                # Name
                ax.annotate(
                    line1,
                    xy=(centroid.x, centroid.y + 0.01),
                    ha="center", va="bottom",
                    fontsize=6.5, color="white", fontweight="bold",
                    path_effects=stroke,
                )
                # FSI value in amber below the name
                if line2:
                    ax.annotate(
                        line2,
                        xy=(centroid.x, centroid.y - 0.01),
                        ha="center", va="top",
                        fontsize=6, color="#fbbf24", fontweight="600",
                        path_effects=stroke,
                    )

        # Dynamic legend
        patches = [
            mpatches.Patch(
                facecolor=cmap[i][1],
                edgecolor="#ffffff",
                linewidth=0.4,
                label=f"{band_labels[i]}  {cmap[i][0]:.3f}–{cmap[i + 1][0]:.3f}",
            )
            for i in range(4)
        ]
        legend = ax.legend(
            handles=patches,
            loc="lower left",
            fontsize=7,
            framealpha=0.88,
            facecolor="#0f172a",
            labelcolor="white",
            edgecolor="#475569",
            title="Flood Susceptibility Index",
            title_fontsize=7.5,
        )
        legend.get_title().set_color("#94a3b8")

        ax.set_title(
            f"Flood Susceptibility Index — {settings.study_area_name}",
            color="white", fontsize=11, fontweight="bold", pad=12,
        )
        ax.tick_params(colors="#64748b", labelsize=7)
        ax.set_xlabel("Longitude", color="#64748b", fontsize=8)
        ax.set_ylabel("Latitude",  color="#64748b", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        plt.tight_layout(pad=1.2)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as exc:
        log.warning("Static map PNG generation failed: %s", exc)
        try:
            plt.close("all")
        except Exception:
            pass
        return None


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

    # Build dynamic colour scale from actual FSI values
    colormap, is_dynamic = _build_dynamic_colormap(fsi_ranking)

    name_col = next(
        (c for c in ["upazila_name", "NAME_3", "NAME_2", "name"] if c in admin_gdf.columns),
        None,
    )

    for _, row in admin_gdf.iterrows():
        unit_name = str(row[name_col]) if name_col else "Unknown"
        info = ranking_map.get(unit_name, {})
        mean_fsi = info.get("mean_fsi", 0.0)
        rank = info.get("rank", "?")
        color = _fsi_to_color(float(mean_fsi), colormap)

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

    # Dynamic legend — shows actual value ranges from the data
    band_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    legend_rows = "".join(
        f'<span style="color:{band_colors[i]}">■</span>'
        f' {_CATEGORY_LABELS[i]}'
        f' ({colormap[i][0]:.3f}–{colormap[i + 1][0]:.3f})<br>'
        for i in range(4)
    )
    note = (
        '<span style="font-size:9px;color:#888">Scale fitted to data range</span>'
        if is_dynamic else ""
    )
    legend_html = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
        'padding:10px 14px;border-radius:6px;border:1px solid #ccc;font-size:11px;'
        'line-height:1.7;">'
        "<b>Flood Susceptibility Index</b><br>"
        f"{legend_rows}"
        f"{note}"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()


def render_html_report(
    run_id: str,
    query: str,
    answer: str | None,
    ranking: list[dict[str, Any]],
    agent_trace: list[dict[str, Any]],
    map_html: str,
    map_png_b64: str | None = None,
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
        map_png_b64=map_png_b64,
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

    # Build dynamic colour scale once so both maps use the same stops
    colormap, _ = _build_dynamic_colormap(ranking)

    # Interactive Folium map (for HTML/browser view)
    try:
        map_html = build_folium_map(admin_gdf, river_gdf, ranking)
    except Exception as exc:
        log.warning("Folium map generation failed: %s", exc)
        map_html = ""

    # Static matplotlib PNG (embedded in PDF via base64 data URI)
    map_png_b64: str | None = None
    if not admin_gdf.empty:
        map_png_b64 = generate_static_map_png(
            admin_gdf, river_gdf, ranking, colormap
        )

    html_content = render_html_report(
        run_id=run_id,
        query=query,
        answer=answer,
        ranking=ranking,
        agent_trace=agent_trace,
        map_html=map_html,
        map_png_b64=map_png_b64,
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
