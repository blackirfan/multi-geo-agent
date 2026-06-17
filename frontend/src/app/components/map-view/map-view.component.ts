import {
  Component,
  ElementRef,
  OnDestroy,
  afterNextRender,
  effect,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import * as L from 'leaflet';
import { AppStateService } from '../../services/app-state.service';
import { GeoReasonerService } from '../../services/georeasoner.service';
import { RankingItem } from '../../models/types';

interface GeoFeature {
  properties: Record<string, unknown>;
  geometry: unknown;
  type: string;
}

// ── Dynamic colour scale ────────────────────────────────────────────────────────

interface LegendBand {
  color: string;   // CSS rgb(...)
  label: string;   // "Low (0.418–0.521)"
  category: string;
  lo: number;
  hi: number;
}

interface ColorScale {
  bands: LegendBand[];
  dynamic: boolean;          // true → scale fitted to data range
}

// Fixed semantic colors for the four bands
const BAND_RGB: [number, number, number][] = [
  [ 46, 204, 113],   // green  — Low
  [241, 196,  15],   // yellow — Moderate
  [230, 126,  34],   // orange — High
  [231,  76,  60],   // red    — Very High
];
const CAP_RGB: [number, number, number] = [142, 68, 173]; // purple (above scale)

const CATEGORY_LABELS = ['Low', 'Moderate', 'High', 'Very High'] as const;

function lerpRgb(
  lo: [number, number, number],
  hi: [number, number, number],
  t: number,
): string {
  return `rgb(${Math.round(lo[0] + t * (hi[0] - lo[0]))},${Math.round(lo[1] + t * (hi[1] - lo[1]))},${Math.round(lo[2] + t * (hi[2] - lo[2]))})`;
}

function rgbCss(c: [number, number, number]): string {
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/**
 * Build a four-band colour scale.
 *
 * When FSI values span a meaningful range (≥ 0.02), the scale is stretched
 * from the actual data min → max so that closely-spaced values produce
 * visibly different colours.  If the range is negligible, the fixed 0–1
 * fallback is used.
 */
function buildColorScale(ranking: RankingItem[]): ColorScale {
  const FIXED_THRESHOLDS = [0, 0.25, 0.50, 0.75, 1.0];

  let thresholds: number[];
  let dynamic = false;

  if (ranking.length >= 2) {
    const vals = ranking.map(r => r.mean_fsi ?? 0).filter(Number.isFinite);
    const dataMin = Math.min(...vals);
    const dataMax = Math.max(...vals);
    const range   = dataMax - dataMin;

    if (range >= 0.02) {
      const step = range / 4;
      thresholds = [
        dataMin,
        dataMin + step,
        dataMin + step * 2,
        dataMin + step * 3,
        dataMax,
      ];
      dynamic = true;
    } else {
      thresholds = FIXED_THRESHOLDS;
    }
  } else {
    thresholds = FIXED_THRESHOLDS;
  }

  const bands: LegendBand[] = BAND_RGB.map((c, i) => ({
    color:    rgbCss(c),
    category: CATEGORY_LABELS[i],
    label:    `${CATEGORY_LABELS[i]} (${thresholds[i].toFixed(3)}–${thresholds[i + 1].toFixed(3)})`,
    lo:       thresholds[i],
    hi:       thresholds[i + 1],
  }));

  return { bands, dynamic };
}

/** Map a single FSI value to a CSS colour using the current scale. */
function applyScale(value: number | null | undefined, scale: ColorScale): string {
  const val = value ?? 0;

  for (let i = 0; i < scale.bands.length; i++) {
    const { lo, hi } = scale.bands[i];
    if (val <= hi) {
      const range = hi - lo;
      const t = range > 0 ? Math.max(0, Math.min(1, (val - lo) / range)) : 0;
      return lerpRgb(BAND_RGB[i], BAND_RGB[i + 1] ?? CAP_RGB, t);
    }
  }
  return rgbCss(CAP_RGB);
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function featureName(props: Record<string, unknown>): string {
  return (
    (props['upazila_name'] as string | undefined) ||
    (props['NAME_3']       as string | undefined) ||
    (props['NAME_2']       as string | undefined) ||
    (props['name']         as string | undefined) ||
    'Unknown'
  );
}

// ── Component ──────────────────────────────────────────────────────────────────

@Component({
  selector: 'gr-map-view',
  standalone: true,
  template: `
    <div class="relative w-full h-full">
      <div #mapEl class="absolute inset-0"></div>

      @if (loading()) {
        <div class="absolute inset-0 flex items-center justify-center
                    bg-slate-900/60 backdrop-blur-sm z-[9999] pointer-events-none">
          <div class="text-slate-400 text-xs flex items-center gap-2">
            <span class="animate-spin text-lg">◌</span>
            Updating map…
          </div>
        </div>
      }

      <!-- Dynamic legend -->
      <div class="absolute bottom-8 right-4 z-[400] bg-slate-900/90
                  border border-slate-700 rounded-lg p-3 text-xs backdrop-blur-sm min-w-[186px]">
        <p class="text-[10px] uppercase tracking-widest text-slate-400 font-semibold mb-2">FSI Scale</p>

        @for (band of scale().bands; track band.category) {
          <div class="flex items-center gap-2 mb-1">
            <div class="w-3 h-3 rounded-sm flex-shrink-0" [style.background]="band.color"></div>
            <span class="text-slate-300 text-[10px]">{{ band.label }}</span>
          </div>
        }

        @if (scale().dynamic) {
          <p class="text-[9px] text-slate-500 mt-2 pt-1.5 border-t border-slate-700/60 leading-tight">
            Scale fitted to data range
          </p>
        }
      </div>
    </div>
  `,
})
export class MapViewComponent implements OnDestroy {
  private mapEl = viewChild.required<ElementRef<HTMLDivElement>>('mapEl');
  private state = inject(AppStateService);
  private geo   = inject(GeoReasonerService);

  protected loading = signal(false);
  protected scale   = signal<ColorScale>(buildColorScale([]));

  private map: L.Map | null = null;
  private adminLayer: L.GeoJSON | null = null;
  private mapReady = signal(false);

  // Keep a snapshot so the GeoJSON style callback always uses the scale
  // that was current when the layer was built (avoids stale closure).
  private activeScale: ColorScale = buildColorScale([]);

  constructor() {
    afterNextRender(() => { this.boot(); });

    effect(() => {
      if (!this.mapReady()) return;
      const ranking = this.state.ranking();
      if (ranking.length) this.updateFsiChoropleth(ranking);
    });
  }

  private boot(): void {
    this.map = L.map(this.mapEl().nativeElement).setView([24.9, 91.87], 9);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org">OSM</a> &copy; <a href="https://carto.com">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 18,
    }).addTo(this.map);

    this.geo.getRiversLayer().subscribe({
      next: geo => {
        L.geoJSON(geo as Parameters<typeof L.geoJSON>[0], {
          style: { color: '#1e6fa8', weight: 1.5, opacity: 0.7 },
        }).addTo(this.map!);
      },
    });

    this.geo.getAdminLayer().subscribe({
      next: geo => {
        this.adminLayer = L.geoJSON(geo as Parameters<typeof L.geoJSON>[0], {
          style: { fillColor: '#334155', color: '#475569', weight: 1, fillOpacity: 0.35 },
          onEachFeature: (f, layer) => {
            const name = featureName((f as GeoFeature).properties);
            // Permanent name label at centroid
            layer.bindTooltip(
              `<span class="lbl-name">${name}</span>`,
              { permanent: true, direction: 'center', className: 'upazila-label' }
            );
          },
        }).addTo(this.map!);

        if (this.adminLayer.getBounds().isValid()) {
          this.map!.fitBounds(this.adminLayer.getBounds(), { padding: [24, 24] });
        }
        this.mapReady.set(true);
      },
    });
  }

  private updateFsiChoropleth(ranking: RankingItem[]): void {
    if (!this.map) return;
    this.loading.set(true);

    // Build the scale from actual data BEFORE the HTTP call so the legend
    // updates immediately while the layer fetch is in flight.
    const colorScale = buildColorScale(ranking);
    this.activeScale = colorScale;
    this.scale.set(colorScale);

    this.geo.getFsiLayer(ranking).subscribe({
      next: geo => {
        if (this.adminLayer) {
          this.map!.removeLayer(this.adminLayer);
          this.adminLayer = null;
        }

        // Capture scale in a local variable — style callbacks are synchronous
        // during layer construction so no stale-closure risk here.
        const cs = this.activeScale;

        this.adminLayer = L.geoJSON(geo as Parameters<typeof L.geoJSON>[0], {
          style: f => {
            const props  = (f as GeoFeature | undefined)?.properties ?? {};
            const meanFsi = props['mean_fsi'] as number | undefined;
            return {
              fillColor:   applyScale(meanFsi, cs),
              color:       '#0f172a',
              weight:      1,
              fillOpacity: meanFsi != null ? 0.72 : 0.25,
            };
          },
          onEachFeature: (f, layer) => {
            const props   = (f as GeoFeature).properties;
            const name    = featureName(props);
            const fsi     = props['mean_fsi'] as number | undefined;
            const rank    = props['fsi_rank'] as number | undefined;
            const catBand = fsi != null
              ? cs.bands.find(b => fsi >= b.lo && fsi <= b.hi) ?? cs.bands.at(-1)
              : undefined;

            // Permanent label: name on top, FSI value below in amber
            const fsiPart = fsi != null
              ? `<span class="lbl-fsi">${fsi.toFixed(3)}</span>`
              : '';
            layer.bindTooltip(
              `<span class="lbl-name">${name}</span>${fsiPart}`,
              { permanent: true, direction: 'center', className: 'upazila-label' }
            );

            // Detailed popup on click
            layer.bindPopup(
              `<b>${name}</b><br>` +
              `Rank: ${rank ?? '?'}<br>` +
              `Mean FSI: ${fsi != null ? fsi.toFixed(4) : 'n/a'}` +
              (catBand ? `<br>Category: <b>${catBand.category}</b>` : ''),
              { maxWidth: 220 }
            );

            (layer as L.Path).on('mouseover', () =>
              (layer as L.Path).setStyle({ weight: 2.5, color: '#e94560' })
            );
            (layer as L.Path).on('mouseout', () =>
              this.adminLayer!.resetStyle(layer as L.Path)
            );
          },
        }).addTo(this.map!);

        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  ngOnDestroy(): void {
    this.map?.remove();
  }
}
