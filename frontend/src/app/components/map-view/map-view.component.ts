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

// GeoJSON feature shape expected from the API
interface GeoFeature {
  properties: Record<string, unknown>;
  geometry: unknown;
  type: string;
}

const FSI_STOPS: [number, [number, number, number]][] = [
  [0.00, [46,  204, 113]],
  [0.25, [241, 196,  15]],
  [0.50, [230, 126,  34]],
  [0.75, [231,  76,  60]],
  [1.00, [142,  68, 173]],
];

function fsiColor(v: number | null | undefined): string {
  const val = Math.max(0, Math.min(1, v ?? 0));
  for (let i = 0; i < FSI_STOPS.length - 1; i++) {
    const [lo, lc] = FSI_STOPS[i];
    const [hi, hc] = FSI_STOPS[i + 1];
    if (val <= hi) {
      const t = (val - lo) / (hi - lo);
      const r = Math.round(lc[0] + t * (hc[0] - lc[0]));
      const g = Math.round(lc[1] + t * (hc[1] - lc[1]));
      const b = Math.round(lc[2] + t * (hc[2] - lc[2]));
      return `rgb(${r},${g},${b})`;
    }
  }
  return 'rgb(142,68,173)';
}

function featureName(props: Record<string, unknown>): string {
  return (
    (props['upazila_name'] as string | undefined) ||
    (props['NAME_3']      as string | undefined) ||
    (props['NAME_2']      as string | undefined) ||
    (props['name']        as string | undefined) ||
    'Unknown'
  );
}

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

      <!-- Legend -->
      <div class="absolute bottom-8 right-4 z-[400] bg-slate-900/90
                  border border-slate-700 rounded-lg p-3 text-xs backdrop-blur-sm">
        <p class="text-[10px] uppercase tracking-widest text-slate-400 font-semibold mb-2">FSI Scale</p>
        @for (stop of legend; track stop.label) {
          <div class="flex items-center gap-2 mb-1">
            <div class="w-3 h-3 rounded-sm flex-shrink-0" [style.background]="stop.color"></div>
            <span class="text-slate-400 text-[10px]">{{ stop.label }}</span>
          </div>
        }
      </div>
    </div>
  `,
})
export class MapViewComponent implements OnDestroy {
  private mapEl  = viewChild.required<ElementRef<HTMLDivElement>>('mapEl');
  private state  = inject(AppStateService);
  private geo    = inject(GeoReasonerService);

  protected loading = signal(false);

  protected legend = [
    { color: fsiColor(0.1),  label: 'Low (0–0.25)' },
    { color: fsiColor(0.35), label: 'Moderate (0.25–0.5)' },
    { color: fsiColor(0.6),  label: 'High (0.5–0.75)' },
    { color: fsiColor(0.9),  label: 'Very High (0.75–1)' },
  ];

  private map: L.Map | null = null;
  private adminLayer: L.GeoJSON | null = null;
  private mapReady = signal(false);

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
            layer.bindTooltip(
              `<span style="color:#e94560;font-weight:700">${name}</span>`,
              { sticky: true }
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

    this.geo.getFsiLayer(ranking).subscribe({
      next: geo => {
        if (this.adminLayer) {
          this.map!.removeLayer(this.adminLayer);
          this.adminLayer = null;
        }

        this.adminLayer = L.geoJSON(geo as Parameters<typeof L.geoJSON>[0], {
          style: f => {
            const props = (f as GeoFeature | undefined)?.properties ?? {};
            const meanFsi = props['mean_fsi'] as number | undefined;
            return {
              fillColor: fsiColor(meanFsi),
              color: '#0f172a',
              weight: 1,
              fillOpacity: meanFsi != null ? 0.72 : 0.25,
            };
          },
          onEachFeature: (f, layer) => {
            const props = (f as GeoFeature).properties;
            const name  = featureName(props);
            const fsi   = props['mean_fsi'] as number | undefined;
            const rank  = props['fsi_rank'] as number | undefined;

            layer.bindPopup(
              `<b>${name}</b><br>` +
              `Rank: ${rank ?? '?'}<br>` +
              `Mean FSI: ${fsi != null ? fsi.toFixed(4) : 'n/a'}`,
              { maxWidth: 200 }
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
