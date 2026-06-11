import { Component, inject } from '@angular/core';
import { AppStateService } from '../../services/app-state.service';

@Component({
  selector: 'gr-ranking-table',
  standalone: true,
  template: `
    <div class="flex flex-col h-full">
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-2
                  bg-slate-900/80 border-b border-slate-700 flex-shrink-0">
        <span class="text-[10px] uppercase tracking-widest font-semibold text-slate-400">
          FSI Ranking
        </span>
        @if (state.ranking().length) {
          <span class="text-[10px] text-slate-500">{{ state.ranking().length }} units</span>
        }
      </div>

      <!-- List -->
      <div class="flex-1 overflow-y-auto">
        @if (!state.ranking().length && !state.isRunning()) {
          <div class="flex flex-col items-center justify-center h-32 text-slate-600 text-xs text-center px-4">
            <span class="text-3xl mb-2 opacity-40">⬡</span>
            Run an analysis to see the flood susceptibility ranking.
          </div>
        }

        @if (state.isRunning() && !state.ranking().length) {
          @for (_ of placeholder; track $index) {
            <div class="px-4 py-3 border-b border-slate-700/50 animate-pulse">
              <div class="h-3 bg-slate-700 rounded w-3/4 mb-2"></div>
              <div class="h-2 bg-slate-700 rounded w-1/2"></div>
            </div>
          }
        }

        @for (item of state.ranking(); track item.rank) {
          <div class="flex items-center gap-3 px-4 py-2.5 border-b border-slate-700/40
                      hover:bg-slate-700/30 transition-colors cursor-default group">
            <!-- Rank badge -->
            <div [class]="rankBadgeClass(item.rank)"
                 class="w-6 h-6 rounded-full flex items-center justify-center
                        text-[10px] font-bold text-white flex-shrink-0">
              {{ item.rank }}
            </div>

            <!-- Info -->
            <div class="flex-1 min-w-0">
              <p class="text-sm font-medium text-slate-200 truncate">{{ item.name }}</p>
              <div class="flex items-center gap-2 mt-1">
                <!-- FSI bar -->
                <div class="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div class="h-full rounded-full transition-all duration-700"
                       [style.width.%]="item.mean_fsi * 100"
                       [style.background]="fsiColor(item.mean_fsi)">
                  </div>
                </div>
                <span class="text-[10px] text-slate-400 font-mono flex-shrink-0">
                  {{ item.mean_fsi.toFixed(4) }}
                </span>
              </div>
            </div>
          </div>
        }
      </div>
    </div>
  `,
})
export class RankingTableComponent {
  protected state = inject(AppStateService);
  protected placeholder = Array(8);

  rankBadgeClass(rank: number): string {
    if (rank === 1) return 'bg-rose-500';
    if (rank === 2) return 'bg-orange-500';
    if (rank === 3) return 'bg-amber-500';
    return 'bg-slate-600';
  }

  fsiColor(v: number): string {
    v = Math.max(0, Math.min(1, v));
    const stops: [number, [number, number, number]][] = [
      [0.0,  [46,  204, 113]],
      [0.25, [241, 196, 15]],
      [0.5,  [230, 126, 34]],
      [0.75, [231, 76,  60]],
      [1.0,  [142, 68,  173]],
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      const [lo, lc] = stops[i];
      const [hi, hc] = stops[i + 1];
      if (v <= hi) {
        const t = (v - lo) / (hi - lo);
        const r = Math.round(lc[0] + t * (hc[0] - lc[0]));
        const g = Math.round(lc[1] + t * (hc[1] - lc[1]));
        const b = Math.round(lc[2] + t * (hc[2] - lc[2]));
        return `rgb(${r},${g},${b})`;
      }
    }
    return 'rgb(142,68,173)';
  }
}
