import { Component, inject, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AppStateService } from '../../services/app-state.service';
import { GeoReasonerService } from '../../services/georeasoner.service';

type ExportFormat = 'pdf' | 'html';

@Component({
  selector: 'gr-query-panel',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="p-4 border-b border-slate-700 bg-slate-800/60">

      <!-- Brand row -->
      <div class="flex items-center gap-2 mb-4">
        <div class="w-7 h-7 rounded-full bg-crimson flex items-center justify-center text-white text-xs font-black">G</div>
        <div>
          <p class="text-sm font-bold text-white leading-none">GeoReasoner</p>
          <p class="text-[10px] text-slate-400 mt-0.5">Multi-Agent Geospatial AI</p>
        </div>
      </div>

      <!-- Query textarea -->
      <label class="block text-[10px] uppercase tracking-widest text-slate-400 mb-1.5 font-semibold">
        Natural-Language Query
      </label>
      <textarea
        [(ngModel)]="state.query"
        [disabled]="state.isRunning()"
        rows="3"
        placeholder="Which upazila in Sylhet have the highest flood risk?"
        class="w-full px-3 py-2 rounded-md bg-slate-700 border border-slate-600
               text-slate-100 text-sm resize-none
               placeholder:text-slate-500
               focus:outline-none focus:border-crimson focus:ring-1 focus:ring-crimson/40
               disabled:opacity-50 disabled:cursor-not-allowed
               transition-colors"
      ></textarea>

      <!-- Run button -->
      <div class="mt-2.5">
        <button
          (click)="runAnalysis()"
          [disabled]="state.isRunning() || !state.query().trim()"
          class="w-full flex items-center justify-center gap-2
                 px-3 py-2 rounded-md text-sm font-semibold
                 bg-crimson hover:bg-crimson/90 text-white
                 disabled:opacity-40 disabled:cursor-not-allowed
                 transition-all active:scale-95">
          @if (state.isRunning()) {
            <span class="animate-spin text-base">◌</span>
            <span>Analysing…</span>
          } @else {
            <span>▶</span>
            <span>Run Analysis</span>
          }
        </button>
      </div>

      <!-- Export buttons — visible only after a completed run -->
      @if (state.runId()) {
        <div class="mt-2.5">
          <p class="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-1.5">
            Export Report
          </p>
          <div class="grid grid-cols-2 gap-2">

            <!-- PDF -->
            <button
              (click)="exportReport('pdf')"
              [disabled]="state.isRunning() || exporting() !== null"
              title="Download full report as PDF"
              class="flex items-center justify-center gap-1.5
                     px-2 py-2 rounded-md text-xs font-semibold
                     bg-rose-700 hover:bg-rose-600 text-white
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all active:scale-95">
              @if (exporting() === 'pdf') {
                <span class="animate-spin">◌</span>
              } @else {
                <span>↓</span>
              }
              PDF
            </button>

            <!-- HTML -->
            <button
              (click)="exportReport('html')"
              [disabled]="state.isRunning() || exporting() !== null"
              title="Open interactive HTML report in a new tab"
              class="flex items-center justify-center gap-1.5
                     px-2 py-2 rounded-md text-xs font-semibold
                     bg-sky-700 hover:bg-sky-600 text-white
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all active:scale-95">
              @if (exporting() === 'html') {
                <span class="animate-spin">◌</span>
              } @else {
                <span>↓</span>
              }
              HTML
            </button>

          </div>
        </div>
      }

      <!-- Error banner -->
      @if (state.error()) {
        <div class="mt-3 px-3 py-2 rounded-md bg-red-900/50 border border-red-700 text-red-300 text-xs">
          ✕ {{ state.error() }}
        </div>
      }
    </div>
  `,
})
export class QueryPanelComponent {
  protected state    = inject(AppStateService);
  private   geo      = inject(GeoReasonerService);

  analysisStarted = output<void>();
  protected exporting = signal<ExportFormat | null>(null);

  runAnalysis(): void {
    const q = this.state.query().trim();
    if (!q || this.state.isRunning()) return;
    this.state.setRunning();
    this.analysisStarted.emit();

    this.geo.runQuery(q).subscribe({
      next:  res => this.state.setDone(res),
      error: err => this.state.setError(
        err?.error?.detail ?? err?.message ?? 'Unknown error'
      ),
    });
  }

  exportReport(format: ExportFormat): void {
    const id = this.state.runId();
    if (!id || this.exporting() !== null) return;

    this.exporting.set(format);

    this.geo.generateReport(
      id,
      this.state.query(),
      this.state.answer(),
      this.state.ranking(),
      this.state.trace(),
    ).subscribe({
      next:  () => this.openDownload(id, format),
      error: () => this.openDownload(id, format),
    });
  }

  private openDownload(id: string, format: ExportFormat): void {
    this.exporting.set(null);
    window.open(this.geo.reportDownloadUrl(id, format), '_blank');
  }
}
