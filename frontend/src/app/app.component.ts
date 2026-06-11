import { Component } from '@angular/core';
import { LogPanelComponent } from './components/log-panel/log-panel.component';
import { MapViewComponent } from './components/map-view/map-view.component';
import { QueryPanelComponent } from './components/query-panel/query-panel.component';
import { RankingTableComponent } from './components/ranking-table/ranking-table.component';
import { WorkflowViewComponent } from './components/workflow-view/workflow-view.component';

@Component({
  selector: 'gr-root',
  standalone: true,
  imports: [
    QueryPanelComponent,
    RankingTableComponent,
    MapViewComponent,
    WorkflowViewComponent,
    LogPanelComponent,
  ],
  template: `
    <div class="h-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden">

      <!-- ── Header ──────────────────────────────────────────────── -->
      <header class="flex-shrink-0 flex items-center justify-between
                     px-5 py-2.5 border-b border-slate-800
                     bg-slate-900/80 backdrop-blur-sm z-10">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-full bg-crimson flex items-center justify-center
                      text-white font-black text-sm shadow-lg shadow-crimson/30">G</div>
          <div>
            <h1 class="text-base font-bold text-white leading-none">
              GeoReasoner
              <span class="ml-2 text-[10px] font-semibold px-1.5 py-0.5 rounded
                           bg-crimson/20 text-crimson border border-crimson/30">β</span>
            </h1>
            <p class="text-[10px] text-slate-500 mt-0.5">
              LLM-Orchestrated Multi-Agent Geospatial Intelligence · Sylhet, Bangladesh
            </p>
          </div>
        </div>

        <div class="flex items-center gap-4 text-[10px] text-slate-500">
          <span class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            FastAPI backend
          </span>
          <span>Gemma 3 12B via LM Studio</span>
          <span>LangGraph pipeline</span>
        </div>
      </header>

      <!-- ── Main three-column layout ────────────────────────────── -->
      <main class="flex-1 flex overflow-hidden min-h-0">

        <!-- Left: query + ranking -->
        <aside class="w-72 flex-shrink-0 flex flex-col border-r border-slate-800
                      bg-slate-900/50 overflow-hidden">
          <gr-query-panel />
          <gr-ranking-table class="flex-1 min-h-0" />
        </aside>

        <!-- Center: Leaflet map -->
        <div class="flex-1 relative overflow-hidden min-w-0">
          <gr-map-view class="absolute inset-0 block" />
        </div>

        <!-- Right: workflow + log -->
        <aside class="w-80 flex-shrink-0 flex flex-col border-l border-slate-800
                      bg-slate-900/50 overflow-hidden">
          <gr-workflow-view class="flex-shrink-0" />
          <gr-log-panel class="flex-1 min-h-0" />
        </aside>

      </main>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100vh; }
    .bg-crimson { background-color: #e94560; }
    .text-crimson { color: #e94560; }
    .border-crimson { border-color: #e94560; }
    .shadow-crimson\\/30 { --tw-shadow-color: rgba(233,69,96,.3); }
    .bg-crimson\\/20 { background-color: rgba(233,69,96,.2); }
    .border-crimson\\/30 { border-color: rgba(233,69,96,.3); }
  `],
})
export class AppComponent {}
