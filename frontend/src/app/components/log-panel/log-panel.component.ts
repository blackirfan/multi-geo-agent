import { Component, inject } from '@angular/core';
import { AppStateService } from '../../services/app-state.service';

const AGENT_COLORS: Record<string, string> = {
  planner:        '#a78bfa',
  gis_analyst:    '#34d399',
  remote_sensing: '#60a5fa',
  hydrology:      '#f59e0b',
  reasoner:       '#f472b6',
};

@Component({
  selector: 'gr-log-panel',
  standalone: true,
  template: `
    <div class="flex flex-col h-full min-h-0">
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-2
                  border-b border-slate-700 bg-slate-900/60 flex-shrink-0">
        <span class="text-[10px] uppercase tracking-widest font-semibold text-slate-400">
          Execution Log
        </span>
        @if (state.trace().length) {
          <span class="text-[10px] text-slate-500">{{ state.trace().length }} events</span>
        }
      </div>

      <!-- Answer summary (when available) -->
      @if (state.answer()) {
        <div class="px-4 py-3 border-b border-slate-700 bg-slate-800/40 flex-shrink-0">
          <p class="text-[10px] uppercase tracking-widest text-slate-500 mb-1.5 font-semibold">
            Analysis Summary
          </p>
          <p class="text-xs text-slate-300 leading-relaxed line-clamp-4">
            {{ answerText() }}
          </p>
        </div>
      }

      <!-- Trace entries -->
      <div class="flex-1 overflow-y-auto min-h-0">
        @if (!state.trace().length && !state.isRunning()) {
          <div class="flex flex-col items-center justify-center h-24 text-slate-600 text-xs text-center px-4">
            Agent execution events will appear here.
          </div>
        }

        @if (state.isRunning() && !state.trace().length) {
          <div class="px-4 py-4 text-xs text-slate-500 flex items-center gap-2">
            <span class="animate-spin">◌</span>
            Pipeline executing…
          </div>
        }

        @for (entry of state.trace(); track $index) {
          <div class="px-4 py-2.5 border-b border-slate-700/30 hover:bg-slate-700/20 transition-colors">
            <div class="flex items-center gap-2 mb-0.5">
              <span class="text-[10px] font-bold uppercase tracking-wide"
                    [style.color]="agentColor(entry.agent)">
                {{ entry.agent }}
              </span>
              <span class="text-[10px] text-slate-500">→</span>
              <span class="text-[10px] font-mono text-slate-400 truncate">{{ entry.tool }}</span>
            </div>
            @if (entry.timestamp) {
              <p class="text-[9px] text-slate-600 font-mono">{{ entry.timestamp }}</p>
            }
            @if (hasError(entry.result)) {
              <p class="text-[10px] text-red-400 mt-0.5 font-mono truncate">{{ entry.result }}</p>
            }
          </div>
        }
      </div>
    </div>
  `,
})
export class LogPanelComponent {
  protected state = inject(AppStateService);

  agentColor(agent: string): string {
    return AGENT_COLORS[agent] ?? '#94a3b8';
  }

  hasError(result: string): boolean {
    return typeof result === 'string' && result.startsWith('ERROR');
  }

  answerText(): string {
    return (this.state.answer() ?? '').replace(/\*\*/g, '').slice(0, 300);
  }
}
