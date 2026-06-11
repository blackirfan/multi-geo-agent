import { Component, inject } from '@angular/core';
import { AppStateService } from '../../services/app-state.service';
import {
  AGENT_ICONS,
  AGENT_KEYS,
  AGENT_LABELS,
  AgentKey,
  AgentStatus,
} from '../../models/types';

interface NodeDef {
  key: AgentKey;
  label: string;
  icon: string;
  cx: number;
}

const NODES: NodeDef[] = AGENT_KEYS.map((key, i) => ({
  key,
  label: AGENT_LABELS[key],
  icon: AGENT_ICONS[key],
  cx: 56 + i * 118,
}));

const CY = 62;
const R  = 28;

const STATUS_FILL: Record<AgentStatus, string> = {
  idle:    '#334155',
  running: '#3b82f6',
  done:    '#22c55e',
  error:   '#ef4444',
};

const STATUS_RING: Record<AgentStatus, string> = {
  idle:    'none',
  running: '#3b82f6',
  done:    '#22c55e',
  error:   '#ef4444',
};

@Component({
  selector: 'gr-workflow-view',
  standalone: true,
  template: `
    <div class="border-b border-slate-700 bg-slate-900/60 px-2 py-3">
      <p class="text-[10px] uppercase tracking-widest text-slate-400 font-semibold px-2 mb-2">
        Agent Pipeline
      </p>

      <svg [attr.viewBox]="'0 0 ' + svgW + ' 140'" class="w-full" style="height:130px">

        <!-- Connection lines -->
        @for (node of nodes.slice(0, -1); track node.key) {
          <line
            [attr.x1]="node.cx + R + 4"
            [attr.y1]="CY"
            [attr.x2]="node.cx + 118 - R - 4"
            [attr.y2]="CY"
            stroke="#334155" stroke-width="1.5"
            stroke-dasharray="4 3"/>
          <!-- Arrow head -->
          <polygon
            [attr.points]="arrowPoints(node.cx + 118 - R - 1)"
            fill="#475569"/>
        }

        <!-- Nodes -->
        @for (node of nodes; track node.key) {
          <!-- Pulse ring (running) -->
          @if (statusOf(node.key) === 'running') {
            <circle
              [attr.cx]="node.cx" [attr.cy]="CY"
              [attr.r]="R + 8"
              [attr.fill]="STATUS_RING['running']"
              opacity="0.18"
              class="animate-pulse-ring"/>
          }

          <!-- Main circle -->
          <circle
            [attr.cx]="node.cx" [attr.cy]="CY" [attr.r]="R"
            [attr.fill]="STATUS_FILL[statusOf(node.key)]"
            class="transition-colors duration-500"
            stroke="#0f172a" stroke-width="2"/>

          <!-- Icon -->
          <text
            [attr.x]="node.cx" [attr.y]="CY + 6"
            text-anchor="middle"
            font-size="16" fill="white"
            font-family="system-ui, sans-serif"
            class="select-none pointer-events-none">
            {{ node.icon }}
          </text>

          <!-- Status tick / X overlay -->
          @if (statusOf(node.key) === 'done') {
            <text [attr.x]="node.cx + R - 6" [attr.y]="CY - R + 10"
                  font-size="11" fill="#22c55e" font-weight="700">✓</text>
          }
          @if (statusOf(node.key) === 'error') {
            <text [attr.x]="node.cx + R - 6" [attr.y]="CY - R + 10"
                  font-size="11" fill="#ef4444" font-weight="700">✕</text>
          }

          <!-- Label -->
          <text
            [attr.x]="node.cx" [attr.y]="CY + R + 18"
            text-anchor="middle"
            font-size="9.5" fill="#94a3b8"
            font-family="system-ui, sans-serif"
            class="select-none">
            {{ node.label }}
          </text>

          <!-- Sub-label (status) -->
          <text
            [attr.x]="node.cx" [attr.y]="CY + R + 31"
            text-anchor="middle"
            font-size="8" [attr.fill]="STATUS_RING[statusOf(node.key)] === 'none' ? '#475569' : STATUS_RING[statusOf(node.key)]"
            font-family="system-ui, sans-serif"
            class="select-none transition-all duration-300">
            {{ statusOf(node.key) }}
          </text>
        }
      </svg>
    </div>
  `,
})
export class WorkflowViewComponent {
  protected state = inject(AppStateService);
  protected nodes = NODES;
  protected STATUS_FILL = STATUS_FILL;
  protected STATUS_RING = STATUS_RING;
  protected CY = CY;
  protected R = R;
  protected svgW = 56 + (AGENT_KEYS.length - 1) * 118 + 56;

  statusOf(key: AgentKey): AgentStatus {
    return this.state.statuses()[key];
  }

  arrowPoints(tipX: number): string {
    const y = CY;
    return `${tipX},${y} ${tipX - 7},${y - 4} ${tipX - 7},${y + 4}`;
  }
}
