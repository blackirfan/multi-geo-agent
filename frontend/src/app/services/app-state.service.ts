import { Injectable, computed, signal } from '@angular/core';
import {
  AGENT_KEYS,
  AgentKey,
  AgentStatus,
  QueryResponse,
} from '../models/types';

type StatusMap = Record<AgentKey, AgentStatus>;

const idleStatuses = (): StatusMap =>
  Object.fromEntries(AGENT_KEYS.map(k => [k, 'idle'])) as StatusMap;

@Injectable({ providedIn: 'root' })
export class AppStateService {
  readonly query   = signal('Which unions in Sylhet have the highest flood risk?');
  readonly isRunning = signal(false);
  readonly result  = signal<QueryResponse | null>(null);
  readonly statuses = signal<StatusMap>(idleStatuses());
  readonly error   = signal<string | null>(null);

  readonly ranking = computed(() => this.result()?.fsi_ranking ?? []);
  readonly trace   = computed(() => this.result()?.agent_trace ?? []);
  readonly answer  = computed(() => this.result()?.answer ?? null);
  readonly runId   = computed(() => this.result()?.run_id ?? null);

  setRunning(): void {
    this.isRunning.set(true);
    this.error.set(null);
    this.result.set(null);
    this.statuses.set(
      Object.fromEntries(AGENT_KEYS.map(k => [k, 'running'])) as StatusMap
    );
  }

  setDone(response: QueryResponse): void {
    this.result.set(response);
    this.isRunning.set(false);
    // Animate agents to 'done' in trace order
    const seen = new Set<AgentKey>();
    const ordered: AgentKey[] = [];
    for (const e of response.agent_trace) {
      const key = e.agent as AgentKey;
      if (AGENT_KEYS.includes(key) && !seen.has(key)) {
        seen.add(key);
        ordered.push(key);
      }
    }
    // Fill in any missing agents (they still completed)
    for (const k of AGENT_KEYS) {
      if (!seen.has(k)) ordered.push(k);
    }
    ordered.forEach((key, i) => {
      setTimeout(() => {
        this.statuses.update(s => ({ ...s, [key]: 'done' }));
      }, 350 * (i + 1));
    });
  }

  setError(msg: string): void {
    this.isRunning.set(false);
    this.error.set(msg);
    this.statuses.set(
      Object.fromEntries(AGENT_KEYS.map(k => [k, 'error'])) as StatusMap
    );
  }

  reset(): void {
    this.isRunning.set(false);
    this.result.set(null);
    this.error.set(null);
    this.statuses.set(idleStatuses());
  }
}
