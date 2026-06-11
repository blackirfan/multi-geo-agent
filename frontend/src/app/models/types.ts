export interface RankingItem {
  rank: number;
  name: string;
  mean_fsi: number;
  max_fsi: number;
}

export interface TraceEntry {
  agent: string;
  tool: string;
  result: string;
  timestamp: string;
}

export interface QueryResponse {
  run_id: string;
  answer: string | null;
  fsi_ranking: RankingItem[];
  agent_trace: TraceEntry[];
  error: string | null;
}

export interface ReportResponse {
  run_id: string;
  report_url: string;
  format: string;
}

export type AgentStatus = 'idle' | 'running' | 'done' | 'error';

export const AGENT_KEYS = [
  'planner',
  'gis_analyst',
  'remote_sensing',
  'hydrology',
  'reasoner',
] as const;

export type AgentKey = (typeof AGENT_KEYS)[number];

export const AGENT_LABELS: Record<AgentKey, string> = {
  planner:        'Planner',
  gis_analyst:    'GIS Analyst',
  remote_sensing: 'Remote Sensing',
  hydrology:      'Hydrology',
  reasoner:       'Reasoner',
};

export const AGENT_ICONS: Record<AgentKey, string> = {
  planner:        '◈',
  gis_analyst:    '⬡',
  remote_sensing: '⊙',
  hydrology:      '◉',
  reasoner:       '✦',
};
