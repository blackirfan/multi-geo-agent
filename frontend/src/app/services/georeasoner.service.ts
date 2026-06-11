import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { QueryResponse, RankingItem, ReportResponse } from '../models/types';

@Injectable({ providedIn: 'root' })
export class GeoReasonerService {
  private http = inject(HttpClient);

  runQuery(query: string, runId?: string): Observable<QueryResponse> {
    return this.http.post<QueryResponse>('/query', {
      query,
      ...(runId ? { run_id: runId } : {}),
    });
  }

  getAdminLayer(): Observable<object> {
    return this.http.get<object>('/layers/admin');
  }

  getRiversLayer(): Observable<object> {
    return this.http.get<object>('/layers/rivers');
  }

  getFsiLayer(ranking: RankingItem[]): Observable<object> {
    return this.http.post<object>('/layers/fsi', { fsi_ranking: ranking });
  }

  generateReport(
    runId: string,
    query: string,
    answer: string | null,
    ranking: RankingItem[],
    trace: object[],
  ): Observable<ReportResponse> {
    return this.http.post<ReportResponse>('/reports', {
      run_id: runId,
      query,
      answer,
      fsi_ranking: ranking,
      agent_trace: trace,
    });
  }

  reportDownloadUrl(runId: string, format: 'pdf' | 'html' = 'pdf'): string {
    return `/reports/${runId}?format=${format}`;
  }
}
