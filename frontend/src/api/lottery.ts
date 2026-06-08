import { apiClient } from './client'

import type {
  LatestLotteryResponse,
  LotteryHistoryQuery,
  LotteryHistoryResponse,
  LotterySyncSummary,
  SandboxAnalysisQuery,
  SandboxAnalysisResponse,
  SandboxFilterResponse,
  SandboxIntervalsResponse,
  SandboxSummaryResponse,
} from '@/types'

export async function autoSyncLatestLottery(): Promise<LotterySyncSummary> {
  return apiClient.post<LotterySyncSummary, LotterySyncSummary>('/lottery/auto-sync')
}

export async function fetchLatestLotteryResults(limit = 10): Promise<LatestLotteryResponse> {
  return apiClient.get<LatestLotteryResponse, LatestLotteryResponse>('/lottery/latest', {
    params: { limit },
  })
}

export async function fetchLotteryHistory(query: LotteryHistoryQuery = {}): Promise<LotteryHistoryResponse> {
  const { page, page_size, ...rest } = query
  const limit = page_size ?? 20
  const offset = Math.max(0, ((page ?? 1) - 1) * limit)

  return apiClient.get<LotteryHistoryResponse, LotteryHistoryResponse>('/lottery/history', {
    params: {
      ...rest,
      limit,
      offset,
    },
  })
}

export async function fetchSandboxAnalysis(query: SandboxAnalysisQuery): Promise<SandboxAnalysisResponse> {
  const params = buildSandboxParams(query)
  const [filterPayload, intervalPayload, summaryPayload] = await Promise.all([
    apiClient.get<SandboxFilterResponse, SandboxFilterResponse>('/lottery/sandbox/filter', { params }),
    apiClient.get<SandboxIntervalsResponse, SandboxIntervalsResponse>('/lottery/sandbox/intervals', { params }),
    apiClient.get<SandboxSummaryResponse, SandboxSummaryResponse>('/lottery/sandbox/summary', { params }),
  ])

  return {
    window_size: filterPayload.window_size,
    actual_periods: filterPayload.actual_periods,
    events: filterPayload.events ?? filterPayload.results ?? [],
    intervals: intervalPayload.rows ?? [],
    summary: summaryPayload.summary,
    total: filterPayload.total,
  }
}

function buildSandboxParams(query: SandboxAnalysisQuery) {
  return {
    periods: query.recent_periods,
    issue: query.issue,
    start_date: query.start_date,
    end_date: query.end_date,
    event_type: query.event_type === 'interval' ? 'consecutive' : query.event_type,
    level: query.level,
    scope: query.scope,
    zones: query.zones?.join(','),
    limit: Math.max(query.recent_periods ?? 100, query.page_size ?? 20),
    offset: 0,
  }
}
