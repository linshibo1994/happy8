import { apiClient } from './client'

import type { LatestLotteryResponse, LotterySyncSummary } from '@/types'

export async function autoSyncLatestLottery(): Promise<LotterySyncSummary> {
  return apiClient.post<LotterySyncSummary, LotterySyncSummary>('/lottery/auto-sync')
}

export async function fetchLatestLotteryResults(limit = 10): Promise<LatestLotteryResponse> {
  return apiClient.get<LatestLotteryResponse, LatestLotteryResponse>('/lottery/latest', {
    params: { limit },
  })
}
