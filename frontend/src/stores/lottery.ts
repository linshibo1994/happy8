import { defineStore } from 'pinia'

import { autoSyncLatestLottery, fetchLatestLotteryResults } from '@/api'
import type { LotteryResult, LotteryResultPayload } from '@/types'

interface LotteryState {
  latestResult: LotteryResult
  history: LotteryResult[]
  lastUpdatedAt: string
  syncStatus: 'idle' | 'syncing' | 'success' | 'error'
  syncError?: string
}

const latestResult: LotteryResult = {
  issue: '20260607001',
  numbers: [1, 5, 8, 12, 16, 20, 24, 27, 33, 36, 41, 45, 49, 53, 58, 62, 66, 70, 74, 79],
  openedAt: '2026-06-07T10:00:00+08:00',
  sum: 789,
  oddCount: 10,
  evenCount: 10,
  bigCount: 10,
  smallCount: 10,
  zoneDistribution: {
    '1-20': 6,
    '21-40': 4,
    '41-60': 5,
    '61-80': 5,
  },
}

export const useLotteryStore = defineStore('lottery', {
  state: (): LotteryState => ({
    latestResult,
    history: [latestResult],
    lastUpdatedAt: latestResult.openedAt,
    syncStatus: 'idle',
    syncError: undefined,
  }),
  getters: {
    nextIssue: (state) => String(Number(state.latestResult.issue) + 1),
  },
  actions: {
    setLatestResult(result: LotteryResult) {
      this.latestResult = result
      this.lastUpdatedAt = new Date().toISOString()
    },
    async refreshLatestResults(options: { autoSync?: boolean } = {}) {
      this.syncStatus = 'syncing'
      this.syncError = undefined

      try {
        if (options.autoSync ?? true) {
          const syncSummary = await autoSyncLatestLottery()
          if (syncSummary.latest_result) {
            this.setLatestResult(mapLotteryResult(syncSummary.latest_result))
          }
        }

        const payload = await fetchLatestLotteryResults(20)
        const results = payload.results.map(mapLotteryResult)

        if (results.length > 0) {
          this.history = results
          this.setLatestResult(results[0])
        }

        this.syncStatus = 'success'
      } catch (error) {
        this.syncStatus = 'error'
        this.syncError = error instanceof Error ? error.message : '开奖数据同步失败'
      }
    },
  },
})

export function mapLotteryResult(payload: LotteryResultPayload): LotteryResult {
  const numbers = payload.numbers.map(Number)
  return {
    issue: String(payload.issue),
    numbers,
    openedAt: payload.openedAt ?? payload.draw_date ?? new Date().toISOString(),
    sum: payload.sum ?? payload.sum_value ?? numbers.reduce((total, number) => total + number, 0),
    oddCount: payload.oddCount ?? payload.odd_count ?? numbers.filter((number) => number % 2 === 1).length,
    evenCount: payload.evenCount ?? payload.even_count ?? numbers.filter((number) => number % 2 === 0).length,
    bigCount: payload.bigCount ?? payload.big_count ?? numbers.filter((number) => number >= 41).length,
    smallCount: payload.smallCount ?? payload.small_count ?? numbers.filter((number) => number <= 40).length,
    zoneDistribution: normalizeZoneDistribution(
      payload.zoneDistribution ?? payload.zone_distribution,
      numbers,
    ),
  }
}

function normalizeZoneDistribution(
  distribution: Record<string, number> | undefined,
  numbers: number[],
): Record<string, number> {
  if (distribution && '1-20' in distribution) {
    return {
      '1-20': distribution['1-20'] ?? 0,
      '21-40': distribution['21-40'] ?? 0,
      '41-60': distribution['41-60'] ?? 0,
      '61-80': distribution['61-80'] ?? 0,
    }
  }

  if (distribution && 'zone_1' in distribution) {
    return {
      '1-20': distribution.zone_1 ?? 0,
      '21-40': distribution.zone_2 ?? 0,
      '41-60': distribution.zone_3 ?? 0,
      '61-80': distribution.zone_4 ?? 0,
    }
  }

  return buildZoneDistribution(numbers)
}

function buildZoneDistribution(numbers: number[]): Record<string, number> {
  return numbers.reduce<Record<string, number>>(
    (distribution, number) => {
      if (number <= 20) {
        distribution['1-20'] += 1
      } else if (number <= 40) {
        distribution['21-40'] += 1
      } else if (number <= 60) {
        distribution['41-60'] += 1
      } else {
        distribution['61-80'] += 1
      }
      return distribution
    },
    { '1-20': 0, '21-40': 0, '41-60': 0, '61-80': 0 },
  )
}
