import { defineStore } from 'pinia'

import type { LotteryResult } from '@/types'

interface LotteryState {
  latestResult: LotteryResult
  history: LotteryResult[]
  lastUpdatedAt: string
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
  }),
  getters: {
    nextIssue: (state) => String(Number(state.latestResult.issue) + 1),
  },
  actions: {
    setLatestResult(result: LotteryResult) {
      this.latestResult = result
      this.lastUpdatedAt = new Date().toISOString()
    },
  },
})
