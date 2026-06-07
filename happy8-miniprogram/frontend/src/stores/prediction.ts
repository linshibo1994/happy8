import { defineStore } from 'pinia'

import type { PredictionExecutionState, PredictionResult } from '@/types'

interface PredictionState {
  selectedAlgorithm: string
  analysisPeriods: number
  predictCount: number
  executionState: PredictionExecutionState
  history: PredictionResult[]
}

const mockResult: PredictionResult = {
  id: 'prediction-mock-001',
  targetIssue: '20260607002',
  algorithm: 'frequency',
  analysisPeriods: 120,
  predictCount: 10,
  numbers: [3, 8, 14, 21, 29, 36, 48, 57, 66, 72],
  confidence: 0.62,
  elapsedMs: 860,
  createdAt: '2026-06-07T10:05:00+08:00',
  explanation: '基于频率、遗漏和区间均衡的演示预测结果。',
}

export const usePredictionStore = defineStore('prediction', {
  state: (): PredictionState => ({
    selectedAlgorithm: 'frequency',
    analysisPeriods: 120,
    predictCount: 10,
    executionState: {
      id: 'execution-idle',
      algorithm: 'frequency',
      targetIssue: '20260607002',
      progress: 0,
      phase: 'permission',
      message: '等待开始预测',
      startedAt: Date.now(),
    },
    history: [mockResult],
  }),
  actions: {
    markRunning(message: string) {
      this.executionState = {
        ...this.executionState,
        progress: 15,
        phase: 'data',
        message,
        startedAt: Date.now(),
        endedAt: undefined,
        error: undefined,
      }
    },
    markDone(result: PredictionResult) {
      this.executionState = {
        ...this.executionState,
        progress: 100,
        phase: 'done',
        message: '预测完成',
        endedAt: Date.now(),
        result,
      }
      this.history.unshift(result)
    },
  },
})
