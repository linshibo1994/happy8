export type PrizeLevel = '一等奖' | '二等奖' | '三等奖' | '四等奖' | '五等奖' | '六等奖'

export type TestingStrategy = 'progressive' | 'random'

export interface TestingRunParams {
  methods: string[]
  strategy: TestingStrategy
  target_prize: PrizeLevel
  periods_start: number
  periods_end: number
  count_start: number
  count_end: number
  max_tests: number
  parallel: boolean
  workers: number
}

export interface MethodTestingStat {
  total_tests: number
  winning_tests: number
  winning_rate: number
  best_prize: PrizeLevel | null
}

export interface TestingStats {
  session_id: string
  test_time: string
  total_tests: number
  winning_tests: number
  winning_rate: number
  method_stats: Record<string, MethodTestingStat>
  prize_stats: Record<string, number>
  best_methods: string[]
}

export interface TestingRunResult {
  session_id: string
  strategy: TestingStrategy
  target_prize: PrizeLevel
  tested_methods: string[]
  successful_methods: string[]
  stats: TestingStats
  report_files: {
    json: string
    text: string
  }
  time: string
}

// SSE 实时事件类型

export interface SseLogEvent {
  message: string
  level: 'info' | 'warning' | 'error' | 'debug'
  timestamp: string
}

export interface SseProgressEvent {
  method: string
  periods: number
  strategy: string
  attempt?: number
  total?: number
  range_start?: number
  range_end?: number
}

export interface SsePrediction {
  predicted_reds: number[]
  predicted_blue: number
  prize_level: string | null
}

export interface SseResultEvent {
  method: string
  periods: number
  has_winning: boolean
  best_prize: string | null
  total_prizes: number
  predictions: SsePrediction[]
}

export interface SseWinningEvent {
  method: string
  periods: number
  prize_level: string
  predicted_reds: number[]
  predicted_blue: number
  winning_reds: number[]
  winning_blue: number
  issue: string
  date: string
}

export interface SseCompleteEvent {
  success: boolean
  message?: string
  session_id?: string
  strategy?: string
  target_prize?: string
  tested_methods?: string[]
  successful_methods?: string[]
  stats?: TestingStats
  report_files?: { json: string; text: string }
  time?: string
}
