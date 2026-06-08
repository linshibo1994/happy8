export type MembershipLevel = 'free' | 'vip' | 'premium'

export type AlgorithmCategory = '基础统计' | '马尔可夫' | '机器学习' | '深度学习' | '概率与综合'

export type PredictionPhase =
  | 'permission'
  | 'data'
  | 'feature'
  | 'compute'
  | 'validate'
  | 'done'
  | 'error'

export interface LotteryResult {
  issue: string
  numbers: number[]
  openedAt: string
  sum: number
  oddCount: number
  evenCount: number
  bigCount: number
  smallCount: number
  zoneDistribution: Record<string, number>
}

export interface LotteryResultPayload {
  issue: string
  numbers: number[]
  draw_date?: string
  openedAt?: string
  sum_value?: number
  sum?: number
  odd_count?: number
  oddCount?: number
  even_count?: number
  evenCount?: number
  big_count?: number
  bigCount?: number
  small_count?: number
  smallCount?: number
  zone_distribution?: Record<string, number>
  zoneDistribution?: Record<string, number>
}

export interface LatestLotteryResponse {
  results: LotteryResultPayload[]
  total: number
}

export interface LotterySyncSummary {
  updated_count: number
  latest_result?: LotteryResultPayload | null
  synced_at?: string | null
  skipped?: boolean
  reason?: string
  trigger?: string
}

export type SandboxEventType = 'consecutive' | 'gap' | 'mixed' | 'interval'
export type SandboxScope = 'global' | 'zone'
export type SandboxConsecutiveLevel = 2 | 3 | 4
export type SandboxTableMode = 'history' | 'events' | 'intervals'

export interface LotteryHistoryQuery {
  page?: number
  page_size?: number
  issue?: string
  start_date?: string
  end_date?: string
}

export interface LotteryHistoryResponse {
  results: LotteryResultPayload[]
  total: number
  page?: number
  page_size?: number
  has_more?: boolean
}

export interface SandboxAnalysisQuery {
  recent_periods?: number
  issue?: string
  start_date?: string
  end_date?: string
  event_type?: SandboxEventType
  level?: SandboxConsecutiveLevel
  scope?: SandboxScope
  zones?: number[]
  page?: number
  page_size?: number
}

export interface SandboxEventMatch {
  issue: string
  draw_date?: string
  openedAt?: string
  numbers: number[]
  event_type: SandboxEventType
  scope: SandboxScope
  zones?: number[]
  groups: number[][]
  longest_length?: number
  group_count?: number
  label?: string
}

export interface SandboxIntervalRow {
  issue: string
  next_issue?: string | null
  gap?: number | null
  distance?: number | null
  draw_date?: string
}

export interface SandboxSummary {
  sample_periods: number
  event_level?: SandboxConsecutiveLevel
  hit_periods: number
  hit_rate: number
  total_groups: number
  avg_gap?: number | null
  median_gap?: number | null
  max_gap?: number | null
  current_missing?: number | null
  latest_issue?: string | null
  top_zones?: Array<{ zone: number; count: number }>
  baseline_delta?: number | null
  updated_at?: string | null
}

export interface SandboxAnalysisResponse {
  window_size: number
  actual_periods: number
  events: SandboxEventMatch[]
  intervals: SandboxIntervalRow[]
  summary: SandboxSummary
  total: number
}

export interface SandboxFilterResponse {
  results?: SandboxEventMatch[]
  events?: SandboxEventMatch[]
  total: number
  limit: number
  offset: number
  window_size: number
  actual_periods: number
}

export interface SandboxIntervalsResponse {
  matched_count: number
  sample_size: number
  is_sample_sufficient: boolean
  intervals: number[]
  rows?: SandboxIntervalRow[]
  avg_interval?: number | null
  min_interval?: number | null
  max_interval?: number | null
  message?: string | null
  window_size?: number
  actual_periods?: number
}

export interface SandboxSummaryResponse {
  periods: number
  matched_count: number
  match_rate: number
  highlights: string[]
  latest_matches: SandboxEventMatch[]
  summary: SandboxSummary
  window_size?: number
  actual_periods?: number
}

export interface Algorithm {
  name: string
  displayName: string
  category: AlgorithmCategory
  permissionLevel: MembershipLevel
  description: string
  averageCostMs: number
  successRate: number
  enabled: boolean
  complexity: '低' | '中' | '高'
  recommendedScenario: string
}

export interface PredictionResult {
  id: string
  targetIssue: string
  algorithm: string
  analysisPeriods: number
  predictCount: number
  numbers: number[]
  confidence: number
  elapsedMs: number
  createdAt: string
  explanation: string
}

export interface PredictionExecutionState {
  id: string
  algorithm: string
  targetIssue: string
  progress: number
  phase: PredictionPhase
  message: string
  startedAt: number
  endedAt?: number
  result?: PredictionResult
  error?: string
}

export interface MembershipStatus {
  level: MembershipLevel
  levelName: string
  remainingPredictions: number
  dailyLimit: number
  expiresAt?: string
  benefits: string[]
}

export interface UserProfile {
  id: string
  nickname: string
  avatarUrl?: string
  role: 'user' | 'admin'
}

export interface ApiEnvelope<T> {
  code?: number
  message?: string
  data?: T
}

export interface ApiErrorPayload {
  status: number
  message: string
  raw?: unknown
}
