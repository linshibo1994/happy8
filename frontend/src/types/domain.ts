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
