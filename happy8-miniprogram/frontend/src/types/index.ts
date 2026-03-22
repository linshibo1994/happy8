/**
 * 通用API响应结构
 */
export interface ApiResponse<T = unknown> {
  code: number
  message: string
  data: T
  success?: boolean
  error_code?: string | null
  timestamp?: string
}

export type MembershipLevel = 'free' | 'vip' | 'premium'

export interface LoginRequest {
  code: string
  user_info?: Record<string, unknown>
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in?: number
  user_info: UserInfo
}

export interface RefreshTokenRequest {
  refresh_token: string
}

export interface RefreshTokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface UserInfo {
  id: number
  openid?: string
  wechat_openid?: string
  nickname?: string
  avatar_url?: string
  phone?: string
  email?: string
  created_at?: string
  updated_at?: string
  real_name?: string | null
  gender?: 'male' | 'female' | 'unknown' | null
  birthday?: string | null
  address?: string | null
  preferences?: Record<string, unknown> | null
  is_new?: boolean
}

export interface UserProfile {
  real_name?: string
  gender?: 'male' | 'female' | 'unknown'
  birthday?: string
  address?: string
  preferences?: Record<string, unknown>
}

export interface MembershipSummary {
  id: number
  user_id: number
  level: MembershipLevel
  expire_date?: string | null
  auto_renew?: boolean | null
  predictions_today: number
  predictions_total: number
  is_valid: boolean
  days_remaining?: number | null
}

export interface MembershipStatus {
  membership: MembershipSummary
  permissions: Record<string, boolean>
  limits: {
    daily_predictions_limit: number | null
    daily_predictions_used: number
    total_predictions: number
  }
}

export interface MembershipPlan {
  id: number
  name: string
  level: MembershipLevel
  duration_days: number
  price: number
  original_price?: number | null
  features: string[]
  max_predictions_per_day?: number | null
  available_algorithms?: string[] | null
  is_active: boolean
  sort_order?: number | null
  description?: string
}

export type OrderStatus = 'pending' | 'paid' | 'cancelled' | 'refunded' | 'expired'

export interface MembershipOrder {
  id: number
  order_no: string
  user_id: number
  plan_id: number
  plan_name: string
  amount: number
  original_amount?: number | null
  discount_amount?: number | null
  status: OrderStatus
  pay_method?: string | null
  transaction_id?: string | null
  expire_at?: string | null
  paid_at?: string | null
  created_at: string
  updated_at: string
}

export interface OrderListResult {
  orders: MembershipOrder[]
  total: number
  has_more: boolean
}

export interface AlgorithmInfo {
  id: number
  algorithm_name: string
  display_name: string
  description: string
  required_level: MembershipLevel
  has_permission: boolean
  default_params: Record<string, unknown>
  avg_execution_time?: number | null
  success_rate?: number | null
  usage_count: number
}

export interface PredictionRequest {
  algorithm: string
  target_issue: string
  periods: number
  count: number
  params?: Record<string, unknown>
}

export interface PredictionAnalysis {
  algorithm: string
  engine?: string
  [key: string]: unknown
}

export interface PredictionResult {
  predicted_numbers: number[]
  confidence_score: number
  analysis_data: PredictionAnalysis
  algorithm: string
  target_issue: string
  periods: number
  execution_time: number
  is_cached: boolean
}

export interface PredictionHistory {
  id: number
  algorithm: string
  target_issue: string
  periods: number
  count: number
  predicted_numbers: number[]
  confidence_score?: number | null
  actual_numbers?: number[] | null
  hit_count?: number | null
  hit_rate?: number | null
  is_hit?: boolean | null
  execution_time: number
  is_cached: boolean
  created_at: string
}

export interface LotteryResult {
  id: number
  issue: string
  draw_date: string
  numbers: number[]
  sum_value: number
  odd_count: number
  even_count: number
  big_count: number
  small_count: number
  zone_distribution: Record<string, number>
}

export interface PredictionStats {
  user_id: number
  total_predictions: number
  today_predictions: number
  hit_predictions: number
  overall_hit_rate: number
  favorite_algorithm?: string | null
  algorithm_stats: Record<string, PredictionStatDetail>
  recent_performance: PredictionRecentPerformance[]
}

export interface PredictionStatDetail {
  count: number
  hit_count: number
  hit_rate: number
}

export interface PredictionRecentPerformance {
  date: string
  algorithm: string
  target_issue: string
  is_hit: boolean | null
  hit_count: number | null
  confidence_score: number | null
}

export interface PredictionLimit {
  can_predict: boolean
  predictions_today: number
  daily_limit: number | null
  membership_level: MembershipLevel | 'NONE'
  remaining_predictions: number | null
  next_reset_time: string | null
}

export interface SystemInfo {
  model: string
  pixelRatio: number
  windowWidth: number
  windowHeight: number
  system: string
  platform: string
  version: string
  statusBarHeight?: number
  safeArea?: {
    left: number
    right: number
    top: number
    bottom: number
    width: number
    height: number
  }
}

export interface AppConfig {
  api_base_url: string
  version: string
  debug: boolean
  cache_duration: number
  max_retry_times: number
}

export interface CreateOrderRequest {
  plan_id: number
}

export interface CreatePaymentRequest {
  order_no: string
  payment_method?: 'wechat_pay'
}

export interface CancelPaymentRequest {
  order_no: string
}

export interface PaymentParams {
  timeStamp: string
  nonceStr: string
  package: string
  signType: string
  paySign: string
}

export interface PaymentResponse {
  prepay_id: string
  payment_params: PaymentParams
}

export interface PaymentStatus {
  order_no: string
  status: string
  transaction_id?: string | null
  paid_at?: string | null
  created_at?: string | null
  expire_at?: string | null
  reason?: string | null
  trade_state?: string | null
}

export interface PaginatedQuery {
  limit?: number
  offset?: number
}
