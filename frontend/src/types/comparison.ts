// 批量对比请求参数
export interface BatchCompareParams {
  target_issue: string
  method_name: string
  periods_mode: 'fixed' | 'random'
  periods_value: number
  comparison_times: number
}

// 单轮对比结果
export interface CompareRoundResult {
  round: number
  analysis_periods: number
  predicted_reds: number[] | null
  predicted_blue: number | null
  prize_level: number
  prize_name: string
  red_matches: number
  blue_match: boolean
  success: boolean
}

// 奖级统计
export interface PrizeStats {
  count: number
  probability: number
  prize_name: string
  prize_money: string
}

// 期数统计
export interface PeriodsStats {
  mode: string
  fixed_value: number | null
  used_periods: number[]
  min_periods: number
  max_periods: number
  avg_periods: number
}

// 实际开奖结果
export interface ActualResult {
  issue: string
  date: string
  red_balls: number[]
  blue_ball: number
}

// 批量对比完整结果
export interface BatchCompareResult {
  success: boolean
  error?: string
  target_issue: string
  actual_result: ActualResult
  method_name: string
  comparison_times: number
  success_predictions: number
  success_rate: number
  periods_stats: PeriodsStats
  probability_stats: Record<string, PrizeStats>
  detailed_results: CompareRoundResult[]
  generated_time: string
}

// SSE 进度事件
export interface CompareProgressEvent {
  round: number
  total: number
  percentage: number
}

// SSE 结果事件
export interface CompareResultEvent {
  round: number
  predicted_reds: number[]
  predicted_blue: number
  prize_level: number
  prize_name: string
  red_matches: number
  blue_match: boolean
  analysis_periods: number
}

// SSE 完成事件
export interface CompareCompleteEvent {
  success: boolean
  summary: BatchCompareResult
}

// 算法选项
export interface AlgorithmOption {
  label: string
  value: string
  category: string
}
