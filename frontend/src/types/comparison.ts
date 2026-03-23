// 批量对比请求参数
export interface BatchCompareParams {
  target_issue: string
  method_name: string
  periods_mode: 'fixed' | 'random'
  periods_value: number
  comparison_times: number
}

// 单轮对比结果（快乐8）
export interface CompareRoundResult {
  round: number
  analysis_periods: number
  predicted_numbers: number[]
  hit_numbers: number[]
  hit_count: number
  hit_rate: number
  success: boolean
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
  numbers: number[]
}

// 批量对比汇总
export interface BatchCompareResult {
  success: boolean
  error?: string
  target_issue: string
  actual_result: ActualResult
  method_name: string
  comparison_times: number
  success_predictions: number
  success_rate: number
  avg_hit_count: number
  avg_hit_rate: number
  best_hit_count: number
  best_hit_rate: number
  periods_stats: PeriodsStats
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
export interface CompareResultEvent extends CompareRoundResult {}

// SSE 完成事件
export interface CompareCompleteEvent {
  success: boolean
  summary: BatchCompareResult
  message?: string
}

// 算法选项
export interface AlgorithmOption {
  label: string
  value: string
  category: string
}
