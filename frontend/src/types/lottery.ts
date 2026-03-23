// 快乐8开奖结果
export interface LotteryResult {
  period: string
  date: string
  numbers: number[]
  sum_value?: number
  odd_count?: number
  even_count?: number
  big_count?: number
  small_count?: number
}

// 历史记录项
export interface HistoryItem extends LotteryResult {
  week?: string
}

// 号码统计
export interface NumberStats {
  number: number
  count: number
  frequency: number
  lastAppear: number
  maxMissing: number
}

// 总体统计
export interface Statistics {
  total_periods: number
  frequency: Record<number, number>
  most_frequent: [number, number][]
  least_frequent: [number, number][]
}

// 趋势点
export interface TrendPoint {
  period: string
  date?: string
  sum_value: number
  odd_count: number
  even_count: number
  big_count?: number
  small_count?: number
}

// 频率分析结果
export interface FrequencyAnalysisResult {
  type: 'frequency'
  periods: number
  frequency: Record<number, number>
  top_numbers: [number, number][]
}

// 走势分析结果
export interface TrendsAnalysisResult {
  type: 'trends'
  periods: number
  trend_data: TrendPoint[]
  patterns: {
    average_sum: number
    max_sum: number
    min_sum: number
    hot_numbers: [number, number][]
  }
}

// 冷热号分析结果
export interface HotColdAnalysisResult {
  periods: number
  hot_numbers: number[]
  cold_numbers: number[]
  frequency: Record<number, number>
}

// 遗漏分析结果
export interface MissingAnalysisResult {
  periods: number
  missing_map: Record<number, number>
  max_missing: [number, number]
}

// 历史数据响应
export interface HistoryDataResponse {
  success: boolean
  data: HistoryItem[]
  total: number
  message?: string
  timestamp: string
}
