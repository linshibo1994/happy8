// 彩票开奖结果 (匹配后端返回结构)
export interface LotteryResult {
  period: string         // 期号
  date: string           // 开奖日期
  red_balls: number[]    // 红球数组 (6个)
  blue_ball: number      // 蓝球 (1个)
  sales?: string         // 销售额
  pool?: string          // 奖池金额
}

// 前端展示用 (兼容旧字段名)
export interface LotteryDisplay {
  code: string           // 期号 (别名)
  date: string           // 开奖日期
  red: number[]          // 红球数组
  blue: number           // 蓝球
  sales?: string
  pool?: string
}

export interface PrizeDetail {
  name: string           // 奖项名称
  count: number          // 中奖注数
  money: number          // 单注奖金
}

export interface HistoryItem extends LotteryResult {
  week?: string          // 星期
}

// 号码统计
export interface NumberStats {
  number: number
  count: number
  frequency: number      // 频率百分比
  lastAppear: number     // 上次出现距今期数
  maxMissing: number     // 最大遗漏期数
}

// 统计数据 (匹配后端返回结构)
export interface Statistics {
  total_periods: number
  red_frequency: Record<number, number>   // 红球频率 {1: 50, 2: 48, ...}
  blue_frequency: Record<number, number>  // 蓝球频率
  most_frequent_red: [number, number][]   // [[号码, 次数], ...]
  most_frequent_blue: [number, number][]
}

// 趋势数据点
export interface TrendPoint {
  period: string
  date?: string
  red_sum: number
  odd_count: number
  even_count: number
  max_red?: number
  min_red?: number
  blue_ball?: number
}

// 辅助函数：转换后端数据为前端展示格式
export function toLotteryDisplay(result: LotteryResult): LotteryDisplay {
  return {
    code: result.period,
    date: result.date,
    red: result.red_balls,
    blue: result.blue_ball,
    sales: result.sales,
    pool: result.pool
  }
}

// 频率分析结果
export interface FrequencyAnalysisResult {
  type: 'frequency'
  periods: number
  red_frequency: Record<number, number>
  blue_frequency: Record<number, number>
  most_frequent_red: [number, number][]
  most_frequent_blue: [number, number][]
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
  red_balls: {
    hot: number[]
    warm: number[]
    cold: number[]
    frequency: Record<number, number>
  }
  blue_balls: {
    hot: number[]
    warm: number[]
    cold: number[]
    frequency: Record<number, number>
  }
}

// 遗漏值分析结果
export interface MissingAnalysisResult {
  periods: number
  red_missing: Record<number, number>
  blue_missing: Record<number, number>
  max_red_missing: [number, number]
  max_blue_missing: [number, number]
}

// 历史数据响应 (修复嵌套问题)
export interface HistoryDataResponse {
  success: boolean
  data: HistoryItem[]
  total: number
  message?: string
  timestamp: string
}
