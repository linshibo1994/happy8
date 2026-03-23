import { request } from './request'
import type { ApiResponse, HealthResponse } from '@/types'
import type {
  LotteryResult,
  HistoryItem,
  Statistics,
  HistoryDataResponse,
  FrequencyAnalysisResult,
  TrendsAnalysisResult,
  HotColdAnalysisResult,
  MissingAnalysisResult,
  TrendPoint,
} from '@/types/lottery'
import { API_PREFIX } from '@/utils/constants'

type BackendResponse<T> = ApiResponse<T>

interface BackendLatestItem {
  issue: string
  draw_date?: string
  date?: string
  numbers: number[]
  sum?: number
  odd_count?: number
}

interface BackendHistoryPayload {
  items: Array<{
    issue: string
    date: string
    numbers: number[]
    sum?: number
    odd_count?: number
  }>
  pagination?: {
    total?: number
  }
}

interface BackendFrequencyItem {
  number: number
  count: number
}

const toLotteryResult = (item: BackendLatestItem): LotteryResult => {
  const numbers = Array.isArray(item.numbers) ? item.numbers : []
  const oddCount = Number(item.odd_count ?? numbers.filter((num) => num % 2 === 1).length)

  return {
    period: String(item.issue || ''),
    date: String(item.date || item.draw_date || ''),
    numbers,
    sum_value: Number(item.sum ?? numbers.reduce((sum, num) => sum + num, 0)),
    odd_count: oddCount,
    even_count: Math.max(0, numbers.length - oddCount),
  }
}

const buildFrequencyMap = (items: BackendFrequencyItem[]): Record<number, number> => {
  const map: Record<number, number> = {}
  for (const item of items) {
    const num = Number(item.number)
    const count = Number(item.count)
    if (Number.isFinite(num) && Number.isFinite(count)) {
      map[num] = count
    }
  }
  return map
}

const toTrendPoints = (historyItems: HistoryItem[]): TrendPoint[] => {
  return historyItems.map((item) => {
    const sumValue = item.sum_value ?? item.numbers.reduce((sum, num) => sum + num, 0)
    const oddCount = item.odd_count ?? item.numbers.filter((num) => num % 2 === 1).length
    return {
      period: item.period,
      date: item.date,
      sum_value: sumValue,
      odd_count: oddCount,
      even_count: Math.max(0, item.numbers.length - oddCount),
    }
  })
}

// 健康检查
export const checkHealth = async (): Promise<HealthResponse> => {
  const res = await request.get<BackendResponse<{
    status: string
    version?: string
    analyzer?: {
      loaded?: boolean
      total_records?: number
      latest_issue?: string
    }
  }>>('/health')

  const data = res.data || {}
  const analyzer = data.analyzer || {}

  return {
    status: data.status || (res.success ? 'healthy' : 'unhealthy'),
    timestamp: res.timestamp || new Date().toISOString(),
    version: data.version || 'unknown',
    analyzer_available: Boolean(analyzer.loaded),
    total_periods: Number(analyzer.total_records || 0),
    last_period: analyzer.latest_issue,
    last_update: res.timestamp,
  }
}

// 简单健康检查
export const pingHealth = async (): Promise<{ status: string }> => {
  const health = await checkHealth()
  return { status: health.status }
}

// 获取最新开奖数据
export const getLatestData = async (): Promise<ApiResponse<LotteryResult>> => {
  const res = await request.get<BackendResponse<BackendLatestItem>>(`${API_PREFIX}/data/latest`)
  return {
    ...res,
    data: res.data ? toLotteryResult(res.data) : (null as unknown as LotteryResult),
  }
}

// 获取历史数据
export const getHistoryData = async (params: {
  periods?: number
  start_period?: string
  end_period?: string
}): Promise<HistoryDataResponse> => {
  const pageSize = Math.max(1, Math.min(Number(params.periods || 100), 500))
  const payload = {
    page: 1,
    page_size: pageSize,
    start_issue: params.start_period,
    end_issue: params.end_period,
  }

  const res = await request.post<BackendResponse<BackendHistoryPayload>>(`${API_PREFIX}/data/history`, payload)
  const items = Array.isArray(res.data?.items) ? res.data.items : []
  const mappedItems: HistoryItem[] = items.map((item) => toLotteryResult(item))
  const total = Number(res.data?.pagination?.total || mappedItems.length)

  return {
    success: res.success,
    message: res.message,
    timestamp: res.timestamp || new Date().toISOString(),
    data: mappedItems,
    total,
  }
}

// 刷新数据
export const refreshData = (): Promise<ApiResponse<null>> => {
  return request.post(`${API_PREFIX}/data/refresh`, { reload_from_local: false })
}

// 追加最近N期数据
export const appendRecentData = (count = 5): Promise<ApiResponse<null>> => {
  return request.post(`${API_PREFIX}/data/append`, { count })
}

// 获取统计数据
export const getStatistics = async (): Promise<ApiResponse<Statistics>> => {
  const res = await request.get<BackendResponse<{
    total_records: number
    frequency: BackendFrequencyItem[]
  }>>(`${API_PREFIX}/data/statistics`)

  const frequency = buildFrequencyMap(Array.isArray(res.data?.frequency) ? res.data.frequency : [])
  const sorted = Object.entries(frequency)
    .map(([num, count]) => [Number(num), Number(count)] as [number, number])
    .sort((a, b) => b[1] - a[1])

  return {
    ...res,
    data: {
      total_periods: Number(res.data?.total_records || 0),
      frequency,
      most_frequent: sorted.slice(0, 10),
      least_frequent: sorted.slice(-10),
    },
  }
}

// 获取频率分析
export const getFrequencyAnalysis = async (params: {
  periods?: number
}): Promise<ApiResponse<FrequencyAnalysisResult>> => {
  const res = await request.get<BackendResponse<{
    periods: number
    all_numbers: BackendFrequencyItem[]
    top_numbers: BackendFrequencyItem[]
  }>>(`${API_PREFIX}/analysis/frequency`, {
    params: { periods: params.periods || 100 },
  })

  const allNumbers = Array.isArray(res.data?.all_numbers) ? res.data.all_numbers : []
  const frequency = buildFrequencyMap(allNumbers)
  const topNumbers = Array.isArray(res.data?.top_numbers) ? res.data.top_numbers : []

  return {
    ...res,
    data: {
      type: 'frequency',
      periods: Number(res.data?.periods || 0),
      frequency,
      top_numbers: topNumbers.map((item) => [Number(item.number), Number(item.count)] as [number, number]),
    },
  }
}

// 获取走势分析（基于历史数据计算）
export const getTrendsAnalysis = async (params: {
  periods?: number
}): Promise<ApiResponse<TrendsAnalysisResult>> => {
  const history = await getHistoryData({ periods: params.periods || 100 })
  const trendData = toTrendPoints(history.data || [])
  const sums = trendData.map((item) => item.sum_value)
  const averageSum = sums.length > 0 ? sums.reduce((sum, value) => sum + value, 0) / sums.length : 0

  return {
    success: history.success,
    message: history.message || '走势分析完成',
    timestamp: history.timestamp,
    data: {
      type: 'trends',
      periods: trendData.length,
      trend_data: trendData,
      patterns: {
        average_sum: averageSum,
        max_sum: sums.length > 0 ? Math.max(...sums) : 0,
        min_sum: sums.length > 0 ? Math.min(...sums) : 0,
        hot_numbers: [],
      },
    },
  }
}

// 获取趋势分析（别名）
export const getTrendAnalysis = (params: {
  periods?: number
}): Promise<ApiResponse<TrendsAnalysisResult>> => {
  return getTrendsAnalysis({ periods: params.periods })
}

// 获取冷热号分析
export const getHotColdAnalysis = async (periods?: number): Promise<ApiResponse<HotColdAnalysisResult>> => {
  const res = await request.get<BackendResponse<{
    periods: number
    hot_numbers: Array<{ number: number; count: number }>
    cold_numbers: Array<{ number: number; count: number }>
  }>>(`${API_PREFIX}/analysis/hot-cold`, {
    params: { periods: periods || 50 },
  })

  const hot = (res.data?.hot_numbers || []).map((item) => Number(item.number)).filter((n) => Number.isFinite(n))
  const cold = (res.data?.cold_numbers || []).map((item) => Number(item.number)).filter((n) => Number.isFinite(n))
  const frequency: Record<number, number> = {}

  for (const item of res.data?.hot_numbers || []) {
    frequency[Number(item.number)] = Number(item.count || 0)
  }
  for (const item of res.data?.cold_numbers || []) {
    const num = Number(item.number)
    if (!(num in frequency)) {
      frequency[num] = Number(item.count || 0)
    }
  }

  return {
    ...res,
    data: {
      periods: Number(res.data?.periods || 0),
      hot_numbers: hot,
      cold_numbers: cold,
      frequency,
    },
  }
}

// 获取遗漏分析
export const getMissingAnalysis = async (periods?: number): Promise<ApiResponse<MissingAnalysisResult>> => {
  const res = await request.get<BackendResponse<{
    periods: number
    all_numbers: Array<{ number: number; missing_periods: number }>
  }>>(`${API_PREFIX}/analysis/missing`, {
    params: { periods: periods || 100 },
  })

  const allNumbers = Array.isArray(res.data?.all_numbers) ? res.data.all_numbers : []
  const missingMap: Record<number, number> = {}

  for (let i = 1; i <= 80; i++) {
    const found = allNumbers.find((item) => Number(item.number) === i)
    missingMap[i] = Number(found?.missing_periods || 0)
  }

  const maxMissing = Object.entries(missingMap)
    .map(([num, value]) => [Number(num), Number(value)] as [number, number])
    .sort((a, b) => b[1] - a[1])[0] || [0, 0]

  return {
    ...res,
    data: {
      periods: Number(res.data?.periods || 0),
      missing_map: missingMap,
      max_missing: maxMissing,
    },
  }
}
