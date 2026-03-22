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
  MissingAnalysisResult
} from '@/types/lottery'
import { API_PREFIX } from '@/utils/constants'

// 健康检查 (使用详细健康检查端点)
export const checkHealth = (): Promise<HealthResponse> => {
  return request.get(`${API_PREFIX}/health`)
}

// 简单健康检查 (根路径)
export const pingHealth = (): Promise<{ status: string }> => {
  return request.get('/health')
}

// 获取最新开奖数据
export const getLatestData = (): Promise<ApiResponse<LotteryResult>> => {
  return request.get(`${API_PREFIX}/data/latest`)
}

// 获取历史数据
export const getHistoryData = (params: {
  periods?: number
  start_period?: string
  end_period?: string
}): Promise<HistoryDataResponse> => {
  return request.post(`${API_PREFIX}/data/history`, params)
}

// 刷新数据
export const refreshData = (): Promise<ApiResponse<null>> => {
  return request.post(`${API_PREFIX}/data/refresh`)
}

// 追加最近N期数据
export const appendRecentData = (count = 5): Promise<ApiResponse<null>> => {
  return request.post(`${API_PREFIX}/data/append`, { count })
}

// 获取统计数据
export const getStatistics = (): Promise<ApiResponse<Statistics>> => {
  return request.get(`${API_PREFIX}/data/statistics`)
}

// 获取频率分析
export const getFrequencyAnalysis = (params: {
  analysis_type?: string
  periods?: number
}): Promise<ApiResponse<FrequencyAnalysisResult>> => {
  return request.post(`${API_PREFIX}/analysis/frequency`, {
    analysis_type: params.analysis_type || 'frequency',
    periods: params.periods || 100
  })
}

// 获取走势分析
export const getTrendsAnalysis = (params: {
  analysis_type?: string
  periods?: number
}): Promise<ApiResponse<TrendsAnalysisResult>> => {
  return request.post(`${API_PREFIX}/analysis/trends`, {
    analysis_type: params.analysis_type || 'trends',
    periods: params.periods || 100
  })
}

// 获取趋势分析
export const getTrendAnalysis = (params: {
  analysis_type: string
  periods?: number
}): Promise<ApiResponse<TrendsAnalysisResult>> => {
  return request.post(`${API_PREFIX}/analysis/trend`, {
    analysis_type: params.analysis_type,
    periods: params.periods || 100
  })
}

// 获取冷热号分析
export const getHotColdAnalysis = (periods?: number): Promise<ApiResponse<HotColdAnalysisResult>> => {
  return request.get(`${API_PREFIX}/analysis/hot_cold`, {
    params: { periods: periods || 50 }
  })
}

// 获取遗漏值分析
export const getMissingAnalysis = (periods?: number): Promise<ApiResponse<MissingAnalysisResult>> => {
  return request.get(`${API_PREFIX}/analysis/missing`, {
    params: { periods: periods || 100 }
  })
}
