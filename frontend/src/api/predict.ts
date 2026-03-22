import { request } from './request'
import { getApiBaseUrl } from './request'
import type { ApiResponse } from '@/types'
import type { PredictMethod, PredictParams, PredictionResponse } from '@/types/predict'
import { API_PREFIX } from '@/utils/constants'

// 获取预测方法列表
export const getMethods = (): Promise<ApiResponse<PredictMethod[]>> => {
  return request.get(`${API_PREFIX}/methods`)
}

// 获取算法列表(分类)
export const getAlgorithms = (): Promise<ApiResponse<{
  id: number
  name: string
  category: string
  description: string
  params: Record<string, unknown>
  supported_features: {
    gpu: boolean
    parallel: boolean
  }
}[]>> => {
  return request.get(`${API_PREFIX}/algorithms`)
}

// 获取算法详情
export const getAlgorithmDetail = (id: number): Promise<ApiResponse<unknown>> => {
  return request.get(`${API_PREFIX}/algorithms/${id}`)
}

// 执行预测
export const runPredict = (params: PredictParams): Promise<PredictionResponse> => {
  return request.post(`${API_PREFIX}/predict`, {
    method: params.method,
    periods: params.periods || 100,
    count: params.count || 1,
    duplex: params.duplex || false,
    red_count: params.redCount || 8,
    blue_count: params.blueCount || 2,
    explain: params.explain !== false,
    use_gpu: params.useGpu || false,
    parallel: params.parallel || false
  })
}

// 流式执行预测（SSE）
export const createPredictStream = (params: PredictParams): EventSource => {
  const searchParams = new URLSearchParams({
    method: params.method,
    periods: String(params.periods || 100),
    count: String(params.count || 1),
    duplex: String(Boolean(params.duplex)),
    red_count: String(params.redCount || 8),
    blue_count: String(params.blueCount || 2),
    explain: String(params.explain !== false),
    use_gpu: String(Boolean(params.useGpu)),
    parallel: String(Boolean(params.parallel))
  })
  const baseUrl = getApiBaseUrl()
  const url = `${baseUrl}${API_PREFIX}/predict/stream?${searchParams.toString()}`
  return new EventSource(url)
}

// 获取系统配置
export const getSystemConfig = (): Promise<ApiResponse<{
  app_name: string
  version: string
  description: string
  features: {
    gpu_enabled: boolean
    parallel_enabled: boolean
    max_workers: number
  }
  limits: {
    max_periods: number
    max_predictions: number
    max_duplex_red: number
    max_duplex_blue: number
  }
}>> => {
  return request.get(`${API_PREFIX}/system/config`)
}

// 获取版本信息
export const getVersion = (): Promise<ApiResponse<{
  version: string
  app_name: string
  build_time: string
  features: string[]
}>> => {
  return request.get(`${API_PREFIX}/version`)
}
