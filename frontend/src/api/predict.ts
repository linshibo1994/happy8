import { request } from './request'
import { getApiBaseUrl } from './request'
import type { ApiResponse } from '@/types'
import type { PredictMethod, PredictParams, PredictionResponse } from '@/types/predict'
import { API_PREFIX } from '@/utils/constants'

const LEGACY_METHOD_ALIASES: Record<string, string> = {
  super: 'super_predictor',
  graph_nn: 'gnn',
  dynamic_bayes: 'bayesian',
  adaptive_ensemble: 'advanced_ensemble',
  high_confidence_full: 'high_confidence',
  high_confidence_advanced: 'high_confidence',
  high_confidence_lite: 'high_confidence',
  high_confidence_complete: 'high_confidence',
  hybrid: 'super_predictor',
  hybrid_v2: 'super_predictor',
  stats: 'frequency',
  probability: 'bayesian',
  decision_tree: 'advanced_ensemble',
  patterns: 'clustering',
  frequency_cons2: 'frequency',
  consensus_halving: 'ensemble',
}

const normalizeMethod = (method: string): string => {
  return LEGACY_METHOD_ALIASES[method] || method
}

const categoryToKey = (category: string): PredictMethod['category'] => {
  if (category.includes('马尔可夫')) return 'markov'
  if (category.includes('深度学习')) return 'deep_learning'
  if (category.includes('机器学习')) return 'machine_learning'
  if (category.includes('智能')) return 'intelligent'
  return 'statistical'
}

// 获取预测方法列表
export const getMethods = async (): Promise<ApiResponse<PredictMethod[]>> => {
  const res = await request.get<ApiResponse<{
    methods: Array<{
      method: string
      display_name?: string
      category?: string
      mapped_function?: string
    }>
  }>>(`${API_PREFIX}/methods`)

  const items = Array.isArray(res.data?.methods) ? res.data.methods : []
  const mapped: PredictMethod[] = items.map((item) => {
    const method = String(item.method || '')
    return {
      id: method,
      name: String(item.display_name || method),
      description: `后端映射: ${item.mapped_function || '未知'}`,
      category: categoryToKey(String(item.category || '统计类')),
      supportsGpu: ['transformer', 'gnn', 'lstm'].includes(method),
      supportsParallel: true,
    }
  })

  return {
    ...res,
    data: mapped,
  }
}

// 获取算法列表(分类)
export const getAlgorithms = async (): Promise<ApiResponse<{
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
  const methodRes = await getMethods()
  const data = methodRes.data.map((item, index) => ({
    id: index + 1,
    name: item.name,
    category: item.category,
    description: item.description,
    params: {},
    supported_features: {
      gpu: Boolean(item.supportsGpu),
      parallel: Boolean(item.supportsParallel),
    },
  }))

  return {
    success: methodRes.success,
    message: methodRes.message,
    timestamp: methodRes.timestamp,
    data,
  }
}

// 获取算法详情
export const getAlgorithmDetail = async (id: number): Promise<ApiResponse<unknown>> => {
  const list = await getAlgorithms()
  const item = list.data.find((algo) => algo.id === id) || null
  return {
    success: list.success,
    message: item ? '获取算法详情成功' : '未找到算法详情',
    timestamp: list.timestamp,
    data: item,
  }
}

// 执行预测
export const runPredict = (params: PredictParams): Promise<PredictionResponse> => {
  return request.post(`${API_PREFIX}/predict`, {
    method: normalizeMethod(params.method),
    periods: params.periods || 100,
    count: params.count || 1,
  })
}

// 流式执行预测（SSE）
export const createPredictStream = (params: PredictParams): EventSource => {
  const searchParams = new URLSearchParams({
    method: normalizeMethod(params.method),
    periods: String(params.periods || 100),
    count: String(params.count || 1),
  })
  const baseUrl = getApiBaseUrl()
  const url = `${baseUrl}${API_PREFIX}/predict/stream?${searchParams.toString()}`
  return new EventSource(url)
}

// 获取系统配置
export const getSystemConfig = async (): Promise<ApiResponse<{
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
    max_numbers: number
  }
}>> => {
  const res = await request.get<ApiResponse<{
    app?: { name?: string; version?: string }
  }>>(`${API_PREFIX}/system/info`)

  const app = res.data?.app || {}
  return {
    ...res,
    data: {
      app_name: String(app.name || 'Happy8 Prediction API'),
      version: String(app.version || '1.0.0'),
      description: '快乐8预测系统运行配置',
      features: {
        gpu_enabled: true,
        parallel_enabled: true,
        max_workers: 8,
      },
      limits: {
        max_periods: 5000,
        max_predictions: 30,
        max_numbers: 30,
      },
    },
  }
}

// 获取版本信息
export const getVersion = async (): Promise<ApiResponse<{
  version: string
  app_name: string
  build_time: string
  features: string[]
}>> => {
  const res = await request.get<ApiResponse<{
    app?: { name?: string; version?: string }
    algorithms?: { methods?: string[] }
  }>>(`${API_PREFIX}/system/info`)

  const app = res.data?.app || {}
  const algorithms = res.data?.algorithms?.methods || []

  return {
    ...res,
    data: {
      version: String(app.version || '1.0.0'),
      app_name: String(app.name || 'Happy8 Prediction API'),
      build_time: res.timestamp || new Date().toISOString(),
      features: Array.isArray(algorithms) ? algorithms : [],
    },
  }
}
