import { apiService } from './api'
import type { AlgorithmInfo, PredictionStats, ApiResponse } from '@/types'

export const algorithmApi = {
  // 获取可用算法列表
  getAvailableAlgorithms: (): Promise<ApiResponse<AlgorithmInfo[]>> => {
    return apiService.get('/algorithms')
  },

  // 获取算法详情
  getAlgorithmDetail: (algorithmName: string): Promise<ApiResponse<AlgorithmInfo>> => {
    return apiService.get(`/algorithms/${algorithmName}`)
  },

  // 获取用户统计信息
  getUserStats: (): Promise<ApiResponse<PredictionStats>> => {
    return apiService.get('/algorithms/user-stats')
  },

  // 获取算法统计信息
  getAlgorithmStats: (algorithmName: string): Promise<ApiResponse<Record<string, unknown>>> => {
    return apiService.get(`/algorithms/${algorithmName}/stats`)
  },

  // 更新算法参数
  updateAlgorithmParams: (
    algorithmName: string,
    params: Record<string, unknown>
  ): Promise<ApiResponse<null>> => {
    return apiService.put(`/algorithms/${algorithmName}/params`, { params })
  },
}
