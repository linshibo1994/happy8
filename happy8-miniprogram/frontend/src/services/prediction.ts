import { apiService } from './api'
import type {
  PredictionRequest,
  PredictionResult,
  PredictionHistory,
  PredictionStats,
  PredictionLimit,
  LotteryResult,
  ApiResponse,
} from '@/types'

export const predictionApi = {
  // 生成预测
  generatePrediction: (data: PredictionRequest): Promise<ApiResponse<PredictionResult>> => {
    return apiService.post('/predictions/predict', data)
  },

  // 获取预测历史
  getPredictionHistory: (params: {
    limit?: number
    offset?: number
    algorithm?: string
  }): Promise<
    ApiResponse<{ history: PredictionHistory[]; total: number; limit: number; offset: number }>
  > => {
    return apiService.get('/predictions/history', params)
  },

  // 批量生成预测
  generateBatchPredictions: (data: {
    algorithms: string[]
    periods: number
    count: number
    target_issue: string
  }): Promise<
    ApiResponse<{
      results: PredictionResult[]
      total_count: number
      success_count: number
      failed_count: number
    }>
  > => {
    return apiService.post('/predictions/batch-predict', data)
  },

  // 获取预测统计
  getPredictionStats: (): Promise<ApiResponse<PredictionStats>> => {
    return apiService.get('/predictions/stats')
  },

  // 检查预测限制
  getPredictionLimit: (): Promise<ApiResponse<PredictionLimit>> => {
    return apiService.get('/predictions/limit')
  },

  // 获取最新开奖
  getLatestLotteryResults: (
    limit = 10
  ): Promise<ApiResponse<{ results: LotteryResult[]; total: number }>> => {
    return apiService.get('/predictions/lottery-results', { limit })
  },
}
