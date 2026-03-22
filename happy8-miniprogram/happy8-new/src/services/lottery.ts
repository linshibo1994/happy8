import { apiService } from './api'
import type { LotteryResult, ApiResponse } from '@/types'

export const lotteryApi = {
  // 获取最新开奖结果
  getLatestResults: (limit: number = 10): Promise<ApiResponse<{ results: LotteryResult[] }>> => {
    return apiService.get('/lottery/latest', { limit })
  },

  // 获取历史开奖结果
  getHistoricalResults: (params: {
    limit?: number
    offset?: number
    start_date?: string
    end_date?: string
    issue?: string
  }): Promise<ApiResponse<{ results: LotteryResult[] }>> => {
    return apiService.get('/lottery/history', params)
  },

  // 获取指定期号的开奖结果
  getResultByIssue: (issue: string): Promise<ApiResponse<LotteryResult>> => {
    return apiService.get(`/lottery/results/${issue}`)
  },

  // 获取开奖统计信息
  getStatistics: (params: {
    periods?: number
    type?: 'frequency' | 'hot_cold' | 'missing' | 'zone'
  }): Promise<ApiResponse<Record<string, unknown>>> => {
    return apiService.get('/lottery/statistics', params)
  },

  // 获取号码走势
  getTrends: (params: {
    periods?: number
    numbers?: number[]
  }): Promise<ApiResponse<Record<string, unknown>>> => {
    return apiService.get('/lottery/trends', params)
  },

  // 搜索开奖结果
  searchResults: (params: {
    numbers?: number[]
    start_date?: string
    end_date?: string
    limit?: number
    offset?: number
  }): Promise<ApiResponse<{ results: LotteryResult[] }>> => {
    return apiService.get('/lottery/search', params)
  },

  // 同步最新数据
  syncLatestData: (): Promise<ApiResponse<{ updated_count: number }>> => {
    return apiService.post('/lottery/sync')
  },
}
