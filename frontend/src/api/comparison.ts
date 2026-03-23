import { request } from './request'
import { getApiBaseUrl } from './request'
import type { ApiResponse } from '@/types'
import type { BatchCompareParams, BatchCompareResult } from '@/types/comparison'
import { API_PREFIX } from '@/utils/constants'

// 获取可用期号列表
export const getAvailableIssues = (limit = 100): Promise<ApiResponse<{
  issues: string[]
  total: number
}>> => {
  return request.get(`${API_PREFIX}/comparison/issues`, { params: { limit } })
}

// 执行批量对比（POST，非流式）
export const runBatchCompare = (params: BatchCompareParams): Promise<ApiResponse<BatchCompareResult>> => {
  return request.post(`${API_PREFIX}/comparison/batch`, {
    target_issue: params.target_issue,
    method: params.method_name,
    periods: params.periods_value,
    count: 20,
    comparison_times: params.comparison_times,
    max_parallel: 1,
    timeout_seconds: 30,
  })
}

// 创建 SSE 流式批量对比连接
export const createBatchCompareStream = (params: BatchCompareParams): EventSource => {
  const searchParams = new URLSearchParams({
    target_issue: params.target_issue,
    method_name: params.method_name,
    periods_mode: params.periods_mode,
    periods_value: String(params.periods_value),
    comparison_times: String(params.comparison_times)
  })

  const baseUrl = getApiBaseUrl()
  const url = `${baseUrl}${API_PREFIX}/comparison/batch/stream?${searchParams.toString()}`
  return new EventSource(url)
}
