import { request } from './request'
import { getApiBaseUrl } from './request'
import type { ApiResponse } from '@/types'
import type { TestingRunParams, TestingRunResult } from '@/types/testing'
import { API_PREFIX } from '@/utils/constants'

// 运行测试系统（保留兼容）
export const runPredictionTests = (params: TestingRunParams): Promise<ApiResponse<TestingRunResult>> => {
  return request.post(`${API_PREFIX}/testing/run`, params, { timeout: 0 })
}

// 获取测试系统选项
export const getTestingOptions = (): Promise<ApiResponse<{
  available_methods: string[]
  target_prizes: string[]
}>> => {
  return request.get(`${API_PREFIX}/testing/options`)
}

// 创建 SSE 流式测试连接
export const createTestingStream = (params: TestingRunParams): EventSource => {
  const searchParams = new URLSearchParams({
    methods: params.methods.join(','),
    strategy: params.strategy,
    target_prize: params.target_prize,
    periods_start: String(params.periods_start),
    periods_end: String(params.periods_end),
    count_start: String(params.count_start),
    count_end: String(params.count_end),
    max_tests: String(params.max_tests),
    parallel: String(params.parallel),
    workers: String(params.workers)
  })
  const baseUrl = getApiBaseUrl()
  const url = `${baseUrl}${API_PREFIX}/testing/stream?${searchParams.toString()}`
  return new EventSource(url)
}
