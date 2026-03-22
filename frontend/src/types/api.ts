// API响应基础结构
export interface ApiResponse<T = unknown> {
  success: boolean
  message: string
  data: T
  timestamp?: string
}

// 分页请求参数
export interface PaginationParams {
  page?: number
  pageSize?: number
}

// 分页响应
export interface PaginatedResponse<T> {
  total: number
  page: number
  page_size: number
  items: T[]
}

// 健康检查响应
export interface HealthResponse {
  status: string
  timestamp: string
  version: string
  analyzer_available: boolean
  total_periods: number
  last_period?: string
  last_update?: string
}

// 错误响应
export interface ErrorResponse {
  success: false
  message: string
  error?: string
  code?: number
}
