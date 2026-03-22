import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { API_BASE_URL, API_TIMEOUT } from '@/utils/constants'

// 错误处理回调类型
type ErrorCallback = (message: string, status?: number) => void

// 错误处理器（由外部注入，用于连接通知系统）
let errorHandler: ErrorCallback | null = null

const normalizeBaseUrl = (url: string): string => {
  const normalized = (url || '').trim()
  if (!normalized) return ''
  return normalized.replace(/\/+$/, '')
}

const loadInitialBaseUrl = (): string => {
  try {
    const saved = localStorage.getItem('app_api_url')
    if (saved !== null) {
      return normalizeBaseUrl(saved)
    }
  } catch {
    // 忽略本地存储异常
  }
  return normalizeBaseUrl(API_BASE_URL)
}

// 设置错误处理器
export const setErrorHandler = (handler: ErrorCallback) => {
  errorHandler = handler
}

export const getApiBaseUrl = (): string => normalizeBaseUrl(String(instance.defaults.baseURL || ''))

export const setApiBaseUrl = (url: string) => {
  const normalized = normalizeBaseUrl(url)
  instance.defaults.baseURL = normalized
  try {
    localStorage.setItem('app_api_url', normalized)
  } catch {
    // 忽略本地存储异常
  }
}

// 创建axios实例
const instance: AxiosInstance = axios.create({
  baseURL: loadInitialBaseUrl(),
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
instance.interceptors.request.use(
  (config) => {
    // 可在此添加token等认证信息
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
instance.interceptors.response.use(
  (response: AxiosResponse) => {
    return response.data
  },
  (error) => {
    let message = '未知错误'
    let status: number | undefined

    if (error.response) {
      status = error.response.status
      const data = error.response.data

      // 根据状态码生成友好的错误消息
      switch (status) {
        case 400:
          message = data?.message || '请求参数错误'
          break
        case 401:
          message = '未授权，请重新登录'
          break
        case 403:
          message = '拒绝访问'
          break
        case 404:
          message = data?.message || '请求的资源不存在'
          break
        case 500:
          message = '服务器内部错误'
          break
        case 502:
          message = '网关错误'
          break
        case 503:
          message = '服务暂时不可用'
          break
        case 504:
          message = '网关超时'
          break
        default:
          message = data?.message || `服务器错误 (${status})`
      }
    } else if (error.request) {
      message = '网络错误：无法连接到服务器'
    } else if (error.code === 'ECONNABORTED') {
      message = '请求超时，请稍后重试'
    } else {
      message = error.message || '请求配置错误'
    }

    // 控制台日志
    console.error(`API错误 [${status || 'NETWORK'}]:`, message)

    // 调用错误处理器（如果已设置）
    if (errorHandler) {
      errorHandler(message, status)
    }

    return Promise.reject(error)
  }
)

// 封装请求方法
export const request = {
  get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return instance.get(url, config)
  },
  post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return instance.post(url, data, config)
  },
  put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return instance.put(url, data, config)
  },
  delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return instance.delete(url, config)
  }
}

export default instance
