import axios, { AxiosError, type AxiosResponse } from 'axios'

import type { ApiEnvelope, ApiErrorPayload } from '@/types'

const TOKEN_STORAGE_KEY = 'happy8.auth.token'

export class Happy8ApiError extends Error {
  status: number
  raw?: unknown

  constructor(payload: ApiErrorPayload) {
    super(payload.message)
    this.name = 'Happy8ApiError'
    this.status = payload.status
    this.raw = payload.raw
  }
}

export const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
})

apiClient.interceptors.request.use((config) => {
  const token = window.localStorage.getItem(TOKEN_STORAGE_KEY)

  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  return config
})

apiClient.interceptors.response.use(
  (response: AxiosResponse<ApiEnvelope<unknown> | unknown>) => unwrapResponse(response),
  (error: AxiosError<ApiEnvelope<unknown>>) => {
    throw normalizeApiError(error)
  },
)

function unwrapResponse<T>(response: AxiosResponse<ApiEnvelope<T> | T>): T {
  const body = response.data

  if (isEnvelope<T>(body)) {
    if (typeof body.code === 'number' && body.code !== 0 && body.code !== 200) {
      throw new Happy8ApiError({
        status: response.status,
        message: body.message || '接口返回业务错误',
        raw: body,
      })
    }

    return body.data as T
  }

  return body as T
}

function normalizeApiError(error: AxiosError<ApiEnvelope<unknown>>): Happy8ApiError {
  if (error.response) {
    const message =
      error.response.data?.message ||
      statusMessageMap[error.response.status] ||
      '接口请求失败，请稍后重试'

    return new Happy8ApiError({
      status: error.response.status,
      message,
      raw: error.response.data,
    })
  }

  if (error.code === 'ECONNABORTED') {
    return new Happy8ApiError({
      status: 408,
      message: '接口请求超时，请检查网络或稍后重试',
      raw: error,
    })
  }

  return new Happy8ApiError({
    status: 0,
    message: '网络连接异常，请确认服务是否可用',
    raw: error,
  })
}

function isEnvelope<T>(value: unknown): value is ApiEnvelope<T> {
  return Boolean(value && typeof value === 'object' && ('data' in value || 'code' in value))
}

const statusMessageMap: Record<number, string> = {
  400: '请求参数不正确，请检查输入内容',
  401: '登录状态已失效，请重新登录',
  403: '当前会员权限不足',
  404: '请求的数据不存在',
  409: '操作冲突，请刷新后重试',
  429: '请求过于频繁，请稍后再试',
  500: '服务端异常，请稍后重试',
}

export function setAuthToken(token: string): void {
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token)
}

export function clearAuthToken(): void {
  window.localStorage.removeItem(TOKEN_STORAGE_KEY)
}
