import { API_CONFIG, ERROR_MESSAGES, STATUS_CODES, STORAGE_KEYS } from '@/constants'
import type { ApiResponse } from '@/types'

class ApiService {
  private baseURL: string
  private timeout: number

  constructor() {
    this.baseURL = API_CONFIG.FULL_BASE_URL
    this.timeout = API_CONFIG.TIMEOUT
  }

  private async request<T>(
    url: string,
    options: {
      method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
      data?: UniApp.RequestOptions['data']
      headers?: Record<string, string>
      timeout?: number
      skipAuth?: boolean
    } = {}
  ): Promise<ApiResponse<T>> {
    const { method = 'GET', data, headers = {}, timeout = this.timeout, skipAuth = false } = options

    try {
      // 获取认证令牌
      if (!skipAuth) {
        const token = uni.getStorageSync(STORAGE_KEYS.ACCESS_TOKEN)
        if (token) {
          headers['Authorization'] = `Bearer ${token}`
        }
      }

      // 设置默认headers
      headers['Content-Type'] = 'application/json'

      const requestConfig: UniApp.RequestOptions = {
        url: `${this.baseURL}${url}`,
        method,
        data,
        header: headers,
        timeout,
        success: () => {},
        fail: () => {},
        complete: () => {},
      }

      const response = await new Promise<UniApp.RequestSuccessCallbackResult>((resolve, reject) => {
        uni.request({
          ...requestConfig,
          success: resolve,
          fail: reject,
        })
      })

      // 检查HTTP状态码
      if (response.statusCode >= 200 && response.statusCode < 300) {
        return response.data as ApiResponse<T>
      }

      if (response.statusCode === STATUS_CODES.UNAUTHORIZED) {
        // Token过期，尝试刷新
        if (!skipAuth && !url.includes('/auth/refresh')) {
          const refreshResult = await this.refreshToken()
          if (refreshResult) {
            // 重新发起请求
            return this.request<T>(url, options)
          } else {
            // 刷新失败，跳转登录
            this.handleAuthError()
          }
        }
        throw new Error(ERROR_MESSAGES.AUTH_EXPIRED)
      }

      const body = response.data as Partial<ApiResponse<unknown>> | undefined
      const message =
        typeof body?.message === 'string' && body.message
          ? body.message
          : `请求失败: ${response.statusCode}`
      throw new Error(message)
    } catch (error) {
      console.error('API请求失败:', error)

      // 网络错误处理
      const errMsg =
        typeof error === 'object' &&
        error !== null &&
        'errMsg' in error &&
        typeof (error as { errMsg: string }).errMsg === 'string'
          ? (error as { errMsg: string }).errMsg
          : ''
      if (errMsg.includes('timeout')) {
        throw new Error(ERROR_MESSAGES.TIMEOUT_ERROR)
      }
      if (errMsg.includes('fail')) {
        throw new Error(ERROR_MESSAGES.NETWORK_ERROR)
      }

      if (error instanceof Error) {
        throw error
      }
      throw new Error(ERROR_MESSAGES.UNKNOWN_ERROR)
    }
  }

  private async refreshToken(): Promise<boolean> {
    try {
      const refreshToken = uni.getStorageSync(STORAGE_KEYS.REFRESH_TOKEN)
      if (!refreshToken) return false

      const response = await this.request<{ access_token: string; refresh_token: string }>(
        '/auth/refresh',
        {
          method: 'POST',
          data: { refresh_token: refreshToken },
          skipAuth: true,
        }
      )

      if (response.code === STATUS_CODES.SUCCESS) {
        uni.setStorageSync(STORAGE_KEYS.ACCESS_TOKEN, response.data.access_token)
        uni.setStorageSync(STORAGE_KEYS.REFRESH_TOKEN, response.data.refresh_token)
        return true
      }

      return false
    } catch {
      return false
    }
  }

  private handleAuthError() {
    // 清除本地认证信息
    uni.removeStorageSync(STORAGE_KEYS.ACCESS_TOKEN)
    uni.removeStorageSync(STORAGE_KEYS.REFRESH_TOKEN)
    uni.removeStorageSync(STORAGE_KEYS.USER_INFO)

    // 跳转到登录页面
    uni.showModal({
      title: '登录过期',
      content: '您的登录已过期，请重新登录',
      showCancel: false,
      success: () => {
        uni.switchTab({
          url: '/pages/profile/profile',
        })
      },
    })
  }

  // GET请求
  async get<T>(
    url: string,
    params?: Record<string, string | number | boolean | Array<string | number>>
  ): Promise<ApiResponse<T>> {
    let fullUrl = url
    if (params) {
      const queryParts: string[] = []
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          if (Array.isArray(value)) {
            value.forEach(item => {
              queryParts.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(item))}`)
            })
          } else {
            queryParts.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
          }
        }
      })
      if (queryParts.length > 0) {
        fullUrl += `?${queryParts.join('&')}`
      }
    }

    return this.request<T>(fullUrl, { method: 'GET' })
  }

  // POST请求
  async post<T>(url: string, data?: UniApp.RequestOptions['data']): Promise<ApiResponse<T>> {
    return this.request<T>(url, { method: 'POST', data })
  }

  // PUT请求
  async put<T>(url: string, data?: UniApp.RequestOptions['data']): Promise<ApiResponse<T>> {
    return this.request<T>(url, { method: 'PUT', data })
  }

  // DELETE请求
  async delete<T>(url: string): Promise<ApiResponse<T>> {
    return this.request<T>(url, { method: 'DELETE' })
  }

  // 文件上传
  async upload<T>(
    url: string,
    filePath: string,
    name: string = 'file',
    formData?: UniApp.UploadFileOption['formData']
  ): Promise<ApiResponse<T>> {
    try {
      const token = uni.getStorageSync('access_token')
      const headers: Record<string, string> = {}

      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await new Promise<UniApp.UploadFileSuccessCallbackResult>(
        (resolve, reject) => {
          uni.uploadFile({
            url: `${this.baseURL}${url}`,
            filePath,
            name,
            formData,
            header: headers,
            success: resolve,
            fail: reject,
          })
        }
      )

      if (response.statusCode >= 200 && response.statusCode < 300) {
        return JSON.parse(response.data) as ApiResponse<T>
      } else {
        throw new Error(`上传失败: ${response.statusCode}`)
      }
    } catch (error) {
      console.error('文件上传失败:', error)
      throw error
    }
  }
}

export const apiService = new ApiService()
