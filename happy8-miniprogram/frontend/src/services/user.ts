import { apiService } from './api'
import type {
  LoginRequest,
  LoginResponse,
  RefreshTokenRequest,
  UserInfo,
  UserProfile,
  ApiResponse,
  PredictionStats,
} from '@/types'

export const userApi = {
  // 微信登录
  wechatLogin: (data: LoginRequest): Promise<ApiResponse<LoginResponse>> => {
    return apiService.post('/auth/wechat-login', data)
  },

  // 刷新令牌
  refreshToken: (
    data: RefreshTokenRequest
  ): Promise<ApiResponse<{ access_token: string; refresh_token: string }>> => {
    return apiService.post('/auth/refresh', data)
  },

  // 登出
  logout: (): Promise<ApiResponse<null>> => {
    return apiService.post('/auth/logout')
  },

  // 获取用户信息
  getProfile: (): Promise<ApiResponse<UserInfo>> => {
    return apiService.get('/users/profile')
  },

  // 更新个人资料
  updateProfile: (data: Partial<UserProfile>): Promise<ApiResponse<UserInfo>> => {
    return apiService.put('/users/profile', data)
  },

  // 获取用户统计
  getStatistics: (): Promise<ApiResponse<PredictionStats>> => {
    return apiService.get('/users/statistics')
  },
}
