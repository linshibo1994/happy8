import { apiService } from './api'
import type {
  MembershipStatus,
  MembershipPlan,
  MembershipOrder,
  OrderListResult,
  MembershipSummary,
  CreateOrderRequest,
  ApiResponse,
} from '@/types'

export const membershipApi = {
  // 获取会员状态
  getMembershipStatus: (): Promise<ApiResponse<MembershipStatus>> => {
    return apiService.get('/memberships/status')
  },

  // 获取会员套餐列表
  getMembershipPlans: (): Promise<ApiResponse<MembershipPlan[]>> => {
    return apiService.get('/memberships/plans')
  },

  // 创建订单
  createOrder: (data: CreateOrderRequest): Promise<ApiResponse<MembershipOrder>> => {
    return apiService.post('/memberships/orders', data)
  },

  // 获取订单列表
  getOrders: (params: {
    limit?: number
    offset?: number
  }): Promise<ApiResponse<OrderListResult>> => {
    return apiService.get('/memberships/orders', params)
  },

  // 获取订单详情
  getOrderDetail: (orderNo: string): Promise<ApiResponse<MembershipOrder>> => {
    return apiService.get(`/memberships/orders/${orderNo}`)
  },

  // 升级会员（支付成功后调用）
  upgradeMembership: (orderNo: string): Promise<ApiResponse<MembershipSummary>> => {
    return apiService.post('/memberships/upgrade', { order_no: orderNo })
  },

  // 权限检查
  getPermissions: (): Promise<ApiResponse<Record<string, boolean>>> => {
    return apiService.get('/memberships/permissions')
  },
}
