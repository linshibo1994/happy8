import { apiService } from './api'
import type { ApiResponse, PaymentResponse, PaymentStatus } from '@/types'

type CancelPayload = {
  order_no: string
  reason?: string
}

export const paymentApi = {
  createPayment: (orderNo: string): Promise<ApiResponse<PaymentResponse>> => {
    return apiService.post('/payments/create', {
      order_no: orderNo,
      payment_method: 'wechat_pay',
    })
  },

  getPaymentStatus: (orderNo: string): Promise<ApiResponse<PaymentStatus>> => {
    return apiService.get(`/payments/status/${orderNo}`)
  },

  cancelPayment: (payload: CancelPayload): Promise<ApiResponse<null>> => {
    return apiService.post('/payments/cancel', payload)
  },
}
