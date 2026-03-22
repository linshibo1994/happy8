import { apiService } from './api'
import { userApi } from './user'
import { membershipApi } from './membership'
import { algorithmApi } from './algorithm'
import { predictionApi } from './prediction'
import { lotteryApi } from './lottery'
import { paymentApi } from './payment'

export { apiService, userApi, membershipApi, algorithmApi, predictionApi, lotteryApi, paymentApi }

// 统一导出，便于使用
export const api = {
  user: userApi,
  membership: membershipApi,
  algorithm: algorithmApi,
  prediction: predictionApi,
  lottery: lotteryApi,
  payment: paymentApi,
}
