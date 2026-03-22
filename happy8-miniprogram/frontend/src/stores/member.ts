import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { MembershipStatus, MembershipPlan, MembershipOrder, MembershipLevel } from '@/types'
import { MEMBERSHIP, STATUS_CODES } from '@/constants'
import { membershipApi } from '@/services/membership'
import { paymentApi } from '@/services/payment'

const LEVEL_PRIORITY: Record<MembershipLevel, number> = {
  free: 0,
  vip: 1,
  premium: 2,
}

const STATUS_TEXT: Record<string, string> = {
  pending: '待支付',
  paid: '已支付',
  cancelled: '已取消',
  refunded: '已退款',
  expired: '已过期',
}

const STATUS_COLOR: Record<string, string> = {
  pending: '#ff9800',
  paid: '#4caf50',
  cancelled: '#999999',
  refunded: '#2196f3',
  expired: '#f44336',
}

const ensureArray = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === 'string')
  }
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value)
      return Array.isArray(parsed)
        ? parsed.filter((item: unknown): item is string => typeof item === 'string')
        : []
    } catch (_) {
      return value ? [value] : []
    }
  }
  return []
}

export const useMembershipStore = defineStore(
  'membership',
  () => {
    const membershipStatus = ref<MembershipStatus | null>(null)
    const membershipPlans = ref<MembershipPlan[]>([])
    const orders = ref<MembershipOrder[]>([])
    const loading = ref(false)

    const currentLevel = computed<MembershipLevel>(
      () => membershipStatus.value?.membership.level ?? 'free'
    )
    const isActive = computed(() => membershipStatus.value?.membership.is_valid ?? false)
    const remainingDays = computed(() => membershipStatus.value?.membership.days_remaining ?? 0)

    const dailyPredictionLimit = computed(() => {
      const limit = membershipStatus.value?.limits.daily_predictions_limit
      if (limit === null || limit === undefined) {
        return MEMBERSHIP.DAILY_LIMITS[currentLevel.value] ?? 0
      }
      return limit
    })

    const usedPredictions = computed(() => {
      if (membershipStatus.value?.limits.daily_predictions_used !== undefined) {
        return membershipStatus.value.limits.daily_predictions_used
      }
      return membershipStatus.value?.membership.predictions_today ?? 0
    })

    const remainingPredictions = computed(() => {
      const limit = dailyPredictionLimit.value
      if (limit === null || limit < 0) return -1
      return Math.max(0, limit - usedPredictions.value)
    })

    const isExpiringSoon = computed(() => remainingDays.value > 0 && remainingDays.value <= 7)

    const normalizePlan = (plan: MembershipPlan): MembershipPlan => ({
      ...plan,
      features: ensureArray(plan.features),
      available_algorithms: plan.available_algorithms
        ? ensureArray(plan.available_algorithms)
        : null,
    })

    const loadMembershipStatus = async () => {
      try {
        loading.value = true
        const response = await membershipApi.getMembershipStatus()
        if (response.code === STATUS_CODES.SUCCESS) {
          membershipStatus.value = response.data
        }
      } catch (error) {
        console.error('加载会员状态失败:', error)
      } finally {
        loading.value = false
      }
    }

    const loadMembershipPlans = async () => {
      try {
        const response = await membershipApi.getMembershipPlans()
        if (response.code === STATUS_CODES.SUCCESS) {
          membershipPlans.value = response.data.map(normalizePlan)
        }
      } catch (error) {
        console.error('加载会员套餐失败:', error)
      }
    }

    const loadOrders = async (limit = 20, offset = 0) => {
      try {
        const response = await membershipApi.getOrders({ limit, offset })
        if (response.code === STATUS_CODES.SUCCESS) {
          const nextOrders = response.data.orders
          if (offset === 0) {
            orders.value = nextOrders
          } else {
            orders.value = [...orders.value, ...nextOrders]
          }
        }
      } catch (error) {
        console.error('加载订单失败:', error)
      }
    }

    const createOrder = async (planId: number) => {
      try {
        loading.value = true
        const response = await membershipApi.createOrder({ plan_id: planId })
        if (response.code === STATUS_CODES.SUCCESS) {
          orders.value = [response.data, ...orders.value]
          return response.data
        }
        throw new Error(response.message)
      } catch (error) {
        uni.showToast({ title: '创建订单失败', icon: 'error' })
        throw error
      } finally {
        loading.value = false
      }
    }

    const payOrder = async (orderNo: string) => {
      const paymentResponse = await paymentApi.createPayment(orderNo)
      if (paymentResponse.code !== STATUS_CODES.SUCCESS || !paymentResponse.data.payment_params) {
        throw new Error(paymentResponse.message)
      }

      const paymentParams = paymentResponse.data.payment_params
        return new Promise<void>((resolve, reject) => {
          uni.requestPayment({
            provider: 'wxpay',
            orderInfo: '',
            ...paymentParams,
            success: async () => {
              uni.showToast({ title: '支付成功', icon: 'success' })
              try {
                await membershipApi.upgradeMembership(orderNo)
                await loadMembershipStatus()
                await loadOrders()
              } finally {
                resolve()
              }
            },
            fail: (err: unknown) => {
              console.error('支付失败:', err)
              uni.showToast({ title: '支付失败', icon: 'error' })
              reject(err)
            }
          })
        })
    }

    const cancelOrder = async (orderNo: string, reason?: string) => {
      try {
        const response = await paymentApi.cancelPayment({ order_no: orderNo, reason })
        if (response.code === STATUS_CODES.SUCCESS) {
          orders.value = orders.value.map(order =>
            order.order_no === orderNo ? { ...order, status: 'cancelled' } : order
          )
        }
        return response
      } catch (error) {
        console.error('取消订单失败:', error)
        throw error
      }
    }

    const checkPermission = (requiredLevel: MembershipLevel): boolean => {
      const current = LEVEL_PRIORITY[currentLevel.value]
      const required = LEVEL_PRIORITY[requiredLevel]
      return (isActive.value || requiredLevel === 'free') && current >= required
    }

    const canUsePrediction = (): boolean => {
      if (!isActive.value && currentLevel.value !== 'free') {
        return false
      }
      if (remainingPredictions.value === -1) return true
      return remainingPredictions.value > 0
    }

    const updateDailyPredictionCount = () => {
      if (!membershipStatus.value) return
      if (membershipStatus.value.limits.daily_predictions_used !== undefined) {
        membershipStatus.value.limits.daily_predictions_used += 1
      }
      membershipStatus.value.membership.predictions_today += 1
      membershipStatus.value.membership.predictions_total += 1
    }

    const getLevelName = (level: MembershipLevel): string => {
      return MEMBERSHIP.LEVEL_NAMES[level] || level
    }

    const getLevelColor = (level: MembershipLevel): string => {
      return MEMBERSHIP.LEVEL_COLORS[level] || '#999999'
    }

    const getLevelFeatures = (level: MembershipLevel): string[] => {
      return MEMBERSHIP.FEATURES[level] || []
    }

    const formatPrice = (price: number): string => `¥${(price / 100).toFixed(2)}`

    const formatDuration = (durationDays: number): string => {
      if (durationDays >= 30) {
        const months = Math.round(durationDays / 30)
        return `${months}个月`
      }
      return `${durationDays}天`
    }

    const getOrderStatusText = (status: string): string => STATUS_TEXT[status] || status
    const getOrderStatusColor = (status: string): string => STATUS_COLOR[status] || '#999999'

    const recommendedPlan = computed(() => {
      return (
        membershipPlans.value.find(plan => plan.level !== 'free' && plan.is_active) ??
        membershipPlans.value[0]
      )
    })

    const upgradeOptions = computed(() => {
      const current = LEVEL_PRIORITY[currentLevel.value]
      return membershipPlans.value.filter(plan => LEVEL_PRIORITY[plan.level] > current)
    })

  const loadAll = async () => {
    await Promise.all([loadMembershipStatus(), loadMembershipPlans()])
  }

  const resetStore = () => {
    membershipStatus.value = null
    membershipPlans.value = []
    orders.value = []
    loading.value = false
    try {
      uni.removeStorageSync('membership-store')
    } catch (error) {
      console.error('清理会员缓存失败:', error)
    }
  }

  return {
    membershipStatus,
    membershipPlans,
    orders,
    loading,

    currentLevel,
    isActive,
    remainingDays,
    dailyPredictionLimit,
    remainingPredictions,
    isExpiringSoon,
    recommendedPlan,
    upgradeOptions,

    loadAll,
    loadMembershipStatus,
    loadMembershipPlans,
    loadOrders,
    createOrder,
    payOrder,
    cancelOrder,
    checkPermission,
    canUsePrediction,
    updateDailyPredictionCount,

    getLevelName,
    getLevelColor,
    getLevelFeatures,
    formatPrice,
    formatDuration,
    getOrderStatusText,
    getOrderStatusColor,

    resetStore,
  }
},
{
  persist: {
    key: 'membership-store',
    storage: {
      getItem: (key: string) => uni.getStorageSync(key),
      setItem: (key: string, value: string) => uni.setStorageSync(key, value),
    },
    paths: ['membershipStatus', 'membershipPlans'],
  },
}
)
