import { defineStore } from 'pinia'

import type { MembershipStatus } from '@/types'

interface MembershipState {
  status: MembershipStatus
}

export const useMembershipStore = defineStore('membership', {
  state: (): MembershipState => ({
    status: {
      level: 'premium',
      levelName: '青铜会员',
      remainingPredictions: 18,
      dailyLimit: 30,
      benefits: ['单算法预测', '批量预测', '历史复盘'],
    },
  }),
  getters: {
    hasQuota: (state) => state.status.remainingPredictions > 0,
  },
  actions: {
    consumePredictionQuota(count = 1) {
      this.status.remainingPredictions = Math.max(0, this.status.remainingPredictions - count)
    },
  },
})
