import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { LotteryResult } from '@/types'
import { STATUS_CODES } from '@/constants'
import { lotteryApi } from '@/services'

export const useLotteryStore = defineStore(
  'lottery',
  () => {
    const latestResults = ref<LotteryResult[]>([])
    const historicalResults = ref<LotteryResult[]>([])
    const loading = ref(false)

    const loadLatestResults = async (limit = 10) => {
      loading.value = true
      try {
        const response = await lotteryApi.getLatestResults(limit)
        if (response.code === STATUS_CODES.SUCCESS) {
          latestResults.value = response.data.results
        }
      } finally {
        loading.value = false
      }
    }

    const loadHistoricalResults = async (params: { limit?: number; offset?: number } = {}) => {
      loading.value = true
      try {
        const response = await lotteryApi.getHistoricalResults(params)
        if (response.code === STATUS_CODES.SUCCESS) {
          if ((params.offset ?? 0) === 0) {
            historicalResults.value = response.data.results
          } else {
            historicalResults.value = [...historicalResults.value, ...response.data.results]
          }
        }
      } finally {
        loading.value = false
      }
    }

    const fetchResultByIssue = async (issue: string) => {
      const response = await lotteryApi.getResultByIssue(issue)
      if (response.code === STATUS_CODES.SUCCESS) {
        return response.data
      }
      return null
    }

    return {
      latestResults,
      historicalResults,
      loading,
      loadLatestResults,
      loadHistoricalResults,
      fetchResultByIssue,
    }
  },
  {
    persist: {
      key: 'lottery-store',
      storage: {
        getItem: (key: string) => uni.getStorageSync(key),
        setItem: (key: string, value: string) => uni.setStorageSync(key, value),
      },
      paths: ['latestResults'],
    },
  }
)
