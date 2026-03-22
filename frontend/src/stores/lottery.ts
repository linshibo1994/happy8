import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ApiResponse } from '@/types'
import type { LotteryResult, HistoryItem, Statistics } from '@/types/lottery'
import { getLatestData, getHistoryData, getStatistics, refreshData, appendRecentData } from '@/api/lottery'

export const useLotteryStore = defineStore('lottery', () => {
  // 最新开奖结果
  const latestResult = ref<LotteryResult | null>(null)

  // 历史数据
  const historyData = ref<HistoryItem[]>([])

  // 统计数据
  const statistics = ref<Statistics | null>(null)

  // 加载状态
  const loading = ref(false)

  // 总期数
  const totalPeriods = ref(0)

  // 加载最新数据
  const fetchLatest = async (silent = false) => {
    if (!silent) {
      loading.value = true
    }
    try {
      const res = await getLatestData()
      if (res.success && res.data) {
        latestResult.value = res.data
      }
    } catch (error) {
      console.error('获取最新数据失败:', error)
    } finally {
      if (!silent) {
        loading.value = false
      }
    }
  }

  // 加载历史数据
  const fetchHistory = async (periods = 100, silent = false) => {
    if (!silent) {
      loading.value = true
    }
    try {
      const res = await getHistoryData({ periods })
      if (res.success && res.data) {
        // 后端直接返回数组，或者 { data: [...], total: number }
        if (Array.isArray(res.data)) {
          historyData.value = res.data
          totalPeriods.value = res.data.length
        } else {
          historyData.value = res.data.data || []
          totalPeriods.value = res.data.total || 0
        }
      }
    } catch (error) {
      console.error('获取历史数据失败:', error)
    } finally {
      if (!silent) {
        loading.value = false
      }
    }
  }

  // 加载统计数据
  const fetchStatistics = async (silent = false) => {
    if (!silent) {
      loading.value = true
    }
    try {
      const res = await getStatistics()
      if (res.success && res.data) {
        statistics.value = res.data
      }
    } catch (error) {
      console.error('获取统计数据失败:', error)
    } finally {
      if (!silent) {
        loading.value = false
      }
    }
  }

  // 刷新数据
  const refresh = async (): Promise<ApiResponse<null>> => {
    loading.value = true
    try {
      const res = await refreshData()
      if (res.success) {
        // 刷新成功后重新获取数据
        await fetchLatest()
        await fetchHistory()
      }
      return res
    } catch (error) {
      console.error('刷新数据失败:', error)
      return {
        success: false,
        message: '刷新数据失败，请稍后重试',
        data: null
      }
    } finally {
      loading.value = false
    }
  }

  // 追加最近N期数据
  const appendRecent = async (count = 5): Promise<ApiResponse<null>> => {
    loading.value = true
    try {
      const res = await appendRecentData(count)
      if (res.success) {
        await fetchLatest()
        await fetchHistory()
      }
      return res
    } catch (error) {
      console.error('追加最近数据失败:', error)
      return {
        success: false,
        message: '追加最近数据失败，请稍后重试',
        data: null
      }
    } finally {
      loading.value = false
    }
  }

  return {
    latestResult,
    historyData,
    statistics,
    loading,
    totalPeriods,
    fetchLatest,
    fetchHistory,
    fetchStatistics,
    refresh,
    appendRecent
  }
})
