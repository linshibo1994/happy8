import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type {
  AlgorithmInfo,
  PredictionResult,
  PredictionHistory,
  LotteryResult,
  PredictionStats,
  PredictionRequest,
} from '@/types'
import { STATUS_CODES } from '@/constants'
import { algorithmApi, predictionApi, lotteryApi } from '@/services'

export const usePredictStore = defineStore('predict', () => {
  const availableAlgorithms = ref<AlgorithmInfo[]>([])
  const selectedAlgorithm = ref<AlgorithmInfo | null>(null)
  const predictionParams = ref({
    periods: 30,
    count: 5,
    target_issue: '',
  })
  const currentPrediction = ref<PredictionResult | null>(null)
  const predictionHistory = ref<PredictionHistory[]>([])
  const historyMeta = ref({ total: 0, limit: 20, offset: 0 })
  const latestResults = ref<LotteryResult[]>([])
  const stats = ref<PredictionStats | null>(null)
  const loading = ref(false)
  const predicting = ref(false)

  const hasSelectedAlgorithm = computed(() => !!selectedAlgorithm.value)
  const canPredict = computed(
    () => hasSelectedAlgorithm.value && !!predictionParams.value.target_issue && !predicting.value
  )

  const algorithmsByLevel = computed(() => {
    return availableAlgorithms.value.reduce<Record<'free' | 'vip' | 'premium', AlgorithmInfo[]>>(
      (acc, algo) => {
        acc[algo.required_level].push(algo)
        return acc
      },
      { free: [], vip: [], premium: [] }
    )
  })

  const sortedPredictionHistory = computed(() => {
    return [...predictionHistory.value].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )
  })

  const loadAvailableAlgorithms = async () => {
    try {
      loading.value = true
      const response = await algorithmApi.getAvailableAlgorithms()
      if (response.code === STATUS_CODES.SUCCESS) {
        availableAlgorithms.value = response.data
        if (!selectedAlgorithm.value && response.data.length > 0) {
          selectedAlgorithm.value = response.data[0]
        }
      }
    } catch (error) {
      console.error('加载算法列表失败:', error)
      uni.showToast({ title: '加载算法失败', icon: 'error' })
    } finally {
      loading.value = false
    }
  }

  const loadPredictionHistory = async (limit = 20, offset = 0) => {
    try {
      const response = await predictionApi.getPredictionHistory({ limit, offset })
      if (response.code === STATUS_CODES.SUCCESS) {
        historyMeta.value = {
          total: response.data.total,
          limit: response.data.limit,
          offset: response.data.offset,
        }
        if (offset === 0) {
          predictionHistory.value = response.data.history
        } else {
          predictionHistory.value = [...predictionHistory.value, ...response.data.history]
        }
      }
    } catch (error) {
      console.error('加载预测历史失败:', error)
    }
  }

  const loadLatestLotteryResults = async (limit = 10) => {
    try {
      const response = await lotteryApi.getLatestResults(limit)
      if (response.code === STATUS_CODES.SUCCESS) {
        latestResults.value = response.data.results
        if (!predictionParams.value.target_issue && response.data.results.length > 0) {
          predictionParams.value.target_issue = computeNextIssue(response.data.results[0].issue)
        }
      }
    } catch (error) {
      console.error('加载开奖结果失败:', error)
    }
  }

  const loadPredictionStats = async () => {
    try {
      const response = await predictionApi.getPredictionStats()
      if (response.code === STATUS_CODES.SUCCESS) {
        stats.value = response.data
      }
    } catch (error) {
      console.error('加载统计数据失败:', error)
    }
  }

  const setSelectedAlgorithm = (algorithm: AlgorithmInfo) => {
    selectedAlgorithm.value = algorithm
  }

  const updatePredictionParams = (params: Partial<typeof predictionParams.value>) => {
    predictionParams.value = { ...predictionParams.value, ...params }
  }

  const resetPredictionParams = () => {
    predictionParams.value = {
      periods: 30,
      count: 5,
      target_issue:
        latestResults.value.length > 0 ? computeNextIssue(latestResults.value[0].issue) : '',
    }
  }

  const generatePrediction = async () => {
    if (!canPredict.value || !selectedAlgorithm.value) {
      uni.showToast({ title: '请选择可用算法', icon: 'none' })
      return
    }

    const payload: PredictionRequest = {
      algorithm: selectedAlgorithm.value.algorithm_name,
      target_issue: predictionParams.value.target_issue,
      periods: predictionParams.value.periods,
      count: predictionParams.value.count,
      params: selectedAlgorithm.value.default_params || {},
    }

    try {
      predicting.value = true
      const response = await predictionApi.generatePrediction(payload)
      if (response.code === STATUS_CODES.SUCCESS) {
        currentPrediction.value = response.data
        uni.showToast({ title: '预测完成', icon: 'success' })
        await loadPredictionHistory(historyMeta.value.limit, 0)
        return response.data
      }
      throw new Error(response.message)
    } catch (error) {
      console.error('生成预测失败:', error)
      uni.showToast({ title: '预测失败', icon: 'error' })
      throw error
    } finally {
      predicting.value = false
    }
  }

  const clearCurrentPrediction = () => {
    currentPrediction.value = null
  }

  const formatPredictionNumbers = (numbers: number[]): string =>
    numbers
      .slice()
      .sort((a, b) => a - b)
      .join(', ')
  const formatConfidenceScore = (score?: number | null): string => {
    if (score === undefined || score === null) return '--'
    return `${(score * 100).toFixed(2)}%`
  }

  const getNumberZone = (number: number): string => {
    if (number >= 1 && number <= 20) return 'zone-1'
    if (number >= 21 && number <= 40) return 'zone-2'
    if (number >= 41 && number <= 60) return 'zone-3'
    if (number >= 61 && number <= 80) return 'zone-4'
    return ''
  }

  const getZoneColor = (zone: string): string => {
    const map: Record<string, string> = {
      'zone-1': '#f44336',
      'zone-2': '#ff9800',
      'zone-3': '#4caf50',
      'zone-4': '#2196f3',
    }
    return map[zone] || '#999999'
  }

  const calculateHitRate = (predicted: number[], actual: number[] | undefined | null): number => {
    if (!actual || actual.length === 0) return 0
    const hits = predicted.filter(num => actual.includes(num)).length
    return Number(((hits / predicted.length) * 100).toFixed(2))
  }

  const computeNextIssue = (issue: string): string => {
    const next = String(Number(issue) + 1)
    return next.padStart(issue.length, '0')
  }

  const loadAll = async () => {
    await Promise.all([
      loadAvailableAlgorithms(),
      loadPredictionHistory(),
      loadLatestLotteryResults(),
      loadPredictionStats(),
    ])
  }

  return {
    availableAlgorithms,
    selectedAlgorithm,
    predictionParams,
    currentPrediction,
    predictionHistory,
    historyMeta,
    latestResults,
    stats,
    loading,
    predicting,

    hasSelectedAlgorithm,
    canPredict,
    algorithmsByLevel,
    sortedPredictionHistory,

    loadAll,
    loadAvailableAlgorithms,
    loadPredictionHistory,
    loadLatestLotteryResults,
    loadPredictionStats,
    setSelectedAlgorithm,
    updatePredictionParams,
    resetPredictionParams,
    generatePrediction,
    clearCurrentPrediction,

    formatPredictionNumbers,
    formatConfidenceScore,
    getNumberZone,
    getZoneColor,
    calculateHitRate,
  }
})
