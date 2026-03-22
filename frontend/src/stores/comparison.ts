import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  BatchCompareParams,
  BatchCompareResult,
  CompareRoundResult,
  AlgorithmOption
} from '@/types/comparison'
import { getAvailableIssues, createBatchCompareStream } from '@/api/comparison'

// 算法选项列表
const ALGORITHM_OPTIONS: AlgorithmOption[] = [
  // 马尔可夫链
  { label: '马尔可夫链(一阶)', value: 'markov', category: '马尔可夫' },
  { label: '马尔可夫链(二阶)', value: 'markov_2nd', category: '马尔可夫' },
  { label: '马尔可夫链(三阶)', value: 'markov_3rd', category: '马尔可夫' },
  { label: '自适应马尔可夫链', value: 'adaptive_markov', category: '马尔可夫' },
  // 统计分析
  { label: '统计分析', value: 'stats', category: '统计分析' },
  { label: '概率分析', value: 'probability', category: '统计分析' },
  { label: '决策树', value: 'decision_tree', category: '统计分析' },
  { label: '模式识别', value: 'patterns', category: '统计分析' },
  { label: '频率分析', value: 'frequency', category: '统计分析' },
  { label: '冷热号分析', value: 'hot_cold', category: '统计分析' },
  { label: '共识减半', value: 'consensus_halving', category: '统计分析' },
  // 机器学习
  { label: '集成学习', value: 'ensemble', category: '机器学习' },
  { label: '自适应集成', value: 'adaptive_ensemble', category: '机器学习' },
  { label: '聚类分析', value: 'clustering', category: '机器学习' },
  { label: '蒙特卡洛模拟', value: 'monte_carlo', category: '机器学习' },
  // 深度学习
  { label: 'LSTM 神经网络', value: 'lstm', category: '深度学习' },
  { label: 'Transformer', value: 'transformer', category: '深度学习' },
  { label: '图神经网络', value: 'graph_nn', category: '深度学习' },
  // 贝叶斯
  { label: '动态贝叶斯网络', value: 'dynamic_bayes', category: '贝叶斯' },
  // 智能预测
  { label: 'Super 预测器', value: 'super', category: '智能预测' },
  { label: 'Super Predictor', value: 'super_predictor', category: '智能预测' },
  { label: '混合分析', value: 'hybrid', category: '智能预测' },
  { label: '混合分析 V2', value: 'hybrid_v2', category: '智能预测' },
]

export const useComparisonStore = defineStore('comparison', () => {
  // 可用期号列表
  const availableIssues = ref<string[]>([])
  const issuesLoading = ref(false)

  // 对比参数
  const params = ref<BatchCompareParams>({
    target_issue: '',
    method_name: 'frequency',
    periods_mode: 'fixed',
    periods_value: 100,
    comparison_times: 50
  })

  // 对比状态
  const isComparing = ref(false)
  const progress = ref(0)
  const currentRound = ref(0)
  const roundResults = ref<CompareRoundResult[]>([])
  const finalResult = ref<BatchCompareResult | null>(null)
  const error = ref<string | null>(null)

  // SSE 连接
  let eventSource: EventSource | null = null

  // 算法列表
  const algorithmOptions = ALGORITHM_OPTIONS

  // 算法分类
  const algorithmCategories = computed(() => {
    const categories: Record<string, AlgorithmOption[]> = {}
    for (const opt of ALGORITHM_OPTIONS) {
      if (!categories[opt.category]) {
        categories[opt.category] = []
      }
      categories[opt.category].push(opt)
    }
    return categories
  })

  // 中奖轮次
  const winningRounds = computed(() =>
    roundResults.value.filter(r => r.prize_level > 0 && r.prize_level <= 6)
  )

  // 中奖率
  const winRate = computed(() => {
    const total = roundResults.value.length
    if (total === 0) return 0
    return (winningRounds.value.length / total) * 100
  })

  // 获取可用期号
  const fetchIssues = async (limit = 100) => {
    issuesLoading.value = true
    try {
      const res = await getAvailableIssues(limit)
      if (res.success && res.data) {
        availableIssues.value = res.data.issues
        // 默认选中最新期号
        if (res.data.issues.length > 0 && !params.value.target_issue) {
          params.value.target_issue = res.data.issues[0]
        }
      }
    } catch (err) {
      console.error('获取期号列表失败:', err)
    } finally {
      issuesLoading.value = false
    }
  }

  // 参数校验
  const validateParams = (): string | null => {
    const p = params.value
    if (!p.target_issue) return '请选择目标期号'
    const times = Number(p.comparison_times)
    if (isNaN(times) || times < 1 || times > 500) return '对比次数需在 1-500 之间'
    if (p.periods_mode === 'fixed') {
      const periods = Number(p.periods_value)
      if (isNaN(periods) || periods < 20 || periods > 1000) return '分析期数需在 20-1000 之间'
    }
    return null
  }

  // 开始 SSE 流式对比
  const startCompare = () => {
    if (isComparing.value) return

    // 参数校验
    const validationError = validateParams()
    if (validationError) {
      error.value = validationError
      return
    }

    // 防御性清理残留连接
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }

    // 重置状态
    isComparing.value = true
    progress.value = 0
    currentRound.value = 0
    roundResults.value = []
    finalResult.value = null
    error.value = null

    // 创建 SSE 连接
    eventSource = createBatchCompareStream(params.value)

    eventSource.addEventListener('progress', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data)
        currentRound.value = data.round
        progress.value = data.percentage
      } catch (err) {
        console.error('解析进度数据失败:', err)
      }
    })

    eventSource.addEventListener('result', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data)
        roundResults.value.push({
          round: data.round,
          analysis_periods: data.analysis_periods,
          predicted_reds: data.predicted_reds,
          predicted_blue: data.predicted_blue,
          prize_level: data.prize_level,
          prize_name: data.prize_name,
          red_matches: data.red_matches,
          blue_match: data.blue_match,
          success: data.predicted_reds !== null
        })
      } catch (err) {
        console.error('解析结果数据失败:', err)
      }
    })

    eventSource.addEventListener('complete', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data)
        if (data.success && data.summary) {
          finalResult.value = data.summary
        } else if (!data.success && data.message) {
          error.value = data.message
        }
      } catch (err) {
        console.error('解析完成数据失败:', err)
      }
      stopCompare()
    })

    // onerror 处理：EventSource 会自动重连，必须立即 close 防止重连风暴
    eventSource.onerror = () => {
      // 立即关闭，阻止自动重连
      if (eventSource) {
        eventSource.close()
        eventSource = null
      }
      if (isComparing.value) {
        error.value = 'SSE 连接断开'
        isComparing.value = false
        if (progress.value === 0) {
          progress.value = 0
        }
      }
    }
  }

  // 停止对比
  const stopCompare = () => {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
    isComparing.value = false
    if (roundResults.value.length > 0) {
      progress.value = 100
    }
  }

  // 重置
  const reset = () => {
    stopCompare()
    progress.value = 0
    currentRound.value = 0
    roundResults.value = []
    finalResult.value = null
    error.value = null
  }

  // 导出 JSON
  const exportJSON = () => {
    if (!finalResult.value && roundResults.value.length === 0) return

    const exportData = finalResult.value || {
      params: params.value,
      results: roundResults.value,
      exported_time: new Date().toISOString()
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `comparison_${params.value.target_issue}_${params.value.method_name}_${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return {
    availableIssues,
    issuesLoading,
    params,
    isComparing,
    progress,
    currentRound,
    roundResults,
    finalResult,
    error,
    algorithmOptions,
    algorithmCategories,
    winningRounds,
    winRate,
    fetchIssues,
    validateParams,
    startCompare,
    stopCompare,
    reset,
    exportJSON
  }
})
