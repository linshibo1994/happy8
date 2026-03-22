import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { PredictMethod, PredictResult, PredictParams, ProgressMessage } from '@/types/predict'
import { getMethods, createPredictStream } from '@/api/predict'
import { PREDICT_STEPS } from '@/utils/constants'

export const usePredictStore = defineStore('predict', () => {
  // 预测方法列表
  const methods = ref<PredictMethod[]>([])

  // 当前选中的方法
  const selectedMethod = ref<string>('super')

  // 预测参数
  const params = ref<PredictParams>({
    method: 'super',
    periods: 100,
    count: 1,
    duplex: false,
    redCount: 8,
    blueCount: 2,
    useGpu: true,
    parallel: true,
    explain: true
  })

  // 预测状态
  const isPredicting = ref(false)

  // 当前步骤 (0-6)
  const currentStep = ref(0)

  // 进度百分比 (0-100)
  const progress = ref(0)

  // 日志消息
  const logs = ref<string[]>([])

  // 预测结果
  const results = ref<PredictResult[]>([])

  // 错误信息
  const error = ref<string | null>(null)

  // 统计信息
  const predictionCount = ref(0)
  const confidenceTotal = ref(0)
  const confidenceSamples = ref(0)

  // SSE 连接
  let eventSource: EventSource | null = null

  // 计算属性
  const currentStepInfo = computed(() => PREDICT_STEPS[currentStep.value])

  const stepLabels = computed(() => PREDICT_STEPS.map(s => s.label))
  const averageConfidence = computed(() => {
    if (confidenceSamples.value <= 0) return 0
    return confidenceTotal.value / confidenceSamples.value
  })

  const loadPredictStats = () => {
    try {
      predictionCount.value = Number(localStorage.getItem('predict_count') || 0)
      confidenceTotal.value = Number(localStorage.getItem('predict_confidence_total') || 0)
      confidenceSamples.value = Number(localStorage.getItem('predict_confidence_samples') || 0)
    } catch {
      predictionCount.value = 0
      confidenceTotal.value = 0
      confidenceSamples.value = 0
    }
  }

  const savePredictStats = () => {
    try {
      localStorage.setItem('predict_count', String(predictionCount.value))
      localStorage.setItem('predict_confidence_total', String(confidenceTotal.value))
      localStorage.setItem('predict_confidence_samples', String(confidenceSamples.value))
    } catch {
      // 忽略本地存储异常
    }
  }

  // 加载方法列表
  const fetchMethods = async () => {
    try {
      const res = await getMethods()
      if (res.success) {
        methods.value = res.data
      }
    } catch (err) {
      console.error('获取方法列表失败:', err)
    }
  }

  const combination = (n: number, k: number): number => {
    if (k < 0 || n < k) return 0
    if (k === 0 || n === k) return 1
    const kk = Math.min(k, n - k)
    let result = 1
    for (let i = 1; i <= kk; i++) {
      result = (result * (n - kk + i)) / i
    }
    return Math.round(result)
  }

  const normalizeResultItem = (item: Record<string, unknown>, method: string, fallbackConfidence: number): PredictResult => {
    const redBalls = Array.isArray(item.red_balls)
      ? item.red_balls.map((value) => Number(value)).filter((num) => Number.isFinite(num))
      : []

    const blueBalls = Array.isArray(item.blue_balls)
      ? item.blue_balls.map((value) => Number(value)).filter((num) => Number.isFinite(num))
      : []

    const blueBall = item.blue_ball !== undefined ? Number(item.blue_ball) : (blueBalls[0] ?? undefined)
    const duplex = Boolean(item.duplex) || blueBalls.length > 1
    const redCount = Number(item.red_count || redBalls.length || 6)
    const blueCount = Number(item.blue_count || (blueBalls.length > 0 ? blueBalls.length : (blueBall ? 1 : 0)))
    const stakeCount = Number(item.stake_count || (duplex ? combination(redCount, 6) * Math.max(blueCount, 1) : 1))
    const totalCost = Number(item.total_cost || stakeCount * 2)
    const confidence = Number(item.confidence ?? fallbackConfidence ?? 0)

    return {
      method: String(item.method || method),
      red_balls: redBalls,
      blue_ball: Number.isFinite(blueBall) ? blueBall : undefined,
      blue_balls: blueBalls.length > 0 ? blueBalls : undefined,
      duplex,
      red_count: redCount,
      blue_count: blueCount,
      stake_count: stakeCount,
      total_cost: totalCost,
      confidence: Number.isFinite(confidence) ? confidence : 0
    }
  }

  const normalizeResults = (data: unknown, method: string, fallbackConfidence: number): PredictResult[] => {
    if (Array.isArray(data)) {
      return data
        .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
        .map((item) => normalizeResultItem(item, method, fallbackConfidence))
    }

    if (data && typeof data === 'object') {
      const payload = data as Record<string, unknown>
      const predictions = payload.predictions
      if (Array.isArray(predictions)) {
        return predictions
          .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
          .map((item) => normalizeResultItem(item, method, fallbackConfidence))
      }
      return [normalizeResultItem(payload, method, fallbackConfidence)]
    }

    return []
  }

  const updateStatsFromResults = (items: PredictResult[]) => {
    if (items.length === 0) return
    predictionCount.value += items.length
    items.forEach((item) => {
      if (typeof item.confidence === 'number' && Number.isFinite(item.confidence)) {
        confidenceTotal.value += item.confidence
        confidenceSamples.value += 1
      }
    })
    savePredictStats()
  }

  const closeEventSource = () => {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
  }

  // 开始预测
  const startPredict = async () => {
    if (isPredicting.value) {
      return
    }

    isPredicting.value = true
    currentStep.value = 0
    progress.value = 0
    logs.value = []
    results.value = []
    error.value = null

    closeEventSource()
    const streamParams: PredictParams = {
      ...params.value,
      method: selectedMethod.value
    }
    eventSource = createPredictStream(streamParams)

    const pushLog = (message: string) => {
      const line = `[${new Date().toLocaleTimeString()}] ${message}`
      if (logs.value[logs.value.length - 1] !== line) {
        logs.value.push(line)
      }
    }

    eventSource.addEventListener('progress', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data) as ProgressMessage
        handleProgress(data)
        if (data.details) {
          pushLog(data.details)
        }
      } catch (err) {
        console.error('解析进度事件失败:', err)
      }
    })

    eventSource.addEventListener('result', (e: MessageEvent) => {
      try {
        const payload = JSON.parse(e.data) as {
          success: boolean
          data?: unknown
          confidence?: number
          method?: string
          message?: string
        }
        if (payload.success && payload.data) {
          const normalized = normalizeResults(
            payload.data,
            payload.method || selectedMethod.value,
            payload.confidence || 0
          )
          results.value = normalized
          updateStatsFromResults(normalized)
          pushLog(payload.message || '预测完成')
        } else {
          error.value = payload.message || '预测失败'
          pushLog(error.value)
        }
      } catch (err) {
        console.error('解析结果事件失败:', err)
      }
    })

    eventSource.addEventListener('error_event', (e: MessageEvent) => {
      try {
        const payload = JSON.parse(e.data) as { message?: string }
        error.value = payload.message || '预测失败'
      } catch {
        error.value = '预测失败'
      }
      pushLog(error.value)
    })

    eventSource.addEventListener('complete', () => {
      isPredicting.value = false
      progress.value = 100
      closeEventSource()
    })

    eventSource.onerror = () => {
      if (isPredicting.value) {
        error.value = error.value || '预测连接中断'
        logs.value.push(`[${new Date().toLocaleTimeString()}] ${error.value}`)
        isPredicting.value = false
      }
      closeEventSource()
    }
  }

  // 处理进度消息
  const handleProgress = (message: ProgressMessage) => {
    if (message.type === 'progress') {
      progress.value = message.percentage
      if (message.step) {
        const stepIndex = PREDICT_STEPS.findIndex(s => s.key === message.step)
        if (stepIndex >= 0) {
          currentStep.value = stepIndex
        }
      }
    } else if (message.type === 'log') {
      logs.value.push(message.details)
    } else if (message.type === 'error') {
      error.value = message.details
    }
  }

  // 取消预测
  const cancelPredict = () => {
    closeEventSource()
    isPredicting.value = false
    logs.value.push(`[${new Date().toLocaleTimeString()}] 预测已取消`)
  }

  // 重置状态
  const reset = () => {
    closeEventSource()
    currentStep.value = 0
    progress.value = 0
    logs.value = []
    results.value = []
    error.value = null
  }

  loadPredictStats()

  return {
    methods,
    selectedMethod,
    params,
    isPredicting,
    currentStep,
    progress,
    logs,
    results,
    error,
    predictionCount,
    averageConfidence,
    currentStepInfo,
    stepLabels,
    fetchMethods,
    startPredict,
    handleProgress,
    cancelPredict,
    reset
  }
})
