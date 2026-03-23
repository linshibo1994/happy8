import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { PredictMethod, PredictResult, PredictParams, ProgressMessage } from '@/types/predict'
import { getMethods, createPredictStream } from '@/api/predict'
import { PREDICT_STEPS } from '@/utils/constants'

const toNumberList = (value: unknown): number[] => {
  if (!Array.isArray(value)) return []
  return value.map((item) => Number(item)).filter((num) => Number.isFinite(num))
}

export const usePredictStore = defineStore('predict', () => {
  const methods = ref<PredictMethod[]>([])
  const selectedMethod = ref<string>('super_predictor')

  const params = ref<PredictParams>({
    method: 'super_predictor',
    periods: 100,
    count: 20,
    useGpu: true,
    parallel: true,
    explain: true,
    compareIssue: '',
  })

  const isPredicting = ref(false)
  const currentStep = ref(0)
  const progress = ref(0)
  const logs = ref<string[]>([])
  const results = ref<PredictResult[]>([])
  const error = ref<string | null>(null)

  const predictionCount = ref(0)
  const confidenceTotal = ref(0)
  const confidenceSamples = ref(0)

  let eventSource: EventSource | null = null

  const currentStepInfo = computed(() => PREDICT_STEPS[currentStep.value])
  const stepLabels = computed(() => PREDICT_STEPS.map((step) => step.label))
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

  const fetchMethods = async () => {
    try {
      const res = await getMethods()
      if (res.success) {
        methods.value = res.data
        const availableIds = new Set(res.data.map((item) => item.id))
        if (!availableIds.has(selectedMethod.value) && res.data.length > 0) {
          selectedMethod.value = res.data[0].id
        }
      }
    } catch (err) {
      console.error('获取方法列表失败:', err)
    }
  }

  const normalizeResultItem = (item: Record<string, unknown>, method: string, fallbackConfidence: number): PredictResult => {
    const numbers = toNumberList(item.numbers) || toNumberList(item.predicted_numbers)
    const confidence = Number(item.confidence ?? item.confidence_score ?? fallbackConfidence ?? 0)
    // 优先从 comparison 嵌套对象中提取命中数据，兼容顶层字段
    const comparison = (item.comparison && typeof item.comparison === 'object') ? item.comparison as Record<string, unknown> : null
    const hitNumbers = toNumberList(comparison?.hit_numbers ?? item.hit_numbers)
    const hitCountRaw = Number(comparison?.hit_count ?? item.hit_count ?? hitNumbers.length)
    const hitRateRaw = Number(comparison?.hit_rate ?? item.hit_rate ?? (numbers.length > 0 ? hitCountRaw / numbers.length : 0))

    return {
      method: String(item.algorithm || item.method || method),
      numbers,
      confidence: Number.isFinite(confidence) ? confidence : 0,
      execution_time: Number(item.execution_time || 0),
      target_issue: typeof item.target_issue === 'string' ? item.target_issue : undefined,
      periods: Number.isFinite(Number(item.periods)) ? Number(item.periods) : undefined,
      hit_numbers: hitNumbers,
      hit_count: Number.isFinite(hitCountRaw) ? hitCountRaw : 0,
      hit_rate: Number.isFinite(hitRateRaw) ? hitRateRaw : 0,
      compare_issue: comparison?.target_issue ? String(comparison.target_issue) : undefined,
      actual_numbers: toNumberList(comparison?.actual_numbers),
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

  const handleProgress = (message: ProgressMessage) => {
    const extended = message as ProgressMessage & { progress?: number; message?: string }
    const inferredType =
      message.type ||
      (typeof extended.progress === 'number' || message.step ? 'progress' : undefined) ||
      (extended.message ? 'log' : undefined)

    if (inferredType === 'progress') {
      const nextProgress =
        typeof message.percentage === 'number'
          ? message.percentage
          : typeof extended.progress === 'number'
            ? extended.progress
            : progress.value
      progress.value = nextProgress

      if (message.step) {
        const normalizedStep = message.step === 'LOAD_DATA' ? 'FETCH_DATA' : message.step
        const stepIndex = PREDICT_STEPS.findIndex((step) => step.key === normalizedStep)
        if (stepIndex >= 0) {
          currentStep.value = stepIndex
        }
      }
    } else if (inferredType === 'log') {
      logs.value.push(message.details || extended.message || '')
    } else if (inferredType === 'error') {
      error.value = message.details || extended.message || '预测失败'
    }
  }

  const startPredict = async () => {
    if (isPredicting.value) return

    isPredicting.value = true
    currentStep.value = 0
    progress.value = 0
    logs.value = []
    results.value = []
    error.value = null

    closeEventSource()
    const streamParams: PredictParams = {
      ...params.value,
      method: selectedMethod.value,
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
        const data = JSON.parse(e.data) as ProgressMessage & { progress?: number; message?: string }
        handleProgress(data)
        if (data.details) {
          pushLog(data.details)
        } else if (data.message) {
          pushLog(data.message)
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
          const normalized = normalizeResults(payload.data, payload.method || selectedMethod.value, payload.confidence || 0)
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

    eventSource.addEventListener('error', (e: MessageEvent) => {
      try {
        const payload = JSON.parse(e.data) as { message?: string }
        if (payload.message) {
          error.value = payload.message
          pushLog(payload.message)
        }
      } catch {
        // SSE 标准 error 事件由 onerror 兜底处理
      }
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

  const cancelPredict = () => {
    closeEventSource()
    isPredicting.value = false
    logs.value.push(`[${new Date().toLocaleTimeString()}] 预测已取消`)
  }

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
    reset,
  }
})
