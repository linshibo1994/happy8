<script setup lang="ts">
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'

import PredictionExecutionPanel, {
  type BatchPredictionSummary,
  type PredictionActionState,
} from '@/components/prediction/PredictionExecutionPanel.vue'
import PredictionParameterPanel, {
  type PredictionParameterSettings,
} from '@/components/prediction/PredictionParameterPanel.vue'
import PredictionReviewPanel from '@/components/prediction/PredictionReviewPanel.vue'
import { useAlgorithmStore } from '@/stores/algorithm'
import { useLotteryStore } from '@/stores/lottery'
import { useMembershipStore } from '@/stores/membership'
import { usePredictionStore } from '@/stores/prediction'
import type { Algorithm, PredictionExecutionState, PredictionPhase, PredictionResult } from '@/types'

type PredictionMode = 'single' | 'batch'

const phasePlan: Array<{ phase: PredictionPhase; progress: number; message: string; delay: number }> = [
  { phase: 'permission', progress: 8, message: '校验会员权限与剩余次数', delay: 260 },
  { phase: 'data', progress: 23, message: '读取最近 N 期开奖数据', delay: 320 },
  { phase: 'feature', progress: 48, message: '分析频率、遗漏、区间与结构', delay: 420 },
  { phase: 'compute', progress: 78, message: '执行当前算法模型', delay: 520 },
  { phase: 'validate', progress: 94, message: '去重、排序、生成置信度', delay: 280 },
  { phase: 'done', progress: 100, message: '生成预测结果', delay: 180 },
]

const algorithmStore = useAlgorithmStore()
const lotteryStore = useLotteryStore()
const membershipStore = useMembershipStore()
const predictionStore = usePredictionStore()

const mode = ref<PredictionMode>('single')
const targetIssue = ref(lotteryStore.nextIssue)
const selectedAlgorithm = ref(predictionStore.selectedAlgorithm)
const selectedBatchAlgorithms = ref<string[]>(['frequency', 'hot_cold', 'missing'])
const analysisPeriods = ref(predictionStore.analysisPeriods)
const predictCount = ref(predictionStore.predictCount)
const parameterSettings = reactive<PredictionParameterSettings>({
  recentWeight: 65,
  balanceWeight: 45,
  confidenceFloor: 55,
  excludeRecentHits: true,
})
const executionState = ref<PredictionExecutionState>({
  ...predictionStore.executionState,
  targetIssue: targetIssue.value,
})
const batchExecutions = ref<PredictionExecutionState[]>([])
const currentResult = ref<PredictionResult | null>(predictionStore.executionState.result ?? null)
const batchSummary = ref<BatchPredictionSummary | null>(null)
const logs = ref<string[]>(['等待选择算法并开始预测'])
const isRunning = ref(false)
const actionState = ref<PredictionActionState>('idle')
const retryMode = ref<PredictionMode>('single')
const completedExecutionSignature = ref('')
const timers = new Set<ReturnType<typeof window.setTimeout>>()

const enabledAlgorithms = computed(() => algorithmStore.enabledAlgorithms)
const selectedAlgorithmItem = computed(
  () => enabledAlgorithms.value.find((algorithm) => algorithm.name === selectedAlgorithm.value) ?? null,
)
const selectedBatchAlgorithmItems = computed(() =>
  enabledAlgorithms.value.filter((algorithm) => selectedBatchAlgorithms.value.includes(algorithm.name)),
)
const requiredQuota = computed(() => (mode.value === 'single' ? 1 : selectedBatchAlgorithms.value.length))
const executionSignature = computed(() =>
  JSON.stringify({
    mode: mode.value,
    targetIssue: targetIssue.value,
    algorithm: mode.value === 'single' ? selectedAlgorithm.value : [...selectedBatchAlgorithms.value].sort().join(','),
    analysisPeriods: analysisPeriods.value,
    predictCount: predictCount.value,
    recentWeight: parameterSettings.recentWeight,
    balanceWeight: parameterSettings.balanceWeight,
    confidenceFloor: parameterSettings.confidenceFloor,
    excludeRecentHits: parameterSettings.excludeRecentHits,
  }),
)
const recentPredictions = computed(() => {
  const algorithmNames =
    mode.value === 'single' ? [selectedAlgorithm.value] : selectedBatchAlgorithms.value.length ? selectedBatchAlgorithms.value : []

  return predictionStore.history.filter((item) => algorithmNames.includes(item.algorithm)).slice(0, 3)
})
const currentModeHasFreshResult = computed(() => {
  if (mode.value === 'single') {
    return Boolean(
      completedExecutionSignature.value === executionSignature.value &&
      currentResult.value &&
        currentResult.value.algorithm === selectedAlgorithm.value &&
        currentResult.value.targetIssue === targetIssue.value &&
        currentResult.value.analysisPeriods === analysisPeriods.value &&
        currentResult.value.predictCount === predictCount.value,
    )
  }

  const completedAlgorithmNames = batchExecutions.value
    .filter((execution) => execution.result)
    .map((execution) => execution.algorithm)
    .sort()
  const selectedAlgorithmNames = [...selectedBatchAlgorithms.value].sort()

  return Boolean(
    completedExecutionSignature.value === executionSignature.value &&
    batchSummary.value &&
      selectedAlgorithmNames.length > 0 &&
      completedAlgorithmNames.length === selectedAlgorithmNames.length &&
      completedAlgorithmNames.every((name, index) => name === selectedAlgorithmNames[index]),
  )
})

const clearTimers = () => {
  timers.forEach((timer) => window.clearTimeout(timer))
  timers.clear()
}

const schedule = (handler: () => void, delay: number) => {
  const timer = window.setTimeout(() => {
    timers.delete(timer)
    handler()
  }, delay)

  timers.add(timer)
}

const clampNumber = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const normalizeNumbers = (numbers: number[]) => [...new Set(numbers)].sort((a, b) => a - b)

const seededNumber = (seed: string, index: number) => {
  const charSum = seed.split('').reduce((sum, char, charIndex) => sum + char.charCodeAt(0) * (charIndex + 3), 0)

  return ((charSum + index * 17 + analysisPeriods.value * 3 + predictCount.value * 11) % 80) + 1
}

const generateCandidateNumbers = (algorithm: Algorithm) => {
  const numbers: number[] = []
  let cursor = 0

  while (numbers.length < predictCount.value && cursor < 160) {
    const candidate = seededNumber(`${algorithm.name}-${targetIssue.value}`, cursor)

    if (!parameterSettings.excludeRecentHits || !lotteryStore.latestResult.numbers.includes(candidate) || cursor > 80) {
      numbers.push(candidate)
    }

    cursor += 1
  }

  return normalizeNumbers(numbers).slice(0, predictCount.value)
}

const createResult = (algorithm: Algorithm, startedAt: number): PredictionResult => {
  const confidenceBoost = parameterSettings.recentWeight * 0.0008 + parameterSettings.balanceWeight * 0.0006
  const confidence = clampNumber(
    algorithm.successRate + confidenceBoost - (predictCount.value > 12 ? 0.025 : 0),
    parameterSettings.confidenceFloor / 100,
    0.88,
  )
  const elapsedMs = Math.max(320, Math.round(algorithm.averageCostMs * (0.78 + (analysisPeriods.value - 10) / 520)))

  return {
    id: `${algorithm.name}-${targetIssue.value}-${startedAt}`,
    targetIssue: targetIssue.value,
    algorithm: algorithm.name,
    analysisPeriods: analysisPeriods.value,
    predictCount: predictCount.value,
    numbers: generateCandidateNumbers(algorithm),
    confidence,
    elapsedMs,
    createdAt: new Date().toISOString(),
    explanation: `${algorithm.displayName}基于 ${analysisPeriods.value} 期历史数据，结合近期权重 ${parameterSettings.recentWeight}% 与区间均衡 ${parameterSettings.balanceWeight}% 生成候选。`,
  }
}

const createInitialExecution = (algorithmName: string): PredictionExecutionState => ({
  id: `${algorithmName}-${targetIssue.value}-${Date.now()}`,
  algorithm: algorithmName,
  targetIssue: targetIssue.value,
  progress: 0,
  phase: 'permission',
  message: '等待权限校验',
  startedAt: Date.now(),
})

const addLog = (message: string) => {
  const timeText = new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date())

  logs.value = [`${timeText} ${message}`, ...logs.value].slice(0, 12)
}

const syncActionStateForCurrentMode = () => {
  if (isRunning.value || actionState.value === 'error') {
    return
  }

  actionState.value = currentModeHasFreshResult.value ? 'success' : 'idle'
}

const validateBeforeRun = () => {
  if (!targetIssue.value.trim()) {
    actionState.value = 'error'
    addLog('目标期号不能为空')
    return false
  }

  if (requiredQuota.value <= 0) {
    actionState.value = 'error'
    addLog('请至少选择一个算法')
    return false
  }

  if (membershipStore.status.remainingPredictions < requiredQuota.value) {
    actionState.value = 'error'
    addLog('剩余预测次数不足，已阻止本次提交')
    return false
  }

  return true
}

const updateStoreParameters = () => {
  predictionStore.selectedAlgorithm = selectedAlgorithm.value
  predictionStore.analysisPeriods = analysisPeriods.value
  predictionStore.predictCount = predictCount.value
}

const updateParameterSettings = (settings: PredictionParameterSettings) => {
  Object.assign(parameterSettings, settings)
}

const startSinglePrediction = () => {
  const algorithm = selectedAlgorithmItem.value

  if (!algorithm) {
    actionState.value = 'error'
    addLog('当前算法不可用')
    return
  }

  clearTimers()
  updateStoreParameters()
  isRunning.value = true
  actionState.value = 'loading'
  currentResult.value = null
  batchSummary.value = null
  logs.value = []
  const execution = createInitialExecution(algorithm.name)
  executionState.value = execution
  predictionStore.executionState = execution
  addLog(`开始执行 ${algorithm.displayName}`)

  let totalDelay = 0
  phasePlan.forEach((step) => {
    totalDelay += step.delay
    schedule(() => {
      executionState.value = {
        ...executionState.value,
        progress: step.progress,
        phase: step.phase,
        message: step.message.replace('N', String(analysisPeriods.value)),
      }
      predictionStore.executionState = executionState.value
      addLog(`${algorithm.displayName}：${executionState.value.message}`)

      if (step.phase === 'done') {
        const result = createResult(algorithm, execution.startedAt)
        currentResult.value = result
        executionState.value = {
          ...executionState.value,
          progress: 100,
          phase: 'done',
          message: '预测完成，结果已输出',
          endedAt: Date.now(),
          result,
        }
        predictionStore.markDone(result)
        membershipStore.consumePredictionQuota(1)
        isRunning.value = false
        completedExecutionSignature.value = executionSignature.value
        actionState.value = 'success'
        retryMode.value = 'single'
        addLog(`${algorithm.displayName} 输出 ${result.numbers.length} 个候选号码`)
      }
    }, totalDelay)
  })
}

const updateBatchExecution = (executionId: string, patch: Partial<PredictionExecutionState>) => {
  batchExecutions.value = batchExecutions.value.map((execution) =>
    execution.id === executionId ? { ...execution, ...patch } : execution,
  )
}

const createBatchSummary = (results: PredictionResult[]): BatchPredictionSummary => {
  const frequency = new Map<number, number>()

  results.forEach((result) => {
    result.numbers.forEach((number) => {
      frequency.set(number, (frequency.get(number) ?? 0) + 1)
    })
  })

  const highFrequencyNumbers = [...frequency.entries()]
    .map(([number, count]) => ({ number, count }))
    .sort((a, b) => b.count - a.count || a.number - b.number)
    .slice(0, 10)
  const intersectionNumbers = highFrequencyNumbers
    .filter((item) => item.count === results.length)
    .map((item) => item.number)
    .slice(0, 10)
  const allNumbers = [...frequency.keys()]
  const zoneGroups = [
    allNumbers.filter((number) => number <= 20).length,
    allNumbers.filter((number) => number > 20 && number <= 40).length,
    allNumbers.filter((number) => number > 40 && number <= 60).length,
    allNumbers.filter((number) => number > 60).length,
  ]
  const maxZone = Math.max(...zoneGroups)
  const minZone = Math.min(...zoneGroups)
  const fastest = results.reduce((fastestResult, result) =>
    result.elapsedMs < fastestResult.elapsedMs ? result : fastestResult,
  )
  const fastestAlgorithmName =
    enabledAlgorithms.value.find((algorithm) => algorithm.name === fastest.algorithm)?.displayName ?? fastest.algorithm

  return {
    intersectionNumbers,
    highFrequencyNumbers,
    disagreementText:
      maxZone - minZone >= 4
        ? '算法候选在号码区间上存在明显偏移，建议重点复盘区间均衡。'
        : '算法候选区间分布较均衡，分歧主要来自具体号码排序。',
    averageConfidence: results.reduce((sum, result) => sum + result.confidence, 0) / results.length,
    fastestAlgorithmName,
  }
}

const maybeCompleteBatch = () => {
  if (!batchExecutions.value.length) {
    return
  }

  const isComplete = batchExecutions.value.every((execution) => execution.result || execution.error)

  if (!isComplete) {
    return
  }

  const results = batchExecutions.value
    .map((execution) => execution.result)
    .filter((result): result is PredictionResult => Boolean(result))

  if (!results.length) {
    isRunning.value = false
    actionState.value = 'error'
    addLog('批量预测没有成功结果，可点击重试')
    return
  }

  batchSummary.value = createBatchSummary(results)
  membershipStore.consumePredictionQuota(results.length)
  isRunning.value = false
  completedExecutionSignature.value = executionSignature.value
  actionState.value = 'success'
  retryMode.value = 'batch'
  addLog(`批量预测完成，成功 ${results.length} 个算法`)
}

const startBatchPrediction = () => {
  clearTimers()
  updateStoreParameters()
  isRunning.value = true
  actionState.value = 'loading'
  currentResult.value = null
  batchSummary.value = null
  logs.value = []
  const algorithms = selectedBatchAlgorithmItems.value
  batchExecutions.value = algorithms.map((algorithm) => createInitialExecution(algorithm.name))
  addLog(`开始批量预测：${algorithms.map((algorithm) => algorithm.displayName).join('、')}`)

  algorithms.forEach((algorithm, algorithmIndex) => {
    const execution = batchExecutions.value[algorithmIndex]
    let totalDelay = algorithmIndex * 110

    phasePlan.forEach((step) => {
      totalDelay += Math.round(step.delay * (0.85 + algorithmIndex * 0.05))
      schedule(() => {
        updateBatchExecution(execution.id, {
          progress: step.progress,
          phase: step.phase,
          message: step.message.replace('N', String(analysisPeriods.value)),
        })

        if (step.phase === 'compute') {
          addLog(`${algorithm.displayName} 正在计算候选权重`)
        }

        if (step.phase === 'done') {
          const result = createResult(algorithm, execution.startedAt)
          updateBatchExecution(execution.id, {
            progress: 100,
            phase: 'done',
            message: '预测完成',
            endedAt: Date.now(),
            result,
          })
          predictionStore.history.unshift(result)
          addLog(`${algorithm.displayName} 完成，置信度 ${Math.round(result.confidence * 100)}%`)
          maybeCompleteBatch()
        }
      }, totalDelay)
    })
  })
}

const startPrediction = () => {
  if (isRunning.value || !validateBeforeRun()) {
    return
  }

  if (mode.value === 'single') {
    startSinglePrediction()
  } else {
    startBatchPrediction()
  }
}

const cancelPrediction = () => {
  if (!isRunning.value) {
    return
  }

  clearTimers()
  isRunning.value = false
  actionState.value = 'error'
  retryMode.value = mode.value
  addLog('用户取消执行，参数已保留')

  if (mode.value === 'single') {
    executionState.value = {
      ...executionState.value,
      phase: 'error',
      message: '已取消，可重试',
      error: '用户取消执行',
      endedAt: Date.now(),
    }
    predictionStore.executionState = executionState.value
  } else {
    batchExecutions.value = batchExecutions.value.map((execution) =>
      execution.result
        ? execution
        : {
            ...execution,
            phase: 'error',
            message: '已取消，可重试',
            error: '用户取消执行',
            endedAt: Date.now(),
          },
    )
  }
}

const retryPrediction = () => {
  mode.value = retryMode.value
  startPrediction()
}

const copyCurrentNumbers = async () => {
  if (!currentResult.value) {
    return
  }

  const text = currentResult.value.numbers.map((number) => String(number).padStart(2, '0')).join(' ')

  try {
    await navigator.clipboard.writeText(text)
    addLog('预测号码已复制到剪贴板')
  } catch {
    addLog(`复制失败，请手动记录：${text}`)
  }
}

watch(analysisPeriods, (value) => {
  analysisPeriods.value = clampNumber(value, 10, 200)
})

watch(predictCount, (value) => {
  predictCount.value = clampNumber(value, 1, 20)
})

watch(
  [
    mode,
    selectedAlgorithm,
    selectedBatchAlgorithms,
    targetIssue,
    analysisPeriods,
    predictCount,
    () => parameterSettings.recentWeight,
    () => parameterSettings.balanceWeight,
    () => parameterSettings.confidenceFloor,
    () => parameterSettings.excludeRecentHits,
  ],
  syncActionStateForCurrentMode,
)

onBeforeUnmount(() => {
  clearTimers()
})
</script>

<template>
  <section class="prediction-page" aria-labelledby="prediction-page-title">
    <div class="prediction-page__intro">
      <span class="section-kicker">预测执行</span>
      <h2 id="prediction-page-title">第 {{ targetIssue }} 期模型工作台</h2>
      <p>选择算法、配置参数并观察阶段轨道；批量预测会逐个算法展示状态和共识摘要。</p>
      <p class="prediction-page__disclaimer">
        预测结果仅基于历史数据和算法模型生成，不代表确定开奖结果，请理性使用。
      </p>
    </div>

    <div class="prediction-page__layout">
      <div class="prediction-page__panel prediction-page__panel--parameters">
        <PredictionParameterPanel
          v-model:target-issue="targetIssue"
          v-model:analysis-periods="analysisPeriods"
          v-model:predict-count="predictCount"
          :settings="parameterSettings"
          :mode="mode"
          :membership="membershipStore.status"
          :required-quota="requiredQuota"
          :is-running="isRunning"
          @update:settings="updateParameterSettings"
        />
      </div>

      <div class="prediction-page__panel prediction-page__panel--execution">
        <PredictionExecutionPanel
          v-model:mode="mode"
          v-model:selected-algorithm="selectedAlgorithm"
          v-model:selected-batch-algorithms="selectedBatchAlgorithms"
          :algorithms="enabledAlgorithms"
          :execution-state="executionState"
          :batch-executions="batchExecutions"
          :batch-summary="batchSummary"
          :current-result="currentResult"
          :is-running="isRunning"
          :action-state="actionState"
          :logs="logs"
          @start="startPrediction"
          @cancel="cancelPrediction"
          @retry="retryPrediction"
          @copy="copyCurrentNumbers"
        />
      </div>

      <div class="prediction-page__panel prediction-page__panel--review">
        <PredictionReviewPanel
          :mode="mode"
          :selected-algorithm="selectedAlgorithmItem"
          :selected-algorithms="selectedBatchAlgorithmItems"
          :recent-predictions="recentPredictions"
        />
      </div>
    </div>
  </section>
</template>

<style scoped>
.prediction-page {
  display: grid;
  gap: 22px;
}

.prediction-page__intro {
  max-width: 920px;
}

.prediction-page__intro h2 {
  margin: 3px 0 0;
  font-family: var(--h8-font-title);
  font-size: 30px;
  line-height: 1.2;
}

.prediction-page__intro p {
  margin: 8px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.6;
}

.prediction-page__disclaimer {
  display: inline-flex;
  max-width: 100%;
  border-left: 3px solid var(--h8-color-risk-orange);
  background: color-mix(in srgb, var(--h8-color-risk-orange) 8%, transparent);
  padding: 8px 10px;
}

.prediction-page__layout {
  display: grid;
  grid-template-columns: minmax(252px, 0.86fr) minmax(0, 2fr) minmax(260px, 0.95fr);
  gap: 16px;
  align-items: start;
}

.prediction-page__panel {
  min-width: 0;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-surface-strong) 94%, transparent);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 18px;
}

.prediction-page__panel--execution {
  padding: 20px;
}

@media (max-width: 1320px) {
  .prediction-page__layout {
    grid-template-columns: minmax(250px, 0.9fr) minmax(0, 1.8fr);
  }

  .prediction-page__panel--review {
    grid-column: 1 / -1;
  }
}

@media (max-width: 900px) {
  .prediction-page__layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .prediction-page__intro h2 {
    font-size: 24px;
  }

  .prediction-page__panel,
  .prediction-page__panel--execution {
    padding: 15px;
  }
}
</style>
