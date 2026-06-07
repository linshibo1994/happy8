<script setup lang="ts">
import { computed } from 'vue'
import {
  CheckCircle2,
  Copy,
  LoaderCircle,
  Play,
  RefreshCcw,
  RotateCcw,
  Save,
  Square,
  XCircle,
} from 'lucide-vue-next'

import type { Algorithm, PredictionExecutionState, PredictionResult } from '@/types'

import PredictionNumberBall from './PredictionNumberBall.vue'
import PredictionStageTrack from './PredictionStageTrack.vue'

export interface BatchPredictionSummary {
  intersectionNumbers: number[]
  highFrequencyNumbers: Array<{ number: number; count: number }>
  disagreementText: string
  averageConfidence: number
  fastestAlgorithmName: string
}

export type PredictionActionState = 'idle' | 'loading' | 'success' | 'error'

const props = defineProps<{
  mode: 'single' | 'batch'
  algorithms: Algorithm[]
  selectedAlgorithm: string
  selectedBatchAlgorithms: string[]
  executionState: PredictionExecutionState
  batchExecutions: PredictionExecutionState[]
  batchSummary: BatchPredictionSummary | null
  currentResult: PredictionResult | null
  isRunning: boolean
  actionState: PredictionActionState
  logs: string[]
}>()

const emit = defineEmits<{
  'update:mode': [value: 'single' | 'batch']
  'update:selectedAlgorithm': [value: string]
  'update:selectedBatchAlgorithms': [value: string[]]
  start: []
  cancel: []
  retry: []
  copy: []
}>()

const selectedAlgorithmItem = computed(() => props.algorithms.find((algorithm) => algorithm.name === props.selectedAlgorithm))

const successfulBatchResults = computed(() =>
  props.batchExecutions
    .map((execution) => execution.result)
    .filter((result): result is PredictionResult => Boolean(result)),
)

const canStart = computed(() => {
  if (props.isRunning) {
    return false
  }

  if (props.mode === 'batch') {
    return props.selectedBatchAlgorithms.length > 0
  }

  return Boolean(props.selectedAlgorithm)
})

const actionLabel = computed(() => {
  if (props.actionState === 'loading') {
    return props.mode === 'single' ? '预测中' : '批量执行中'
  }

  if (props.actionState === 'success') {
    return '执行完成'
  }

  if (props.actionState === 'error') {
    return '重新预测'
  }

  return props.mode === 'single' ? '开始预测' : '开始批量预测'
})

const actionIcon = computed(() => {
  if (props.actionState === 'loading') {
    return LoaderCircle
  }

  if (props.actionState === 'success') {
    return CheckCircle2
  }

  if (props.actionState === 'error') {
    return RefreshCcw
  }

  return Play
})

const isAlgorithmSelectedInBatch = (algorithmName: string) => props.selectedBatchAlgorithms.includes(algorithmName)

const toggleBatchAlgorithm = (algorithmName: string) => {
  const nextValue = isAlgorithmSelectedInBatch(algorithmName)
    ? props.selectedBatchAlgorithms.filter((name) => name !== algorithmName)
    : [...props.selectedBatchAlgorithms, algorithmName]

  emit('update:selectedBatchAlgorithms', nextValue)
}

const executionStatusText = (execution: PredictionExecutionState) => {
  if (execution.error) {
    return execution.error
  }

  if (execution.result) {
    return `${Math.round(execution.result.confidence * 100)}% / ${execution.result.elapsedMs}ms`
  }

  return execution.message
}

const getAlgorithmName = (algorithmName: string) =>
  props.algorithms.find((algorithm) => algorithm.name === algorithmName)?.displayName ?? algorithmName
</script>

<template>
  <section class="execution-panel" aria-label="预测执行区">
    <header class="execution-panel__header">
      <div>
        <span class="section-kicker">执行区</span>
        <h2>算法预测</h2>
        <p>
          {{ mode === 'single' ? '单算法输出完整阶段轨道和结果解释。' : '批量模式逐个算法落位，并生成交集与分歧摘要。' }}
        </p>
      </div>

      <div class="mode-tabs" role="tablist" aria-label="预测模式">
        <button
          type="button"
          role="tab"
          :aria-selected="mode === 'single'"
          :disabled="isRunning"
          @click="emit('update:mode', 'single')"
        >
          单算法
        </button>
        <button
          type="button"
          role="tab"
          :aria-selected="mode === 'batch'"
          :disabled="isRunning"
          @click="emit('update:mode', 'batch')"
        >
          批量预测
        </button>
      </div>
    </header>

    <div v-if="mode === 'single'" class="algorithm-picker" aria-label="单算法选择">
      <button
        v-for="algorithm in algorithms"
        :key="algorithm.name"
        type="button"
        class="algorithm-card"
        :class="{ 'is-active': selectedAlgorithm === algorithm.name }"
        :disabled="isRunning || !algorithm.enabled"
        @click="emit('update:selectedAlgorithm', algorithm.name)"
      >
        <span>
          <strong>{{ algorithm.displayName }}</strong>
          <small>{{ algorithm.category }} / {{ algorithm.complexity }}复杂度</small>
        </span>
        <em>{{ Math.round(algorithm.successRate * 100) }}%</em>
      </button>
    </div>

    <div v-else class="batch-picker" aria-label="批量算法选择">
      <label
        v-for="algorithm in algorithms"
        :key="algorithm.name"
        class="batch-picker__item"
        :class="{ 'is-active': isAlgorithmSelectedInBatch(algorithm.name) }"
      >
        <input
          type="checkbox"
          :checked="isAlgorithmSelectedInBatch(algorithm.name)"
          :disabled="isRunning || !algorithm.enabled"
          @change="toggleBatchAlgorithm(algorithm.name)"
        />
        <span>
          <strong>{{ algorithm.displayName }}</strong>
          <small>{{ algorithm.permissionLevel }} / {{ algorithm.averageCostMs }}ms</small>
        </span>
      </label>
    </div>

    <div class="execution-panel__actions">
      <button
        class="primary-action"
        type="button"
        :class="[`primary-action--${actionState}`]"
        :disabled="!canStart"
        @click="actionState === 'error' ? emit('retry') : emit('start')"
      >
        <component :is="actionIcon" :size="18" :class="{ 'is-spinning': actionState === 'loading' }" aria-hidden="true" />
        <span>{{ actionLabel }}</span>
      </button>

      <button class="secondary-action" type="button" :disabled="!isRunning" @click="emit('cancel')">
        <Square :size="16" aria-hidden="true" />
        <span>取消</span>
      </button>

      <button class="secondary-action" type="button" :disabled="isRunning || actionState !== 'error'" @click="emit('retry')">
        <RotateCcw :size="16" aria-hidden="true" />
        <span>重试</span>
      </button>
    </div>

    <PredictionStageTrack
      v-if="mode === 'single'"
      :progress="executionState.progress"
      :phase="executionState.phase"
      :message="executionState.message"
      :status="actionState"
    />

    <section v-else class="batch-board" aria-label="批量预测状态">
      <div class="batch-board__summary">
        <strong>总进度</strong>
        <span>{{ batchExecutions.length ? Math.round(batchExecutions.reduce((sum, execution) => sum + execution.progress, 0) / batchExecutions.length) : 0 }}%</span>
      </div>

      <div class="batch-board__grid">
        <article v-for="execution in batchExecutions" :key="execution.id" class="batch-execution-card">
          <header>
            <strong>{{ getAlgorithmName(execution.algorithm) }}</strong>
            <span v-if="execution.error" class="status-badge status-badge--error">
              <XCircle :size="14" aria-hidden="true" />
              异常
            </span>
            <span v-else-if="execution.result" class="status-badge status-badge--success">
              <CheckCircle2 :size="14" aria-hidden="true" />
              完成
            </span>
            <span v-else class="status-badge status-badge--running">
              <LoaderCircle :size="14" class="is-spinning" aria-hidden="true" />
              执行
            </span>
          </header>

          <PredictionStageTrack
            :progress="execution.progress"
            :phase="execution.phase"
            :message="executionStatusText(execution)"
            :status="execution.error ? 'error' : execution.result ? 'success' : 'running'"
            compact
          />

          <div v-if="execution.result" class="mini-balls" aria-label="算法预测号码">
            <PredictionNumberBall
              v-for="number in execution.result.numbers"
              :key="`${execution.id}-${number}`"
              :value="number"
              size="small"
            />
          </div>
        </article>
      </div>
    </section>

    <section class="log-panel" aria-label="当前执行日志">
      <header>
        <strong>当前日志</strong>
        <small>{{ logs.length }} 条</small>
      </header>
      <ol>
        <li v-for="log in logs.slice(0, 5)" :key="log">{{ log }}</li>
      </ol>
    </section>

    <section v-if="mode === 'single' && currentResult" class="result-panel" aria-label="单算法预测结果">
      <header>
        <div>
          <span class="section-kicker">结果号码</span>
          <h3>{{ selectedAlgorithmItem?.displayName ?? currentResult.algorithm }}</h3>
        </div>
        <div class="result-panel__metrics">
          <span>{{ Math.round(currentResult.confidence * 100) }}% 置信度</span>
          <span>{{ currentResult.elapsedMs }}ms</span>
          <span>未命中缓存</span>
        </div>
      </header>

      <div class="result-balls">
        <PredictionNumberBall v-for="number in currentResult.numbers" :key="number" :value="number" />
      </div>

      <p>{{ currentResult.explanation }}</p>

      <div class="result-panel__tools">
        <button type="button" @click="emit('copy')">
          <Copy :size="16" aria-hidden="true" />
          复制号码
        </button>
        <button type="button">
          <Save :size="16" aria-hidden="true" />
          保存结果
        </button>
      </div>
    </section>

    <section v-if="mode === 'batch' && batchSummary" class="batch-result-panel" aria-label="批量预测汇总">
      <header>
        <span class="section-kicker">批量汇总</span>
        <h3>多算法共识</h3>
      </header>

      <div class="summary-grid">
        <article>
          <strong>交集号码</strong>
          <div class="summary-balls">
            <PredictionNumberBall
              v-for="number in batchSummary.intersectionNumbers"
              :key="`intersection-${number}`"
              :value="number"
              variant="intersection"
              size="small"
            />
            <span v-if="batchSummary.intersectionNumbers.length === 0" class="empty-text">暂无完全交集</span>
          </div>
        </article>

        <article>
          <strong>高频候选</strong>
          <div class="summary-balls">
            <span v-for="item in batchSummary.highFrequencyNumbers" :key="item.number" class="frequency-chip">
              {{ String(item.number).padStart(2, '0') }} x{{ item.count }}
            </span>
          </div>
        </article>

        <article>
          <strong>算法分歧</strong>
          <p>{{ batchSummary.disagreementText }}</p>
        </article>

        <article>
          <strong>平均置信度</strong>
          <p>{{ Math.round(batchSummary.averageConfidence * 100) }}%</p>
        </article>

        <article>
          <strong>最快算法</strong>
          <p>{{ batchSummary.fastestAlgorithmName }}</p>
        </article>

        <article>
          <strong>成功算法</strong>
          <p>{{ successfulBatchResults.length }} / {{ batchExecutions.length }}</p>
        </article>
      </div>
    </section>

    <p class="prediction-disclaimer">
      预测结果仅基于历史数据和算法模型生成，不代表确定开奖结果，请理性使用。
    </p>
  </section>
</template>

<style scoped>
.execution-panel {
  display: grid;
  gap: 18px;
  min-width: 0;
}

.execution-panel__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.execution-panel__header h2 {
  margin: 3px 0 0;
  font-family: var(--h8-font-title);
  font-size: 24px;
  line-height: 1.2;
}

.execution-panel__header p {
  margin: 8px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.55;
}

.mode-tabs {
  display: inline-grid;
  grid-template-columns: repeat(2, minmax(84px, 1fr));
  flex: 0 0 auto;
  overflow: hidden;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
}

.mode-tabs button,
.algorithm-card,
.primary-action,
.secondary-action,
.result-panel__tools button {
  cursor: pointer;
}

.mode-tabs button:focus-visible,
.algorithm-card:focus-visible,
.batch-picker__item:focus-within,
.primary-action:focus-visible,
.secondary-action:focus-visible,
.result-panel__tools button:focus-visible {
  outline: 0;
  box-shadow: var(--h8-focus-ring);
}

.mode-tabs button {
  min-height: 36px;
  border: 0;
  background: transparent;
  color: var(--h8-color-text-muted);
  padding: 0 12px;
  line-height: 1.2;
}

.mode-tabs button[aria-selected='true'] {
  background: var(--h8-color-cinnabar);
  color: #fff;
}

.mode-tabs button:disabled,
.algorithm-card:disabled,
.primary-action:disabled,
.secondary-action:disabled,
.result-panel__tools button:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.algorithm-picker {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.algorithm-card {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  min-height: 74px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-text);
  padding: 12px;
  text-align: left;
  overflow-wrap: anywhere;
  transition:
    border-color 160ms ease,
    background 160ms ease,
    transform 160ms ease;
}

.algorithm-card:hover:not(:disabled),
.algorithm-card.is-active {
  border-color: color-mix(in srgb, var(--h8-color-cinnabar) 55%, var(--h8-color-line));
  background: color-mix(in srgb, var(--h8-color-cinnabar) 7%, var(--h8-color-surface-strong));
}

.algorithm-card strong,
.batch-picker__item strong {
  display: block;
  font-size: 14px;
  line-height: 1.25;
}

.algorithm-card small,
.batch-picker__item small {
  display: block;
  margin-top: 5px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.35;
}

.algorithm-card em {
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-size: 13px;
  font-style: normal;
  font-weight: 700;
  white-space: nowrap;
}

.batch-picker {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 9px;
}

.batch-picker__item {
  display: flex;
  min-width: 0;
  align-items: flex-start;
  gap: 9px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 10px;
  cursor: pointer;
}

.batch-picker__item.is-active {
  border-color: color-mix(in srgb, var(--h8-color-data-blue) 58%, var(--h8-color-line));
  background: color-mix(in srgb, var(--h8-color-data-blue) 7%, var(--h8-color-surface-strong));
}

.batch-picker__item input {
  flex: 0 0 auto;
  width: 16px;
  height: 16px;
  margin-top: 1px;
  accent-color: var(--h8-color-data-blue);
}

.execution-panel__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.primary-action,
.secondary-action,
.result-panel__tools button {
  display: inline-flex;
  max-width: 100%;
  min-height: 38px;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border-radius: var(--h8-radius-control);
  padding: 0 14px;
  line-height: 1.25;
  text-align: center;
  transition:
    border-color 160ms ease,
    background 160ms ease,
    color 160ms ease;
}

.primary-action {
  border: 1px solid var(--h8-color-cinnabar);
  background: var(--h8-color-cinnabar);
  color: #fff;
  font-weight: 700;
}

.primary-action--success {
  border-color: var(--h8-color-turquoise);
  background: var(--h8-color-turquoise);
}

.primary-action--error {
  border-color: var(--h8-color-risk-orange);
  background: var(--h8-color-risk-orange);
}

.secondary-action,
.result-panel__tools button {
  border: 1px solid var(--h8-color-line);
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-text);
}

.batch-board,
.log-panel,
.result-panel,
.batch-result-panel {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 15px;
}

.batch-board {
  display: grid;
  gap: 12px;
}

.batch-board__summary,
.log-panel header,
.result-panel header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.batch-board__summary span {
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-size: 20px;
  font-weight: 700;
}

.batch-board__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.batch-execution-card {
  display: grid;
  gap: 10px;
  min-width: 0;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  padding: 12px;
}

.batch-execution-card header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.batch-execution-card header strong {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 14px;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  flex: 0 0 auto;
  font-size: 12px;
  font-weight: 700;
}

.status-badge--running {
  color: var(--h8-color-data-blue);
}

.status-badge--success {
  color: var(--h8-color-turquoise);
}

.status-badge--error {
  color: var(--h8-color-risk-orange);
}

.mini-balls,
.result-balls,
.summary-balls {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.log-panel {
  display: grid;
  gap: 10px;
}

.log-panel strong,
.result-panel h3,
.batch-result-panel h3 {
  margin: 0;
}

.log-panel small {
  color: var(--h8-color-text-muted);
}

.log-panel ol {
  display: grid;
  gap: 6px;
  margin: 0;
  padding-left: 18px;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.45;
}

.result-panel,
.batch-result-panel {
  display: grid;
  gap: 14px;
}

.result-panel h3,
.batch-result-panel h3 {
  margin-top: 3px;
  font-family: var(--h8-font-title);
  font-size: 19px;
}

.result-panel__metrics {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 7px;
}

.result-panel__metrics span {
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  color: var(--h8-color-text-muted);
  padding: 5px 8px;
  font-size: 12px;
  line-height: 1.25;
}

.result-panel p,
.summary-grid p {
  margin: 0;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.55;
}

.result-panel__tools {
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.summary-grid article {
  display: grid;
  gap: 8px;
  min-height: 88px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  padding: 12px;
}

.summary-grid strong {
  font-size: 13px;
}

.frequency-chip {
  display: inline-flex;
  min-height: 24px;
  align-items: center;
  border: 1px solid color-mix(in srgb, var(--h8-color-cinnabar) 48%, var(--h8-color-line));
  border-radius: 999px;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 12px;
  font-weight: 700;
  padding: 0 8px;
}

.empty-text {
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.prediction-disclaimer {
  margin: 0;
  border-left: 3px solid var(--h8-color-risk-orange);
  background: color-mix(in srgb, var(--h8-color-risk-orange) 9%, var(--h8-color-surface-strong));
  color: var(--h8-color-text-muted);
  padding: 10px 12px;
  font-size: 13px;
  line-height: 1.5;
}

.is-spinning {
  animation: h8-spin 900ms linear infinite;
}

@keyframes h8-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 760px) {
  .execution-panel__header {
    display: grid;
  }

  .mode-tabs {
    width: 100%;
  }

  .algorithm-picker,
  .batch-picker,
  .batch-board__grid,
  .summary-grid {
    grid-template-columns: 1fr;
  }

  .batch-board__summary,
  .result-panel header {
    align-items: flex-start;
    flex-direction: column;
  }

  .result-panel__metrics {
    justify-content: flex-start;
  }
}

@media (prefers-reduced-motion: reduce) {
  .is-spinning {
    animation: none;
  }

  .algorithm-card,
  .primary-action,
  .secondary-action,
  .result-panel__tools button,
  .stage-track__bar span {
    transition: none;
  }
}

@media (max-width: 1180px) {
  .algorithm-picker,
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .batch-picker {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .execution-panel__header,
  .result-panel header {
    display: grid;
  }

  .mode-tabs,
  .algorithm-picker,
  .batch-picker,
  .batch-board__grid,
  .summary-grid {
    grid-template-columns: 1fr;
  }

  .result-panel__metrics {
    justify-content: flex-start;
  }
}
</style>
