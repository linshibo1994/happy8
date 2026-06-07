<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { ArrowRight, Layers3 } from 'lucide-vue-next'

import HotColdSummaryPanel from '@/components/dashboard/HotColdSummaryPanel.vue'
import LatestDrawPanel from '@/components/dashboard/LatestDrawPanel.vue'
import QuickPredictionPanel from '@/components/dashboard/QuickPredictionPanel.vue'
import RecentPredictionsPanel from '@/components/dashboard/RecentPredictionsPanel.vue'
import RecommendedAlgorithmsPanel from '@/components/dashboard/RecommendedAlgorithmsPanel.vue'
import SystemStatusPanel from '@/components/dashboard/SystemStatusPanel.vue'
import TrendSummaryPanel from '@/components/dashboard/TrendSummaryPanel.vue'
import UsagePerformancePanel from '@/components/dashboard/UsagePerformancePanel.vue'
import { useAlgorithmStore } from '@/stores/algorithm'
import { useLotteryStore } from '@/stores/lottery'
import { useMembershipStore } from '@/stores/membership'
import { usePredictionStore } from '@/stores/prediction'
import type { Algorithm, PredictionResult } from '@/types'

type QuickActionStatus = 'idle' | 'running' | 'success' | 'error'

const algorithmStore = useAlgorithmStore()
const lotteryStore = useLotteryStore()
const membershipStore = useMembershipStore()
const predictionStore = usePredictionStore()

const quickStatus = ref<QuickActionStatus>('idle')
const quickStatusMessage = ref('等待开始预测')
const isSubmitting = ref(false)

const recommendedAlgorithms = computed(() =>
  [...algorithmStore.enabledAlgorithms].sort((a, b) => {
    if (b.successRate !== a.successRate) {
      return b.successRate - a.successRate
    }

    return a.averageCostMs - b.averageCostMs
  }).slice(0, 4),
)

const activeAlgorithm = computed<Algorithm>(() => {
  return (
    algorithmStore.enabledAlgorithms.find((algorithm) => algorithm.name === predictionStore.selectedAlgorithm) ??
    recommendedAlgorithms.value[0] ??
    algorithmStore.algorithms[0]
  )
})

const openedText = computed(() => formatDateTime(lotteryStore.latestResult.openedAt))
const todayUsedCount = computed(() =>
  Math.max(0, membershipStore.status.dailyLimit - membershipStore.status.remainingPredictions),
)

const recentPerformance = computed(() => {
  const records = predictionStore.history
  const hitNumbers = new Set(lotteryStore.latestResult.numbers)
  const scoredRecords = records.map((record) => {
    const hitCount = record.numbers.filter((number) => hitNumbers.has(number)).length
    const algorithm = findAlgorithmDisplayName(record.algorithm)

    return {
      record,
      algorithm,
      hitCount,
    }
  })

  const best = scoredRecords.reduce(
    (currentBest, item) => (item.hitCount > currentBest.hitCount ? item : currentBest),
    scoredRecords[0] ?? {
      record: undefined,
      algorithm: activeAlgorithm.value.displayName,
      hitCount: 0,
    },
  )

  const averageConfidence = records.length
    ? records.reduce((sum, record) => sum + record.confidence, 0) / records.length
    : 0

  return {
    predictionCount: records.length,
    hitCount: scoredRecords.reduce((sum, item) => sum + item.hitCount, 0),
    bestAlgorithm: best.algorithm,
    bestHitCount: best.hitCount,
    averageConfidence,
  }
})

const recentPredictionRecords = computed(() => {
  const latestNumbers = new Set(lotteryStore.latestResult.numbers)

  return predictionStore.history.slice(0, 3).map((record) => {
    const hitNumbers = record.numbers.filter((number) => latestNumbers.has(number))

    return {
      id: record.id,
      targetIssue: record.targetIssue,
      algorithmName: findAlgorithmDisplayName(record.algorithm),
      createdText: formatTime(record.createdAt),
      confidenceText: `${Math.round(record.confidence * 100)}%`,
      elapsedText: `${record.elapsedMs}ms`,
      numbers: record.numbers,
      hitNumbers,
      statusText: hitNumbers.length ? `命中 ${hitNumbers.length}` : '待开奖',
    }
  })
})

const hotNumbers = computed(() => lotteryStore.latestResult.numbers.slice(0, 8))
const coldNumbers = computed(() => {
  const drawnNumbers = new Set(lotteryStore.latestResult.numbers)
  const result: number[] = []

  for (let number = 1; number <= 80 && result.length < 8; number += 1) {
    if (!drawnNumbers.has(number)) {
      result.push(number)
    }
  }

  return result
})

const intersectionNumbers = computed(() => {
  const recentRecords = predictionStore.history.slice(0, 3)

  if (recentRecords.length < 2) {
    return recentRecords[0]?.numbers.slice(0, 4) ?? []
  }

  const counts = new Map<number, number>()
  recentRecords.forEach((record) => {
    record.numbers.forEach((number) => {
      counts.set(number, (counts.get(number) ?? 0) + 1)
    })
  })

  return Array.from(counts.entries())
    .filter(([, count]) => count >= 2)
    .map(([number]) => number)
    .sort((a, b) => a - b)
    .slice(0, 8)
})

const systemStatusItems = computed(() => [
  {
    name: '开奖数据',
    state: '已同步',
    detail: `最新第 ${lotteryStore.latestResult.issue} 期`,
    tone: 'success' as const,
  },
  {
    name: '预测引擎',
    state: predictionStore.executionState.phase === 'error' ? '异常' : '可用',
    detail: predictionStore.executionState.message,
    tone: predictionStore.executionState.phase === 'error' ? ('warning' as const) : ('blue' as const),
  },
  {
    name: '会员权益',
    state: membershipStore.hasQuota ? '正常' : '不足',
    detail: `${membershipStore.status.levelName}，剩余 ${membershipStore.status.remainingPredictions} 次`,
    tone: membershipStore.hasQuota ? ('success' as const) : ('warning' as const),
  },
])

function findAlgorithmDisplayName(name: string) {
  return algorithmStore.algorithms.find((algorithm) => algorithm.name === name)?.displayName ?? name
}

function formatDateTime(value: string) {
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function selectAlgorithm(name: string) {
  predictionStore.selectedAlgorithm = name
  predictionStore.executionState = {
    ...predictionStore.executionState,
    algorithm: name,
    targetIssue: lotteryStore.nextIssue,
  }
  quickStatus.value = 'idle'
  quickStatusMessage.value = `已选择 ${findAlgorithmDisplayName(name)}`
}

function startQuickPrediction() {
  if (!membershipStore.hasQuota || isSubmitting.value) {
    quickStatus.value = 'error'
    quickStatusMessage.value = membershipStore.hasQuota ? '预测正在执行' : '今日预测次数不足'
    return
  }

  isSubmitting.value = true
  quickStatus.value = 'running'
  quickStatusMessage.value = '正在读取最近开奖数据'
  predictionStore.markRunning('正在读取最近开奖数据')

  window.setTimeout(() => {
    const result = createQuickPredictionResult(activeAlgorithm.value)

    membershipStore.consumePredictionQuota()
    predictionStore.markDone(result)
    quickStatus.value = 'success'
    quickStatusMessage.value = '预测完成，可在最近预测中复盘'
    isSubmitting.value = false
  }, 620)
}

function createQuickPredictionResult(algorithm: Algorithm): PredictionResult {
  const seedNumbers = [...lotteryStore.latestResult.numbers]
  const offset = algorithm.averageCostMs % 17
  const numbers = Array.from({ length: predictionStore.predictCount }, (_, index) => {
    const base = seedNumbers[(index * 2 + offset) % seedNumbers.length]
    return ((base + offset + index * 3 - 1) % 80) + 1
  })
    .filter((number, index, array) => array.indexOf(number) === index)
    .sort((a, b) => a - b)

  for (let number = 1; numbers.length < predictionStore.predictCount && number <= 80; number += 1) {
    if (!numbers.includes(number)) {
      numbers.push(number)
    }
  }

  return {
    id: `prediction-${Date.now()}`,
    targetIssue: lotteryStore.nextIssue,
    algorithm: algorithm.name,
    analysisPeriods: predictionStore.analysisPeriods,
    predictCount: predictionStore.predictCount,
    numbers: numbers.slice(0, predictionStore.predictCount).sort((a, b) => a - b),
    confidence: Math.min(0.86, algorithm.successRate + 0.04),
    elapsedMs: algorithm.averageCostMs,
    createdAt: new Date().toISOString(),
    explanation: `${algorithm.displayName} 基于近期走势、区间结构和热冷号平衡生成候选预测。`,
  }
}
</script>

<template>
  <section class="dashboard-page" aria-labelledby="dashboard-title">
    <header class="dashboard-page__intro">
      <div>
        <span class="dashboard-page__kicker">首页工作台</span>
        <h2 id="dashboard-title">开奖、预测与复盘总览</h2>
        <p>首屏直接聚合高频操作信息，展示最新开奖、预测入口、会员次数和近期复盘表现。</p>
      </div>

      <RouterLink class="dashboard-page__secondary-action" to="/prediction">
        <Layers3 :size="17" aria-hidden="true" />
        <span>进入批量预测</span>
        <ArrowRight :size="16" aria-hidden="true" />
      </RouterLink>
    </header>

    <div class="dashboard-page__layout">
      <section class="dashboard-page__column dashboard-page__column--primary" aria-label="开奖与预测入口">
        <LatestDrawPanel :result="lotteryStore.latestResult" :opened-text="openedText" />
        <QuickPredictionPanel
          :next-issue="lotteryStore.nextIssue"
          :algorithm-name="activeAlgorithm.displayName"
          :analysis-periods="predictionStore.analysisPeriods"
          :predict-count="predictionStore.predictCount"
          :remaining="membershipStore.status.remainingPredictions"
          :has-quota="membershipStore.hasQuota"
          :is-submitting="isSubmitting"
          :status="quickStatus"
          :status-message="quickStatusMessage"
          @start="startQuickPrediction"
        />
      </section>

      <section class="dashboard-page__column dashboard-page__column--middle" aria-label="算法与走势摘要">
        <RecommendedAlgorithmsPanel
          :algorithms="recommendedAlgorithms"
          :active-name="predictionStore.selectedAlgorithm"
          @select="selectAlgorithm"
        />
        <TrendSummaryPanel :result="lotteryStore.latestResult" />
        <HotColdSummaryPanel
          :hot-numbers="hotNumbers"
          :cold-numbers="coldNumbers"
          :intersection-numbers="intersectionNumbers"
        />
      </section>

      <section class="dashboard-page__column dashboard-page__column--side" aria-label="账户、状态与复盘">
        <UsagePerformancePanel
          :remaining="membershipStore.status.remainingPredictions"
          :daily-limit="membershipStore.status.dailyLimit"
          :used-count="todayUsedCount"
          :prediction-count="recentPerformance.predictionCount"
          :hit-count="recentPerformance.hitCount"
          :best-algorithm="recentPerformance.bestAlgorithm"
          :best-hit-count="recentPerformance.bestHitCount"
          :average-confidence="recentPerformance.averageConfidence"
        />
        <SystemStatusPanel :items="systemStatusItems" />
        <RecentPredictionsPanel :records="recentPredictionRecords" />
      </section>
    </div>
  </section>
</template>

<style scoped>
.dashboard-page {
  display: grid;
  gap: 20px;
}

.dashboard-page__intro {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  min-width: 0;
}

.dashboard-page__kicker {
  color: var(--h8-color-cinnabar);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
}

.dashboard-page__intro h2 {
  margin: 4px 0 0;
  color: var(--h8-color-text);
  font-family: var(--h8-font-title);
  font-size: 28px;
  line-height: 1.2;
  letter-spacing: 0;
}

.dashboard-page__intro p {
  max-width: 720px;
  margin: 8px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 14px;
  line-height: 1.55;
}

.dashboard-page__secondary-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  min-height: 38px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-text);
  padding: 0 12px;
  font-size: 13px;
  font-weight: 700;
  line-height: 1.2;
  white-space: nowrap;
  transition: border-color 160ms ease, color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.dashboard-page__secondary-action:hover {
  border-color: var(--h8-color-data-blue);
  color: var(--h8-color-data-blue);
  transform: translateY(-1px);
}

.dashboard-page__secondary-action:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--h8-color-data-blue) 34%, transparent);
  outline-offset: 3px;
}

.dashboard-page__layout {
  display: grid;
  grid-template-columns: minmax(320px, 1.18fr) minmax(300px, 1fr) minmax(280px, 0.86fr);
  align-items: start;
  gap: 16px;
}

.dashboard-page__column {
  display: grid;
  min-width: 0;
  gap: 16px;
}

.dashboard-page__column--primary {
  grid-template-rows: auto auto;
}

@media (max-width: 1220px) {
  .dashboard-page__layout {
    grid-template-columns: minmax(0, 1fr) minmax(280px, 0.82fr);
  }

  .dashboard-page__column--middle {
    grid-column: 1 / 2;
  }

  .dashboard-page__column--side {
    grid-column: 2 / 3;
    grid-row: 1 / span 2;
  }
}

@media (max-width: 900px) {
  .dashboard-page__intro {
    align-items: flex-start;
    flex-direction: column;
  }

  .dashboard-page__layout {
    grid-template-columns: 1fr;
  }

  .dashboard-page__column--middle,
  .dashboard-page__column--side {
    grid-column: auto;
    grid-row: auto;
  }
}

@media (max-width: 760px) {
  .dashboard-page {
    gap: 16px;
  }

  .dashboard-page__intro h2 {
    font-size: 24px;
  }

  .dashboard-page__secondary-action {
    width: 100%;
    white-space: normal;
  }
}

@media (prefers-reduced-motion: reduce) {
  .dashboard-page__secondary-action {
    transition: none;
  }
}
</style>
