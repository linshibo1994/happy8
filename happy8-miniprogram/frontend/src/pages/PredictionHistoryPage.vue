<script setup lang="ts">
import { computed, ref, watchEffect } from 'vue'
import { RotateCcw, Search, Table2 } from 'lucide-vue-next'

import PredictionComparePanel from '@/components/history/PredictionComparePanel.vue'
import PredictionRecordRow, { type PredictionRecordView } from '@/components/history/PredictionRecordRow.vue'
import { useAlgorithmStore } from '@/stores/algorithm'
import { useLotteryStore } from '@/stores/lottery'
import { usePredictionStore } from '@/stores/prediction'

const algorithmStore = useAlgorithmStore()
const lotteryStore = useLotteryStore()
const predictionStore = usePredictionStore()

const algorithmFilter = ref('全部')
const issueFilter = ref('')
const selectedRecordId = ref('')

const formatDateTime = (value: string) =>
  new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))

const algorithmLabelByName = computed<Record<string, string>>(() =>
  Object.fromEntries(algorithmStore.algorithms.map((algorithm) => [algorithm.name, algorithm.displayName])),
)

const demoRecords = computed<PredictionRecordView[]>(() => {
  const storeRecords: PredictionRecordView[] = predictionStore.history.map((record) => {
    const actualNumbers = lotteryStore.latestResult.numbers
    const hitNumbers = record.numbers.filter((number) => actualNumbers.includes(number))

    return {
      id: record.id,
      createdAt: formatDateTime(record.createdAt),
      targetIssue: record.targetIssue,
      algorithmName: record.algorithm,
      algorithmLabel: algorithmLabelByName.value[record.algorithm] ?? record.algorithm,
      predictedNumbers: record.numbers,
      actualNumbers,
      hitNumbers,
      hitCount: hitNumbers.length,
      hitRate: hitNumbers.length / Math.max(record.numbers.length, 1),
      confidence: record.confidence,
      elapsedMs: record.elapsedMs,
      cached: false,
    }
  })

  return [
    ...storeRecords,
    {
      id: 'history-20260606002-markov',
      createdAt: '06/06 21:20',
      targetIssue: '20260606002',
      algorithmName: 'markov',
      algorithmLabel: '马尔可夫链',
      predictedNumbers: [2, 9, 14, 22, 31, 38, 44, 55, 63, 78],
      actualNumbers: [1, 4, 10, 17, 24, 29, 33, 40, 46, 51, 57, 60, 64, 69, 72, 75, 76, 77, 79, 80],
      hitNumbers: [],
      hitCount: 0,
      hitRate: 0,
      confidence: 0.59,
      elapsedMs: 1380,
      cached: true,
    },
    {
      id: 'history-20260605001-hot-cold',
      createdAt: '06/05 20:55',
      targetIssue: '20260605001',
      algorithmName: 'hot_cold',
      algorithmLabel: '冷热分析',
      predictedNumbers: [5, 8, 16, 23, 34, 41, 49, 58, 70, 74],
      actualNumbers: [3, 5, 8, 11, 16, 20, 27, 32, 34, 39, 41, 47, 49, 53, 58, 61, 66, 70, 74, 79],
      hitNumbers: [5, 8, 16, 34, 41, 49, 58, 70, 74],
      hitCount: 9,
      hitRate: 0.9,
      confidence: 0.64,
      elapsedMs: 910,
      cached: false,
    },
    {
      id: 'history-20260604001-transformer',
      createdAt: '06/04 21:05',
      targetIssue: '20260604001',
      algorithmName: 'transformer',
      algorithmLabel: 'Transformer',
      predictedNumbers: [6, 12, 19, 25, 33, 36, 43, 52, 68, 80],
      actualNumbers: [2, 6, 12, 15, 18, 21, 25, 28, 33, 37, 43, 48, 52, 54, 59, 63, 68, 73, 76, 80],
      hitNumbers: [6, 12, 25, 33, 43, 52, 68, 80],
      hitCount: 8,
      hitRate: 0.8,
      confidence: 0.67,
      elapsedMs: 3440,
      cached: false,
    },
  ]
})

const algorithmOptions = computed(() => [
  { label: '全部算法', value: '全部' },
  ...algorithmStore.algorithms.map((algorithm) => ({ label: algorithm.displayName, value: algorithm.name })),
])

const filteredRecords = computed(() =>
  demoRecords.value.filter((record) => {
    const algorithmMatched = algorithmFilter.value === '全部' || record.algorithmName === algorithmFilter.value
    const issueMatched = !issueFilter.value.trim() || record.targetIssue.includes(issueFilter.value.trim())

    return algorithmMatched && issueMatched
  }),
)

const selectedRecord = computed(() => filteredRecords.value.find((record) => record.id === selectedRecordId.value))

const totalHitCount = computed(() => filteredRecords.value.reduce((sum, record) => sum + record.hitCount, 0))
const zeroRateCount = computed(() => filteredRecords.value.filter((record) => record.hitRate === 0).length)
const averageHitRate = computed(() => {
  if (!filteredRecords.value.length) {
    return '0%'
  }

  const total = filteredRecords.value.reduce((sum, record) => sum + record.hitRate, 0)
  return `${Math.round((total / filteredRecords.value.length) * 100)}%`
})

const resetFilters = () => {
  algorithmFilter.value = '全部'
  issueFilter.value = ''
}

watchEffect(() => {
  if (!filteredRecords.value.length) {
    selectedRecordId.value = ''
    return
  }

  if (!filteredRecords.value.some((record) => record.id === selectedRecordId.value)) {
    selectedRecordId.value = filteredRecords.value[0].id
  }
})
</script>

<template>
  <section class="history-page" aria-labelledby="history-title">
    <header class="history-page__hero">
      <div>
        <span class="section-kicker">Prediction Review</span>
        <h2 id="history-title">预测历史与命中复盘</h2>
        <p>按算法和期号筛选预测记录，实际开奖号码与预测号码并排展示。命中率 0% 会作为有效复盘结果保留。</p>
      </div>
    </header>

    <div class="history-page__metrics" aria-label="复盘指标">
      <div>
        <span>记录数</span>
        <strong>{{ filteredRecords.length }}</strong>
      </div>
      <div>
        <span>累计命中</span>
        <strong>{{ totalHitCount }}</strong>
      </div>
      <div>
        <span>平均命中率</span>
        <strong>{{ averageHitRate }}</strong>
      </div>
      <div>
        <span>0% 复盘</span>
        <strong>{{ zeroRateCount }}</strong>
      </div>
    </div>

    <section class="history-page__filters" aria-label="预测历史筛选">
      <label>
        <span>算法</span>
        <select v-model="algorithmFilter">
          <option v-for="option in algorithmOptions" :key="option.value" :value="option.value">
            {{ option.label }}
          </option>
        </select>
      </label>

      <label>
        <span>期号</span>
        <div class="history-page__search">
          <Search :size="17" aria-hidden="true" />
          <input v-model="issueFilter" type="search" placeholder="输入完整或部分期号" />
        </div>
      </label>

      <button type="button" @click="resetFilters">
        <RotateCcw :size="16" aria-hidden="true" />
        重置
      </button>
    </section>

    <div class="history-page__layout">
      <main class="history-page__table-panel">
        <div class="history-page__table-title">
          <span>
            <Table2 :size="18" aria-hidden="true" />
            预测记录表
          </span>
          <small>点击期号查看详情</small>
        </div>

        <div v-if="filteredRecords.length" class="history-page__table-scroll">
          <table>
            <thead>
              <tr>
                <th>期号/时间</th>
                <th>算法</th>
                <th>预测号码</th>
                <th>实际开奖</th>
                <th>命中</th>
                <th>命中率</th>
                <th>置信度</th>
                <th>耗时</th>
                <th>来源</th>
              </tr>
            </thead>
            <tbody>
              <PredictionRecordRow
                v-for="record in filteredRecords"
                :key="record.id"
                :record="record"
                :selected="selectedRecordId === record.id"
                @select="selectedRecordId = $event"
              />
            </tbody>
          </table>
        </div>

        <div v-else class="history-page__empty">
          <h3>没有符合条件的预测记录</h3>
          <p>调整算法或期号筛选，也可以先执行一次预测生成新记录。</p>
          <button type="button" @click="resetFilters">查看全部记录</button>
          <RouterLink to="/prediction">去执行预测</RouterLink>
        </div>
      </main>

      <PredictionComparePanel :record="selectedRecord" />
    </div>
  </section>
</template>

<style scoped lang="scss">
.history-page {
  display: grid;
  gap: 22px;
}

.history-page__hero h2 {
  margin: 4px 0 0;
  font-family: var(--h8-font-title);
  font-size: 31px;
  line-height: 1.2;
}

.history-page__hero p {
  max-width: 780px;
  margin: 9px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.7;
}

.history-page__metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.history-page__metrics div {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 16px;
}

.history-page__metrics span {
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.history-page__metrics strong {
  display: block;
  margin-top: 6px;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 28px;
  line-height: 1;
}

.history-page__filters {
  display: grid;
  grid-template-columns: minmax(180px, 240px) minmax(220px, 1fr) auto;
  gap: 12px;
  align-items: end;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 16px;
}

.history-page__filters label {
  display: grid;
  gap: 7px;
}

.history-page__filters label > span {
  color: var(--h8-color-text-muted);
  font-size: 13px;
  font-weight: 700;
}

.history-page__filters select,
.history-page__search,
.history-page__filters button {
  min-height: 40px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
}

.history-page__filters select {
  padding: 0 10px;
}

.history-page__search {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 10px;
}

.history-page__search input {
  width: 100%;
  min-width: 0;
  border: 0;
  background: transparent;
  color: inherit;
  outline: none;
}

.history-page__filters select:focus-visible,
.history-page__search:focus-within,
.history-page__filters button:focus-visible,
.history-page__empty button:focus-visible,
.history-page__empty a:focus-visible {
  outline: 0;
  box-shadow: var(--h8-focus-ring);
}

.history-page__filters button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 0 14px;
  cursor: pointer;
}

.history-page__layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 460px);
  gap: 18px;
  align-items: start;
}

.history-page__table-panel {
  min-width: 0;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
}

.history-page__table-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--h8-color-line);
  padding: 16px;
}

.history-page__table-title span {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--h8-color-cinnabar);
  font-weight: 800;
}

.history-page__table-title small {
  color: var(--h8-color-text-muted);
}

.history-page__table-scroll {
  overflow: auto;
}

.history-page table {
  width: 100%;
  min-width: 980px;
  border-collapse: collapse;
  font-size: 14px;
}

.history-page th {
  border-bottom: 1px solid var(--h8-color-line);
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 800;
  padding: 10px;
  text-align: left;
  white-space: nowrap;
}

.history-page__empty {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  padding: 22px;
}

.history-page__empty h3,
.history-page__empty p {
  width: 100%;
  margin: 0;
}

.history-page__empty p {
  color: var(--h8-color-text-muted);
}

.history-page__empty button,
.history-page__empty a {
  display: inline-flex;
  min-height: 38px;
  align-items: center;
  border: 1px solid var(--h8-color-cinnabar);
  border-radius: var(--h8-radius-control);
  padding: 0 14px;
  font-weight: 800;
  cursor: pointer;
}

.history-page__empty button {
  background: var(--h8-color-cinnabar);
  color: #fff;
}

.history-page__empty a {
  color: var(--h8-color-cinnabar);
}

@media (max-width: 1180px) {
  .history-page__layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .history-page__metrics,
  .history-page__filters {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 460px) {
  .history-page__metrics {
    grid-template-columns: 1fr;
  }
}
</style>
