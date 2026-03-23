<template>
  <div class="comparison-page">
    <DataVBorder :type="1">
      <div class="page-content">
        <div class="page-header">
          <h2 class="page-title">批量对比</h2>
          <button
            class="btn-export"
            @click="store.exportJSON()"
            :disabled="store.roundResults.length === 0"
          >
            导出 JSON
          </button>
        </div>

        <div class="config-panel">
          <div class="config-row">
            <div class="config-item">
              <label>目标期号</label>
              <select v-model="store.params.target_issue" :disabled="store.isComparing">
                <option value="" disabled>请选择期号</option>
                <option v-for="issue in store.availableIssues" :key="issue" :value="issue">
                  {{ issue }}
                </option>
              </select>
            </div>
            <div class="config-item">
              <label>预测算法</label>
              <select v-model="store.params.method_name" :disabled="store.isComparing">
                <optgroup
                  v-for="(methods, category) in store.algorithmCategories"
                  :key="category"
                  :label="category"
                >
                  <option v-for="method in methods" :key="method.value" :value="method.value">
                    {{ method.label }}
                  </option>
                </optgroup>
              </select>
            </div>
            <div class="config-item">
              <label>对比次数</label>
              <input
                type="number"
                v-model.number="store.params.comparison_times"
                :min="1"
                :max="500"
                :disabled="store.isComparing"
              />
            </div>
            <div class="config-item">
              <label>分析期数</label>
              <input
                type="number"
                v-model.number="store.params.periods_value"
                :min="20"
                :max="1000"
                :disabled="store.isComparing || store.params.periods_mode === 'random'"
              />
            </div>
            <div class="config-item config-switch">
              <label>随机期数</label>
              <label class="switch">
                <input
                  type="checkbox"
                  :checked="store.params.periods_mode === 'random'"
                  @change="togglePeriodsMode"
                  :disabled="store.isComparing"
                />
                <span class="slider"></span>
              </label>
            </div>
          </div>

          <div class="config-actions">
            <button
              class="btn-primary"
              @click="handleStart"
              :disabled="store.isComparing || !store.params.target_issue"
            >
              {{ store.isComparing ? '对比中...' : '开始对比' }}
            </button>
            <button class="btn-secondary" @click="store.reset()" :disabled="store.isComparing">
              重置
            </button>
            <button v-if="store.isComparing" class="btn-danger" @click="store.stopCompare()">
              停止
            </button>
          </div>
        </div>

        <div v-if="store.isComparing || store.progress > 0" class="progress-section">
          <div class="progress-header">
            <span>对比进度</span>
            <span>
              {{ store.currentRound }} / {{ store.params.comparison_times }} 轮
              ({{ Math.round(store.progress) }}%)
            </span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: store.progress + '%' }"></div>
          </div>
        </div>

        <div v-if="store.error" class="error-banner">{{ store.error }}</div>
      </div>
    </DataVBorder>

    <div v-if="store.roundResults.length > 0" class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{{ store.roundResults.length }}</div>
        <div class="stat-label">对比轮次</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ successCount }}</div>
        <div class="stat-label">成功预测</div>
      </div>
      <div class="stat-card">
        <div class="stat-value highlight-rate">{{ store.averageHitCount.toFixed(2) }}</div>
        <div class="stat-label">平均命中数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value highlight-win">{{ store.bestHitCount }}</div>
        <div class="stat-label">最高命中数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ store.positiveHitRate.toFixed(1) }}%</div>
        <div class="stat-label">命中轮次占比</div>
      </div>
    </div>

    <DataVBorder v-if="store.hitRounds.length > 0" :type="1">
      <div class="page-content">
        <h3 class="section-title">高命中记录（Top 10）</h3>
        <div class="winning-records">
          <div v-for="record in topHitRounds" :key="'hit-' + record.round" class="winning-record">
            <div class="record-header">
              <span class="record-round">第 {{ record.round }} 轮</span>
              <span class="record-hit">命中 {{ record.hit_count }} 个 ({{ formatRate(record.hit_rate) }})</span>
            </div>
            <div class="record-balls">
              <span class="record-label">预测:</span>
              <div class="balls-row">
                <span
                  v-for="(num, i) in record.predicted_numbers"
                  :key="'pr-' + record.round + '-' + i"
                  class="ball-wrap"
                  :class="{ 'is-hit': record.hit_numbers.includes(num) }"
                >
                  <LotteryBall :number="num" size="sm" />
                </span>
              </div>
            </div>
            <div class="record-info">
              <span>分析期数: {{ record.analysis_periods }}</span>
              <span>命中号码: {{ formatNumbers(record.hit_numbers) }}</span>
            </div>
          </div>
        </div>
      </div>
    </DataVBorder>

    <DataVBorder v-show="store.roundResults.length > 0" :type="1">
      <div class="page-content">
        <h3 class="section-title">命中数量分布</h3>
        <div ref="chartRef" class="chart-container"></div>
      </div>
    </DataVBorder>

    <DataVBorder v-if="store.roundResults.length > 0" :type="1">
      <div class="page-content">
        <h3 class="section-title">详细结果 ({{ store.roundResults.length }} 条)</h3>

        <div v-if="store.finalResult?.actual_result?.numbers?.length" class="actual-result">
          <span class="actual-label">目标期开奖号码:</span>
          <div class="balls-row balls-row-compact">
            <span
              v-for="(num, i) in store.finalResult.actual_result.numbers"
              :key="'actual-' + i"
              class="ball-wrap is-actual"
            >
              <LotteryBall :number="num" size="sm" />
            </span>
          </div>
        </div>

        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>轮次</th>
                <th>分析期数</th>
                <th>预测号码</th>
                <th>命中号码</th>
                <th>命中数</th>
                <th>命中率</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="result in paginatedResults" :key="result.round" :class="{ 'row-winning': result.hit_count > 0 }">
                <td>{{ result.round }}</td>
                <td>{{ result.analysis_periods }}</td>
                <td class="balls-cell">
                  <div v-if="result.predicted_numbers.length > 0" class="balls-row balls-row-compact">
                    <span
                      v-for="(num, i) in result.predicted_numbers"
                      :key="'tr-' + result.round + '-' + i"
                      class="ball-wrap"
                      :class="{ 'is-hit': result.hit_numbers.includes(num) }"
                    >
                      <LotteryBall :number="num" size="sm" />
                    </span>
                  </div>
                  <span v-else class="text-muted">-</span>
                </td>
                <td>{{ formatNumbers(result.hit_numbers) }}</td>
                <td>{{ result.success ? result.hit_count : '-' }}</td>
                <td>{{ result.success ? formatRate(result.hit_rate) : '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="pagination" v-if="totalPages > 1">
          <button class="page-btn" :disabled="currentPage <= 1" @click="currentPage--">上一页</button>
          <span class="page-info">第 {{ currentPage }} / {{ totalPages }} 页</span>
          <button class="page-btn" :disabled="currentPage >= totalPages" @click="currentPage++">下一页</button>
        </div>
      </div>
    </DataVBorder>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { useComparisonStore } from '@/stores/comparison'
import DataVBorder from '@/components/common/DataVBorder.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'

const store = useComparisonStore()

const chartRef = ref<HTMLElement | null>(null)
let chartInstance: echarts.ECharts | null = null

const currentPage = ref(1)
const pageSize = 20

const totalPages = computed(() => Math.ceil(store.roundResults.length / pageSize))

const paginatedResults = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return store.roundResults.slice(start, start + pageSize)
})

const successCount = computed(() => store.roundResults.filter((row) => row.success).length)

const topHitRounds = computed(() => store.hitRounds.slice(0, 10))

const togglePeriodsMode = () => {
  store.params.periods_mode = store.params.periods_mode === 'fixed' ? 'random' : 'fixed'
}

const handleStart = () => {
  const validationError = store.validateParams()
  if (validationError) {
    store.error = validationError
    return
  }
  currentPage.value = 1
  store.startCompare()
}

const formatNumbers = (numbers: number[]) => {
  if (!numbers || numbers.length === 0) return '-'
  return numbers.map((num) => String(num).padStart(2, '0')).join(', ')
}

const formatRate = (rate: number) => `${(rate * 100).toFixed(1)}%`

const hitCountDistribution = computed(() => {
  const dist: Record<number, number> = {}
  for (let i = 0; i <= 20; i++) {
    dist[i] = 0
  }
  for (const row of store.roundResults) {
    const hitCount = Math.min(20, Math.max(0, Number(row.hit_count || 0)))
    dist[hitCount] += 1
  }
  return dist
})

let chartUpdateTimer: ReturnType<typeof setTimeout> | null = null

const scheduleChartUpdate = () => {
  if (chartUpdateTimer) return
  chartUpdateTimer = setTimeout(() => {
    chartUpdateTimer = null
    updateChart()
  }, 250)
}

const updateChart = () => {
  if (!chartRef.value) return

  if (!chartInstance) {
    chartInstance = echarts.init(chartRef.value)
    requestAnimationFrame(() => chartInstance?.resize())
  }

  const dist = hitCountDistribution.value
  const categories = Array.from({ length: 21 }, (_, i) => `${i}个`)
  const values = Array.from({ length: 21 }, (_, i) => dist[i] || 0)

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '3%', bottom: '3%', top: '8%', containLabel: true },
    xAxis: {
      type: 'category',
      data: categories,
      axisLabel: { color: '#aaa', interval: 1 },
      axisLine: { lineStyle: { color: '#444' } }
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#aaa' },
      splitLine: { lineStyle: { color: '#333' } }
    },
    series: [{
      type: 'bar',
      barWidth: '56%',
      data: values,
      itemStyle: {
        color: '#00d4ff'
      },
      label: {
        show: true,
        position: 'top',
        color: '#ccc',
        formatter: '{c}'
      }
    }]
  }

  chartInstance.setOption(option)
}

watch(
  () => store.roundResults.length,
  () => {
    scheduleChartUpdate()
  }
)

watch(
  () => store.isComparing,
  (val) => {
    if (!val && store.roundResults.length > 0) {
      if (chartUpdateTimer) {
        clearTimeout(chartUpdateTimer)
        chartUpdateTimer = null
      }
      nextTick(() => updateChart())
    }
  }
)

const handleResize = () => {
  chartInstance?.resize()
}

onMounted(() => {
  store.fetchIssues()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (chartUpdateTimer) {
    clearTimeout(chartUpdateTimer)
    chartUpdateTimer = null
  }
  chartInstance?.dispose()
  chartInstance = null
  if (store.isComparing) {
    store.stopCompare()
  }
})
</script>

<style scoped>
.comparison-page {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.page-content {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-title {
  font-size: 20px;
  color: var(--text-primary);
}

.section-title {
  font-size: 16px;
  color: var(--text-primary);
  margin-bottom: 16px;
}

.config-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.config-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 160px;
}

.config-item label {
  font-size: 13px;
  color: var(--text-secondary);
}

.config-item select,
.config-item input[type="number"] {
  padding: 8px 12px;
  background: var(--bg-tertiary, #1a1a2e);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 14px;
  outline: none;
  transition: border-color 0.3s;
}

.config-item select:focus,
.config-item input[type="number"]:focus {
  border-color: var(--color-primary);
}

.config-item select:disabled,
.config-item input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.config-switch {
  flex-direction: row;
  align-items: center;
  gap: 8px;
  min-width: auto;
}

.switch {
  position: relative;
  display: inline-block;
  width: 44px;
  height: 24px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background-color: #444;
  border-radius: 24px;
  transition: 0.3s;
}

.slider::before {
  content: '';
  position: absolute;
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: #fff;
  border-radius: 50%;
  transition: 0.3s;
}

.switch input:checked + .slider {
  background-color: var(--color-primary);
}

.switch input:checked + .slider::before {
  transform: translateX(20px);
}

.switch input:disabled + .slider {
  opacity: 0.5;
  cursor: not-allowed;
}

.config-actions {
  display: flex;
  gap: 12px;
}

.btn-primary,
.btn-secondary,
.btn-danger,
.btn-export {
  padding: 10px 20px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
}

.btn-primary {
  background: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}

.btn-primary:hover:not(:disabled) {
  opacity: 0.85;
}

.btn-primary:disabled,
.btn-secondary:disabled,
.btn-export:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary,
.btn-export {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.btn-secondary:hover:not(:disabled),
.btn-export:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.btn-danger {
  background: transparent;
  color: #e74c3c;
  border-color: #e74c3c;
}

.btn-danger:hover {
  background: #e74c3c;
  color: #fff;
}

.progress-section {
  margin-top: 16px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.progress-bar {
  height: 8px;
  background: var(--bg-tertiary, #1a1a2e);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-primary), #00d4ff);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.error-banner {
  margin-top: 16px;
  padding: 12px 16px;
  background: rgba(231, 76, 60, 0.15);
  border: 1px solid rgba(231, 76, 60, 0.3);
  border-radius: 6px;
  color: #e74c3c;
  font-size: 14px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px;
}

.stat-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 18px;
  text-align: center;
}

.stat-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.stat-value.highlight-rate {
  color: #2ecc71;
}

.stat-value.highlight-win {
  color: #f1c40f;
}

.stat-label {
  font-size: 13px;
  color: var(--text-secondary);
}

.winning-records {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.winning-record {
  padding: 12px 16px;
  background: rgba(0, 212, 255, 0.04);
  border: 1px solid rgba(0, 212, 255, 0.25);
  border-radius: 8px;
}

.record-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.record-round {
  font-size: 14px;
  color: var(--text-secondary);
}

.record-hit {
  font-size: 13px;
  color: #00d4ff;
}

.record-balls {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.record-label {
  font-size: 13px;
  color: var(--text-secondary);
  min-width: 40px;
}

.record-info {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 12px;
  color: var(--text-secondary);
}

.actual-result {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.actual-label {
  font-size: 13px;
  color: var(--text-secondary);
  padding-top: 6px;
}

.chart-container {
  width: 100%;
  height: 300px;
}

.table-container {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  min-width: 960px;
}

.data-table th,
.data-table td {
  padding: 10px 8px;
  text-align: left;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-primary);
  font-size: 13px;
  vertical-align: middle;
}

.data-table thead th {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
}

.row-winning {
  background: rgba(46, 204, 113, 0.06);
}

.balls-cell {
  min-width: 460px;
}

.balls-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.balls-row-compact {
  gap: 6px;
}

.ball-wrap {
  position: relative;
  display: inline-flex;
  border-radius: 50%;
  padding: 2px;
}

.ball-wrap.is-hit {
  box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.9);
}

.ball-wrap.is-actual {
  box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.85);
}

.text-muted {
  color: var(--text-secondary);
}

.pagination {
  margin-top: 16px;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
}

.page-btn {
  padding: 6px 12px;
  border: 1px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-radius: 6px;
  cursor: pointer;
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-info {
  color: var(--text-secondary);
  font-size: 13px;
}

@media (max-width: 768px) {
  .comparison-page {
    padding: 16px;
  }

  .page-content {
    padding: 16px;
  }

  .config-actions {
    flex-wrap: wrap;
  }

  .record-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }

  .record-balls {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
