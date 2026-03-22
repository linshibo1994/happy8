<template>
  <div class="comparison-page">
    <!-- 配置面板 -->
    <DataVBorder :type="1">
      <div class="page-content">
        <div class="page-header">
          <h2 class="page-title">批量对比</h2>
          <div class="header-actions">
            <button
              class="btn-export"
              @click="store.exportJSON()"
              :disabled="store.roundResults.length === 0"
            >
              导出 JSON
            </button>
          </div>
        </div>

        <!-- 配置区域 -->
        <div class="config-panel">
          <div class="config-row">
            <div class="config-item">
              <label>目标期号</label>
              <select v-model="store.params.target_issue" :disabled="store.isComparing">
                <option value="" disabled>请选择期号</option>
                <option
                  v-for="issue in store.availableIssues"
                  :key="issue"
                  :value="issue"
                >
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
                  <option
                    v-for="method in methods"
                    :key="method.value"
                    :value="method.value"
                  >
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
            <button
              class="btn-secondary"
              @click="store.reset()"
              :disabled="store.isComparing"
            >
              重置
            </button>
            <button
              v-if="store.isComparing"
              class="btn-danger"
              @click="store.stopCompare()"
            >
              停止
            </button>
          </div>
        </div>

        <!-- 进度条 -->
        <div v-if="store.isComparing || store.progress > 0" class="progress-section">
          <div class="progress-header">
            <span>对比进度</span>
            <span>{{ store.currentRound }} / {{ store.params.comparison_times }} 轮 ({{ Math.round(store.progress) }}%)</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: store.progress + '%' }"></div>
          </div>
        </div>

        <!-- 错误提示 -->
        <div v-if="store.error" class="error-banner">
          {{ store.error }}
        </div>
      </div>
    </DataVBorder>

    <!-- 统计卡片 -->
    <div v-if="store.roundResults.length > 0" class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{{ store.roundResults.length }}</div>
        <div class="stat-label">对比次数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ successCount }}</div>
        <div class="stat-label">成功预测</div>
      </div>
      <div class="stat-card">
        <div class="stat-value highlight-win">{{ store.winningRounds.length }}</div>
        <div class="stat-label">中奖次数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value highlight-rate">{{ store.winRate.toFixed(1) }}%</div>
        <div class="stat-label">中奖率</div>
      </div>
    </div>

    <!-- 中奖记录 -->
    <DataVBorder v-if="store.winningRounds.length > 0" :type="1">
      <div class="page-content">
        <h3 class="section-title">中奖记录</h3>
        <div class="winning-records">
          <div
            v-for="record in store.winningRounds"
            :key="'win-' + record.round"
            class="winning-record"
          >
            <div class="record-header">
              <span class="record-round">第 {{ record.round }} 轮</span>
              <span class="record-prize" :class="'prize-' + record.prize_level">
                {{ record.prize_name }}
              </span>
            </div>
            <div v-if="record.predicted_reds" class="record-balls">
              <span class="record-label">预测:</span>
              <div class="balls-row">
                <LotteryBall
                  v-for="(num, i) in record.predicted_reds"
                  :key="'pr-' + i"
                  :number="num"
                  :type="isRedHit(num) ? 'red' : 'gray'"
                  size="sm"
                />
                <LotteryBall
                  v-if="record.predicted_blue !== null && record.predicted_blue !== undefined"
                  :number="record.predicted_blue"
                  :type="record.blue_match ? 'blue' : 'gray'"
                  size="sm"
                />
              </div>
            </div>
            <div class="record-info">
              <span>红球命中: {{ record.red_matches }}/6</span>
              <span>蓝球: {{ record.blue_match ? '命中' : '未中' }}</span>
            </div>
          </div>
        </div>
      </div>
    </DataVBorder>

    <!-- 奖级分布图 -->
    <DataVBorder v-show="store.roundResults.length > 0" :type="1">
      <div class="page-content">
        <h3 class="section-title">奖级分布</h3>
        <div ref="chartRef" class="chart-container"></div>
      </div>
    </DataVBorder>

    <!-- 详细结果表格 -->
    <DataVBorder v-if="store.roundResults.length > 0" :type="1">
      <div class="page-content">
        <h3 class="section-title">详细结果 ({{ store.roundResults.length }} 条)</h3>
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>轮次</th>
                <th>分析期数</th>
                <th>预测红球</th>
                <th>预测蓝球</th>
                <th>红球命中</th>
                <th>蓝球匹配</th>
                <th>中奖等级</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="result in paginatedResults"
                :key="result.round"
                :class="{ 'row-winning': result.prize_level > 0 && result.prize_level <= 6 }"
              >
                <td>{{ result.round }}</td>
                <td>{{ result.analysis_periods }}</td>
                <td class="balls-cell">
                  <div v-if="result.predicted_reds" class="balls-row balls-row-compact">
                    <LotteryBall
                      v-for="(num, i) in result.predicted_reds"
                      :key="'tr-' + result.round + '-' + i"
                      :number="num"
                      :type="isRedHit(num) ? 'red' : 'gray'"
                      size="sm"
                    />
                  </div>
                  <span v-else class="text-muted">-</span>
                </td>
                <td class="balls-cell">
                  <LotteryBall
                    v-if="result.predicted_blue !== null"
                    :number="result.predicted_blue"
                    :type="result.blue_match ? 'blue' : 'gray'"
                    size="sm"
                  />
                  <span v-else class="text-muted">-</span>
                </td>
                <td>{{ result.success ? result.red_matches + '/6' : '-' }}</td>
                <td>
                  <span v-if="result.success" :class="result.blue_match ? 'text-success' : 'text-muted'">
                    {{ result.blue_match ? '命中' : '未中' }}
                  </span>
                  <span v-else class="text-muted">-</span>
                </td>
                <td>
                  <span :class="'prize-tag prize-' + result.prize_level">
                    {{ result.prize_name }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 分页 -->
        <div class="pagination" v-if="totalPages > 1">
          <button class="page-btn" :disabled="currentPage <= 1" @click="currentPage--">
            上一页
          </button>
          <span class="page-info">
            第 {{ currentPage }} / {{ totalPages }} 页
          </span>
          <button class="page-btn" :disabled="currentPage >= totalPages" @click="currentPage++">
            下一页
          </button>
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

// 图表引用
const chartRef = ref<HTMLElement | null>(null)
let chartInstance: echarts.ECharts | null = null

// 分页
const currentPage = ref(1)
const pageSize = 20

const totalPages = computed(() => Math.ceil(store.roundResults.length / pageSize))

const paginatedResults = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return store.roundResults.slice(start, start + pageSize)
})

// 成功预测数
const successCount = computed(() =>
  store.roundResults.filter(r => r.success).length
)

// 实际开奖红球集合（用于高亮命中号码）
const actualRedSet = computed(() => {
  if (store.finalResult?.actual_result) {
    return new Set(store.finalResult.actual_result.red_balls)
  }
  return new Set<number>()
})

// 判断红球是否命中
const isRedHit = (num: number) => actualRedSet.value.has(num)

// 切换期数模式
const togglePeriodsMode = () => {
  store.params.periods_mode = store.params.periods_mode === 'fixed' ? 'random' : 'fixed'
}

// 开始对比
const handleStart = () => {
  const validationError = store.validateParams()
  if (validationError) {
    store.error = validationError
    return
  }
  currentPage.value = 1
  store.startCompare()
}

// 奖级分布图表数据
const prizeDistribution = computed(() => {
  const dist: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 0: 0 }
  for (const r of store.roundResults) {
    if (r.prize_level in dist) {
      dist[r.prize_level]++
    }
  }
  return dist
})

// 奖级名称映射
const prizeNames: Record<number, string> = {
  1: '一等奖', 2: '二等奖', 3: '三等奖',
  4: '四等奖', 5: '五等奖', 6: '六等奖', 0: '未中奖'
}

// 图表更新节流
let chartUpdateTimer: ReturnType<typeof setTimeout> | null = null

const scheduleChartUpdate = () => {
  if (chartUpdateTimer) return
  chartUpdateTimer = setTimeout(() => {
    chartUpdateTimer = null
    updateChart()
  }, 300)
}

// 更新图表
const updateChart = () => {
  if (!chartRef.value) return

  if (!chartInstance) {
    chartInstance = echarts.init(chartRef.value)
    // 等浏览器完成 layout 后 resize，防止零尺寸问题
    requestAnimationFrame(() => chartInstance?.resize())
  }

  const dist = prizeDistribution.value
  const categories = [1, 2, 3, 4, 5, 6, 0].map(k => prizeNames[k])
  const values = [1, 2, 3, 4, 5, 6, 0].map(k => dist[k])
  const colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#95a5a6']

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '10%', containLabel: true },
    xAxis: {
      type: 'category',
      data: categories,
      axisLabel: { color: '#aaa' },
      axisLine: { lineStyle: { color: '#444' } }
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#aaa' },
      splitLine: { lineStyle: { color: '#333' } }
    },
    series: [{
      type: 'bar',
      data: values.map((v, i) => ({
        value: v,
        itemStyle: { color: colors[i] }
      })),
      barWidth: '50%',
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

// 监听结果变化更新图表（节流）
watch(
  () => store.roundResults.length,
  () => {
    scheduleChartUpdate()
  }
)

// 对比完成时立即更新图表
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

// 窗口resize
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
  // 停止可能正在进行的对比
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

.header-actions {
  display: flex;
  align-items: center;
  gap: 16px;
}

.section-title {
  font-size: 16px;
  color: var(--text-primary);
  margin-bottom: 16px;
}

/* 配置面板 */
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

/* Toggle 开关 */
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
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #444;
  border-radius: 24px;
  transition: 0.3s;
}

.slider::before {
  content: "";
  position: absolute;
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
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

/* 操作按钮 */
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

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.btn-secondary:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.btn-secondary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
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

.btn-export {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.btn-export:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.btn-export:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 进度条 */
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

/* 错误提示 */
.error-banner {
  margin-top: 16px;
  padding: 12px 16px;
  background: rgba(231, 76, 60, 0.15);
  border: 1px solid rgba(231, 76, 60, 0.3);
  border-radius: 6px;
  color: #e74c3c;
  font-size: 14px;
}

/* 统计卡片 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.stat-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 20px;
  text-align: center;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.stat-value.highlight-win {
  color: #f1c40f;
}

.stat-value.highlight-rate {
  color: #2ecc71;
}

.stat-label {
  font-size: 13px;
  color: var(--text-secondary);
}

/* 中奖记录 */
.winning-records {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.winning-record {
  padding: 12px 16px;
  background: rgba(241, 196, 15, 0.05);
  border: 1px solid rgba(241, 196, 15, 0.2);
  border-radius: 8px;
}

.record-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.record-round {
  font-size: 14px;
  color: var(--text-secondary);
}

.record-prize {
  font-size: 13px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: bold;
}

.prize-1 { background: rgba(231, 76, 60, 0.2); color: #e74c3c; }
.prize-2 { background: rgba(230, 126, 34, 0.2); color: #e67e22; }
.prize-3 { background: rgba(241, 196, 15, 0.2); color: #f1c40f; }
.prize-4 { background: rgba(46, 204, 113, 0.2); color: #2ecc71; }
.prize-5 { background: rgba(52, 152, 219, 0.2); color: #3498db; }
.prize-6 { background: rgba(155, 89, 182, 0.2); color: #9b59b6; }

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
  gap: 16px;
  font-size: 12px;
  color: var(--text-secondary);
}

/* 图表 */
.chart-container {
  width: 100%;
  height: 300px;
}

/* 表格 */
.table-container {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th,
.data-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border-color);
  text-align: center;
  font-size: 13px;
}

.data-table th {
  background: var(--bg-tertiary, #1a1a2e);
  color: var(--text-secondary);
  font-weight: 600;
  white-space: nowrap;
}

.data-table td {
  color: var(--text-primary);
}

.row-winning {
  background: rgba(241, 196, 15, 0.05);
}

.balls-cell {
  white-space: nowrap;
}

.balls-row {
  display: flex;
  gap: 4px;
  align-items: center;
  justify-content: center;
}

.balls-row-compact {
  gap: 2px;
}

.prize-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.prize-tag.prize-0 {
  color: var(--text-muted);
}

.text-success {
  color: #2ecc71;
}

.text-muted {
  color: var(--text-muted);
}

/* 分页 */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  padding: 16px 0;
}

.page-btn {
  padding: 8px 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.3s;
}

.page-btn:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-info {
  font-size: 14px;
  color: var(--text-secondary);
}

/* 响应式 */
@media (max-width: 768px) {
  .comparison-page {
    padding: 12px;
  }

  .config-row {
    flex-direction: column;
  }

  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
