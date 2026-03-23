<template>
  <div class="dashboard-page">
    <section class="latest-result">
      <DataVBorder :type="1">
        <div class="result-content">
          <h2 class="section-title">最新开奖结果</h2>

          <div v-if="lotteryStore.loading" class="loading-state">
            <span class="loading-text">数据加载中...</span>
          </div>

          <template v-else-if="latestResult">
            <div class="result-info">
              <span class="period">第 {{ latestResult.period }} 期</span>
              <span class="date">{{ latestResult.date }}</span>
            </div>
            <div class="balls-container">
              <LotteryBall
                v-for="(num, index) in latestResult.numbers"
                :key="'num-' + index"
                :number="num"
                size="lg"
                :animate="animate"
                :delay="index * 70"
              />
            </div>
          </template>

          <div v-else class="empty-state">
            <span class="empty-text">暂无开奖数据</span>
          </div>
        </div>
      </DataVBorder>
    </section>

    <section class="frequency-section">
      <DataVBorder :type="1">
        <div class="result-content">
          <h2 class="section-title">号码频率 Top 20</h2>
          <div ref="frequencyChartRef" class="frequency-chart"></div>
        </div>
      </DataVBorder>
    </section>

    <section class="stats-grid">
      <StatCard
        v-for="stat in statsData"
        :key="stat.title"
        :title="stat.title"
        :value="stat.value"
        :icon="stat.icon"
        :trend="stat.trend"
        :color="stat.color"
      />
    </section>

    <section class="quick-actions">
      <router-link to="/predict" class="action-btn primary">
        <span class="action-icon" v-html="'&#9733;'"></span>
        <span class="action-text">开始预测</span>
      </router-link>
      <router-link to="/history" class="action-btn">
        <span class="action-icon" v-html="'&#128196;'"></span>
        <span class="action-text">历史数据</span>
      </router-link>
      <router-link to="/analysis" class="action-btn">
        <span class="action-icon" v-html="'&#128200;'"></span>
        <span class="action-text">数据分析</span>
      </router-link>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import { useLotteryStore } from '@/stores/lottery'
import { usePredictStore } from '@/stores/predict'
import LotteryBall from '@/components/common/LotteryBall.vue'
import StatCard from '@/components/common/StatCard.vue'
import DataVBorder from '@/components/common/DataVBorder.vue'

const lotteryStore = useLotteryStore()
const predictStore = usePredictStore()

const animate = ref(false)
const frequencyChartRef = ref<HTMLElement | null>(null)
let frequencyChart: echarts.ECharts | null = null

const latestResult = computed(() => lotteryStore.latestResult)

const statsData = computed(() => [
  {
    title: '总期数',
    value: lotteryStore.totalPeriods || '--',
    icon: '&#128202;',
    trend: 0,
    color: 'var(--color-primary)',
  },
  {
    title: '数据状态',
    value: lotteryStore.loading ? '加载中' : '已就绪',
    icon: '&#128197;',
    trend: 0,
    color: 'var(--color-success)',
  },
  {
    title: '预测次数',
    value: predictStore.predictionCount,
    icon: '&#9733;',
    trend: 0,
    color: 'var(--color-accent)',
  },
  {
    title: '平均置信度',
    value: `${(predictStore.averageConfidence * 100).toFixed(1)}%`,
    icon: '&#128175;',
    trend: 0,
    color: 'var(--color-primary)',
  },
])

const renderFrequencyChart = () => {
  if (!frequencyChartRef.value) return

  const statistics = lotteryStore.statistics
  const frequency = statistics?.frequency || {}

  const points = Object.entries(frequency)
    .map(([number, count]) => ({ number: Number(number), count: Number(count) }))
    .filter((item) => Number.isFinite(item.number) && Number.isFinite(item.count))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20)
    .sort((a, b) => a.number - b.number)

  if (!frequencyChart) {
    frequencyChart = echarts.init(frequencyChartRef.value)
  }

  frequencyChart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: points.map((item) => String(item.number).padStart(2, '0')),
      axisLine: { lineStyle: { color: 'rgba(0, 120, 212, 0.3)' } },
      axisLabel: { color: '#a0aec0' },
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: 'rgba(0, 120, 212, 0.2)' } },
      splitLine: { lineStyle: { color: 'rgba(0, 120, 212, 0.12)' } },
      axisLabel: { color: '#a0aec0' },
    },
    series: [
      {
        type: 'bar',
        data: points.map((item) => item.count),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#74c0fc' },
            { offset: 1, color: '#228be6' },
          ]),
        },
        barMaxWidth: 22,
      },
    ],
  })
}

const handleResize = () => {
  frequencyChart?.resize()
}

watch(
  () => lotteryStore.statistics,
  () => {
    nextTick(() => renderFrequencyChart())
  },
  { deep: true },
)

onMounted(async () => {
  await Promise.all([lotteryStore.fetchLatest(), lotteryStore.fetchHistory(100), lotteryStore.fetchStatistics()])
  nextTick(() => renderFrequencyChart())
  window.addEventListener('resize', handleResize)
  setTimeout(() => {
    animate.value = true
  }, 100)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  frequencyChart?.dispose()
  frequencyChart = null
})
</script>

<style scoped>
.dashboard-page {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.section-title {
  font-size: 18px;
  color: var(--text-primary);
  margin-bottom: 16px;
}

.result-content {
  padding: 24px;
}

.frequency-chart {
  width: 100%;
  height: 280px;
}

.result-info {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
}

.period {
  font-size: 16px;
  color: var(--color-primary);
  font-weight: 500;
}

.date {
  color: var(--text-secondary);
}

.balls-container {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.loading-state,
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
}

.loading-text {
  color: var(--text-secondary);
  font-size: 16px;
}

.empty-text {
  color: var(--text-muted);
  font-size: 14px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.quick-actions {
  display: flex;
  gap: 16px;
}

.action-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 20px;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  color: var(--text-primary);
  text-decoration: none;
  transition: all 0.3s;
}

.action-btn:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-glow);
}

.action-btn.primary {
  background: var(--gradient-primary);
  color: #fff;
  border-color: transparent;
}

@media (max-width: 992px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .quick-actions {
    flex-direction: column;
  }
}
</style>
