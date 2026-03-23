<template>
  <div class="analysis-page">
    <div
      v-if="loading"
      class="loading-overlay"
      role="status"
      aria-live="polite"
      aria-label="正在加载分析数据"
    >
      <div class="loading-spinner" aria-hidden="true"></div>
      <span class="loading-text">正在加载分析数据...</span>
    </div>

    <div class="analysis-grid" :class="{ 'is-loading': loading }">
      <DataVBorder :type="1">
        <div class="chart-section">
          <h3 class="section-title">号码频率分析（01-80）</h3>
          <div class="chart-container" ref="frequencyChartRef"></div>
        </div>
      </DataVBorder>

      <DataVBorder :type="1">
        <div class="chart-section">
          <h3 class="section-title">冷热号码分布（Top10）</h3>
          <div class="chart-container" ref="hotColdChartRef"></div>
        </div>
      </DataVBorder>

      <DataVBorder :type="1" class="full-width">
        <div class="chart-section">
          <h3 class="section-title">走势分析（和值）</h3>
          <div class="chart-container trend-chart" ref="trendChartRef"></div>
        </div>
      </DataVBorder>

      <DataVBorder :type="2">
        <div class="stats-section">
          <h3 class="section-title">热门号码</h3>
          <div class="number-list hot">
            <div v-for="num in hotNumbers" :key="'hot-' + num" class="number-item">
              <LotteryBall :number="num" size="sm" />
              <span class="number-count">{{ getCount(num) }} 次</span>
            </div>
          </div>
        </div>
      </DataVBorder>

      <DataVBorder :type="2">
        <div class="stats-section">
          <h3 class="section-title">冷门号码</h3>
          <div class="number-list cold">
            <div v-for="num in coldNumbers" :key="'cold-' + num" class="number-item">
              <LotteryBall :number="num" size="sm" />
              <span class="number-count">{{ getCount(num) }} 次</span>
            </div>
          </div>
        </div>
      </DataVBorder>

      <DataVBorder :type="1" class="full-width">
        <div class="omission-section">
          <h3 class="section-title">遗漏值统计（01-80）</h3>
          <div class="omission-grid">
            <div v-for="num in numberPool" :key="'o-' + num" class="omission-item">
              <LotteryBall :number="num" size="sm" />
              <span class="omission-value">{{ getOmission(num) }}</span>
            </div>
          </div>
        </div>
      </DataVBorder>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useAppStore } from '@/stores/app'
import DataVBorder from '@/components/common/DataVBorder.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'
import { getFrequencyAnalysis, getHotColdAnalysis, getMissingAnalysis, getTrendsAnalysis } from '@/api/lottery'
import type { HotColdAnalysisResult, MissingAnalysisResult, TrendPoint } from '@/types/lottery'

const frequencyChartRef = ref<HTMLElement | null>(null)
const hotColdChartRef = ref<HTMLElement | null>(null)
const trendChartRef = ref<HTMLElement | null>(null)

let frequencyChart: echarts.ECharts | null = null
let hotColdChart: echarts.ECharts | null = null
let trendChart: echarts.ECharts | null = null

const appStore = useAppStore()

const numberPool = Array.from({ length: 80 }, (_, idx) => idx + 1)
const hotNumbers = ref<number[]>([])
const coldNumbers = ref<number[]>([])
const frequencyData = ref<Record<number, number>>({})
const omissionData = ref<Record<number, number>>({})
const trendData = ref<TrendPoint[]>([])
const loading = ref(true)

const getCount = (num: number) => frequencyData.value[num] || 0
const getOmission = (num: number) => omissionData.value[num] || 0

const loadFrequencyData = async () => {
  try {
    const res = await getFrequencyAnalysis({ periods: 100 })
    if (res.success && res.data) {
      frequencyData.value = res.data.frequency || {}
      return
    }
    frequencyData.value = {}
    appStore.notify.warning('频率分析接口返回空数据')
  } catch (error) {
    console.error('加载频率数据失败:', error)
    frequencyData.value = {}
    appStore.notify.error('加载频率数据失败')
  }
}

const loadHotColdData = async () => {
  try {
    const res = await getHotColdAnalysis(100)
    if (res.success && res.data) {
      const data = res.data as HotColdAnalysisResult
      hotNumbers.value = data.hot_numbers || []
      coldNumbers.value = data.cold_numbers || []
      if (data.frequency) {
        frequencyData.value = { ...frequencyData.value, ...data.frequency }
      }
      return
    }
    hotNumbers.value = []
    coldNumbers.value = []
    appStore.notify.warning('冷热号接口返回空数据')
  } catch (error) {
    console.error('加载冷热号数据失败:', error)
    hotNumbers.value = []
    coldNumbers.value = []
    appStore.notify.error('加载冷热号数据失败')
  }
}

const loadMissingData = async () => {
  try {
    const res = await getMissingAnalysis(100)
    if (res.success && res.data) {
      const data = res.data as MissingAnalysisResult
      omissionData.value = data.missing_map || {}
      return
    }
    omissionData.value = {}
    appStore.notify.warning('遗漏值接口返回空数据')
  } catch (error) {
    console.error('加载遗漏值数据失败:', error)
    omissionData.value = {}
    appStore.notify.error('加载遗漏值数据失败')
  }
}

const loadTrendData = async () => {
  try {
    const res = await getTrendsAnalysis({ periods: 100 })
    if (res.success && res.data) {
      trendData.value = res.data.trend_data || []
      return
    }
    trendData.value = []
    appStore.notify.warning('走势分析接口返回空数据')
  } catch (error) {
    console.error('加载走势数据失败:', error)
    trendData.value = []
    appStore.notify.error('加载走势数据失败')
  }
}

const initCharts = () => {
  frequencyChart?.dispose()
  hotColdChart?.dispose()
  trendChart?.dispose()

  if (frequencyChartRef.value) {
    frequencyChart = echarts.init(frequencyChartRef.value)
    const frequencyValues = numberPool.map((num) => frequencyData.value[num] || 0)

    frequencyChart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(19, 26, 53, 0.9)',
        borderColor: 'var(--border-color)',
        textStyle: { color: '#fff' }
      },
      grid: { left: 40, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: numberPool.map((num) => String(num).padStart(2, '0')),
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        axisLabel: { color: '#a0aec0', interval: 7 }
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        splitLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.1)' } },
        axisLabel: { color: '#a0aec0' }
      },
      series: [{
        type: 'bar',
        data: frequencyValues,
        barWidth: '72%',
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#00d4ff' },
            { offset: 1, color: '#4ecdc4' }
          ])
        }
      }]
    })
  }

  if (hotColdChartRef.value) {
    hotColdChart = echarts.init(hotColdChartRef.value)

    const hotSet = new Set(hotNumbers.value)
    const coldSet = new Set(coldNumbers.value)

    const hotSeries = numberPool.map((num) => (hotSet.has(num) ? (frequencyData.value[num] || 0) : 0))
    const coldSeries = numberPool.map((num) => (coldSet.has(num) ? (frequencyData.value[num] || 0) : 0))

    hotColdChart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(19, 26, 53, 0.9)',
        borderColor: 'var(--border-color)',
        textStyle: { color: '#fff' }
      },
      legend: {
        data: ['热门号码', '冷门号码'],
        top: 8,
        textStyle: { color: '#a0aec0' }
      },
      grid: { left: 40, right: 20, top: 46, bottom: 40 },
      xAxis: {
        type: 'category',
        data: numberPool.map((num) => String(num).padStart(2, '0')),
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        axisLabel: { color: '#a0aec0', interval: 7 }
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        splitLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.1)' } },
        axisLabel: { color: '#a0aec0' }
      },
      series: [
        {
          name: '热门号码',
          type: 'line',
          smooth: true,
          data: hotSeries,
          lineStyle: { color: '#ff6b6b', width: 2 },
          itemStyle: { color: '#ff6b6b' },
        },
        {
          name: '冷门号码',
          type: 'line',
          smooth: true,
          data: coldSeries,
          lineStyle: { color: '#ffd93d', width: 2 },
          itemStyle: { color: '#ffd93d' },
        }
      ]
    })
  }

  if (trendChartRef.value) {
    trendChart = echarts.init(trendChartRef.value)
    const chartPoints = [...trendData.value].reverse()
    trendChart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(19, 26, 53, 0.9)',
        borderColor: 'var(--border-color)',
        textStyle: { color: '#fff' }
      },
      xAxis: {
        type: 'category',
        data: chartPoints.map((item) => item.period),
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        axisLabel: { color: '#a0aec0', interval: 'auto' }
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        splitLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.1)' } },
        axisLabel: { color: '#a0aec0' }
      },
      series: [{
        type: 'line',
        smooth: true,
        data: chartPoints.map((item) => item.sum_value),
        lineStyle: { color: '#ffd93d', width: 2 },
        itemStyle: { color: '#ffd93d' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(255, 217, 61, 0.35)' },
            { offset: 1, color: 'rgba(255, 217, 61, 0.02)' }
          ])
        }
      }]
    })
  }
}

const handleResize = () => {
  frequencyChart?.resize()
  hotColdChart?.resize()
  trendChart?.resize()
}

const loadAllData = async () => {
  loading.value = true
  await Promise.all([loadFrequencyData(), loadHotColdData(), loadMissingData(), loadTrendData()])
  loading.value = false
  initCharts()
}

onMounted(() => {
  loadAllData()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  frequencyChart?.dispose()
  hotColdChart?.dispose()
  trendChart?.dispose()
})
</script>

<style scoped>
.analysis-page {
  padding: 24px;
  position: relative;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(15, 23, 42, 0.8);
  z-index: 100;
  border-radius: 8px;
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 4px solid rgba(0, 212, 255, 0.2);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  transform: translateZ(0);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-spinner {
    animation: none;
    opacity: 0.8;
  }
}

.loading-text {
  margin-top: 16px;
  color: var(--text-secondary);
  font-size: 14px;
}

.analysis-grid.is-loading {
  opacity: 0.3;
  pointer-events: none;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
  transition: opacity 0.3s;
}

.full-width {
  grid-column: 1 / -1;
}

.section-title {
  font-size: 16px;
  color: var(--text-primary);
  margin-bottom: 16px;
}

.chart-section,
.stats-section,
.omission-section {
  padding: 20px;
}

.chart-container {
  height: 300px;
}

.number-list {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.number-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.number-count {
  font-size: 12px;
  color: var(--text-secondary);
}

.omission-grid {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 12px;
}

.omission-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.omission-value {
  font-size: 12px;
  color: var(--color-warning);
}

@media (max-width: 1024px) {
  .analysis-grid {
    grid-template-columns: 1fr;
  }

  .omission-grid {
    grid-template-columns: repeat(5, 1fr);
  }
}
</style>
