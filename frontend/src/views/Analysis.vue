<template>
  <div class="analysis-page">
    <!-- 加载状态 -->
    <div v-if="loading"
         class="loading-overlay"
         role="status"
         aria-live="polite"
         aria-label="正在加载分析数据">
      <div class="loading-spinner" aria-hidden="true"></div>
      <span class="loading-text">正在加载分析数据...</span>
    </div>

    <div class="analysis-grid" :class="{ 'is-loading': loading }">
      <!-- 频率分析 -->
      <DataVBorder :type="1">
        <div class="chart-section">
          <h3 class="section-title">红球频率分析</h3>
          <div class="chart-container" ref="redChartRef"></div>
        </div>
      </DataVBorder>

      <DataVBorder :type="1">
        <div class="chart-section">
          <h3 class="section-title">蓝球频率分析</h3>
          <div class="chart-container" ref="blueChartRef"></div>
        </div>
      </DataVBorder>

      <DataVBorder :type="1" class="full-width">
        <div class="chart-section">
          <h3 class="section-title">走势分析（红球和值）</h3>
          <div class="chart-container trend-chart" ref="trendChartRef"></div>
        </div>
      </DataVBorder>

      <!-- 冷热号分析 -->
      <DataVBorder :type="2">
        <div class="stats-section">
          <h3 class="section-title">热门号码</h3>
          <div class="number-list hot">
            <div v-for="num in hotNumbers" :key="'hot-' + num" class="number-item">
              <LotteryBall :number="num" type="red" size="sm" />
              <span class="number-count">{{ getCount(num) }}次</span>
            </div>
          </div>
        </div>
      </DataVBorder>

      <DataVBorder :type="2">
        <div class="stats-section">
          <h3 class="section-title">冷门号码</h3>
          <div class="number-list cold">
            <div v-for="num in coldNumbers" :key="'cold-' + num" class="number-item">
              <LotteryBall :number="num" type="red" size="sm" />
              <span class="number-count">{{ getCount(num) }}次</span>
            </div>
          </div>
        </div>
      </DataVBorder>

      <!-- 遗漏分析 -->
      <DataVBorder :type="1" class="full-width">
        <div class="omission-section">
          <h3 class="section-title">遗漏值统计</h3>
          <div class="omission-grid">
            <div v-for="n in 33" :key="'o-' + n" class="omission-item">
              <LotteryBall :number="n" type="red" size="sm" />
              <span class="omission-value">{{ getOmission(n) }}</span>
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

const redChartRef = ref<HTMLElement | null>(null)
const blueChartRef = ref<HTMLElement | null>(null)
const trendChartRef = ref<HTMLElement | null>(null)

let redChart: echarts.ECharts | null = null
let blueChart: echarts.ECharts | null = null
let trendChart: echarts.ECharts | null = null
const appStore = useAppStore()

// 数据状态
const hotNumbers = ref<number[]>([])
const coldNumbers = ref<number[]>([])
const frequencyData = ref<Record<number, number>>({})
const blueFrequencyData = ref<Record<number, number>>({})
const omissionData = ref<Record<number, number>>({})
const trendData = ref<TrendPoint[]>([])
const loading = ref(true)

const getCount = (num: number) => frequencyData.value[num] || 0
const getOmission = (num: number) => omissionData.value[num] || 0

// 加载频率分析数据
const loadFrequencyData = async () => {
  try {
    const res = await getFrequencyAnalysis({ periods: 100 })
    if (res.success && res.data) {
      frequencyData.value = res.data.red_frequency || {}
      blueFrequencyData.value = res.data.blue_frequency || {}
      return
    }
    frequencyData.value = {}
    blueFrequencyData.value = {}
    appStore.notify.warning('频率分析接口返回空数据')
  } catch (error) {
    console.error('加载频率数据失败:', error)
    frequencyData.value = {}
    blueFrequencyData.value = {}
    appStore.notify.error('加载频率数据失败')
  }
}

// 加载冷热号数据
const loadHotColdData = async () => {
  try {
    const res = await getHotColdAnalysis(50)
    if (res.success && res.data) {
      const data = res.data as HotColdAnalysisResult
      hotNumbers.value = data.red_balls?.hot || []
      coldNumbers.value = data.red_balls?.cold || []
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

// 加载遗漏值数据
const loadMissingData = async () => {
  try {
    const res = await getMissingAnalysis(100)
    if (res.success && res.data) {
      const data = res.data as MissingAnalysisResult
      omissionData.value = data.red_missing || {}
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

// 加载走势数据
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

// 初始化图表
const initCharts = () => {
  // 红球频率图表
  if (redChartRef.value) {
    redChart = echarts.init(redChartRef.value)
    const redData = Array.from({ length: 33 }, (_, i) => ({
      name: String(i + 1).padStart(2, '0'),
      value: frequencyData.value[i + 1] || 0
    }))

    redChart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(19, 26, 53, 0.9)',
        borderColor: 'var(--border-color)',
        textStyle: { color: '#fff' }
      },
      xAxis: {
        type: 'category',
        data: redData.map(d => d.name),
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        axisLabel: { color: '#a0aec0' }
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        splitLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.1)' } },
        axisLabel: { color: '#a0aec0' }
      },
      series: [{
        type: 'bar',
        data: redData.map(d => d.value),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#ff8a8a' },
            { offset: 1, color: '#ff6b6b' }
          ])
        }
      }]
    })
  }

  // 蓝球频率图表
  if (blueChartRef.value) {
    blueChart = echarts.init(blueChartRef.value)
    const blueData = Array.from({ length: 16 }, (_, i) => ({
      name: String(i + 1).padStart(2, '0'),
      value: blueFrequencyData.value[i + 1] || 0
    }))

    blueChart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(19, 26, 53, 0.9)',
        borderColor: 'var(--border-color)',
        textStyle: { color: '#fff' }
      },
      xAxis: {
        type: 'category',
        data: blueData.map(d => d.name),
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        axisLabel: { color: '#a0aec0' }
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.3)' } },
        splitLine: { lineStyle: { color: 'rgba(0, 212, 255, 0.1)' } },
        axisLabel: { color: '#a0aec0' }
      },
      series: [{
        type: 'bar',
        data: blueData.map(d => d.value),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#6ee7e7' },
            { offset: 1, color: '#4ecdc4' }
          ])
        }
      }]
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
        data: chartPoints.map((item) => item.red_sum),
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
  redChart?.resize()
  blueChart?.resize()
  trendChart?.resize()
}

// 加载所有数据
const loadAllData = async () => {
  loading.value = true
  await Promise.all([
    loadFrequencyData(),
    loadHotColdData(),
    loadMissingData(),
    loadTrendData()
  ])
  loading.value = false
  initCharts()
}

onMounted(() => {
  loadAllData()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  redChart?.dispose()
  blueChart?.dispose()
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
  grid-template-columns: repeat(11, 1fr);
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
    grid-template-columns: repeat(6, 1fr);
  }
}
</style>
