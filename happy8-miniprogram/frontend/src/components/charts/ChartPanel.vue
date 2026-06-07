<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import type { ECharts, EChartsOption } from 'echarts'
import { Download, Maximize2, Minimize2 } from 'lucide-vue-next'

const props = withDefaults(
  defineProps<{
    title: string
    subtitle?: string
    summary: string
    option: EChartsOption
    loading?: boolean
    empty?: boolean
    error?: string
  }>(),
  {
    subtitle: '',
    loading: false,
    empty: false,
    error: '',
  },
)

const chartRef = ref<HTMLDivElement | null>(null)
const mobileExpanded = ref(false)
let chart: ECharts | null = null
let resizeObserver: ResizeObserver | null = null

const canRender = computed(() => !props.loading && !props.empty && !props.error)

const stateText = computed(() => {
  if (props.loading) {
    return '正在读取历史开奖数据'
  }

  if (props.error) {
    return props.error
  }

  if (props.empty) {
    return '当前筛选范围暂无开奖记录'
  }

  return ''
})

function ensureChart() {
  if (!chartRef.value || !canRender.value) {
    return
  }

  if (!chart) {
    chart = echarts.init(chartRef.value)
  }

  chart.setOption(props.option, true)
}

function resizeChart() {
  chart?.resize()
}

function toggleMobileChart() {
  mobileExpanded.value = !mobileExpanded.value
  void nextTick(resizeChart)
}

function exportImage() {
  if (!chart) {
    return
  }

  const imageUrl = chart.getDataURL({
    type: 'png',
    pixelRatio: 2,
    backgroundColor: '#fff',
  })
  const link = document.createElement('a')
  link.href = imageUrl
  link.download = `${props.title}.png`
  link.click()
}

watch(
  () => [props.option, props.loading, props.empty, props.error],
  () => {
    if (!canRender.value) {
      chart?.clear()
      return
    }

    void nextTick(ensureChart)
  },
  { deep: true },
)

onMounted(() => {
  ensureChart()

  if (chartRef.value) {
    resizeObserver = new ResizeObserver(resizeChart)
    resizeObserver.observe(chartRef.value)
  }
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  chart?.dispose()
  chart = null
})
</script>

<template>
  <article class="chart-panel">
    <header class="chart-panel__header">
      <div>
        <h3>{{ title }}</h3>
        <p v-if="subtitle">{{ subtitle }}</p>
      </div>
      <div class="chart-panel__actions">
        <button
          class="chart-panel__icon-button chart-panel__mobile-toggle"
          type="button"
          :aria-label="mobileExpanded ? '收起图表' : '展开图表'"
          @click="toggleMobileChart"
        >
          <Minimize2 v-if="mobileExpanded" :size="16" aria-hidden="true" />
          <Maximize2 v-else :size="16" aria-hidden="true" />
        </button>
        <button
          class="chart-panel__icon-button"
          type="button"
          :disabled="!canRender"
          aria-label="导出图表图片"
          @click="exportImage"
        >
          <Download :size="16" aria-hidden="true" />
        </button>
      </div>
    </header>

    <p class="chart-panel__summary">{{ summary }}</p>

    <div v-if="stateText" class="chart-panel__state">
      {{ stateText }}
    </div>
    <div
      v-else
      ref="chartRef"
      class="chart-panel__canvas"
      :class="{ 'chart-panel__canvas--expanded': mobileExpanded }"
      role="img"
      :aria-label="`${title}：${summary}`"
    />
  </article>
</template>

<style scoped>
.chart-panel {
  display: grid;
  min-height: 360px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 16px;
}

.chart-panel__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.chart-panel h3 {
  margin: 0;
  color: var(--h8-color-text);
  font-size: 16px;
  line-height: 1.3;
}

.chart-panel p {
  margin: 4px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.45;
}

.chart-panel__actions {
  display: flex;
  gap: 6px;
}

.chart-panel__icon-button {
  display: inline-grid;
  width: 30px;
  height: 30px;
  place-items: center;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text-muted);
  cursor: pointer;
}

.chart-panel__icon-button:hover:not(:disabled) {
  border-color: var(--h8-color-cinnabar);
  color: var(--h8-color-cinnabar);
}

.chart-panel__icon-button:focus-visible {
  outline: 0;
  box-shadow: var(--h8-focus-ring);
}

.chart-panel__icon-button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.chart-panel__mobile-toggle {
  display: none;
}

.chart-panel__summary {
  min-height: 38px;
}

.chart-panel__state {
  display: grid;
  min-height: 250px;
  place-items: center;
  border: 1px dashed var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  color: var(--h8-color-text-muted);
  font-size: 14px;
}

.chart-panel__canvas {
  width: 100%;
  min-height: 270px;
}

@media (max-width: 760px) {
  .chart-panel {
    min-height: auto;
  }

  .chart-panel__mobile-toggle {
    display: inline-grid;
  }

  .chart-panel__summary {
    min-height: auto;
    border-top: 1px solid var(--h8-color-line);
    padding-top: 10px;
  }

  .chart-panel__canvas {
    height: 0;
    min-height: 0;
    overflow: hidden;
    opacity: 0;
    transition:
      height 180ms ease,
      opacity 180ms ease;
  }

  .chart-panel__canvas--expanded {
    height: 260px;
    min-height: 260px;
    opacity: 1;
  }
}
</style>
