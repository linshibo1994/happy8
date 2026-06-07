<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { EChartsOption } from 'echarts'
import {
  BarChart3,
  Database,
  Filter,
  Gauge,
  Grid2X2,
  RefreshCcw,
  RotateCcw,
  Snowflake,
  ThermometerSun,
} from 'lucide-vue-next'

import NumberBall from '@/components/balls/NumberBall.vue'
import ChartPanel from '@/components/charts/ChartPanel.vue'
import HistoryDrawTable from '@/components/data-table/HistoryDrawTable.vue'
import NumberStatsTable from '@/components/data-table/NumberStatsTable.vue'
import ReplayTable from '@/components/data-table/ReplayTable.vue'
import { useAlgorithmStore } from '@/stores/algorithm'
import { useLotteryStore } from '@/stores/lottery'
import { useUiStore } from '@/stores/ui'
import type { LotteryResult } from '@/types'

type PeriodOption = 30 | 50 | 100 | 200
type Density = 'comfortable' | 'compact'
type StatisticType = 'overview' | 'frequency' | 'missing' | 'structure'
type TableMode = 'history' | 'numbers' | 'replay'
type NumberLevel = 'hot' | 'cold' | 'normal'

interface NumberStatsRow {
  number: number
  count: number
  rate: number
  currentMissing: number
  maxMissing: number
  latestIssue: string
  zone: string
  level: NumberLevel
}

interface ReplayRow {
  id: string
  predictedAt: string
  targetIssue: string
  algorithmName: string
  predictedNumbers: number[]
  actualNumbers: number[]
  hits: number[]
  confidence: number
  elapsedMs: number
}

const periodOptions: PeriodOption[] = [30, 50, 100, 200]
const statisticTypes: Array<{ key: StatisticType; label: string }> = [
  { key: 'overview', label: '总览' },
  { key: 'frequency', label: '频率' },
  { key: 'missing', label: '遗漏' },
  { key: 'structure', label: '结构' },
]
const tableModes: Array<{ key: TableMode; label: string }> = [
  { key: 'history', label: '历史开奖表' },
  { key: 'numbers', label: '号码统计表' },
  { key: 'replay', label: '复盘表' },
]
const zoneLabels = ['1-20', '21-40', '41-60', '61-80'] as const
const allNumbers = Array.from({ length: 80 }, (_, index) => index + 1)
const cinnabar = '#C9352B'
const ink = '#171A1F'
const bronze = '#B88A3B'
const dataBlue = '#276EF1'
const turquoise = '#2E8B6D'
const riskOrange = '#D9822B'
const lineColor = '#E3DDD2'

const route = useRoute()
const router = useRouter()
const algorithmStore = useAlgorithmStore()
const lotteryStore = useLotteryStore()
const uiStore = useUiStore()

const latestDate = new Date(lotteryStore.latestResult.openedAt)
const defaultStartDate = new Date(latestDate)
defaultStartDate.setDate(latestDate.getDate() - 199)

const dateStart = ref(queryString(route.query.start) ?? formatDateInput(defaultStartDate))
const dateEnd = ref(queryString(route.query.end) ?? formatDateInput(latestDate))
const selectedPeriod = ref<PeriodOption>(parsePeriod(queryString(route.query.period)))
const selectedStatisticType = ref<StatisticType>(parseStatisticType(queryString(route.query.stat)))
const selectedNumbers = ref<number[]>(parseNumberList(queryString(route.query.numbers)))
const selectedAlgorithmNames = ref<string[]>(
  parseAlgorithmList(
    queryString(route.query.algos),
    algorithmStore.enabledAlgorithms.map((algorithm) => algorithm.name),
  ),
)
const dashboardDensity = ref<Density>(parseDensity(queryString(route.query.density), uiStore.density))
const activeTableMode = ref<TableMode>('history')
const isLoading = ref(true)
const errorMessage = ref('')
let loadingTimer: number | undefined

const enabledAlgorithms = computed(() => algorithmStore.enabledAlgorithms)
const historySeries = computed(() => createHistorySeries(lotteryStore.latestResult, 240))

const filteredHistory = computed(() => {
  const start = new Date(`${dateStart.value}T00:00:00`)
  const end = new Date(`${dateEnd.value}T23:59:59`)

  return historySeries.value
    .filter((draw) => {
      const openedAt = new Date(draw.openedAt)
      const inDateRange = openedAt >= start && openedAt <= end
      const includesNumbers =
        selectedNumbers.value.length === 0 ||
        selectedNumbers.value.every((number) => draw.numbers.includes(number))

      return inDateRange && includesNumbers
    })
    .slice(0, selectedPeriod.value)
})

const chronologicalHistory = computed(() => [...filteredHistory.value].reverse())

const numberStats = computed<NumberStatsRow[]>(() => {
  const rows = filteredHistory.value
  const rawRows = allNumbers.map((number) => {
    const includedRows = rows.filter((draw) => draw.numbers.includes(number))
    return {
      number,
      count: includedRows.length,
      rate: rows.length > 0 ? includedRows.length / rows.length : 0,
      currentMissing: currentMissing(rows, number),
      maxMissing: maxMissing(rows, number),
      latestIssue: includedRows[0]?.issue ?? '',
      zone: zoneOfNumber(number),
      level: 'normal' as NumberLevel,
    }
  })

  const hotNumbers = new Set(
    [...rawRows]
      .sort((left, right) => right.count - left.count || left.number - right.number)
      .slice(0, 8)
      .map((row) => row.number),
  )
  const coldNumbers = new Set(
    [...rawRows]
      .sort((left, right) => left.count - right.count || right.currentMissing - left.currentMissing)
      .slice(0, 8)
      .map((row) => row.number),
  )

  return rawRows.map((row) => ({
    ...row,
    level: hotNumbers.has(row.number) ? 'hot' : coldNumbers.has(row.number) ? 'cold' : 'normal',
  }))
})

const hotTop = computed(() => topNumbers(numberStats.value, 'hot'))
const coldTop = computed(() => topNumbers(numberStats.value, 'cold'))
const maxMissingRow = computed(() => {
  return [...numberStats.value].sort((left, right) => right.maxMissing - left.maxMissing || left.number - right.number)[0]
})

const averageSum = computed(() => {
  if (filteredHistory.value.length === 0) {
    return 0
  }

  const total = filteredHistory.value.reduce((sum, draw) => sum + draw.sum, 0)
  return Math.round(total / filteredHistory.value.length)
})

const zoneTotals = computed(() => {
  return filteredHistory.value.reduce<Record<string, number>>(
    (totals, draw) => {
      zoneLabels.forEach((zone) => {
        totals[zone] += draw.zoneDistribution[zone] ?? 0
      })
      return totals
    },
    { '1-20': 0, '21-40': 0, '41-60': 0, '61-80': 0 },
  )
})

const zoneBiasText = computed(() => {
  if (filteredHistory.value.length === 0) {
    return '无数据'
  }

  const expected = filteredHistory.value.length * 5
  const [zone, count] = Object.entries(zoneTotals.value).sort(
    (left, right) => Math.abs(right[1] - expected) - Math.abs(left[1] - expected),
  )[0]
  const deviation = expected === 0 ? 0 : ((count - expected) / expected) * 100

  return `${zone} ${deviation >= 0 ? '+' : ''}${deviation.toFixed(1)}%`
})

const dashboardMetrics = computed(() => [
  {
    key: 'issues',
    label: '开奖期数',
    value: `${filteredHistory.value.length}`,
    detail: `筛选窗口 ${selectedPeriod.value} 期`,
    icon: Database,
  },
  {
    key: 'sum',
    label: '平均和值',
    value: `${averageSum.value}`,
    detail: '按当前筛选范围计算',
    icon: Gauge,
  },
  {
    key: 'hot',
    label: '热号Top',
    value: hotTop.value.slice(0, 3).map((row) => padNumber(row.number)).join(' '),
    detail: hotTop.value.slice(0, 3).map((row) => `${row.count}次`).join(' / ') || '无数据',
    icon: ThermometerSun,
    rows: hotTop.value.slice(0, 3),
  },
  {
    key: 'cold',
    label: '冷号Top',
    value: coldTop.value.slice(0, 3).map((row) => padNumber(row.number)).join(' '),
    detail: coldTop.value.slice(0, 3).map((row) => `漏${row.currentMissing}`).join(' / ') || '无数据',
    icon: Snowflake,
    rows: coldTop.value.slice(0, 3),
  },
  {
    key: 'missing',
    label: '最大遗漏',
    value: maxMissingRow.value ? `${padNumber(maxMissingRow.value.number)}` : '无',
    detail: maxMissingRow.value ? `${maxMissingRow.value.maxMissing} 期` : '无数据',
    icon: BarChart3,
    rows: maxMissingRow.value ? [maxMissingRow.value] : [],
  },
  {
    key: 'zone',
    label: '区间偏态',
    value: zoneBiasText.value,
    detail: '相对理论均值偏离',
    icon: Grid2X2,
  },
])

const historyRows = computed(() =>
  filteredHistory.value.map((draw) => ({
    issue: draw.issue,
    openedAt: draw.openedAt,
    numbers: draw.numbers,
    sum: draw.sum,
    oddEvenText: `${draw.oddCount}:${draw.evenCount}`,
    bigSmallText: `${draw.bigCount}:${draw.smallCount}`,
    zoneDistribution: draw.zoneDistribution,
  })),
)

const replayRows = computed<ReplayRow[]>(() => {
  const algorithmMap = new Map(enabledAlgorithms.value.map((algorithm) => [algorithm.name, algorithm.displayName]))
  const activeAlgorithms =
    selectedAlgorithmNames.value.length > 0
      ? selectedAlgorithmNames.value
      : enabledAlgorithms.value.map((item) => item.name)

  return filteredHistory.value.slice(0, 16).map((draw, index) => {
    const algorithmName = activeAlgorithms[index % activeAlgorithms.length] ?? 'frequency'
    const predictedNumbers = buildPredictedNumbers(draw, index)
    const hits = predictedNumbers.filter((number) => draw.numbers.includes(number))
    const predictedAt = new Date(draw.openedAt)
    predictedAt.setHours(predictedAt.getHours() - 3 - index)

    return {
      id: `${draw.issue}-${algorithmName}`,
      predictedAt: predictedAt.toISOString(),
      targetIssue: draw.issue,
      algorithmName: algorithmMap.get(algorithmName) ?? algorithmName,
      predictedNumbers,
      actualNumbers: draw.numbers,
      hits,
      confidence: 0.52 + ((index % 8) * 0.03),
      elapsedMs: 820 + ((index * 173) % 2400),
    }
  })
})

const selectedNumbersLabel = computed(() => {
  if (selectedNumbers.value.length === 0) {
    return '未限定号码'
  }

  return selectedNumbers.value.map(padNumber).join('、')
})

const analysisHint = computed(() => {
  const hintMap: Record<StatisticType, string> = {
    overview: '当前总览同步展示频率、和值、结构和遗漏摘要。',
    frequency: '频率视角重点观察热号、冷号和出现率变化。',
    missing: '遗漏视角优先比较当前遗漏与历史最大遗漏。',
    structure: '结构视角关注奇偶、大小和四区间偏态。',
  }

  return hintMap[selectedStatisticType.value]
})

const heatmapSummary = computed(() => {
  const top = hotTop.value[0]
  if (!top) {
    return '暂无热力数据。'
  }

  return `${padNumber(top.number)} 出现 ${top.count} 次，为当前筛选范围最高频号码；已选号码：${selectedNumbersLabel.value}。`
})

const sumTrendSummary = computed(() => {
  const latest = filteredHistory.value[0]
  if (!latest) {
    return '暂无和值走势数据。'
  }

  return `最新一期和值 ${latest.sum}，筛选窗口平均和值 ${averageSum.value}。`
})

const structureTrendSummary = computed(() => {
  const latest = filteredHistory.value[0]
  if (!latest) {
    return '暂无奇偶大小走势数据。'
  }

  return `最新一期奇偶 ${latest.oddCount}:${latest.evenCount}，大小 ${latest.bigCount}:${latest.smallCount}。`
})

const zoneSummary = computed(() => {
  return `四区间累计分布：${zoneLabels.map((zone) => `${zone} ${zoneTotals.value[zone]}`).join('，')}。`
})

const missingSummary = computed(() => {
  const targetRows = missingChartRows.value.slice(0, 3)
  if (targetRows.length === 0) {
    return '暂无遗漏摘要数据。'
  }

  return `遗漏靠前号码：${targetRows.map((row) => `${padNumber(row.number)} 当前漏${row.currentMissing}`).join('，')}。`
})

const heatmapOption = computed<EChartsOption>(() => ({
  tooltip: {
    trigger: 'item',
    formatter: (params: unknown) => {
      const data = (params as { data?: [number, number, number, number] }).data
      if (!data) {
        return ''
      }
      return `号码 ${padNumber(data[3])}<br/>出现 ${data[2]} 次`
    },
  },
  grid: { top: 20, right: 18, bottom: 34, left: 52 },
  xAxis: {
    type: 'category',
    data: Array.from({ length: 20 }, (_, index) => String(index + 1)),
    axisLine: { lineStyle: { color: lineColor } },
    axisLabel: { color: '#6B7280' },
  },
  yAxis: {
    type: 'category',
    data: zoneLabels,
    axisLine: { lineStyle: { color: lineColor } },
    axisLabel: { color: '#6B7280' },
  },
  visualMap: {
    min: 0,
    max: Math.max(1, ...numberStats.value.map((row) => row.count)),
    orient: 'horizontal',
    left: 'center',
    bottom: 0,
    itemWidth: 12,
    itemHeight: 90,
    calculable: false,
    inRange: {
      color: ['#F7F3EA', '#E9B7A9', cinnabar],
    },
  },
  series: [
    {
      type: 'heatmap',
      data: numberStats.value.map((row) => [
        (row.number - 1) % 20,
        Math.floor((row.number - 1) / 20),
        row.count,
        row.number,
      ]),
      itemStyle: {
        borderColor: '#fff',
        borderWidth: 2,
        borderRadius: 3,
      },
      emphasis: {
        itemStyle: {
          borderColor: ink,
          borderWidth: 1,
        },
      },
    },
  ],
}))

const sumTrendOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  grid: { top: 24, right: 18, bottom: 34, left: 46 },
  xAxis: {
    type: 'category',
    data: chronologicalHistory.value.map((draw) => draw.issue.slice(-3)),
    axisLine: { lineStyle: { color: lineColor } },
    axisLabel: { color: '#6B7280' },
  },
  yAxis: {
    type: 'value',
    min: 650,
    max: 950,
    splitLine: { lineStyle: { color: lineColor, type: 'dashed' } },
    axisLabel: { color: '#6B7280' },
  },
  series: [
    {
      name: '和值',
      type: 'line',
      smooth: true,
      symbolSize: 5,
      lineStyle: { color: dataBlue, width: 2 },
      itemStyle: { color: dataBlue },
      areaStyle: { color: 'rgba(39, 110, 241, 0.1)' },
      data: chronologicalHistory.value.map((draw) => draw.sum),
      markLine: {
        symbol: 'none',
        lineStyle: { color: bronze, type: 'dashed' },
        data: [{ yAxis: averageSum.value, name: '平均和值' }],
      },
    },
  ],
}))

const structureTrendOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  legend: { top: 0, textStyle: { color: '#6B7280' } },
  grid: { top: 38, right: 18, bottom: 34, left: 34 },
  xAxis: {
    type: 'category',
    data: chronologicalHistory.value.map((draw) => draw.issue.slice(-3)),
    axisLine: { lineStyle: { color: lineColor } },
    axisLabel: { color: '#6B7280' },
  },
  yAxis: {
    type: 'value',
    min: 0,
    max: 20,
    splitLine: { lineStyle: { color: lineColor, type: 'dashed' } },
    axisLabel: { color: '#6B7280' },
  },
  series: [
    buildLineSeries('奇数', chronologicalHistory.value.map((draw) => draw.oddCount), cinnabar),
    buildLineSeries('偶数', chronologicalHistory.value.map((draw) => draw.evenCount), dataBlue),
    buildLineSeries('大号', chronologicalHistory.value.map((draw) => draw.bigCount), bronze),
    buildLineSeries('小号', chronologicalHistory.value.map((draw) => draw.smallCount), turquoise),
  ],
}))

const zoneDistributionOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  grid: { top: 26, right: 18, bottom: 34, left: 44 },
  xAxis: {
    type: 'category',
    data: [...zoneLabels],
    axisLine: { lineStyle: { color: lineColor } },
    axisLabel: { color: '#6B7280' },
  },
  yAxis: {
    type: 'value',
    splitLine: { lineStyle: { color: lineColor, type: 'dashed' } },
    axisLabel: { color: '#6B7280' },
  },
  series: [
    {
      name: '累计出现',
      type: 'bar',
      barWidth: 34,
      itemStyle: {
        color: (params: { dataIndex: number }) => [cinnabar, dataBlue, turquoise, bronze][params.dataIndex],
        borderRadius: [5, 5, 0, 0],
      },
      data: zoneLabels.map((zone) => zoneTotals.value[zone]),
    },
  ],
}))

const missingChartRows = computed(() => {
  const selectedSet = new Set(selectedNumbers.value)
  const sourceRows =
    selectedNumbers.value.length > 0
      ? numberStats.value.filter((row) => selectedSet.has(row.number)).slice(0, 5)
      : [...numberStats.value].sort((left, right) => right.currentMissing - left.currentMissing).slice(0, 10)

  return sourceRows
})

const missingOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  legend: { top: 0, textStyle: { color: '#6B7280' } },
  grid: { top: 38, right: 18, bottom: 34, left: 42 },
  xAxis: {
    type: 'category',
    data: missingChartRows.value.map((row) => padNumber(row.number)),
    axisLine: { lineStyle: { color: lineColor } },
    axisLabel: { color: '#6B7280' },
  },
  yAxis: {
    type: 'value',
    splitLine: { lineStyle: { color: lineColor, type: 'dashed' } },
    axisLabel: { color: '#6B7280' },
  },
  series: [
    {
      name: '当前遗漏',
      type: 'bar',
      barWidth: 18,
      itemStyle: { color: riskOrange, borderRadius: [4, 4, 0, 0] },
      data: missingChartRows.value.map((row) => row.currentMissing),
    },
    {
      name: '最大遗漏',
      type: 'line',
      symbolSize: 6,
      lineStyle: { color: dataBlue, width: 2 },
      itemStyle: { color: dataBlue },
      data: missingChartRows.value.map((row) => row.maxMissing),
    },
  ],
}))

watch(
  () => [
    dateStart.value,
    dateEnd.value,
    selectedPeriod.value,
    selectedStatisticType.value,
    selectedNumbers.value.join(','),
    selectedAlgorithmNames.value.join(','),
    dashboardDensity.value,
  ],
  () => {
    uiStore.setDensity(dashboardDensity.value)
    void router.replace({
      query: {
        ...route.query,
        start: dateStart.value,
        end: dateEnd.value,
        period: String(selectedPeriod.value),
        stat: selectedStatisticType.value,
        numbers: selectedNumbers.value.join(',') || undefined,
        algos: selectedAlgorithmNames.value.join(',') || undefined,
        density: dashboardDensity.value,
      },
    })
  },
)

onMounted(() => {
  loadingTimer = window.setTimeout(() => {
    isLoading.value = false
  }, 220)
})

onBeforeUnmount(() => {
  if (loadingTimer) {
    window.clearTimeout(loadingTimer)
  }
})

function queryString(value: unknown) {
  if (Array.isArray(value)) {
    return value[0]?.toString()
  }

  return typeof value === 'string' ? value : undefined
}

function parsePeriod(value: string | undefined): PeriodOption {
  const parsed = Number(value)
  return periodOptions.includes(parsed as PeriodOption) ? (parsed as PeriodOption) : 100
}

function parseStatisticType(value: string | undefined): StatisticType {
  return statisticTypes.some((item) => item.key === value) ? (value as StatisticType) : 'overview'
}

function parseDensity(value: string | undefined, fallback: Density): Density {
  return value === 'compact' || value === 'comfortable' ? value : fallback
}

function parseNumberList(value: string | undefined) {
  if (!value) {
    return []
  }

  return Array.from(
    new Set(
      value
        .split(',')
        .map((item) => Number(item))
        .filter((number) => Number.isInteger(number) && number >= 1 && number <= 80),
    ),
  ).sort((left, right) => left - right)
}

function parseAlgorithmList(value: string | undefined, fallbackNames: string[]) {
  const fallback = fallbackNames.slice(0, 3)
  if (!value) {
    return fallback
  }

  const validNames = new Set(fallbackNames)
  const parsed = value.split(',').filter((name) => validNames.has(name))
  return parsed.length > 0 ? parsed : fallback
}

function formatDateInput(date: Date) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function padNumber(number: number) {
  return String(number).padStart(2, '0')
}

function zoneOfNumber(number: number) {
  if (number <= 20) {
    return '1-20'
  }
  if (number <= 40) {
    return '21-40'
  }
  if (number <= 60) {
    return '41-60'
  }
  return '61-80'
}

function buildZoneDistribution(numbers: number[]) {
  return numbers.reduce<Record<string, number>>(
    (distribution, number) => {
      distribution[zoneOfNumber(number)] += 1
      return distribution
    },
    { '1-20': 0, '21-40': 0, '41-60': 0, '61-80': 0 },
  )
}

function createHistorySeries(latestResult: LotteryResult, count: number): LotteryResult[] {
  return Array.from({ length: count }, (_, index) => {
    if (index === 0) {
      return latestResult
    }

    const numbers = buildSeededNumbers(index)
    const openedAt = new Date(latestResult.openedAt)
    openedAt.setDate(openedAt.getDate() - index)
    const issueNumber = Math.max(1, Number(latestResult.issue) - index)

    return {
      issue: String(issueNumber).padStart(latestResult.issue.length, '0'),
      numbers,
      openedAt: openedAt.toISOString(),
      sum: numbers.reduce((sum, number) => sum + number, 0),
      oddCount: numbers.filter((number) => number % 2 === 1).length,
      evenCount: numbers.filter((number) => number % 2 === 0).length,
      bigCount: numbers.filter((number) => number > 40).length,
      smallCount: numbers.filter((number) => number <= 40).length,
      zoneDistribution: buildZoneDistribution(numbers),
    }
  })
}

function buildSeededNumbers(seed: number) {
  return allNumbers
    .map((number) => ({
      number,
      score: pseudoRandomScore(number, seed),
    }))
    .sort((left, right) => left.score - right.score)
    .slice(0, 20)
    .map((item) => item.number)
    .sort((left, right) => left - right)
}

function pseudoRandomScore(number: number, seed: number) {
  const raw = Math.sin((seed + 17) * (number + 3) * 12.9898 + number * 78.233) * 43758.5453
  return raw - Math.floor(raw)
}

function currentMissing(rows: LotteryResult[], number: number) {
  const index = rows.findIndex((draw) => draw.numbers.includes(number))
  return index >= 0 ? index : rows.length
}

function maxMissing(rows: LotteryResult[], number: number) {
  let current = 0
  let max = 0

  rows.forEach((draw) => {
    if (draw.numbers.includes(number)) {
      max = Math.max(max, current)
      current = 0
      return
    }

    current += 1
  })

  return Math.max(max, current)
}

function topNumbers(rows: NumberStatsRow[], type: 'hot' | 'cold') {
  const sorted = [...rows].sort((left, right) => {
    if (type === 'hot') {
      return right.count - left.count || left.number - right.number
    }

    return left.count - right.count || right.currentMissing - left.currentMissing || left.number - right.number
  })

  return sorted.slice(0, 5)
}

function buildPredictedNumbers(draw: LotteryResult, index: number) {
  const hitCount = 3 + (index % 5)
  const hits = draw.numbers.slice(index % 6, (index % 6) + hitCount)
  const candidates = buildSeededNumbers(index + 311).filter((number) => !hits.includes(number))

  return [...new Set([...hits, ...candidates])].slice(0, 10).sort((left, right) => left - right)
}

function buildLineSeries(name: string, data: number[], color: string) {
  return {
    name,
    type: 'line' as const,
    smooth: true,
    symbolSize: 4,
    lineStyle: { color, width: 2 },
    itemStyle: { color },
    data,
  }
}

function toggleNumber(number: number) {
  if (selectedNumbers.value.includes(number)) {
    selectedNumbers.value = selectedNumbers.value.filter((item) => item !== number)
    return
  }

  selectedNumbers.value = [...selectedNumbers.value, number].sort((left, right) => left - right)
}

function selectHotNumbers() {
  selectedNumbers.value = hotTop.value.slice(0, 5).map((row) => row.number)
}

function selectColdNumbers() {
  selectedNumbers.value = coldTop.value.slice(0, 5).map((row) => row.number)
}

function toggleAlgorithm(name: string) {
  if (selectedAlgorithmNames.value.includes(name)) {
    selectedAlgorithmNames.value = selectedAlgorithmNames.value.filter((item) => item !== name)
    return
  }

  selectedAlgorithmNames.value = [...selectedAlgorithmNames.value, name]
}

function resetFilters() {
  dateStart.value = formatDateInput(defaultStartDate)
  dateEnd.value = formatDateInput(latestDate)
  selectedPeriod.value = 100
  selectedStatisticType.value = 'overview'
  selectedNumbers.value = []
  selectedAlgorithmNames.value = enabledAlgorithms.value.slice(0, 3).map((algorithm) => algorithm.name)
  dashboardDensity.value = 'comfortable'
  errorMessage.value = ''
}

function refreshData() {
  errorMessage.value = ''
  isLoading.value = true
  if (loadingTimer) {
    window.clearTimeout(loadingTimer)
  }
  loadingTimer = window.setTimeout(() => {
    isLoading.value = false
  }, 320)
}
</script>

<template>
  <section class="data-dashboard" aria-labelledby="data-dashboard-title">
    <div class="data-dashboard__intro">
      <span class="section-kicker">历史分析</span>
      <h2 id="data-dashboard-title">历史数据看板</h2>
      <p>{{ analysisHint }}</p>
    </div>

    <section class="data-dashboard__filters" aria-label="历史数据筛选">
      <div class="filter-group filter-group--dates">
        <span class="filter-group__label">
          <Filter :size="15" aria-hidden="true" />
          日期范围
        </span>
        <label>
          <span>开始</span>
          <input v-model="dateStart" type="date" />
        </label>
        <label>
          <span>结束</span>
          <input v-model="dateEnd" type="date" />
        </label>
      </div>

      <div class="filter-group">
        <span class="filter-group__label">分析期数</span>
        <div class="segmented-control" role="group" aria-label="分析期数">
          <button
            v-for="period in periodOptions"
            :key="period"
            type="button"
            :aria-pressed="selectedPeriod === period"
            @click="selectedPeriod = period"
          >
            {{ period }}
          </button>
        </div>
      </div>

      <div class="filter-group">
        <span class="filter-group__label">统计类型</span>
        <div class="segmented-control" role="group" aria-label="统计类型">
          <button
            v-for="item in statisticTypes"
            :key="item.key"
            type="button"
            :aria-pressed="selectedStatisticType === item.key"
            @click="selectedStatisticType = item.key"
          >
            {{ item.label }}
          </button>
        </div>
      </div>

      <div class="filter-group">
        <span class="filter-group__label">密度</span>
        <div class="segmented-control" role="group" aria-label="展示密度">
          <button type="button" :aria-pressed="dashboardDensity === 'comfortable'" @click="dashboardDensity = 'comfortable'">
            舒适
          </button>
          <button type="button" :aria-pressed="dashboardDensity === 'compact'" @click="dashboardDensity = 'compact'">
            紧凑
          </button>
        </div>
      </div>

      <div class="filter-group filter-group--numbers">
        <div class="filter-group__row">
          <span class="filter-group__label">号码选择</span>
          <div class="filter-actions">
            <button type="button" @click="selectHotNumbers">热号</button>
            <button type="button" @click="selectColdNumbers">冷号</button>
            <button type="button" :disabled="selectedNumbers.length === 0" @click="selectedNumbers = []">清空</button>
          </div>
        </div>
        <div class="number-picker" aria-label="选择关注号码">
          <button
            v-for="number in allNumbers"
            :key="number"
            type="button"
            :aria-pressed="selectedNumbers.includes(number)"
            @click="toggleNumber(number)"
          >
            {{ padNumber(number) }}
          </button>
        </div>
      </div>

      <div class="filter-group filter-group--algorithms">
        <span class="filter-group__label">算法</span>
        <div class="algorithm-picker" aria-label="复盘算法筛选">
          <button
            v-for="algorithm in enabledAlgorithms"
            :key="algorithm.name"
            type="button"
            :aria-pressed="selectedAlgorithmNames.includes(algorithm.name)"
            @click="toggleAlgorithm(algorithm.name)"
          >
            {{ algorithm.displayName }}
          </button>
        </div>
      </div>

      <div class="filter-group filter-group--commands">
        <button class="command-button command-button--primary" type="button" :disabled="isLoading" @click="refreshData">
          <RefreshCcw :size="16" aria-hidden="true" />
          {{ isLoading ? '同步中' : '刷新' }}
        </button>
        <button class="command-button" type="button" @click="resetFilters">
          <RotateCcw :size="16" aria-hidden="true" />
          重置
        </button>
      </div>
    </section>

    <div v-if="errorMessage" class="dashboard-alert dashboard-alert--error">
      <span>数据获取失败，请稍后重试。</span>
      <button type="button" @click="refreshData">重试</button>
    </div>
    <div v-else-if="!isLoading && filteredHistory.length === 0" class="dashboard-alert">
      <span>当前筛选范围暂无开奖记录。</span>
      <button type="button" @click="resetFilters">重置筛选</button>
    </div>

    <section class="metric-grid" aria-label="历史数据指标">
      <article v-for="metric in dashboardMetrics" :key="metric.key" class="metric-card">
        <div class="metric-card__header">
          <component :is="metric.icon" :size="18" aria-hidden="true" />
          <span>{{ metric.label }}</span>
        </div>
        <div v-if="metric.rows" class="metric-card__balls">
          <NumberBall
            v-for="row in metric.rows"
            :key="row.number"
            :value="row.number"
            :variant="metric.key === 'cold' ? 'muted' : metric.key === 'missing' ? 'selected' : 'outline'"
            size="small"
          />
        </div>
        <strong v-else>{{ metric.value }}</strong>
        <p>{{ metric.detail }}</p>
      </article>
    </section>

    <section class="charts-grid" aria-label="历史数据图表">
      <ChartPanel
        title="号码热力图"
        subtitle="1-80 号码出现频率"
        :summary="heatmapSummary"
        :option="heatmapOption"
        :loading="isLoading"
        :empty="filteredHistory.length === 0"
        :error="errorMessage"
      />
      <ChartPanel
        title="和值趋势"
        subtitle="按期号从旧到新展示"
        :summary="sumTrendSummary"
        :option="sumTrendOption"
        :loading="isLoading"
        :empty="filteredHistory.length === 0"
        :error="errorMessage"
      />
      <ChartPanel
        title="奇偶大小趋势"
        subtitle="奇数、偶数、大号、小号对比"
        :summary="structureTrendSummary"
        :option="structureTrendOption"
        :loading="isLoading"
        :empty="filteredHistory.length === 0"
        :error="errorMessage"
      />
      <ChartPanel
        title="区间分布"
        subtitle="四区间累计出现次数"
        :summary="zoneSummary"
        :option="zoneDistributionOption"
        :loading="isLoading"
        :empty="filteredHistory.length === 0"
        :error="errorMessage"
      />
      <ChartPanel
        class="charts-grid__wide"
        title="遗漏摘要"
        subtitle="当前遗漏与历史最大遗漏对比"
        :summary="missingSummary"
        :option="missingOption"
        :loading="isLoading"
        :empty="filteredHistory.length === 0"
        :error="errorMessage"
      />
    </section>

    <section class="table-section" aria-label="历史数据表格">
      <header class="table-section__header">
        <div>
          <span class="section-kicker">表格分析</span>
          <h3>开奖、统计与复盘</h3>
          <p>表格与上方筛选条件保持一致，移动端可横向滚动查看完整字段。</p>
        </div>
        <div class="segmented-control" role="group" aria-label="表格类型">
          <button
            v-for="mode in tableModes"
            :key="mode.key"
            type="button"
            :aria-pressed="activeTableMode === mode.key"
            @click="activeTableMode = mode.key"
          >
            {{ mode.label }}
          </button>
        </div>
      </header>

      <HistoryDrawTable
        v-if="activeTableMode === 'history'"
        :rows="historyRows"
        :density="dashboardDensity"
        :loading="isLoading"
      />
      <NumberStatsTable
        v-else-if="activeTableMode === 'numbers'"
        :rows="numberStats"
        :density="dashboardDensity"
        :loading="isLoading"
      />
      <ReplayTable v-else :rows="replayRows" :density="dashboardDensity" :loading="isLoading" />
    </section>
  </section>
</template>

<style scoped>
.data-dashboard {
  display: grid;
  gap: 20px;
  animation: data-dashboard-enter 260ms ease both;
}

.data-dashboard__intro {
  max-width: 840px;
}

.data-dashboard__intro h2 {
  margin: 4px 0 0;
  font-family: var(--h8-font-title);
  font-size: 30px;
  letter-spacing: 0;
  line-height: 1.2;
}

.data-dashboard__intro p {
  margin: 8px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.6;
}

.data-dashboard__filters {
  position: sticky;
  top: calc(var(--h8-topbar-height) + 10px);
  z-index: 8;
  display: grid;
  grid-template-columns: minmax(280px, 1.5fr) repeat(3, minmax(180px, 0.8fr)) auto;
  gap: 12px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-surface-strong) 96%, transparent);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 14px;
  backdrop-filter: blur(18px);
}

.filter-group {
  display: grid;
  align-content: start;
  gap: 8px;
  min-width: 0;
}

.filter-group--numbers,
.filter-group--algorithms {
  grid-column: span 2;
}

.filter-group--commands {
  align-content: end;
  justify-content: end;
  grid-column: span 1;
  grid-template-columns: repeat(2, auto);
}

.filter-group--dates {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.filter-group__label {
  display: inline-flex;
  grid-column: 1 / -1;
  align-items: center;
  gap: 6px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
}

.filter-group__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.filter-group label {
  display: grid;
  gap: 4px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.filter-group input {
  min-height: 34px;
  min-width: 0;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  padding: 0 9px;
}

.filter-group input:focus-visible,
.segmented-control button:focus-visible,
.filter-actions button:focus-visible,
.algorithm-picker button:focus-visible,
.number-picker button:focus-visible,
.command-button:focus-visible,
.dashboard-alert button:focus-visible {
  outline: 0;
  box-shadow: var(--h8-focus-ring);
}

.segmented-control,
.filter-actions,
.algorithm-picker {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.segmented-control button,
.filter-actions button,
.algorithm-picker button,
.number-picker button,
.command-button,
.dashboard-alert button {
  min-height: 32px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  cursor: pointer;
  font-size: 13px;
  line-height: 1.2;
}

.segmented-control button,
.filter-actions button,
.algorithm-picker button {
  padding: 0 10px;
}

.segmented-control button[aria-pressed='true'],
.algorithm-picker button[aria-pressed='true'],
.number-picker button[aria-pressed='true'] {
  border-color: var(--h8-color-cinnabar);
  background: color-mix(in srgb, var(--h8-color-cinnabar) 10%, var(--h8-color-surface-strong));
  color: var(--h8-color-cinnabar);
  font-weight: 700;
}

.filter-actions button:disabled,
.command-button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.number-picker {
  display: grid;
  grid-template-columns: repeat(20, 32px);
  gap: 5px;
  overflow-x: auto;
  padding-bottom: 2px;
}

.number-picker button {
  width: 32px;
  min-height: 28px;
  padding: 0;
  font-family: var(--h8-font-number);
  font-size: 12px;
}

.algorithm-picker {
  max-height: 74px;
  overflow: auto;
}

.command-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 12px;
}

.command-button--primary {
  border-color: var(--h8-color-cinnabar);
  background: var(--h8-color-cinnabar);
  color: #fff;
}

.dashboard-alert {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-risk-orange) 10%, var(--h8-color-surface-strong));
  color: var(--h8-color-text);
  padding: 12px 14px;
}

.dashboard-alert--error {
  background: color-mix(in srgb, var(--h8-color-cinnabar) 10%, var(--h8-color-surface-strong));
}

.dashboard-alert button {
  padding: 0 10px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
}

.metric-card {
  display: grid;
  gap: 10px;
  min-height: 128px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 14px;
}

.metric-card__header {
  display: flex;
  align-items: center;
  gap: 7px;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  font-weight: 700;
}

.metric-card__header svg {
  color: var(--h8-color-cinnabar);
}

.metric-card strong {
  overflow-wrap: anywhere;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 27px;
  letter-spacing: 0;
  line-height: 1.1;
}

.metric-card p {
  margin: 0;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.45;
}

.metric-card__balls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  min-height: 32px;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.charts-grid__wide {
  grid-column: 1 / -1;
}

.table-section {
  display: grid;
  gap: 12px;
}

.table-section__header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
}

.table-section__header h3 {
  margin: 3px 0 0;
  font-size: 18px;
}

.table-section__header p {
  margin: 5px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

@keyframes data-dashboard-enter {
  from {
    opacity: 0;
    transform: translateY(8px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 1280px) {
  .data-dashboard__filters {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .filter-group--numbers,
  .filter-group--algorithms {
    grid-column: span 3;
  }

  .metric-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 920px) {
  .data-dashboard__filters {
    grid-template-columns: 1fr 1fr;
  }

  .filter-group--numbers,
  .filter-group--algorithms,
  .filter-group--commands {
    grid-column: 1 / -1;
  }

  .charts-grid {
    grid-template-columns: 1fr;
  }

  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .data-dashboard__intro h2 {
    font-size: 25px;
  }

  .data-dashboard__filters {
    top: 0;
    grid-template-columns: 1fr;
    max-height: 78vh;
    overflow: auto;
  }

  .filter-group--dates,
  .filter-group--numbers,
  .filter-group--algorithms,
  .filter-group--commands {
    grid-column: 1 / -1;
  }

  .filter-group--dates,
  .filter-group--commands {
    grid-template-columns: 1fr 1fr;
  }

  .number-picker {
    grid-template-columns: repeat(10, 32px);
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }

  .table-section__header {
    display: grid;
  }
}
</style>
