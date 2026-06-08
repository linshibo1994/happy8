<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { EChartsOption } from 'echarts'
import {
  Activity,
  BarChart3,
  Database,
  Filter,
  Globe2,
  Layers3,
  RefreshCcw,
  RotateCcw,
  Search,
  Wifi,
} from 'lucide-vue-next'

import { autoSyncLatestLottery, fetchLotteryHistory, fetchSandboxAnalysis } from '@/api'
import ChartPanel from '@/components/charts/ChartPanel.vue'
import HistoryDrawTable from '@/components/data-table/HistoryDrawTable.vue'
import SandboxEventTable from '@/components/data-sandbox/SandboxEventTable.vue'
import SandboxIntervalTable from '@/components/data-sandbox/SandboxIntervalTable.vue'
import { mapLotteryResult, useLotteryStore } from '@/stores/lottery'
import { useUiStore } from '@/stores/ui'
import type {
  LotteryResult,
  LotteryResultPayload,
  SandboxAnalysisResponse,
  SandboxConsecutiveLevel,
  SandboxEventMatch,
  SandboxEventType,
  SandboxIntervalRow,
  SandboxScope,
  SandboxSummary,
  SandboxTableMode,
} from '@/types'

type Density = 'comfortable' | 'compact'
type PeriodOption = 30 | 50 | 100 | 200 | 500

interface SandboxFilters {
  period: PeriodOption
  issue: string
  startDate: string
  endDate: string
  eventType: SandboxEventType
  level: SandboxConsecutiveLevel
  scope: SandboxScope
  zones: number[]
  page: number
  pageSize: 20 | 50 | 100
}

interface EightZoneRow {
  zone: number
  label: string
  count: number
  rate: number
}

const periodOptions: PeriodOption[] = [30, 50, 100, 200, 500]
const pageSizeOptions = [20, 50, 100] as const
const eventTypes: Array<{ key: SandboxEventType; label: string }> = [
  { key: 'consecutive', label: '连号' },
  { key: 'gap', label: '隔号' },
  { key: 'mixed', label: '连号隔号' },
  { key: 'interval', label: '间隔' },
]
const levels: Array<{ key: SandboxConsecutiveLevel; label: string }> = [
  { key: 2, label: '两连' },
  { key: 3, label: '三连' },
  { key: 4, label: '四连' },
]
const zoneOptions = Array.from({ length: 8 }, (_, index) => ({
  key: index + 1,
  label: `${index + 1}区`,
  range: `${index * 10 + 1}-${index * 10 + 10}`,
}))
const allNumbers = Array.from({ length: 80 }, (_, index) => index + 1)
const lineColor = '#E3DDD2'

const route = useRoute()
const router = useRouter()
const lotteryStore = useLotteryStore()
const uiStore = useUiStore()

const latestDate = new Date(lotteryStore.latestResult.openedAt)
const defaultStartDate = new Date(latestDate)
defaultStartDate.setDate(latestDate.getDate() - 199)

const filters = ref<SandboxFilters>({
  period: parsePeriod(queryString(route.query.period)),
  issue: queryString(route.query.issue) ?? '',
  startDate: queryString(route.query.start) ?? formatDateInput(defaultStartDate),
  endDate: queryString(route.query.end) ?? formatDateInput(latestDate),
  eventType: parseEventType(queryString(route.query.event)),
  level: parseLevel(queryString(route.query.level)),
  scope: parseScope(queryString(route.query.scope)),
  zones: parseZones(queryString(route.query.zones)),
  page: parsePositiveInt(queryString(route.query.page), 1),
  pageSize: parsePageSize(queryString(route.query.pageSize)),
})
const dashboardDensity = ref<Density>(parseDensity(queryString(route.query.density), uiStore.density))
const activeTableMode = ref<SandboxTableMode>(parseTableMode(queryString(route.query.table)))
const history = ref<LotteryResult[]>([])
const historyTotal = ref(0)
const remoteAnalysis = ref<SandboxAnalysisResponse | null>(null)
const isLoading = ref(false)
const isRefreshing = ref(false)
const errorMessage = ref('')
const lastSyncedAt = ref(lotteryStore.lastUpdatedAt)
const dataSource = ref<'接口' | '本地样例'>('本地样例')
let sandboxRequestId = 0

const fallbackHistory = computed(() => createHistorySeries(lotteryStore.latestResult, Math.max(filters.value.period, 240)))

const filteredFallbackHistory = computed(() => {
  const start = new Date(`${filters.value.startDate}T00:00:00`)
  const end = new Date(`${filters.value.endDate}T23:59:59`)
  const issueKeyword = filters.value.issue.trim()

  return fallbackHistory.value
    .filter((draw) => {
      const openedAt = new Date(draw.openedAt)
      const inDateRange = openedAt >= start && openedAt <= end
      const matchIssue = !issueKeyword || draw.issue.includes(issueKeyword)
      return inDateRange && matchIssue
    })
    .slice(0, filters.value.period)
})

const analysisSourceHistory = computed(() => {
  const source = history.value.length > 0 ? history.value : filteredFallbackHistory.value
  return source.slice(0, filters.value.period)
})

const localEvents = computed(() => buildEventMatches(analysisSourceHistory.value, filters.value))
const localIntervals = computed(() => buildIntervals(localEvents.value, analysisSourceHistory.value))
const localSummary = computed(() =>
  buildSummary(analysisSourceHistory.value, localEvents.value, localIntervals.value, filters.value),
)

const sandboxEvents = computed(() => remoteAnalysis.value?.events ?? localEvents.value)
const sandboxIntervals = computed(() => remoteAnalysis.value?.intervals ?? localIntervals.value)
const sandboxSummary = computed(() => remoteAnalysis.value?.summary ?? localSummary.value)
const actualPeriods = computed(() => remoteAnalysis.value?.actual_periods ?? analysisSourceHistory.value.length)
const displayHistory = computed(() => (history.value.length > 0 ? history.value : paginate(filteredFallbackHistory.value, filters.value.page, filters.value.pageSize)))
const eventRows = computed(() => paginate(sandboxEvents.value, filters.value.page, filters.value.pageSize))
const intervalRows = computed(() => paginate(sandboxIntervals.value, filters.value.page, filters.value.pageSize))
const totalRows = computed(() => {
  if (activeTableMode.value === 'history') {
    return historyTotal.value || analysisSourceHistory.value.length
  }
  if (activeTableMode.value === 'intervals') {
    return sandboxIntervals.value.length
  }
  return sandboxEvents.value.length
})
const totalPages = computed(() => Math.max(1, Math.ceil(totalRows.value / filters.value.pageSize)))

const historyRows = computed(() =>
  displayHistory.value.map((draw) => ({
    issue: draw.issue,
    openedAt: draw.openedAt,
    numbers: draw.numbers,
    sum: draw.sum,
    oddEvenText: `${draw.oddCount}:${draw.evenCount}`,
    bigSmallText: `${draw.bigCount}:${draw.smallCount}`,
    zoneDistribution: draw.zoneDistribution,
  })),
)

const metricItems = computed(() => [
  {
    key: 'sample',
    label: '开奖期数',
    value: String(sandboxSummary.value.sample_periods),
    detail: `窗口 ${filters.value.period} 期，实际统计 ${actualPeriods.value} 期`,
    icon: Database,
  },
  {
    key: 'hits',
    label: '命中期数',
    value: String(sandboxSummary.value.hit_periods),
    detail: `命中率 ${formatPercent(sandboxSummary.value.hit_rate)}`,
    icon: Activity,
  },
  {
    key: 'groups',
    label: '总组数',
    value: String(sandboxSummary.value.total_groups),
    detail: currentRuleLabel.value,
    icon: Layers3,
  },
  {
    key: 'gap',
    label: '平均空窗',
    value: formatNumberOrText(sandboxSummary.value.avg_gap, '样本不足'),
    detail: `最长空窗 ${formatNumberOrText(sandboxSummary.value.max_gap, '样本不足')}`,
    icon: BarChart3,
  },
  {
    key: 'missing',
    label: '当前遗漏',
    value: formatNumberOrText(sandboxSummary.value.current_missing, '样本不足'),
    detail: `最近命中 ${sandboxSummary.value.latest_issue ?? '暂无'}`,
    icon: Search,
  },
  {
    key: 'sync',
    label: '联网状态',
    value: dataSource.value,
    detail: `更新 ${formatDateTime(lastSyncedAt.value)}`,
    icon: Wifi,
  },
])

const currentRuleLabel = computed(() => {
  const event = eventTypes.find((item) => item.key === filters.value.eventType)?.label ?? '连号'
  const level = filters.value.eventType === 'gap' ? '' : levels.find((item) => item.key === filters.value.level)?.label
  const scope = filters.value.scope === 'global' ? '全局' : `八区 ${filters.value.zones.join('、') || '全部'}`
  return [scope, level, event].filter(Boolean).join(' ')
})

const summaryText = computed(() => {
  const topZones = sandboxSummary.value.top_zones?.slice(0, 3).map((item) => `${item.zone}区 ${item.count}次`).join('、')
  const baseline = typeof sandboxSummary.value.baseline_delta === 'number'
    ? `相对基线 ${sandboxSummary.value.baseline_delta >= 0 ? '+' : ''}${formatPercent(sandboxSummary.value.baseline_delta)}。`
    : '当前未接入长期基线。'

  return [
    `${currentRuleLabel.value} 在当前窗口命中 ${sandboxSummary.value.hit_periods} 期，出现率 ${formatPercent(sandboxSummary.value.hit_rate)}。`,
    `平均空窗 ${formatNumberOrText(sandboxSummary.value.avg_gap, '样本不足')}，当前遗漏 ${formatNumberOrText(sandboxSummary.value.current_missing, '样本不足')}。`,
    topZones ? `八区偏态集中在 ${topZones}。` : '当前口径暂无明显八区集中信息。',
    baseline,
    '历史统计只描述样本结构，不代表未来确定结果。',
  ].join('')
})

const eightZoneRows = computed<EightZoneRow[]>(() => {
  const totals = Array.from({ length: 8 }, (_, index) => ({
    zone: index + 1,
    label: `${index + 1}区`,
    count: 0,
    rate: 0,
  }))

  sandboxEvents.value.forEach((event) => {
    const zones = event.zones?.length ? event.zones : event.groups.flatMap((group) => group.map(zoneOfNumber))
    Array.from(new Set(zones)).forEach((zone) => {
      totals[zone - 1].count += 1
    })
  })

  const max = Math.max(1, ...totals.map((row) => row.count))
  return totals.map((row) => ({ ...row, rate: row.count / max }))
})

const zoneHeatmapOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  grid: { top: 24, right: 18, bottom: 34, left: 42 },
  xAxis: {
    type: 'category',
    data: eightZoneRows.value.map((row) => row.label),
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
      name: '命中期数',
      type: 'bar',
      barWidth: 24,
      itemStyle: {
        color: '#276EF1',
        borderRadius: [5, 5, 0, 0],
      },
      data: eightZoneRows.value.map((row) => row.count),
    },
  ],
}))

const eventTimelineOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  grid: { top: 24, right: 18, bottom: 34, left: 34 },
  xAxis: {
    type: 'category',
    data: [...analysisSourceHistory.value].reverse().map((draw) => draw.issue.slice(-4)),
    axisLine: { lineStyle: { color: lineColor } },
    axisLabel: { color: '#6B7280' },
  },
  yAxis: {
    type: 'value',
    min: 0,
    max: Math.max(1, ...sandboxEvents.value.map((event) => event.group_count ?? event.groups.length)),
    splitLine: { lineStyle: { color: lineColor, type: 'dashed' } },
    axisLabel: { color: '#6B7280' },
  },
  series: [
    {
      name: '事件组数',
      type: 'line',
      smooth: true,
      symbolSize: 5,
      lineStyle: { color: '#C9352B', width: 2 },
      itemStyle: { color: '#C9352B' },
      areaStyle: { color: 'rgba(201, 53, 43, 0.1)' },
      data: [...analysisSourceHistory.value].reverse().map((draw) => {
        const event = sandboxEvents.value.find((item) => item.issue === draw.issue)
        return event?.group_count ?? event?.groups.length ?? 0
      }),
    },
  ],
}))

const intervalOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis' },
  grid: { top: 24, right: 18, bottom: 34, left: 42 },
  xAxis: {
    type: 'category',
    data: sandboxIntervals.value.map((row) => row.issue.slice(-4)),
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
      name: '空窗期数',
      type: 'bar',
      barWidth: 18,
      itemStyle: { color: '#D9822B', borderRadius: [4, 4, 0, 0] },
      data: sandboxIntervals.value.map((row) => row.gap ?? 0),
    },
  ],
}))

watch(
  () => [
    filters.value.period,
    filters.value.issue,
    filters.value.startDate,
    filters.value.endDate,
    filters.value.eventType,
    filters.value.level,
    filters.value.scope,
    filters.value.zones.join(','),
    filters.value.pageSize,
  ],
  () => {
    filters.value.page = 1
    syncQuery()
    void loadSandboxData()
  },
)

watch(
  () => filters.value.page,
  () => {
    syncQuery()
    void loadSandboxData()
  },
)

onMounted(() => {
  void loadSandboxData()
})

async function loadSandboxData() {
  const requestId = ++sandboxRequestId
  isLoading.value = true
  errorMessage.value = ''

  try {
    const [historyPayload, analysisPayload] = await Promise.all([
      fetchLotteryHistory({
        page: filters.value.page,
        page_size: filters.value.pageSize,
        issue: filters.value.issue || undefined,
        start_date: filters.value.startDate || undefined,
        end_date: filters.value.endDate || undefined,
      }),
      fetchSandboxAnalysis({
        recent_periods: filters.value.period,
        issue: filters.value.issue || undefined,
        start_date: filters.value.startDate || undefined,
        end_date: filters.value.endDate || undefined,
        event_type: filters.value.eventType,
        level: filters.value.level,
        scope: filters.value.scope,
        zones: filters.value.scope === 'zone' ? filters.value.zones : undefined,
        page: filters.value.page,
        page_size: filters.value.pageSize,
      }),
    ])

    if (requestId !== sandboxRequestId) {
      return
    }
    history.value = historyPayload.results.map((item) => mapLotteryResult(item as LotteryResultPayload))
    historyTotal.value = historyPayload.total
    remoteAnalysis.value = analysisPayload
    dataSource.value = '接口'
  } catch (error) {
    if (requestId !== sandboxRequestId) {
      return
    }
    remoteAnalysis.value = null
    history.value = []
    historyTotal.value = 0
    dataSource.value = '本地样例'
    errorMessage.value = error instanceof Error ? error.message : '数据沙盘接口暂不可用，已使用本地样例计算'
  } finally {
    if (requestId === sandboxRequestId) {
      isLoading.value = false
    }
  }
}

async function refreshData() {
  isRefreshing.value = true
  try {
    const summary = await autoSyncLatestLottery()
    lastSyncedAt.value = summary.synced_at ?? new Date().toISOString()
    await lotteryStore.refreshLatestResults({ autoSync: false })
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '最新开奖同步失败，已保留最近成功数据'
  } finally {
    isRefreshing.value = false
    await loadSandboxData()
  }
}

function resetFilters() {
  filters.value = {
    period: 100,
    issue: '',
    startDate: formatDateInput(defaultStartDate),
    endDate: formatDateInput(latestDate),
    eventType: 'consecutive',
    level: 3,
    scope: 'global',
    zones: [],
    page: 1,
    pageSize: 20,
  }
  activeTableMode.value = 'events'
  dashboardDensity.value = 'comfortable'
  syncQuery()
}

function toggleZone(zone: number) {
  if (filters.value.zones.includes(zone)) {
    filters.value.zones = filters.value.zones.filter((item) => item !== zone)
    return
  }
  filters.value.zones = [...filters.value.zones, zone].sort((left, right) => left - right)
}

function gotoPage(nextPage: number) {
  filters.value.page = Math.min(Math.max(1, nextPage), totalPages.value)
}

function setTableMode(mode: SandboxTableMode) {
  activeTableMode.value = mode
}

function setDashboardDensity(density: Density) {
  dashboardDensity.value = density
  uiStore.setDensity(density)
}

function syncQuery() {
  uiStore.setDensity(dashboardDensity.value)
  void router.replace({
    query: {
      period: String(filters.value.period),
      issue: filters.value.issue || undefined,
      start: filters.value.startDate,
      end: filters.value.endDate,
      event: filters.value.eventType,
      level: String(filters.value.level),
      scope: filters.value.scope,
      zones: filters.value.zones.join(',') || undefined,
      page: String(filters.value.page),
      pageSize: String(filters.value.pageSize),
    },
  })
}

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

function parseLevel(value: string | undefined): SandboxConsecutiveLevel {
  const parsed = Number(value)
  return parsed === 2 || parsed === 3 || parsed === 4 ? parsed : 3
}

function parseEventType(value: string | undefined): SandboxEventType {
  return eventTypes.some((item) => item.key === value) ? (value as SandboxEventType) : 'consecutive'
}

function parseScope(value: string | undefined): SandboxScope {
  return value === 'zone' ? 'zone' : 'global'
}

function parseZones(value: string | undefined) {
  if (!value) {
    return []
  }
  return value
    .split(',')
    .map((item) => Number(item))
    .filter((zone) => Number.isInteger(zone) && zone >= 1 && zone <= 8)
}

function parsePageSize(value: string | undefined): 20 | 50 | 100 {
  const parsed = Number(value)
  return pageSizeOptions.includes(parsed as 20 | 50 | 100) ? (parsed as 20 | 50 | 100) : 20
}

function parsePositiveInt(value: string | undefined, fallback: number) {
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback
}

function parseDensity(value: string | undefined, fallback: Density): Density {
  return value === 'compact' || value === 'comfortable' ? value : fallback
}

function parseTableMode(value: string | undefined): SandboxTableMode {
  return value === 'history' || value === 'intervals' || value === 'events' ? value : 'events'
}

function formatDateInput(date: Date) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function formatDateTime(value?: string) {
  if (!value) {
    return '-'
  }
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function formatNumberOrText(value: number | null | undefined, fallback: string) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(value % 1 === 0 ? 0 : 1) : fallback
}

function paginate<T>(rows: T[], page: number, pageSize: number) {
  const start = (page - 1) * pageSize
  return rows.slice(start, start + pageSize)
}

function zoneOfNumber(number: number) {
  return Math.ceil(number / 10)
}

function fourZoneOfNumber(number: number) {
  if (number <= 20) return '1-20'
  if (number <= 40) return '21-40'
  if (number <= 60) return '41-60'
  return '61-80'
}

function buildZoneDistribution(numbers: number[]) {
  return numbers.reduce<Record<string, number>>(
    (distribution, number) => {
      distribution[fourZoneOfNumber(number)] += 1
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
    .map((number) => ({ number, score: pseudoRandomScore(number, seed) }))
    .sort((left, right) => left.score - right.score)
    .slice(0, 20)
    .map((item) => item.number)
    .sort((left, right) => left - right)
}

function pseudoRandomScore(number: number, seed: number) {
  const raw = Math.sin((seed + 17) * (number + 3) * 12.9898 + number * 78.233) * 43758.5453
  return raw - Math.floor(raw)
}

function buildEventMatches(rows: LotteryResult[], currentFilters: SandboxFilters): SandboxEventMatch[] {
  return rows
    .map((draw) => {
      const groups = matchGroups(draw.numbers, currentFilters)
      if (groups.length === 0) {
        return null
      }

      return {
        issue: draw.issue,
        openedAt: draw.openedAt,
        numbers: draw.numbers,
        event_type: currentFilters.eventType,
        scope: currentFilters.scope,
        zones: currentFilters.scope === 'zone'
          ? currentFilters.zones.length > 0
            ? currentFilters.zones
            : Array.from(new Set(groups.flat().map(zoneOfNumber)))
          : Array.from(new Set(groups.flat().map(zoneOfNumber))),
        groups,
        longest_length: Math.max(...groups.map((group) => group.length)),
        group_count: groups.length,
        label: eventLabel(currentFilters.eventType, currentFilters.level),
      }
    })
    .filter((row): row is SandboxEventMatch => Boolean(row))
}

function matchGroups(numbers: number[], currentFilters: SandboxFilters) {
  const scopedSets = buildScopedNumberSets(numbers, currentFilters)
  const groups = scopedSets.flatMap((set) => {
    if (currentFilters.eventType === 'gap') {
      return findGapGroups(set)
    }
    if (currentFilters.eventType === 'mixed' || currentFilters.eventType === 'interval') {
      return findMixedGroups(set)
    }
    return findConsecutiveGroups(set, currentFilters.level)
  })

  return uniqueGroups(groups)
}

function buildScopedNumberSets(numbers: number[], currentFilters: SandboxFilters) {
  const sorted = [...new Set(numbers)].sort((left, right) => left - right)
  if (currentFilters.scope === 'global') {
    return [sorted]
  }

  const zones = currentFilters.zones.length > 0 ? currentFilters.zones : zoneOptions.map((zone) => zone.key)
  return zones.map((zone) => sorted.filter((number) => zoneOfNumber(number) === zone))
}

function findConsecutiveGroups(numbers: number[], level: SandboxConsecutiveLevel) {
  const groups: number[][] = []
  let current: number[] = []

  numbers.forEach((number, index) => {
    if (index === 0 || number === numbers[index - 1] + 1) {
      current.push(number)
    } else {
      if (current.length >= level) {
        groups.push([...current])
      }
      current = [number]
    }
  })

  if (current.length >= level) {
    groups.push(current)
  }

  return groups
}

function findGapGroups(numbers: number[]) {
  const set = new Set(numbers)
  return numbers.flatMap((number) => (set.has(number + 2) ? [[number, number + 2]] : []))
}

function findMixedGroups(numbers: number[]) {
  const groups: number[][] = []
  for (let index = 0; index < numbers.length; index += 1) {
    const group3 = numbers.slice(index, index + 3)
    const group4 = numbers.slice(index, index + 4)
    if (isMixedGroup(group3)) groups.push(group3)
    if (isMixedGroup(group4)) groups.push(group4)
  }
  return groups
}

function isMixedGroup(group: number[]) {
  if (group.length < 3) {
    return false
  }
  const diffs = group.slice(1).map((number, index) => number - group[index])
  return diffs.every((diff) => diff === 1 || diff === 2) && diffs.includes(1) && diffs.includes(2)
}

function uniqueGroups(groups: number[][]) {
  const seen = new Set<string>()
  return groups.filter((group) => {
    const key = group.join(',')
    if (seen.has(key)) {
      return false
    }
    seen.add(key)
    return true
  })
}

function buildIntervals(events: SandboxEventMatch[], rows: LotteryResult[]): SandboxIntervalRow[] {
  const chronological = [...events].reverse()
  return chronological.map((event, index) => {
    const next = chronological[index + 1]
    const currentIndex = rows.findIndex((draw) => draw.issue === event.issue)
    const nextIndex = next ? rows.findIndex((draw) => draw.issue === next.issue) : -1
    const distance = nextIndex >= 0 && currentIndex >= 0 ? Math.abs(currentIndex - nextIndex) : null

    return {
      issue: event.issue,
      draw_date: event.openedAt ?? event.draw_date,
      next_issue: next?.issue ?? null,
      gap: typeof distance === 'number' ? Math.max(0, distance - 1) : null,
      distance,
    }
  })
}

function buildSummary(
  rows: LotteryResult[],
  events: SandboxEventMatch[],
  intervals: SandboxIntervalRow[],
  currentFilters: SandboxFilters,
): SandboxSummary {
  const gaps = intervals.map((row) => row.gap).filter((value): value is number => typeof value === 'number')
  const latestEvent = events[0]
  const currentMissing = latestEvent ? rows.findIndex((draw) => draw.issue === latestEvent.issue) : rows.length
  const topZones = Array.from(
    events
      .flatMap((event) => event.zones ?? event.groups.flat().map(zoneOfNumber))
      .reduce<Map<number, number>>((map, zone) => map.set(zone, (map.get(zone) ?? 0) + 1), new Map()),
  )
    .map(([zone, count]) => ({ zone, count }))
    .sort((left, right) => right.count - left.count || left.zone - right.zone)

  return {
    sample_periods: rows.length,
    event_level: currentFilters.level,
    hit_periods: events.length,
    hit_rate: rows.length > 0 ? events.length / rows.length : 0,
    total_groups: events.reduce((sum, event) => sum + (event.group_count ?? event.groups.length), 0),
    avg_gap: gaps.length > 0 ? gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length : null,
    median_gap: median(gaps),
    max_gap: gaps.length > 0 ? Math.max(...gaps) : null,
    current_missing: currentMissing >= 0 ? currentMissing : null,
    latest_issue: latestEvent?.issue ?? null,
    top_zones: topZones,
    baseline_delta: null,
    updated_at: new Date().toISOString(),
  }
}

function median(values: number[]) {
  if (values.length === 0) {
    return null
  }
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

function eventLabel(eventType: SandboxEventType, level: SandboxConsecutiveLevel) {
  if (eventType === 'gap') return '隔号'
  if (eventType === 'mixed') return '连号隔号'
  if (eventType === 'interval') return `${level}连间隔`
  return `${level}连号`
}
</script>

<template>
  <section class="data-sandbox" aria-labelledby="data-sandbox-title">
    <header class="sandbox-topbar">
      <div>
        <span class="section-kicker">历史分析</span>
        <h2 id="data-sandbox-title">数据沙盘</h2>
        <p>{{ currentRuleLabel }}，最近 {{ filters.period }} 期，实际统计 {{ actualPeriods }} 期。</p>
      </div>
      <div class="sandbox-topbar__status" aria-label="联网查询与刷新状态">
        <span><Globe2 :size="15" aria-hidden="true" />{{ dataSource }}</span>
        <span>最新期号 {{ lotteryStore.latestResult.issue }}</span>
        <span>同步 {{ formatDateTime(lastSyncedAt) }}</span>
      </div>
    </header>

    <section class="sandbox-filters" aria-label="数据沙盘筛选">
      <div class="filter-field filter-field--issue">
        <label for="sandbox-issue"><Search :size="15" aria-hidden="true" />期号查询</label>
        <input id="sandbox-issue" v-model.trim="filters.issue" placeholder="输入完整或部分期号" type="search" />
      </div>

      <div class="filter-field filter-field--dates">
        <label><Filter :size="15" aria-hidden="true" />日期范围</label>
        <input v-model="filters.startDate" type="date" aria-label="开始日期" />
        <input v-model="filters.endDate" type="date" aria-label="结束日期" />
      </div>

      <div class="filter-field">
        <label>分析期数</label>
        <div class="segmented-control" role="group" aria-label="分析期数">
          <button
            v-for="period in periodOptions"
            :key="period"
            type="button"
            :aria-pressed="filters.period === period"
            @click="filters.period = period"
          >
            {{ period }}
          </button>
        </div>
      </div>

      <div class="filter-field">
        <label>事件类型</label>
        <div class="segmented-control" role="group" aria-label="事件类型">
          <button
            v-for="item in eventTypes"
            :key="item.key"
            type="button"
            :aria-pressed="filters.eventType === item.key"
            @click="filters.eventType = item.key"
          >
            {{ item.label }}
          </button>
        </div>
      </div>

      <div class="filter-field">
        <label>连号等级</label>
        <div class="segmented-control" role="group" aria-label="连号等级">
          <button
            v-for="level in levels"
            :key="level.key"
            type="button"
            :aria-pressed="filters.level === level.key"
            @click="filters.level = level.key"
          >
            {{ level.label }}
          </button>
        </div>
      </div>

      <div class="filter-field">
        <label>分析口径</label>
        <div class="segmented-control" role="group" aria-label="分析口径">
          <button type="button" :aria-pressed="filters.scope === 'global'" @click="filters.scope = 'global'">全局</button>
          <button type="button" :aria-pressed="filters.scope === 'zone'" @click="filters.scope = 'zone'">八区</button>
        </div>
      </div>

      <div class="filter-field filter-field--zones">
        <label>八区选择</label>
        <div class="zone-picker" aria-label="八区选择">
          <button
            v-for="zone in zoneOptions"
            :key="zone.key"
            type="button"
            :aria-pressed="filters.zones.includes(zone.key)"
            :disabled="filters.scope === 'global'"
            @click="toggleZone(zone.key)"
          >
            <strong>{{ zone.label }}</strong>
            <span>{{ zone.range }}</span>
          </button>
        </div>
      </div>

      <div class="filter-field">
        <label>每页条数</label>
        <select v-model.number="filters.pageSize">
          <option v-for="size in pageSizeOptions" :key="size" :value="size">{{ size }}</option>
        </select>
      </div>

      <div class="filter-field">
        <label>密度</label>
        <div class="segmented-control" role="group" aria-label="展示密度">
          <button type="button" :aria-pressed="dashboardDensity === 'comfortable'" @click="setDashboardDensity('comfortable')">
            舒适
          </button>
          <button type="button" :aria-pressed="dashboardDensity === 'compact'" @click="setDashboardDensity('compact')">
            紧凑
          </button>
        </div>
      </div>

      <div class="filter-field filter-field--commands">
        <button class="command-button command-button--primary" type="button" :disabled="isRefreshing || isLoading" @click="refreshData">
          <RefreshCcw :size="16" aria-hidden="true" :class="{ 'is-spinning': isRefreshing }" />
          {{ isRefreshing ? '刷新中' : '联网刷新' }}
        </button>
        <button class="command-button" type="button" @click="resetFilters">
          <RotateCcw :size="16" aria-hidden="true" />
          重置
        </button>
      </div>
    </section>

    <div v-if="errorMessage" class="sandbox-alert">
      <span>{{ errorMessage }}</span>
      <button type="button" @click="loadSandboxData">重试接口</button>
    </div>

    <section class="metric-strip" aria-label="沙盘指标">
      <article v-for="metric in metricItems" :key="metric.key" class="metric-item">
        <div>
          <component :is="metric.icon" :size="16" aria-hidden="true" />
          <span>{{ metric.label }}</span>
        </div>
        <strong>{{ metric.value }}</strong>
        <p>{{ metric.detail }}</p>
      </article>
    </section>

    <section class="analysis-grid" aria-label="数据沙盘分析图表">
      <ChartPanel
        title="八区命中分布"
        subtitle="按命中事件聚合到 1-8 区"
        :summary="eightZoneRows.map((row) => `${row.label} ${row.count}`).join('，')"
        :option="zoneHeatmapOption"
        :loading="isLoading"
        :empty="sandboxEvents.length === 0"
        :error="''"
      />
      <ChartPanel
        title="事件时间线"
        subtitle="按期号从旧到新展示命中组数"
        :summary="`当前窗口命中 ${sandboxEvents.length} 期，共 ${sandboxSummary.total_groups} 组。`"
        :option="eventTimelineOption"
        :loading="isLoading"
        :empty="analysisSourceHistory.length === 0"
        :error="''"
      />
      <ChartPanel
        title="间隔柱状图"
        subtitle="相邻同类事件之间的空窗期数"
        :summary="`平均空窗 ${formatNumberOrText(sandboxSummary.avg_gap, '样本不足')}，最长空窗 ${formatNumberOrText(sandboxSummary.max_gap, '样本不足')}。`"
        :option="intervalOption"
        :loading="isLoading"
        :empty="sandboxIntervals.length === 0"
        :error="''"
      />
      <aside class="summary-panel" aria-label="规律总结">
        <div>
          <span class="section-kicker">规律总结</span>
          <h3>{{ currentRuleLabel }}</h3>
        </div>
        <p>{{ summaryText }}</p>
        <div class="summary-panel__evidence">
          <button type="button" @click="setTableMode('events')">查看命中表</button>
          <button type="button" @click="setTableMode('intervals')">查看间隔表</button>
        </div>
      </aside>
    </section>

    <section class="table-section" aria-label="沙盘结果表">
      <header class="table-section__header">
        <div>
          <span class="section-kicker">结果表</span>
          <h3>历史开奖、规则命中与间隔分析</h3>
          <p>筛选变化后自动回到第一页，表格支持横向滚动。</p>
        </div>
        <div class="segmented-control" role="group" aria-label="表格类型">
          <button type="button" :aria-pressed="activeTableMode === 'history'" @click="setTableMode('history')">历史开奖</button>
          <button type="button" :aria-pressed="activeTableMode === 'events'" @click="setTableMode('events')">规则命中</button>
          <button type="button" :aria-pressed="activeTableMode === 'intervals'" @click="setTableMode('intervals')">间隔分析</button>
        </div>
      </header>

      <HistoryDrawTable
        v-if="activeTableMode === 'history'"
        :rows="historyRows"
        :density="dashboardDensity"
        :loading="isLoading"
      />
      <SandboxEventTable
        v-else-if="activeTableMode === 'events'"
        :rows="eventRows"
        :density="dashboardDensity"
        :loading="isLoading"
      />
      <SandboxIntervalTable
        v-else
        :rows="intervalRows"
        :density="dashboardDensity"
        :loading="isLoading"
      />

      <footer class="pagination-bar" aria-label="分页">
        <span>第 {{ filters.page }} / {{ totalPages }} 页，共 {{ totalRows }} 条</span>
        <div>
          <button type="button" :disabled="filters.page <= 1" @click="gotoPage(filters.page - 1)">上一页</button>
          <button type="button" :disabled="filters.page >= totalPages" @click="gotoPage(filters.page + 1)">下一页</button>
        </div>
      </footer>
    </section>
  </section>
</template>

<style scoped>
.data-sandbox {
  display: grid;
  gap: 18px;
  animation: sandbox-enter 260ms ease both;
}

.sandbox-topbar {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
}

.sandbox-topbar h2 {
  margin: 4px 0 0;
  font-family: var(--h8-font-title);
  font-size: 30px;
  letter-spacing: 0;
  line-height: 1.2;
}

.sandbox-topbar p {
  margin: 7px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.6;
}

.sandbox-topbar__status {
  display: flex;
  flex-wrap: wrap;
  justify-content: end;
  gap: 8px;
}

.sandbox-topbar__status span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-text-muted);
  padding: 6px 8px;
  font-size: 12px;
}

.sandbox-filters {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(220px, 1.2fr) minmax(260px, 1.4fr) repeat(4, minmax(150px, 0.8fr));
  gap: 12px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-surface-strong) 96%, transparent);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 14px;
  backdrop-filter: blur(18px);
}

.filter-field {
  display: grid;
  align-content: start;
  gap: 7px;
  min-width: 0;
}

.filter-field--dates {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.filter-field--dates label,
.filter-field--zones {
  grid-column: 1 / -1;
}

.filter-field--zones {
  grid-column: span 3;
}

.filter-field--commands {
  align-content: end;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.filter-field label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
}

.filter-field input,
.filter-field select {
  min-height: 34px;
  min-width: 0;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  padding: 0 9px;
}

.segmented-control,
.zone-picker {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.segmented-control button,
.zone-picker button,
.command-button,
.sandbox-alert button,
.summary-panel__evidence button,
.pagination-bar button {
  min-height: 32px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  cursor: pointer;
  font-size: 13px;
  line-height: 1.2;
}

.segmented-control button {
  padding: 0 10px;
}

.segmented-control button[aria-pressed='true'],
.zone-picker button[aria-pressed='true'] {
  border-color: var(--h8-color-cinnabar);
  background: color-mix(in srgb, var(--h8-color-cinnabar) 10%, var(--h8-color-surface-strong));
  color: var(--h8-color-cinnabar);
  font-weight: 700;
}

.zone-picker button {
  display: grid;
  gap: 2px;
  min-width: 66px;
  padding: 5px 8px;
  text-align: left;
}

.zone-picker button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.zone-picker span {
  color: var(--h8-color-text-muted);
  font-family: var(--h8-font-number);
  font-size: 11px;
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

.command-button:disabled,
.pagination-bar button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.filter-field input:focus-visible,
.filter-field select:focus-visible,
.segmented-control button:focus-visible,
.zone-picker button:focus-visible,
.command-button:focus-visible,
.sandbox-alert button:focus-visible,
.summary-panel__evidence button:focus-visible,
.pagination-bar button:focus-visible {
  outline: 0;
  box-shadow: var(--h8-focus-ring);
}

.sandbox-alert {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border: 1px solid color-mix(in srgb, var(--h8-color-risk-orange) 55%, var(--h8-color-line));
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-risk-orange) 10%, var(--h8-color-surface-strong));
  color: var(--h8-color-text);
  padding: 12px 14px;
}

.sandbox-alert button,
.summary-panel__evidence button,
.pagination-bar button {
  padding: 0 10px;
}

.metric-strip {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
}

.metric-item {
  display: grid;
  gap: 8px;
  min-height: 118px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 13px;
}

.metric-item div {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
}

.metric-item svg {
  color: var(--h8-color-cinnabar);
}

.metric-item strong {
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 24px;
  line-height: 1.1;
}

.metric-item p {
  margin: 0;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.45;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.summary-panel {
  display: grid;
  align-content: start;
  gap: 14px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 16px;
}

.summary-panel h3 {
  margin: 4px 0 0;
  font-size: 18px;
}

.summary-panel p {
  margin: 0;
  color: var(--h8-color-text);
  font-size: 13px;
  line-height: 1.75;
}

.summary-panel__evidence {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
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

.pagination-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 10px 12px;
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.pagination-bar div {
  display: flex;
  gap: 8px;
}

.is-spinning {
  animation: spin-once 620ms ease both;
}

@keyframes sandbox-enter {
  from {
    opacity: 0;
    transform: translateY(8px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes spin-once {
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .data-sandbox,
  .is-spinning {
    animation: none;
  }
}

@media (max-width: 1320px) {
  .sandbox-filters {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .filter-field--zones {
    grid-column: 1 / -1;
  }

  .metric-strip {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .analysis-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 920px) {
  .sandbox-topbar,
  .table-section__header {
    display: grid;
  }

  .sandbox-topbar__status {
    justify-content: start;
  }

  .sandbox-filters,
  .analysis-grid {
    grid-template-columns: 1fr;
  }

  .filter-field--zones,
  .filter-field--dates {
    grid-column: 1 / -1;
  }

  .metric-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 680px) {
  .sandbox-topbar h2 {
    font-size: 25px;
  }

  .sandbox-filters {
    max-height: 78vh;
    overflow: auto;
  }

  .filter-field--dates,
  .filter-field--commands {
    grid-template-columns: 1fr;
  }

  .metric-strip {
    grid-template-columns: 1fr;
  }

  .pagination-bar {
    align-items: stretch;
    flex-direction: column;
  }
}
</style>
