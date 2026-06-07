<script setup lang="ts">
import { computed, ref } from 'vue'

import NumberBall from '@/components/balls/NumberBall.vue'

type Density = 'comfortable' | 'compact'
type SortKey = 'count' | 'rate' | 'currentMissing' | 'maxMissing'
type SortDirection = 'asc' | 'desc'

export interface NumberStatsRow {
  number: number
  count: number
  rate: number
  currentMissing: number
  maxMissing: number
  latestIssue: string
  zone: string
  level: 'hot' | 'cold' | 'normal'
}

const props = withDefaults(
  defineProps<{
    rows: NumberStatsRow[]
    density?: Density
    loading?: boolean
    emptyText?: string
  }>(),
  {
    density: 'comfortable',
    loading: false,
    emptyText: '当前筛选范围暂无号码统计',
  },
)

const sortKey = ref<SortKey>('count')
const sortDirection = ref<SortDirection>('desc')

const sortedRows = computed(() => {
  return [...props.rows].sort((left, right) => {
    const result = left[sortKey.value] - right[sortKey.value]
    return sortDirection.value === 'asc' ? result : -result
  })
})

function setSort(key: SortKey) {
  if (sortKey.value === key) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
    return
  }

  sortKey.value = key
  sortDirection.value = 'desc'
}

function sortLabel(key: SortKey) {
  if (sortKey.value !== key) {
    return ''
  }

  return sortDirection.value === 'asc' ? '升序' : '降序'
}

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`
}

function ballVariant(level: NumberStatsRow['level']) {
  if (level === 'hot') {
    return 'outline'
  }

  if (level === 'cold') {
    return 'muted'
  }

  return 'neutral'
}
</script>

<template>
  <div class="h8-stats-table" :class="`h8-stats-table--${density}`">
    <div v-if="loading" class="h8-stats-table__state">正在读取历史开奖数据</div>
    <div v-else-if="rows.length === 0" class="h8-stats-table__state">{{ emptyText }}</div>
    <div v-else class="h8-stats-table__scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">号码</th>
            <th scope="col">
              <button type="button" @click="setSort('count')">出现次数 {{ sortLabel('count') }}</button>
            </th>
            <th scope="col">
              <button type="button" @click="setSort('rate')">出现率 {{ sortLabel('rate') }}</button>
            </th>
            <th scope="col">
              <button type="button" @click="setSort('currentMissing')">
                当前遗漏 {{ sortLabel('currentMissing') }}
              </button>
            </th>
            <th scope="col">
              <button type="button" @click="setSort('maxMissing')">最大遗漏 {{ sortLabel('maxMissing') }}</button>
            </th>
            <th scope="col">最近出现期号</th>
            <th scope="col">区间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in sortedRows" :key="row.number">
            <td>
              <div class="h8-stats-table__number">
                <NumberBall :value="row.number" :variant="ballVariant(row.level)" size="small" />
                <span v-if="row.level === 'hot'">热号</span>
                <span v-else-if="row.level === 'cold'">冷号</span>
                <span v-else>常规</span>
              </div>
            </td>
            <td class="h8-stats-table__mono">{{ row.count }}</td>
            <td class="h8-stats-table__mono">{{ formatPercent(row.rate) }}</td>
            <td class="h8-stats-table__mono">{{ row.currentMissing }}</td>
            <td class="h8-stats-table__mono">{{ row.maxMissing }}</td>
            <td class="h8-stats-table__mono">{{ row.latestIssue || '未出现' }}</td>
            <td>{{ row.zone }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.h8-stats-table {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
}

.h8-stats-table__scroll {
  overflow-x: auto;
}

.h8-stats-table table {
  width: 100%;
  min-width: 760px;
  border-collapse: collapse;
}

.h8-stats-table th,
.h8-stats-table td {
  border-bottom: 1px solid var(--h8-color-line);
  text-align: left;
  vertical-align: middle;
}

.h8-stats-table--comfortable th,
.h8-stats-table--comfortable td {
  padding: 13px 14px;
}

.h8-stats-table--compact th,
.h8-stats-table--compact td {
  padding: 8px 10px;
}

.h8-stats-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--h8-color-surface);
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.h8-stats-table th button {
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  padding: 0;
}

.h8-stats-table td {
  color: var(--h8-color-text);
  font-size: 13px;
}

.h8-stats-table tbody tr:hover {
  background: color-mix(in srgb, var(--h8-color-cinnabar) 5%, transparent);
}

.h8-stats-table__number {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.h8-stats-table__number span {
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.h8-stats-table__mono {
  font-family: var(--h8-font-number);
}

.h8-stats-table__state {
  display: grid;
  min-height: 180px;
  place-items: center;
  color: var(--h8-color-text-muted);
}
</style>
