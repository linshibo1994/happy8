<script setup lang="ts">
import type { SandboxIntervalRow } from '@/types'

type Density = 'comfortable' | 'compact'

withDefaults(
  defineProps<{
    rows: SandboxIntervalRow[]
    density?: Density
    loading?: boolean
  }>(),
  {
    density: 'comfortable',
    loading: false,
  },
)

const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
})

function formatDate(value?: string) {
  return value ? dateFormatter.format(new Date(value)) : '-'
}

function formatNullable(value?: number | null) {
  return typeof value === 'number' ? String(value) : '样本不足'
}
</script>

<template>
  <div class="interval-table" :class="`interval-table--${density}`">
    <div v-if="loading" class="interval-table__state">正在计算事件出现间隔</div>
    <div v-else-if="rows.length === 0" class="interval-table__state">当前事件出现次数不足，无法计算间隔</div>
    <div v-else class="interval-table__scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">本次期号</th>
            <th scope="col">开奖日期</th>
            <th scope="col">下一次期号</th>
            <th scope="col">空窗期数</th>
            <th scope="col">相邻事件距离</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="`${row.issue}-${row.next_issue ?? 'latest'}`">
            <td class="interval-table__mono">{{ row.issue }}</td>
            <td>{{ formatDate(row.draw_date) }}</td>
            <td class="interval-table__mono">{{ row.next_issue ?? '-' }}</td>
            <td>{{ formatNullable(row.gap) }}</td>
            <td>{{ formatNullable(row.distance) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.interval-table {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
}

.interval-table__scroll {
  overflow-x: auto;
}

.interval-table table {
  width: 100%;
  min-width: 720px;
  border-collapse: collapse;
}

.interval-table th,
.interval-table td {
  border-bottom: 1px solid var(--h8-color-line);
  text-align: left;
  vertical-align: middle;
}

.interval-table--comfortable th,
.interval-table--comfortable td {
  padding: 12px 14px;
}

.interval-table--compact th,
.interval-table--compact td {
  padding: 8px 10px;
}

.interval-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--h8-color-surface);
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.interval-table td {
  color: var(--h8-color-text);
  font-size: 13px;
}

.interval-table tbody tr {
  transition: background 140ms ease;
}

.interval-table tbody tr:hover {
  background: color-mix(in srgb, var(--h8-color-risk-orange) 6%, transparent);
}

.interval-table__mono {
  font-family: var(--h8-font-number);
}

.interval-table__state {
  display: grid;
  min-height: 180px;
  place-items: center;
  color: var(--h8-color-text-muted);
}
</style>
