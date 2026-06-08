<script setup lang="ts">
import NumberBall from '@/components/balls/NumberBall.vue'
import type { SandboxEventMatch } from '@/types'

type Density = 'comfortable' | 'compact'

withDefaults(
  defineProps<{
    rows: SandboxEventMatch[]
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

function formatGroup(group: number[]) {
  return group.map((number) => String(number).padStart(2, '0')).join('-')
}

function eventLabel(row: SandboxEventMatch) {
  if (row.label) {
    return row.label
  }

  const labels = {
    consecutive: `${row.longest_length ?? 2}连号`,
    gap: '隔号',
    mixed: '连号隔号',
    interval: '间隔',
  }

  return labels[row.event_type]
}

function scopeLabel(row: SandboxEventMatch) {
  if (row.scope === 'global') {
    return '全局'
  }

  return row.zones?.map((zone) => `${zone}区`).join('、') || '八区'
}

function highlightedNumbers(row: SandboxEventMatch) {
  return new Set(row.groups.flat())
}
</script>

<template>
  <div class="sandbox-table" :class="`sandbox-table--${density}`">
    <div v-if="loading" class="sandbox-table__state">正在分析当前筛选规则</div>
    <div v-else-if="rows.length === 0" class="sandbox-table__state">当前条件下未找到命中期号</div>
    <div v-else class="sandbox-table__scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">期号</th>
            <th scope="col">开奖日期</th>
            <th scope="col">开奖号码</th>
            <th scope="col">事件类型</th>
            <th scope="col">分析口径</th>
            <th scope="col">命中号码组</th>
            <th scope="col">最长结构</th>
            <th scope="col">组数</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="`${row.issue}-${row.event_type}-${row.groups.map(formatGroup).join(',')}`">
            <td class="sandbox-table__mono">{{ row.issue }}</td>
            <td>{{ formatDate(row.openedAt ?? row.draw_date) }}</td>
            <td>
              <div class="sandbox-table__balls" :aria-label="`第 ${row.issue} 期开奖号码`">
                <NumberBall
                  v-for="number in row.numbers"
                  :key="number"
                  :value="number"
                  :variant="highlightedNumbers(row).has(number) ? 'hit' : 'muted'"
                  size="tiny"
                />
              </div>
            </td>
            <td>
              <span class="sandbox-table__tag">{{ eventLabel(row) }}</span>
            </td>
            <td>{{ scopeLabel(row) }}</td>
            <td>
              <div class="sandbox-table__groups">
                <span v-for="group in row.groups" :key="formatGroup(group)">{{ formatGroup(group) }}</span>
              </div>
            </td>
            <td>{{ row.longest_length ? `${row.longest_length}连` : '-' }}</td>
            <td class="sandbox-table__mono">{{ row.group_count ?? row.groups.length }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.sandbox-table {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
}

.sandbox-table__scroll {
  overflow-x: auto;
}

.sandbox-table table {
  width: 100%;
  min-width: 1120px;
  border-collapse: collapse;
}

.sandbox-table th,
.sandbox-table td {
  border-bottom: 1px solid var(--h8-color-line);
  text-align: left;
  vertical-align: middle;
}

.sandbox-table--comfortable th,
.sandbox-table--comfortable td {
  padding: 12px 14px;
}

.sandbox-table--compact th,
.sandbox-table--compact td {
  padding: 8px 10px;
}

.sandbox-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--h8-color-surface);
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.sandbox-table td {
  color: var(--h8-color-text);
  font-size: 13px;
}

.sandbox-table tbody tr {
  transition: background 140ms ease;
}

.sandbox-table tbody tr:hover {
  background: color-mix(in srgb, var(--h8-color-data-blue) 5%, transparent);
}

.sandbox-table__mono {
  font-family: var(--h8-font-number);
}

.sandbox-table__balls,
.sandbox-table__groups {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.sandbox-table__balls {
  min-width: 248px;
}

.sandbox-table__groups span,
.sandbox-table__tag {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: color-mix(in srgb, var(--h8-color-cinnabar) 8%, var(--h8-color-surface-strong));
  color: var(--h8-color-cinnabar);
  padding: 4px 6px;
  font-family: var(--h8-font-number);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.sandbox-table__tag {
  font-family: var(--h8-font-body);
}

.sandbox-table__state {
  display: grid;
  min-height: 180px;
  place-items: center;
  color: var(--h8-color-text-muted);
}
</style>
