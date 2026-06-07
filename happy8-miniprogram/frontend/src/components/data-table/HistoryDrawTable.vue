<script setup lang="ts">
import NumberBall from '@/components/balls/NumberBall.vue'

type Density = 'comfortable' | 'compact'

export interface HistoryDrawRow {
  issue: string
  openedAt: string
  numbers: number[]
  sum: number
  oddEvenText: string
  bigSmallText: string
  zoneDistribution: Record<string, number>
}

withDefaults(
  defineProps<{
    rows: HistoryDrawRow[]
    density?: Density
    loading?: boolean
    emptyText?: string
  }>(),
  {
    density: 'comfortable',
    loading: false,
    emptyText: '当前筛选范围暂无开奖记录',
  },
)

const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
})

function formatDate(value: string) {
  return dateFormatter.format(new Date(value))
}

function copyRow(row: HistoryDrawRow) {
  const text = [
    `期号：${row.issue}`,
    `开奖日期：${formatDate(row.openedAt)}`,
    `号码：${row.numbers.join(', ')}`,
    `和值：${row.sum}`,
    `奇偶：${row.oddEvenText}`,
    `大小：${row.bigSmallText}`,
  ].join('\n')

  void navigator.clipboard?.writeText(text)
}
</script>

<template>
  <div class="h8-table" :class="`h8-table--${density}`">
    <div v-if="loading" class="h8-table__state">正在读取历史开奖数据</div>
    <div v-else-if="rows.length === 0" class="h8-table__state">{{ emptyText }}</div>
    <div v-else class="h8-table__scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">期号</th>
            <th scope="col">开奖日期</th>
            <th scope="col">开奖号码</th>
            <th scope="col">和值</th>
            <th scope="col">奇偶比</th>
            <th scope="col">大小比</th>
            <th scope="col">区间分布</th>
            <th scope="col">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.issue">
            <td class="h8-table__mono">{{ row.issue }}</td>
            <td>{{ formatDate(row.openedAt) }}</td>
            <td>
              <div class="h8-table__balls" :aria-label="`第 ${row.issue} 期开奖号码`">
                <NumberBall
                  v-for="number in row.numbers"
                  :key="number"
                  :value="number"
                  variant="draw"
                  size="tiny"
                />
              </div>
            </td>
            <td class="h8-table__strong">{{ row.sum }}</td>
            <td>{{ row.oddEvenText }}</td>
            <td>{{ row.bigSmallText }}</td>
            <td>
              <div class="h8-table__zones" aria-label="区间分布">
                <span
                  v-for="(count, zone) in row.zoneDistribution"
                  :key="zone"
                  :style="{ width: `${(count / 20) * 100}%` }"
                >
                  {{ zone }} {{ count }}
                </span>
              </div>
            </td>
            <td>
              <button class="h8-table__action" type="button" @click="copyRow(row)">复制</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.h8-table {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
}

.h8-table__scroll {
  overflow-x: auto;
}

.h8-table table {
  width: 100%;
  min-width: 980px;
  border-collapse: collapse;
}

.h8-table th,
.h8-table td {
  border-bottom: 1px solid var(--h8-color-line);
  text-align: left;
  vertical-align: middle;
}

.h8-table--comfortable th,
.h8-table--comfortable td {
  padding: 13px 14px;
}

.h8-table--compact th,
.h8-table--compact td {
  padding: 8px 10px;
}

.h8-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--h8-color-surface);
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.h8-table td {
  color: var(--h8-color-text);
  font-size: 13px;
}

.h8-table tbody tr {
  transition: background 140ms ease;
}

.h8-table tbody tr:hover {
  background: color-mix(in srgb, var(--h8-color-cinnabar) 5%, transparent);
}

.h8-table__balls {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  min-width: 240px;
}

.h8-table__mono,
.h8-table__strong {
  font-family: var(--h8-font-number);
}

.h8-table__strong {
  color: var(--h8-color-cinnabar);
  font-weight: 700;
}

.h8-table__zones {
  display: flex;
  width: 190px;
  height: 24px;
  overflow: hidden;
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  background: var(--h8-color-surface);
}

.h8-table__zones span {
  display: grid;
  min-width: 22px;
  place-items: center;
  color: #fff;
  font-size: 10px;
  line-height: 1;
  white-space: nowrap;
}

.h8-table__zones span:nth-child(1) {
  background: var(--h8-color-cinnabar);
}

.h8-table__zones span:nth-child(2) {
  background: var(--h8-color-data-blue);
}

.h8-table__zones span:nth-child(3) {
  background: var(--h8-color-turquoise);
}

.h8-table__zones span:nth-child(4) {
  background: var(--h8-color-bronze);
}

.h8-table__action {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  padding: 5px 8px;
  cursor: pointer;
  white-space: nowrap;
}

.h8-table__action:hover {
  border-color: var(--h8-color-cinnabar);
  color: var(--h8-color-cinnabar);
}

.h8-table__state {
  display: grid;
  min-height: 180px;
  place-items: center;
  color: var(--h8-color-text-muted);
}
</style>
