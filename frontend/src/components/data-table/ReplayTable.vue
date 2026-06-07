<script setup lang="ts">
import NumberBall from '@/components/balls/NumberBall.vue'

type Density = 'comfortable' | 'compact'

export interface ReplayRow {
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

withDefaults(
  defineProps<{
    rows: ReplayRow[]
    density?: Density
    loading?: boolean
    emptyText?: string
  }>(),
  {
    density: 'comfortable',
    loading: false,
    emptyText: '当前筛选范围暂无复盘记录',
  },
)

const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
})

function formatDate(value: string) {
  return dateFormatter.format(new Date(value))
}

function formatConfidence(value: number) {
  return `${Math.round(value * 100)}%`
}

function isHit(row: ReplayRow, number: number) {
  return row.hits.includes(number)
}
</script>

<template>
  <div class="h8-replay-table" :class="`h8-replay-table--${density}`">
    <div v-if="loading" class="h8-replay-table__state">正在读取历史开奖数据</div>
    <div v-else-if="rows.length === 0" class="h8-replay-table__state">{{ emptyText }}</div>
    <div v-else class="h8-replay-table__scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">预测时间</th>
            <th scope="col">目标期号</th>
            <th scope="col">算法</th>
            <th scope="col">预测号码</th>
            <th scope="col">实际号码</th>
            <th scope="col">命中数</th>
            <th scope="col">置信度</th>
            <th scope="col">耗时</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.id">
            <td>{{ formatDate(row.predictedAt) }}</td>
            <td class="h8-replay-table__mono">{{ row.targetIssue }}</td>
            <td>{{ row.algorithmName }}</td>
            <td>
              <div class="h8-replay-table__balls" aria-label="预测号码">
                <NumberBall
                  v-for="number in row.predictedNumbers"
                  :key="number"
                  :value="number"
                  :variant="isHit(row, number) ? 'hit' : 'outline'"
                  size="tiny"
                />
              </div>
            </td>
            <td>
              <div class="h8-replay-table__balls" aria-label="实际开奖号码">
                <NumberBall
                  v-for="number in row.actualNumbers"
                  :key="number"
                  :value="number"
                  :variant="isHit(row, number) ? 'hit' : 'muted'"
                  size="tiny"
                />
              </div>
            </td>
            <td class="h8-replay-table__hit">{{ row.hits.length }}</td>
            <td class="h8-replay-table__mono">{{ formatConfidence(row.confidence) }}</td>
            <td class="h8-replay-table__mono">{{ row.elapsedMs }}ms</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.h8-replay-table {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
}

.h8-replay-table__scroll {
  overflow-x: auto;
}

.h8-replay-table table {
  width: 100%;
  min-width: 1060px;
  border-collapse: collapse;
}

.h8-replay-table th,
.h8-replay-table td {
  border-bottom: 1px solid var(--h8-color-line);
  text-align: left;
  vertical-align: middle;
}

.h8-replay-table--comfortable th,
.h8-replay-table--comfortable td {
  padding: 13px 14px;
}

.h8-replay-table--compact th,
.h8-replay-table--compact td {
  padding: 8px 10px;
}

.h8-replay-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--h8-color-surface);
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.h8-replay-table td {
  color: var(--h8-color-text);
  font-size: 13px;
}

.h8-replay-table tbody tr:hover {
  background: color-mix(in srgb, var(--h8-color-cinnabar) 5%, transparent);
}

.h8-replay-table__balls {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  min-width: 250px;
}

.h8-replay-table__mono {
  font-family: var(--h8-font-number);
}

.h8-replay-table__hit {
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-weight: 700;
}

.h8-replay-table__state {
  display: grid;
  min-height: 180px;
  place-items: center;
  color: var(--h8-color-text-muted);
}
</style>
