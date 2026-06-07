<script setup lang="ts">
import { History } from 'lucide-vue-next'

import NumberBall from '@/components/balls/NumberBall.vue'

import DashboardPanel from './DashboardPanel.vue'

interface RecentPredictionItem {
  id: string
  targetIssue: string
  algorithmName: string
  createdText: string
  confidenceText: string
  elapsedText: string
  numbers: number[]
  hitNumbers: number[]
  statusText: string
}

const props = defineProps<{
  records: RecentPredictionItem[]
}>()

function numberVariant(record: RecentPredictionItem, number: number) {
  if (record.hitNumbers.includes(number)) {
    return 'hit'
  }

  return record.statusText.includes('命中') ? 'miss' : 'prediction'
}
</script>

<template>
  <DashboardPanel title="最近预测" kicker="复盘入口">
    <template #icon>
      <History :size="16" />
    </template>

    <div v-if="props.records.length" class="recent-predictions" aria-label="最近预测列表">
      <article v-for="record in props.records" :key="record.id" class="recent-predictions__item">
        <header>
          <div>
            <strong>第 {{ record.targetIssue }} 期</strong>
            <span>{{ record.algorithmName }} · {{ record.createdText }}</span>
          </div>
          <em>{{ record.statusText }}</em>
        </header>

        <div class="recent-predictions__balls" aria-label="预测号码">
          <NumberBall
            v-for="number in record.numbers"
            :key="number"
            :value="number"
            :variant="numberVariant(record, number)"
            size="table"
          />
        </div>

        <footer>
          <span>置信度 {{ record.confidenceText }}</span>
          <span>耗时 {{ record.elapsedText }}</span>
        </footer>
      </article>
    </div>

    <p v-else class="recent-predictions__empty">暂无预测记录，完成一次预测后会在这里复盘。</p>
  </DashboardPanel>
</template>

<style scoped>
.recent-predictions {
  display: grid;
  gap: 14px;
}

.recent-predictions__item {
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 14px;
}

.recent-predictions__item:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

.recent-predictions__item header,
.recent-predictions__item footer {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.recent-predictions__item strong {
  display: block;
  color: var(--h8-color-text);
  font-size: 14px;
  line-height: 1.3;
}

.recent-predictions__item span {
  display: block;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.4;
}

.recent-predictions__item em {
  color: var(--h8-color-cinnabar);
  font-size: 12px;
  font-style: normal;
  font-weight: 700;
  white-space: nowrap;
}

.recent-predictions__balls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 10px 0;
}

.recent-predictions__item footer {
  color: var(--h8-color-text-muted);
}

.recent-predictions__empty {
  margin: 0;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.5;
}

@media (max-width: 420px) {
  .recent-predictions__item header,
  .recent-predictions__item footer {
    display: grid;
  }
}
</style>
