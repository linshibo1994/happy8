<script setup lang="ts">
import { computed } from 'vue'
import { CalendarClock, Database, Hash, Sigma } from 'lucide-vue-next'

import NumberBall from '@/components/balls/NumberBall.vue'
import type { LotteryResult } from '@/types'

import DashboardPanel from './DashboardPanel.vue'

const props = defineProps<{
  result: LotteryResult
  openedText: string
}>()

const drawMeta = computed(() => [
  {
    label: '和值',
    value: props.result.sum,
    icon: Sigma,
  },
  {
    label: '奇偶',
    value: `${props.result.oddCount}:${props.result.evenCount}`,
    icon: Hash,
  },
  {
    label: '大小',
    value: `${props.result.bigCount}:${props.result.smallCount}`,
    icon: Hash,
  },
])
</script>

<template>
  <DashboardPanel title="最新开奖" kicker="实时开奖" tone="accent">
    <template #icon>
      <Database :size="16" />
    </template>

    <div class="latest-draw__issue">
      <div>
        <span>第 {{ props.result.issue }} 期</span>
        <strong>已开奖</strong>
      </div>
      <time :datetime="props.result.openedAt">
        <CalendarClock :size="15" aria-hidden="true" />
        {{ props.openedText }}
      </time>
    </div>

    <div class="latest-draw__balls" aria-label="最新开奖号码">
      <NumberBall
        v-for="number in props.result.numbers"
        :key="number"
        :value="number"
        variant="draw"
      />
    </div>

    <dl class="latest-draw__metrics" aria-label="开奖统计">
      <div v-for="item in drawMeta" :key="item.label">
        <dt>
          <component :is="item.icon" :size="14" aria-hidden="true" />
          {{ item.label }}
        </dt>
        <dd>{{ item.value }}</dd>
      </div>
    </dl>
  </DashboardPanel>
</template>

<style scoped>
.latest-draw__issue {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 14px;
}

.latest-draw__issue span,
.latest-draw__issue time {
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.latest-draw__issue strong {
  display: block;
  margin-top: 3px;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 22px;
  line-height: 1.1;
  letter-spacing: 0;
}

.latest-draw__issue time {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}

.latest-draw__balls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}

.latest-draw__metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0 0;
}

.latest-draw__metrics div {
  min-width: 0;
  border-top: 1px solid var(--h8-color-line);
  padding-top: 10px;
}

.latest-draw__metrics dt {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.latest-draw__metrics dd {
  margin: 4px 0 0;
  color: var(--h8-color-text);
  font-family: var(--h8-font-number);
  font-size: 20px;
  font-weight: 700;
  line-height: 1.1;
}

@media (max-width: 420px) {
  .latest-draw__issue {
    align-items: flex-start;
    flex-direction: column;
  }

  .latest-draw__metrics {
    grid-template-columns: 1fr;
  }
}
</style>
