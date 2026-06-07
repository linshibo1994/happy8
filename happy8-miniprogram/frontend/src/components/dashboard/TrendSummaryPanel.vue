<script setup lang="ts">
import { computed } from 'vue'
import { BarChart3 } from 'lucide-vue-next'

import type { LotteryResult } from '@/types'

import DashboardPanel from './DashboardPanel.vue'

const props = defineProps<{
  result: LotteryResult
}>()

const zoneEntries = computed(() => Object.entries(props.result.zoneDistribution))
const maxZoneValue = computed(() => Math.max(...zoneEntries.value.map(([, count]) => count), 1))
</script>

<template>
  <DashboardPanel title="走势摘要" kicker="结构观察">
    <template #icon>
      <BarChart3 :size="16" />
    </template>

    <div class="trend-summary__numbers" aria-label="开奖结构摘要">
      <div>
        <span>和值</span>
        <strong>{{ props.result.sum }}</strong>
      </div>
      <div>
        <span>奇偶比</span>
        <strong>{{ props.result.oddCount }}:{{ props.result.evenCount }}</strong>
      </div>
      <div>
        <span>大小比</span>
        <strong>{{ props.result.bigCount }}:{{ props.result.smallCount }}</strong>
      </div>
    </div>

    <ul class="trend-summary__zones" aria-label="区间分布">
      <li v-for="[zone, count] in zoneEntries" :key="zone">
        <span>{{ zone }}</span>
        <div class="trend-summary__bar">
          <i :style="{ width: `${Math.round((count / maxZoneValue) * 100)}%` }" />
        </div>
        <strong>{{ count }}</strong>
      </li>
    </ul>
  </DashboardPanel>
</template>

<style scoped>
.trend-summary__numbers {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.trend-summary__numbers div {
  min-width: 0;
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 10px;
}

.trend-summary__numbers span {
  display: block;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.trend-summary__numbers strong {
  display: block;
  margin-top: 5px;
  color: var(--h8-color-text);
  font-family: var(--h8-font-number);
  font-size: 20px;
  line-height: 1;
  letter-spacing: 0;
}

.trend-summary__zones {
  display: grid;
  gap: 10px;
  margin: 16px 0 0;
  padding: 0;
  list-style: none;
}

.trend-summary__zones li {
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr) 24px;
  align-items: center;
  gap: 10px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.trend-summary__bar {
  height: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--h8-color-line) 76%, var(--h8-color-gray));
}

.trend-summary__bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--h8-color-data-blue), var(--h8-color-turquoise));
}

.trend-summary__zones strong {
  color: var(--h8-color-text);
  font-family: var(--h8-font-number);
  text-align: right;
}

@media (max-width: 420px) {
  .trend-summary__numbers {
    grid-template-columns: 1fr;
  }
}
</style>
