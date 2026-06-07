<script setup lang="ts">
import { computed } from 'vue'
import { Gauge, Trophy } from 'lucide-vue-next'

import DashboardPanel from './DashboardPanel.vue'

const props = defineProps<{
  remaining: number
  dailyLimit: number
  usedCount: number
  predictionCount: number
  hitCount: number
  bestAlgorithm: string
  bestHitCount: number
  averageConfidence: number
}>()

const quotaPercent = computed(() => {
  if (props.dailyLimit <= 0) {
    return 0
  }

  return Math.round((props.remaining / props.dailyLimit) * 100)
})

const averageConfidenceText = computed(() => `${Math.round(props.averageConfidence * 100)}%`)
</script>

<template>
  <DashboardPanel title="次数与表现" kicker="今日概况" :tone="props.remaining > 0 ? 'success' : 'warning'">
    <template #icon>
      <Gauge :size="16" />
    </template>

    <div class="usage-performance__quota">
      <div>
        <span>今日剩余</span>
        <strong>{{ props.remaining }}</strong>
      </div>
      <span class="usage-performance__limit">已用 {{ props.usedCount }} / {{ props.dailyLimit }}</span>
    </div>

    <div class="usage-performance__bar" aria-label="今日预测次数剩余比例">
      <span :style="{ width: `${quotaPercent}%` }" />
    </div>

    <dl class="usage-performance__stats" aria-label="近期命中表现">
      <div>
        <dt>近7天预测</dt>
        <dd>{{ props.predictionCount }} 次</dd>
      </div>
      <div>
        <dt>命中号码</dt>
        <dd>{{ props.hitCount }} 个</dd>
      </div>
      <div>
        <dt>
          <Trophy :size="13" aria-hidden="true" />
          最高算法
        </dt>
        <dd>{{ props.bestAlgorithm }} · {{ props.bestHitCount }} 中</dd>
      </div>
      <div>
        <dt>平均置信度</dt>
        <dd>{{ averageConfidenceText }}</dd>
      </div>
    </dl>
  </DashboardPanel>
</template>

<style scoped>
.usage-performance__quota {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
}

.usage-performance__quota span {
  display: block;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.usage-performance__quota strong {
  display: block;
  margin-top: 4px;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 38px;
  line-height: 1;
  letter-spacing: 0;
}

.usage-performance__limit {
  white-space: nowrap;
}

.usage-performance__bar {
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--h8-color-line) 72%, var(--h8-color-gray));
  margin-top: 14px;
}

.usage-performance__bar span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--h8-color-turquoise);
}

.usage-performance__stats {
  display: grid;
  gap: 10px;
  margin: 16px 0 0;
}

.usage-performance__stats div {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border-top: 1px solid var(--h8-color-line);
  padding-top: 10px;
}

.usage-performance__stats dt {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.usage-performance__stats dd {
  min-width: 0;
  margin: 0;
  color: var(--h8-color-text);
  font-weight: 700;
  text-align: right;
}
</style>
