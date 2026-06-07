<script setup lang="ts">
import { Flame, Snowflake } from 'lucide-vue-next'

import NumberBall from '@/components/balls/NumberBall.vue'

import DashboardPanel from './DashboardPanel.vue'

const props = defineProps<{
  hotNumbers: number[]
  coldNumbers: number[]
  intersectionNumbers: number[]
}>()
</script>

<template>
  <DashboardPanel title="热冷号摘要" kicker="号码分层">
    <template #icon>
      <Flame :size="16" />
    </template>

    <div class="hot-cold-summary">
      <section aria-label="近期热号">
        <h3>
          <Flame :size="14" aria-hidden="true" />
          近期热号
        </h3>
        <div class="hot-cold-summary__balls">
          <NumberBall v-for="number in props.hotNumbers" :key="number" :value="number" variant="draw" size="small" />
        </div>
      </section>

      <section aria-label="低频冷号">
        <h3>
          <Snowflake :size="14" aria-hidden="true" />
          低频冷号
        </h3>
        <div class="hot-cold-summary__balls">
          <NumberBall v-for="number in props.coldNumbers" :key="number" :value="number" variant="miss" size="small" />
        </div>
      </section>

      <section aria-label="预测交集">
        <h3>预测交集</h3>
        <div v-if="props.intersectionNumbers.length" class="hot-cold-summary__balls">
          <NumberBall
            v-for="number in props.intersectionNumbers"
            :key="number"
            :value="number"
            variant="intersection"
            size="small"
          />
        </div>
        <p v-else>暂无交集，等待更多预测样本。</p>
      </section>
    </div>
  </DashboardPanel>
</template>

<style scoped>
.hot-cold-summary {
  display: grid;
  gap: 16px;
}

.hot-cold-summary section {
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 14px;
}

.hot-cold-summary section:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

.hot-cold-summary h3 {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 10px;
  color: var(--h8-color-text);
  font-size: 13px;
  line-height: 1.3;
}

.hot-cold-summary__balls {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.hot-cold-summary p {
  margin: 0;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.5;
}
</style>
