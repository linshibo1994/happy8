<script setup lang="ts">
import { computed } from 'vue'

import type { PredictionPhase } from '@/types'

const props = withDefaults(
  defineProps<{
    progress: number
    phase: PredictionPhase
    message: string
    status?: 'idle' | 'running' | 'success' | 'error'
    compact?: boolean
  }>(),
  {
    status: 'idle',
    compact: false,
  },
)

const stages: Array<{ phase: PredictionPhase; label: string; range: string }> = [
  { phase: 'permission', label: '权限校验', range: '0-10%' },
  { phase: 'data', label: '数据准备', range: '10-25%' },
  { phase: 'feature', label: '特征分析', range: '25-50%' },
  { phase: 'compute', label: '算法计算', range: '50-80%' },
  { phase: 'validate', label: '结果校验', range: '80-95%' },
  { phase: 'done', label: '完成输出', range: '95-100%' },
]

const activeIndex = computed(() => {
  if (props.phase === 'error') {
    return Math.max(0, stages.findIndex((stage) => props.progress < Number(stage.range.split('-')[1].replace('%', ''))))
  }

  return Math.max(0, stages.findIndex((stage) => stage.phase === props.phase))
})

const normalizedProgress = computed(() => Math.min(100, Math.max(0, Math.round(props.progress))))
</script>

<template>
  <div class="stage-track" :class="{ 'stage-track--compact': compact, 'stage-track--error': status === 'error' }">
    <div class="stage-track__header">
      <div>
        <span class="stage-track__kicker">{{ status === 'error' ? '执行异常' : '阶段进度' }}</span>
        <strong>{{ message }}</strong>
      </div>
      <span class="stage-track__percent">{{ normalizedProgress }}%</span>
    </div>

    <div class="stage-track__bar" role="progressbar" :aria-valuenow="normalizedProgress" aria-valuemin="0" aria-valuemax="100">
      <span :style="{ width: `${normalizedProgress}%` }" />
    </div>

    <ol v-if="!compact" class="stage-track__nodes" aria-label="预测执行阶段">
      <li
        v-for="(stage, index) in stages"
        :key="stage.phase"
        :class="{
          'is-complete': normalizedProgress >= Number(stage.range.split('-')[1].replace('%', '')) || phase === 'done',
          'is-active': index === activeIndex && status !== 'error',
          'is-error': index === activeIndex && status === 'error',
        }"
      >
        <span aria-hidden="true" />
        <div>
          <strong>{{ stage.label }}</strong>
          <small>{{ stage.range }}</small>
        </div>
      </li>
    </ol>
  </div>
</template>

<style scoped>
.stage-track {
  display: grid;
  gap: 14px;
}

.stage-track__header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
}

.stage-track__header strong {
  display: block;
  margin-top: 3px;
  color: var(--h8-color-text);
  font-size: 16px;
  line-height: 1.35;
}

.stage-track__kicker {
  color: var(--h8-color-data-blue);
  font-size: 12px;
  font-weight: 700;
}

.stage-track__percent {
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-size: 28px;
  font-weight: 700;
  line-height: 1;
}

.stage-track__bar {
  height: 9px;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--h8-color-data-blue) 12%, var(--h8-color-line));
}

.stage-track__bar span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--h8-color-data-blue);
  transition: width 220ms ease;
}

.stage-track__nodes {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.stage-track__nodes li {
  display: grid;
  gap: 7px;
  min-width: 0;
  color: var(--h8-color-text-muted);
}

.stage-track__nodes li > span {
  width: 100%;
  height: 5px;
  border-radius: 999px;
  background: var(--h8-color-line);
}

.stage-track__nodes strong,
.stage-track__nodes small {
  display: block;
  overflow-wrap: anywhere;
}

.stage-track__nodes strong {
  color: inherit;
  font-size: 12px;
  line-height: 1.2;
}

.stage-track__nodes small {
  margin-top: 2px;
  font-family: var(--h8-font-number);
  font-size: 11px;
}

.stage-track__nodes li.is-complete,
.stage-track__nodes li.is-active {
  color: var(--h8-color-data-blue);
}

.stage-track__nodes li.is-complete > span,
.stage-track__nodes li.is-active > span {
  background: var(--h8-color-data-blue);
}

.stage-track__nodes li.is-error {
  color: var(--h8-color-risk-orange);
}

.stage-track__nodes li.is-error > span,
.stage-track--error .stage-track__bar span {
  background: var(--h8-color-risk-orange);
}

.stage-track--error .stage-track__kicker,
.stage-track--error .stage-track__percent {
  color: var(--h8-color-risk-orange);
}

.stage-track--compact {
  gap: 8px;
}

.stage-track--compact .stage-track__header strong {
  font-size: 13px;
}

.stage-track--compact .stage-track__percent {
  font-size: 18px;
}

.stage-track--compact .stage-track__bar {
  height: 6px;
}

@media (max-width: 900px) {
  .stage-track__nodes {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}
</style>
