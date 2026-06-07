<script setup lang="ts">
import { computed } from 'vue'
import { AlertCircle, CheckCircle2, LoaderCircle, PlayCircle } from 'lucide-vue-next'

import DashboardPanel from './DashboardPanel.vue'

type QuickActionStatus = 'idle' | 'running' | 'success' | 'error'

const props = defineProps<{
  nextIssue: string
  algorithmName: string
  analysisPeriods: number
  predictCount: number
  remaining: number
  hasQuota: boolean
  isSubmitting: boolean
  status: QuickActionStatus
  statusMessage: string
}>()

const emit = defineEmits<{
  start: []
}>()

const buttonText = computed(() => {
  if (!props.hasQuota) {
    return '次数不足'
  }

  if (props.isSubmitting) {
    return '预测中'
  }

  if (props.status === 'success') {
    return '再次预测'
  }

  return '一键预测'
})
</script>

<template>
  <DashboardPanel title="下期行动" kicker="快捷入口" tone="blue">
    <template #icon>
      <PlayCircle :size="16" />
    </template>

    <div class="quick-predict__target">
      <span>下期期号</span>
      <strong>{{ props.nextIssue }}</strong>
    </div>

    <dl class="quick-predict__params">
      <div>
        <dt>默认算法</dt>
        <dd>{{ props.algorithmName }}</dd>
      </div>
      <div>
        <dt>分析期数</dt>
        <dd>{{ props.analysisPeriods }}</dd>
      </div>
      <div>
        <dt>预测个数</dt>
        <dd>{{ props.predictCount }}</dd>
      </div>
    </dl>

    <button
      class="quick-predict__button"
      type="button"
      :disabled="!props.hasQuota || props.isSubmitting"
      :aria-busy="props.isSubmitting"
      @click="emit('start')"
    >
      <LoaderCircle v-if="props.isSubmitting" class="quick-predict__spin" :size="18" aria-hidden="true" />
      <CheckCircle2 v-else-if="props.status === 'success'" :size="18" aria-hidden="true" />
      <AlertCircle v-else-if="!props.hasQuota || props.status === 'error'" :size="18" aria-hidden="true" />
      <PlayCircle v-else :size="18" aria-hidden="true" />
      <span>{{ buttonText }}</span>
    </button>

    <p class="quick-predict__status" :class="`quick-predict__status--${props.status}`">
      {{ props.statusMessage }}，今日剩余 {{ props.remaining }} 次。
    </p>
    <p class="quick-predict__notice">预测仅作数据分析参考，不构成确定性承诺。</p>
  </DashboardPanel>
</template>

<style scoped>
.quick-predict__target {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 14px;
}

.quick-predict__target span {
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.quick-predict__target strong {
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-size: 24px;
  line-height: 1;
  letter-spacing: 0;
}

.quick-predict__params {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 14px 0;
}

.quick-predict__params div {
  min-width: 0;
}

.quick-predict__params dt {
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.quick-predict__params dd {
  overflow: hidden;
  margin: 4px 0 0;
  color: var(--h8-color-text);
  font-weight: 700;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quick-predict__button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  min-height: 42px;
  border: 1px solid var(--h8-color-cinnabar);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-cinnabar);
  color: #fff;
  cursor: pointer;
  font-weight: 700;
  transition: background 160ms ease, border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.quick-predict__button:hover:not(:disabled) {
  background: color-mix(in srgb, var(--h8-color-cinnabar) 88%, var(--h8-color-ink));
  border-color: color-mix(in srgb, var(--h8-color-cinnabar) 88%, var(--h8-color-ink));
  transform: translateY(-1px);
}

.quick-predict__button:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--h8-color-data-blue) 36%, transparent);
  outline-offset: 3px;
}

.quick-predict__button:disabled {
  border-color: var(--h8-color-line);
  background: color-mix(in srgb, var(--h8-color-gray) 42%, var(--h8-color-line));
  color: var(--h8-color-surface-strong);
  cursor: not-allowed;
  transform: none;
}

.quick-predict__spin {
  animation: spin 900ms linear infinite;
}

.quick-predict__status,
.quick-predict__notice {
  margin: 10px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.quick-predict__status--success {
  color: var(--h8-color-turquoise);
}

.quick-predict__status--error {
  color: var(--h8-color-risk-orange);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 420px) {
  .quick-predict__target {
    display: grid;
  }

  .quick-predict__params {
    grid-template-columns: 1fr;
  }
}

@media (prefers-reduced-motion: reduce) {
  .quick-predict__button,
  .quick-predict__spin {
    animation: none;
    transition: none;
  }
}
</style>
