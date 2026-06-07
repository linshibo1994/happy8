<script setup lang="ts">
import { BrainCircuit, LockKeyhole, Timer } from 'lucide-vue-next'

import type { Algorithm, MembershipLevel } from '@/types'

import DashboardPanel from './DashboardPanel.vue'

const props = defineProps<{
  algorithms: Algorithm[]
  activeName: string
}>()

const emit = defineEmits<{
  select: [name: string]
}>()

function permissionLabel(level: MembershipLevel) {
  const labels: Record<MembershipLevel, string> = {
    free: '免费',
    vip: 'VIP',
    premium: 'Premium',
  }

  return labels[level]
}

function successRateText(rate: number) {
  return `${Math.round(rate * 100)}%`
}
</script>

<template>
  <DashboardPanel title="推荐算法" kicker="策略选择" description="按历史表现、耗时和权限等级综合排序。" tone="blue">
    <template #icon>
      <BrainCircuit :size="16" />
    </template>

    <ul class="recommended-algorithms" aria-label="推荐算法列表">
      <li v-for="algorithm in props.algorithms" :key="algorithm.name">
        <button
          class="recommended-algorithms__row"
          type="button"
          :class="{ 'recommended-algorithms__row--active': algorithm.name === props.activeName }"
          :disabled="!algorithm.enabled"
          :aria-pressed="algorithm.name === props.activeName"
          @click="emit('select', algorithm.name)"
        >
          <span class="recommended-algorithms__main">
            <strong>{{ algorithm.displayName }}</strong>
            <small>{{ algorithm.category }} · {{ algorithm.recommendedScenario }}</small>
          </span>
          <span class="recommended-algorithms__meta">
            <span :class="`recommended-algorithms__level recommended-algorithms__level--${algorithm.permissionLevel}`">
              <LockKeyhole v-if="algorithm.permissionLevel !== 'free'" :size="12" aria-hidden="true" />
              {{ permissionLabel(algorithm.permissionLevel) }}
            </span>
            <span>
              <Timer :size="12" aria-hidden="true" />
              {{ algorithm.averageCostMs }}ms
            </span>
            <span>{{ successRateText(algorithm.successRate) }}</span>
          </span>
        </button>
      </li>
    </ul>
  </DashboardPanel>
</template>

<style scoped>
.recommended-algorithms {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.recommended-algorithms__row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  width: 100%;
  min-height: 72px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: transparent;
  color: var(--h8-color-text);
  cursor: pointer;
  padding: 12px;
  text-align: left;
  transition: border-color 160ms ease, background 160ms ease, box-shadow 160ms ease;
}

.recommended-algorithms__row:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--h8-color-data-blue) 46%, var(--h8-color-line));
  background: color-mix(in srgb, var(--h8-color-data-blue) 5%, transparent);
}

.recommended-algorithms__row:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--h8-color-data-blue) 34%, transparent);
  outline-offset: 3px;
}

.recommended-algorithms__row:disabled {
  color: var(--h8-color-text-muted);
  cursor: not-allowed;
  opacity: 0.62;
}

.recommended-algorithms__row--active {
  border-color: var(--h8-color-data-blue);
  box-shadow: inset 3px 0 0 var(--h8-color-data-blue);
}

.recommended-algorithms__main {
  min-width: 0;
}

.recommended-algorithms__main strong,
.recommended-algorithms__main small {
  display: block;
}

.recommended-algorithms__main strong {
  color: var(--h8-color-text);
  font-size: 15px;
  line-height: 1.3;
}

.recommended-algorithms__main small {
  overflow: hidden;
  margin-top: 5px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.45;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.recommended-algorithms__meta {
  display: grid;
  justify-items: end;
  gap: 5px;
  color: var(--h8-color-text-muted);
  font-family: var(--h8-font-number);
  font-size: 12px;
  white-space: nowrap;
}

.recommended-algorithms__meta span {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.recommended-algorithms__level {
  color: var(--h8-color-text-muted);
  font-family: var(--h8-font-body);
  font-weight: 700;
}

.recommended-algorithms__level--free {
  color: var(--h8-color-turquoise);
}

.recommended-algorithms__level--vip,
.recommended-algorithms__level--premium {
  color: var(--h8-color-bronze);
}

@media (max-width: 420px) {
  .recommended-algorithms__row {
    grid-template-columns: 1fr;
  }

  .recommended-algorithms__meta {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-start;
  }
}
</style>
