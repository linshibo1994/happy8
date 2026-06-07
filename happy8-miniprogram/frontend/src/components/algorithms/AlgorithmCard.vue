<script setup lang="ts">
import { computed } from 'vue'
import { Lock, Timer, TrendingUp } from 'lucide-vue-next'

import type { Algorithm, MembershipLevel } from '@/types'

const props = defineProps<{
  algorithm: Algorithm
  selected: boolean
  locked: boolean
}>()

defineEmits<{
  select: [name: string]
}>()

const permissionText: Record<MembershipLevel, string> = {
  free: '免费',
  vip: 'VIP',
  premium: 'Premium',
}

const successRateText = computed(() => `${Math.round(props.algorithm.successRate * 100)}%`)
const averageCostText = computed(() => `${(props.algorithm.averageCostMs / 1000).toFixed(1)}s`)
</script>

<template>
  <button
    class="algorithm-card"
    :class="{ 'algorithm-card--selected': selected, 'algorithm-card--locked': locked }"
    type="button"
    :aria-pressed="selected"
    @click="$emit('select', algorithm.name)"
  >
    <span class="algorithm-card__top">
      <span>
        <small>{{ algorithm.category }}</small>
        <strong>{{ algorithm.displayName }}</strong>
      </span>
      <span class="algorithm-card__badge" :class="`algorithm-card__badge--${algorithm.permissionLevel}`">
        <Lock v-if="locked" :size="14" aria-hidden="true" />
        {{ permissionText[algorithm.permissionLevel] }}
      </span>
    </span>

    <span class="algorithm-card__scenario">{{ algorithm.recommendedScenario }}</span>

    <span class="algorithm-card__metrics" aria-label="算法指标">
      <span>
        <TrendingUp :size="15" aria-hidden="true" />
        {{ successRateText }}
      </span>
      <span>
        <Timer :size="15" aria-hidden="true" />
        {{ averageCostText }}
      </span>
      <span>复杂度 {{ algorithm.complexity }}</span>
    </span>

    <span v-if="locked" class="algorithm-card__lock-note">升级后可执行，仍可查看算法档案。</span>
  </button>
</template>

<style scoped lang="scss">
.algorithm-card {
  display: grid;
  gap: 14px;
  width: 100%;
  min-height: 168px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 10px 28px color-mix(in srgb, var(--h8-color-shadow) 70%, transparent);
  color: var(--h8-color-text);
  padding: 16px;
  text-align: left;
  cursor: pointer;
  transition:
    border-color 180ms ease,
    box-shadow 180ms ease,
    transform 180ms ease;
}

.algorithm-card:hover,
.algorithm-card:focus-visible {
  border-color: color-mix(in srgb, var(--h8-color-cinnabar) 54%, var(--h8-color-line));
  box-shadow: 0 14px 34px color-mix(in srgb, var(--h8-color-shadow) 90%, transparent);
  outline: none;
  transform: translateY(-1px);
}

.algorithm-card--selected {
  border-color: var(--h8-color-cinnabar);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--h8-color-cinnabar) 13%, transparent);
}

.algorithm-card--locked {
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--h8-color-bronze) 8%, transparent), transparent 42%),
    var(--h8-color-surface-strong);
}

.algorithm-card__top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.algorithm-card small {
  display: block;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.algorithm-card strong {
  display: block;
  margin-top: 3px;
  color: var(--h8-color-text);
  font-family: var(--h8-font-title);
  font-size: 19px;
  line-height: 1.25;
}

.algorithm-card__badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  min-height: 26px;
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  padding: 4px 9px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.algorithm-card__badge--free {
  color: var(--h8-color-turquoise);
}

.algorithm-card__badge--vip,
.algorithm-card__badge--premium {
  border-color: color-mix(in srgb, var(--h8-color-bronze) 54%, var(--h8-color-line));
  color: var(--h8-color-bronze);
}

.algorithm-card__scenario {
  color: var(--h8-color-text-muted);
  font-size: 14px;
  line-height: 1.55;
}

.algorithm-card__metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.algorithm-card__metrics span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 28px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--h8-color-jade) 72%, var(--h8-color-surface-strong));
  color: var(--h8-color-text);
  padding: 5px 8px;
  font-family: var(--h8-font-number);
  font-size: 12px;
}

.algorithm-card__lock-note {
  color: var(--h8-color-risk-orange);
  font-size: 13px;
}
</style>
