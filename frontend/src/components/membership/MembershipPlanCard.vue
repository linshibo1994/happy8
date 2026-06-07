<script setup lang="ts">
import { CheckCircle2 } from 'lucide-vue-next'

import type { MembershipLevel } from '@/types'

export interface MembershipPlanView {
  id: string
  level: MembershipLevel
  name: string
  priceText: string
  quotaText: string
  description: string
  benefits: string[]
  recommended?: boolean
}

defineProps<{
  plan: MembershipPlanView
  current: boolean
}>()
</script>

<template>
  <article class="plan-card" :class="{ 'plan-card--current': current, 'plan-card--recommended': plan.recommended }">
    <div class="plan-card__header">
      <span>{{ current ? '当前等级' : plan.recommended ? '推荐路径' : '可选套餐' }}</span>
      <h3>{{ plan.name }}</h3>
      <strong>{{ plan.priceText }}</strong>
      <p>{{ plan.description }}</p>
    </div>

    <div class="plan-card__quota">{{ plan.quotaText }}</div>

    <ul>
      <li v-for="benefit in plan.benefits" :key="benefit">
        <CheckCircle2 :size="16" aria-hidden="true" />
        <span>{{ benefit }}</span>
      </li>
    </ul>

    <button type="button" :disabled="current">
      {{ current ? '已生效' : '查看订单状态' }}
    </button>
  </article>
</template>

<style scoped lang="scss">
.plan-card {
  display: grid;
  gap: 16px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 10px 28px color-mix(in srgb, var(--h8-color-shadow) 70%, transparent);
  padding: 18px;
}

.plan-card--current {
  border-color: color-mix(in srgb, var(--h8-color-turquoise) 48%, var(--h8-color-line));
}

.plan-card--recommended {
  border-color: color-mix(in srgb, var(--h8-color-bronze) 58%, var(--h8-color-line));
}

.plan-card__header span {
  color: var(--h8-color-cinnabar);
  font-size: 12px;
  font-weight: 800;
}

.plan-card h3 {
  margin: 5px 0 0;
  font-family: var(--h8-font-title);
  font-size: 22px;
}

.plan-card strong {
  display: block;
  margin-top: 8px;
  color: var(--h8-color-bronze);
  font-family: var(--h8-font-number);
  font-size: 24px;
}

.plan-card p {
  margin: 8px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.6;
}

.plan-card__quota {
  width: fit-content;
  border-radius: 999px;
  background: color-mix(in srgb, var(--h8-color-data-blue) 10%, transparent);
  color: var(--h8-color-data-blue);
  padding: 6px 10px;
  font-family: var(--h8-font-number);
  font-weight: 800;
}

.plan-card ul {
  display: grid;
  gap: 9px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.plan-card li {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  color: var(--h8-color-text);
  line-height: 1.45;
}

.plan-card li svg {
  margin-top: 2px;
  color: var(--h8-color-turquoise);
}

.plan-card button {
  min-height: 40px;
  border: 1px solid var(--h8-color-cinnabar);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-cinnabar);
  color: #fff;
  font-weight: 800;
  cursor: pointer;
}

.plan-card button:disabled {
  border-color: var(--h8-color-line);
  background: color-mix(in srgb, var(--h8-color-line) 38%, var(--h8-color-surface));
  color: var(--h8-color-text-muted);
  cursor: default;
}
</style>
