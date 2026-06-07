<script setup lang="ts">
import { computed } from 'vue'
import { LockKeyhole, SlidersHorizontal } from 'lucide-vue-next'

import type { MembershipStatus } from '@/types'

export interface PredictionParameterSettings {
  recentWeight: number
  balanceWeight: number
  confidenceFloor: number
  excludeRecentHits: boolean
}

const props = defineProps<{
  mode: 'single' | 'batch'
  targetIssue: string
  analysisPeriods: number
  predictCount: number
  settings: PredictionParameterSettings
  membership: MembershipStatus
  requiredQuota: number
  isRunning: boolean
}>()

const emit = defineEmits<{
  'update:targetIssue': [value: string]
  'update:analysisPeriods': [value: number]
  'update:predictCount': [value: number]
  'update:settings': [value: PredictionParameterSettings]
}>()

const quotaUsageText = computed(() => {
  const modeText = props.mode === 'single' ? '单算法' : '批量'

  return `${modeText}预计消耗 ${props.requiredQuota} 次，今日剩余 ${props.membership.remainingPredictions} 次`
})

const hasEnoughQuota = computed(() => props.membership.remainingPredictions >= props.requiredQuota)

const updateNumber = (key: 'analysisPeriods' | 'predictCount', value: Event) => {
  const input = value.target as HTMLInputElement
  const numberValue = Number(input.value)

  if (Number.isNaN(numberValue)) {
    return
  }

  emit(`update:${key}`, numberValue)
}

const updateSetting = (key: keyof PredictionParameterSettings, value: number | boolean) => {
  emit('update:settings', {
    ...props.settings,
    [key]: value,
  })
}
</script>

<template>
  <aside class="parameter-panel" aria-label="预测参数面板">
    <header class="parameter-panel__header">
      <span class="section-kicker">参数面板</span>
      <h2>执行条件</h2>
      <p>目标期号、分析窗口和预测个数会同时影响单算法与批量预测。</p>
    </header>

    <div class="parameter-form">
      <label class="field">
        <span>目标期号</span>
        <input
          :value="targetIssue"
          :disabled="isRunning"
          inputmode="numeric"
          autocomplete="off"
          @input="emit('update:targetIssue', ($event.target as HTMLInputElement).value)"
        />
      </label>

      <label class="field">
        <span>分析期数</span>
        <div class="range-control">
          <input
            type="range"
            min="10"
            max="200"
            step="5"
            :value="analysisPeriods"
            :disabled="isRunning"
            @input="updateNumber('analysisPeriods', $event)"
          />
          <input
            class="range-control__number"
            type="number"
            min="10"
            max="200"
            :value="analysisPeriods"
            :disabled="isRunning"
            @input="updateNumber('analysisPeriods', $event)"
          />
        </div>
      </label>

      <label class="field">
        <span>预测个数</span>
        <div class="range-control">
          <input
            type="range"
            min="1"
            max="20"
            step="1"
            :value="predictCount"
            :disabled="isRunning"
            @input="updateNumber('predictCount', $event)"
          />
          <input
            class="range-control__number"
            type="number"
            min="1"
            max="20"
            :value="predictCount"
            :disabled="isRunning"
            @input="updateNumber('predictCount', $event)"
          />
        </div>
      </label>
    </div>

    <section class="advanced-panel" aria-labelledby="advanced-title">
      <div class="advanced-panel__title">
        <SlidersHorizontal :size="17" aria-hidden="true" />
        <h3 id="advanced-title">算法参数</h3>
      </div>

      <label class="field field--compact">
        <span>近期权重</span>
        <div class="range-control">
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            :value="settings.recentWeight"
            :disabled="isRunning"
            @input="updateSetting('recentWeight', Number(($event.target as HTMLInputElement).value))"
          />
          <strong>{{ settings.recentWeight }}%</strong>
        </div>
      </label>

      <label class="field field--compact">
        <span>区间均衡</span>
        <div class="range-control">
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            :value="settings.balanceWeight"
            :disabled="isRunning"
            @input="updateSetting('balanceWeight', Number(($event.target as HTMLInputElement).value))"
          />
          <strong>{{ settings.balanceWeight }}%</strong>
        </div>
      </label>

      <label class="field field--compact">
        <span>置信阈值</span>
        <div class="range-control">
          <input
            type="range"
            min="30"
            max="80"
            step="5"
            :value="settings.confidenceFloor"
            :disabled="isRunning"
            @input="updateSetting('confidenceFloor', Number(($event.target as HTMLInputElement).value))"
          />
          <strong>{{ settings.confidenceFloor }}%</strong>
        </div>
      </label>

      <label class="checkbox-field">
        <input
          type="checkbox"
          :checked="settings.excludeRecentHits"
          :disabled="isRunning"
          @change="updateSetting('excludeRecentHits', ($event.target as HTMLInputElement).checked)"
        />
        <span>降低上期已开号码权重</span>
      </label>
    </section>

    <section class="quota-panel" :class="{ 'quota-panel--warning': !hasEnoughQuota }" aria-label="会员权限">
      <LockKeyhole :size="18" aria-hidden="true" />
      <div>
        <strong>{{ membership.levelName }}</strong>
        <p>{{ quotaUsageText }}</p>
        <small v-if="!hasEnoughQuota">剩余次数不足，请减少批量算法数量或升级权益。</small>
        <small v-else>{{ membership.benefits.join(' / ') }}</small>
      </div>
    </section>
  </aside>
</template>

<style scoped>
.parameter-panel {
  display: grid;
  gap: 18px;
  align-content: start;
}

.parameter-panel__header h2 {
  margin: 3px 0 0;
  font-family: var(--h8-font-title);
  font-size: 22px;
  line-height: 1.2;
}

.parameter-panel__header p {
  margin: 8px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.parameter-form,
.advanced-panel {
  display: grid;
  gap: 14px;
}

.field {
  display: grid;
  gap: 8px;
}

.field > span,
.checkbox-field,
.quota-panel p,
.quota-panel small {
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.field > span {
  font-weight: 700;
}

.field input:not([type='range']) {
  width: 100%;
  min-height: 40px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-text);
  padding: 0 11px;
  outline: none;
}

.field input:not([type='range']):focus-visible {
  border-color: var(--h8-color-data-blue);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--h8-color-data-blue) 18%, transparent);
}

.field input:disabled,
.checkbox-field input:disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

.range-control {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 72px;
  align-items: center;
  gap: 10px;
}

.range-control input[type='range'] {
  accent-color: var(--h8-color-data-blue);
}

.range-control__number {
  text-align: center;
}

.range-control strong {
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-size: 13px;
  text-align: right;
}

.advanced-panel {
  border-top: 1px solid var(--h8-color-line);
  border-bottom: 1px solid var(--h8-color-line);
  padding: 16px 0;
}

.advanced-panel__title,
.quota-panel {
  display: flex;
  align-items: flex-start;
  gap: 9px;
}

.advanced-panel__title {
  color: var(--h8-color-cinnabar);
}

.advanced-panel__title h3 {
  margin: 0;
  color: var(--h8-color-text);
  font-size: 15px;
}

.field--compact {
  gap: 7px;
}

.checkbox-field {
  display: flex;
  align-items: center;
  gap: 8px;
}

.checkbox-field input {
  width: 16px;
  height: 16px;
  accent-color: var(--h8-color-cinnabar);
}

.quota-panel {
  border: 1px solid color-mix(in srgb, var(--h8-color-bronze) 45%, var(--h8-color-line));
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-bronze) 9%, var(--h8-color-surface-strong));
  color: var(--h8-color-bronze);
  padding: 14px;
}

.quota-panel strong {
  display: block;
  color: var(--h8-color-text);
}

.quota-panel p,
.quota-panel small {
  display: block;
  margin: 4px 0 0;
  line-height: 1.45;
}

.quota-panel--warning {
  border-color: color-mix(in srgb, var(--h8-color-risk-orange) 55%, var(--h8-color-line));
  background: color-mix(in srgb, var(--h8-color-risk-orange) 10%, var(--h8-color-surface-strong));
  color: var(--h8-color-risk-orange);
}
</style>
