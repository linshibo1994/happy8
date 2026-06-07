<script setup lang="ts">
import { Activity, ShieldCheck } from 'lucide-vue-next'

import DashboardPanel from './DashboardPanel.vue'

interface SystemStatusItem {
  name: string
  state: string
  detail: string
  tone: 'success' | 'blue' | 'warning' | 'muted'
}

const props = defineProps<{
  items: SystemStatusItem[]
}>()
</script>

<template>
  <DashboardPanel title="系统状态" kicker="运行监控" tone="success">
    <template #icon>
      <ShieldCheck :size="16" />
    </template>

    <ul class="system-status" aria-label="系统状态列表">
      <li v-for="item in props.items" :key="item.name" class="system-status__item">
        <span class="system-status__dot" :class="`system-status__dot--${item.tone}`" aria-hidden="true" />
        <div>
          <strong>{{ item.name }}</strong>
          <small>{{ item.detail }}</small>
        </div>
        <span class="system-status__state">{{ item.state }}</span>
      </li>
    </ul>

    <p class="system-status__note">
      <Activity :size="14" aria-hidden="true" />
      真实接口接入后可替换为同步延迟、引擎耗时和错误率。
    </p>
  </DashboardPanel>
</template>

<style scoped>
.system-status {
  display: grid;
  gap: 12px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.system-status__item {
  display: grid;
  grid-template-columns: 10px minmax(0, 1fr) auto;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 12px;
}

.system-status__dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--h8-color-gray);
}

.system-status__dot--success {
  background: var(--h8-color-turquoise);
}

.system-status__dot--blue {
  background: var(--h8-color-data-blue);
}

.system-status__dot--warning {
  background: var(--h8-color-risk-orange);
}

.system-status__item strong {
  display: block;
  color: var(--h8-color-text);
  font-size: 14px;
  line-height: 1.3;
}

.system-status__item small {
  display: block;
  overflow: hidden;
  margin-top: 2px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.system-status__state {
  color: var(--h8-color-text);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.system-status__note {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  margin: 14px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.45;
}
</style>
