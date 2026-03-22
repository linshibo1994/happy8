<template>
  <header class="app-header">
    <div class="header-left">
      <div class="logo">
        <LotteryBall :number="8" size="sm" />
        <span class="logo-text">Happy8</span>
      </div>
      <div class="system-title">快乐8智能预测系统</div>
    </div>

    <div class="header-right">
      <div class="status-item" :class="apiStatusClass">
        <span class="status-dot"></span>
        <span>API {{ apiStatusText }}</span>
      </div>
      <div class="status-item" v-if="latestPeriod">
        <span class="status-label">最新期号</span>
        <span class="status-value">{{ latestPeriod }}</span>
      </div>
      <button class="theme-toggle" @click="toggleTheme">
        <span v-if="isDark" v-html="'&#9788;'"></span>
        <span v-else v-html="'&#9790;'"></span>
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'
import { useLotteryStore } from '@/stores/lottery'
import LotteryBall from '@/components/common/LotteryBall.vue'

const appStore = useAppStore()
const lotteryStore = useLotteryStore()

const isDark = computed(() => appStore.isDarkTheme)
const latestPeriod = computed(() => lotteryStore.latestResult?.period || '')
const apiStatusText = computed(() => {
  if (appStore.apiStatus === 'connected') return '已连接'
  if (appStore.apiStatus === 'error') return '异常'
  return '未连接'
})
const apiStatusClass = computed(() => ({
  'status-connected': appStore.apiStatus === 'connected',
  'status-error': appStore.apiStatus === 'error',
  'status-disconnected': appStore.apiStatus === 'disconnected'
}))

const toggleTheme = () => appStore.toggleTheme()
</script>

<style scoped>
.app-header {
  height: 60px;
  background: var(--bg-secondary);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-text {
  font-size: 20px;
  font-weight: 700;
  background: var(--gradient-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.system-title {
  color: var(--text-secondary);
  font-size: 14px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-connected .status-dot {
  background: var(--color-success);
  box-shadow: 0 0 8px var(--color-success);
}

.status-disconnected .status-dot {
  background: var(--text-muted);
}

.status-error .status-dot {
  background: var(--color-error);
  box-shadow: 0 0 8px var(--color-error);
}

.status-label {
  color: var(--text-muted);
}

.status-value {
  color: var(--color-primary);
  font-weight: 600;
}

.theme-toggle {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s;
}

.theme-toggle:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-glow);
}
</style>
