<template>
  <footer class="app-footer">
    <div class="footer-left">
      <span class="copyright">Happy8 Predictor &copy; 2026</span>
    </div>

    <div class="footer-center">
      <div class="api-status" :class="statusClass">
        <span class="status-dot"></span>
        <span class="status-text">API {{ statusText }}</span>
      </div>
    </div>

    <div class="footer-right">
      <span class="version">v{{ version }}</span>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

const version = computed(() => appStore.version)
const apiStatus = computed(() => appStore.apiStatus)

const statusClass = computed(() => ({
  'status-connected': apiStatus.value === 'connected',
  'status-disconnected': apiStatus.value === 'disconnected',
  'status-error': apiStatus.value === 'error'
}))

const statusText = computed(() => {
  if (apiStatus.value === 'connected') return '已连接'
  if (apiStatus.value === 'error') return '错误'
  return '未连接'
})
</script>

<style scoped>
.app-footer {
  height: 40px;
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 100;
  font-size: 12px;
}

.footer-left,
.footer-center,
.footer-right {
  display: flex;
  align-items: center;
}

.copyright,
.version {
  color: var(--text-muted);
}

.api-status {
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

.status-connected .status-dot {
  background: var(--color-success);
  box-shadow: 0 0 6px var(--color-success);
}

.status-disconnected .status-dot {
  background: var(--text-muted);
}

.status-error .status-dot {
  background: var(--color-error);
}

.status-text {
  color: var(--text-secondary);
}
</style>
