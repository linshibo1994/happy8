<template>
  <Teleport to="body">
    <TransitionGroup name="toast" tag="div" class="toast-container">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        :class="['toast', `toast-${notification.type}`]"
      >
        <div class="toast-icon">
          <span v-if="notification.type === 'success'" v-html="'&#10003;'"></span>
          <span v-else-if="notification.type === 'error'" v-html="'&#10007;'"></span>
          <span v-else-if="notification.type === 'warning'" v-html="'&#9888;'"></span>
          <span v-else v-html="'&#8505;'"></span>
        </div>
        <div class="toast-content">{{ notification.message }}</div>
        <button class="toast-close" @click="close(notification.id)">&times;</button>
      </div>
    </TransitionGroup>
  </Teleport>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

const notifications = computed(() => appStore.notifications)

const close = (id: number) => {
  appStore.removeNotification(id)
}
</script>

<style scoped>
.toast-container {
  position: fixed;
  top: 80px;
  right: 20px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 12px;
  pointer-events: none;
}

.toast {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 280px;
  max-width: 400px;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  pointer-events: auto;
  backdrop-filter: blur(10px);
}

.toast-success {
  border-left: 4px solid var(--color-success);
}

.toast-error {
  border-left: 4px solid var(--color-error);
}

.toast-warning {
  border-left: 4px solid var(--color-warning);
}

.toast-info {
  border-left: 4px solid var(--color-primary);
}

.toast-icon {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-size: 14px;
}

.toast-success .toast-icon {
  background: rgba(16, 185, 129, 0.2);
  color: var(--color-success);
}

.toast-error .toast-icon {
  background: rgba(239, 68, 68, 0.2);
  color: var(--color-error);
}

.toast-warning .toast-icon {
  background: rgba(245, 158, 11, 0.2);
  color: var(--color-warning);
}

.toast-info .toast-icon {
  background: rgba(0, 212, 255, 0.2);
  color: var(--color-primary);
}

.toast-content {
  flex: 1;
  font-size: 14px;
  color: var(--text-primary);
  line-height: 1.5;
}

.toast-close {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 18px;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s;
}

.toast-close:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

/* 动画 */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100%);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100%);
}

.toast-move {
  transition: transform 0.3s ease;
}
</style>
