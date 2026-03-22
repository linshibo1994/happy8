<template>
  <aside class="app-sidebar" :class="{ collapsed: isCollapsed }">
    <nav class="sidebar-nav">
      <router-link
        v-for="item in menuItems"
        :key="item.path"
        :to="item.path"
        class="nav-item"
        :class="{ active: isActive(item.path) }"
      >
        <span class="nav-icon" v-html="item.icon"></span>
        <span class="nav-label" v-show="!isCollapsed">{{ item.label }}</span>
      </router-link>
    </nav>

    <button class="collapse-btn" @click="toggleSidebar">
      <span class="collapse-icon" v-html="isCollapsed ? '&#9654;' : '&#9664;'"></span>
    </button>
  </aside>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'

const route = useRoute()
const appStore = useAppStore()

const isCollapsed = computed(() => appStore.sidebarCollapsed)

const menuItems = [
  { path: '/', label: '仪表盘', icon: '&#128200;' },
  { path: '/predict', label: '预测中心', icon: '&#9889;' },
  { path: '/history', label: '历史数据', icon: '&#128196;' },
  { path: '/analysis', label: '数据分析', icon: '&#128202;' },
  { path: '/comparison', label: '批量对比', icon: '&#128203;' },
  { path: '/settings', label: '系统设置', icon: '&#9881;' }
]

const isActive = (path: string) => {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}

const toggleSidebar = () => appStore.toggleSidebar()
</script>

<style scoped>
.app-sidebar {
  width: 200px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 60px;
  left: 0;
  bottom: 40px;
  transition: width 0.3s ease;
  z-index: 90;
}

.app-sidebar.collapsed {
  width: 64px;
}

.sidebar-nav {
  flex: 1;
  padding: 16px 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all 0.3s;
}

.nav-item:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.nav-item.active {
  background: rgba(0, 212, 255, 0.12);
  color: var(--color-primary);
  border-left: 3px solid var(--color-primary);
}

.nav-icon {
  font-size: 18px;
  width: 24px;
  text-align: center;
}

.nav-label {
  font-size: 14px;
  white-space: nowrap;
}

.collapsed .nav-item {
  justify-content: center;
  padding: 12px;
}

.collapse-btn {
  padding: 12px;
  background: transparent;
  border: none;
  border-top: 1px solid var(--border-color);
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.3s;
}

.collapse-btn:hover {
  background: var(--bg-hover);
  color: var(--color-primary);
}

.collapse-icon {
  font-size: 12px;
}
</style>
