<template>
  <div class="app-container" :class="{ 'sidebar-collapsed': sidebarCollapsed }">
    <AppHeader />
    <AppSidebar />
    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
    <AppFooter />
    <LoadingOverlay :visible="loading" :message="loadingMessage" />
    <Toast />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { useLotteryStore } from '@/stores/lottery'
import { checkHealth } from '@/api/lottery'
import { setErrorHandler } from '@/api/request'
import AppHeader from '@/components/layout/AppHeader.vue'
import AppSidebar from '@/components/layout/AppSidebar.vue'
import AppFooter from '@/components/layout/AppFooter.vue'
import LoadingOverlay from '@/components/common/LoadingOverlay.vue'
import Toast from '@/components/common/Toast.vue'

const appStore = useAppStore()
const lotteryStore = useLotteryStore()

const loading = ref(false)
const loadingMessage = ref('')

const sidebarCollapsed = computed(() => appStore.sidebarCollapsed)

setErrorHandler((message) => {
  appStore.notify.error(message)
})

const initApp = async () => {
  loading.value = true
  loadingMessage.value = '正在初始化系统...'

  try {
    const health = await checkHealth()
    appStore.setApiStatus(health.status === 'healthy' ? 'connected' : 'disconnected')

    loadingMessage.value = '正在加载开奖数据...'
    await Promise.all([
      lotteryStore.fetchLatest(),
      lotteryStore.fetchHistory(200),
      lotteryStore.fetchStatistics()
    ])
  } catch (error) {
    console.error('初始化失败:', error)
    appStore.setApiStatus('error')
  } finally {
    loading.value = false
    loadingMessage.value = ''
  }
}

onMounted(() => {
  initApp()
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.main-content {
  flex: 1;
  margin-top: 60px;
  margin-bottom: 40px;
  margin-left: 200px;
  min-height: calc(100vh - 100px);
  background: var(--bg-primary);
  transition: margin-left 0.3s ease;
}

.sidebar-collapsed .main-content {
  margin-left: 64px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 768px) {
  .main-content {
    margin-left: 0;
  }
}
</style>
