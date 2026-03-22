import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getApiBaseUrl, setApiBaseUrl } from '@/api/request'

// 通知类型
export interface AppNotification {
  id: number
  type: 'success' | 'error' | 'warning' | 'info'
  message: string
  duration?: number
}

let notificationId = 0
const THEME_STORAGE_KEY = 'app_theme'

const loadInitialTheme = (): 'dark' | 'light' => {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY)
    return saved === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

const applyThemeToDocument = (theme: 'dark' | 'light') => {
  if (typeof document !== 'undefined') {
    document.documentElement.setAttribute('data-theme', theme)
  }
}

export const useAppStore = defineStore('app', () => {
  // 主题
  const theme = ref<'dark' | 'light'>(loadInitialTheme())

  // 侧边栏折叠状态
  const sidebarCollapsed = ref(false)

  // GPU状态
  const gpuEnabled = ref(true)
  const gpuAvailable = ref(false)

  // 并行处理状态
  const parallelEnabled = ref(true)

  // 工作线程数
  const workerCount = ref(4)

  // API 配置
  const apiUrl = ref(getApiBaseUrl())

  // API状态
  const apiStatus = ref<'connected' | 'disconnected' | 'error'>('disconnected')

  // 系统版本
  const version = ref('2.0.0')

  // 通知列表
  const notifications = ref<AppNotification[]>([])

  // 计算属性
  const isDarkTheme = computed(() => theme.value === 'dark')

  // 方法
  const toggleTheme = () => {
    setTheme(theme.value === 'dark' ? 'light' : 'dark')
  }

  const toggleSidebar = () => {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  const setGpuEnabled = (enabled: boolean) => {
    gpuEnabled.value = enabled
  }

  const setParallelEnabled = (enabled: boolean) => {
    parallelEnabled.value = enabled
  }

  const setWorkerCount = (count: number) => {
    workerCount.value = Math.max(1, Math.min(16, count))
  }

  const setTheme = (newTheme: 'dark' | 'light') => {
    theme.value = newTheme
    applyThemeToDocument(newTheme)
    try {
      localStorage.setItem(THEME_STORAGE_KEY, newTheme)
    } catch {
      // 忽略本地存储异常
    }
  }

  const setApiUrl = (url: string) => {
    setApiBaseUrl(url)
    apiUrl.value = getApiBaseUrl()
  }

  const setApiStatus = (status: 'connected' | 'disconnected' | 'error') => {
    apiStatus.value = status
  }

  // 添加通知
  const addNotification = (
    type: AppNotification['type'],
    message: string,
    duration: number = 5000
  ) => {
    const id = ++notificationId
    const notification: AppNotification = { id, type, message, duration }
    notifications.value.push(notification)

    // 自动移除
    if (duration > 0) {
      setTimeout(() => {
        removeNotification(id)
      }, duration)
    }

    return id
  }

  // 移除通知
  const removeNotification = (id: number) => {
    const index = notifications.value.findIndex(n => n.id === id)
    if (index > -1) {
      notifications.value.splice(index, 1)
    }
  }

  // 快捷通知方法
  const notify = {
    success: (message: string, duration?: number) => addNotification('success', message, duration),
    error: (message: string, duration?: number) => addNotification('error', message, duration ?? 8000),
    warning: (message: string, duration?: number) => addNotification('warning', message, duration),
    info: (message: string, duration?: number) => addNotification('info', message, duration)
  }

  // 初始化主题变量
  applyThemeToDocument(theme.value)

  return {
    theme,
    sidebarCollapsed,
    gpuEnabled,
    gpuAvailable,
    parallelEnabled,
    workerCount,
    apiUrl,
    apiStatus,
    version,
    notifications,
    isDarkTheme,
    toggleTheme,
    toggleSidebar,
    setGpuEnabled,
    setParallelEnabled,
    setWorkerCount,
    setTheme,
    setApiUrl,
    setApiStatus,
    addNotification,
    removeNotification,
    notify
  }
})
