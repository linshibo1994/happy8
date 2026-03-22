import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { SystemInfo, AppConfig } from '@/types'
import { API_CONFIG, STORAGE_KEYS } from '@/constants'

export const useAppStore = defineStore(
  'app',
  () => {
    // 状态
    const loading = ref(false)
    const systemInfo = ref<SystemInfo | null>(null)
    const config = ref<AppConfig>({
      api_base_url: API_CONFIG.BASE_URL,
      version: '1.0.0',
      debug: import.meta.env.DEV,
      cache_duration: API_CONFIG.CACHE_DURATION,
      max_retry_times: API_CONFIG.MAX_RETRY,
    })

    // 网络状态
    const networkType = ref<string>('unknown')
    const isOnline = ref(true)

    // 主题相关
    const isDarkMode = ref(false)
    const primaryColor = ref('#d32f2f')

    // 计算属性
    const safeAreaTop = computed(() => {
      return systemInfo.value?.safeArea?.top || 0
    })

    const safeAreaBottom = computed(() => {
      return systemInfo.value?.safeArea?.bottom || 0
    })

    const statusBarHeight = computed(() => {
      return systemInfo.value?.statusBarHeight || 0
    })

    // 方法
    const initApp = async () => {
      try {
        // 获取系统信息
        await getSystemInfo()

        // 监听网络状态
        monitorNetworkStatus()

        // 加载用户配置
        loadUserConfig()

        console.log('应用初始化完成')
      } catch (error) {
        console.error('应用初始化失败:', error)
      }
    }

    const getSystemInfo = () => {
      return new Promise<void>((resolve, reject) => {
        uni.getSystemInfo({
          success: res => {
            systemInfo.value = res as SystemInfo
            resolve()
          },
          fail: reject,
        })
      })
    }

    const monitorNetworkStatus = () => {
      // 获取网络类型
      uni.getNetworkType({
        success: res => {
          networkType.value = res.networkType
          isOnline.value = res.networkType !== 'none'
        },
      })

      // 监听网络状态变化
      uni.onNetworkStatusChange(res => {
        networkType.value = res.networkType
        isOnline.value = res.isConnected

        if (!res.isConnected) {
          showToast('网络连接已断开')
        }
      })
    }

    const loadUserConfig = () => {
      try {
        const savedConfig = uni.getStorageSync(STORAGE_KEYS.APP_CONFIG)
        if (savedConfig) {
          config.value = { ...config.value, ...savedConfig }
        }

        // 加载主题设置
        const savedTheme = uni.getStorageSync(STORAGE_KEYS.THEME_MODE)
        if (savedTheme) {
          isDarkMode.value = savedTheme === 'dark'
        }
      } catch (error) {
        console.error('加载用户配置失败:', error)
      }
    }

    const saveUserConfig = () => {
      try {
        uni.setStorageSync(STORAGE_KEYS.APP_CONFIG, config.value)
        uni.setStorageSync(STORAGE_KEYS.THEME_MODE, isDarkMode.value ? 'dark' : 'light')
      } catch (error) {
        console.error('保存用户配置失败:', error)
      }
    }

    const setLoading = (value: boolean) => {
      loading.value = value
    }

    const setSystemInfo = (info: SystemInfo) => {
      systemInfo.value = info
    }

    const updateConfig = (newConfig: Partial<AppConfig>) => {
      config.value = { ...config.value, ...newConfig }
      saveUserConfig()
    }

    const setTheme = (isDark: boolean) => {
      isDarkMode.value = isDark
      saveUserConfig()

      // 更新页面主题
      if (isDark) {
        uni.setTabBarStyle({
          backgroundColor: '#1a1a1a',
          color: '#999999',
          selectedColor: primaryColor.value,
          borderStyle: 'black',
        })
      } else {
        uni.setTabBarStyle({
          backgroundColor: '#ffffff',
          color: '#999999',
          selectedColor: primaryColor.value,
          borderStyle: 'black',
        })
      }
    }

    const setPrimaryColor = (color: string) => {
      primaryColor.value = color
      saveUserConfig()
    }

    // 工具方法
    const showToast = (title: string, icon: 'success' | 'error' | 'loading' | 'none' = 'none') => {
      uni.showToast({
        title,
        icon,
        duration: 2000,
      })
    }

    const showLoading = (title: string = '加载中...') => {
      setLoading(true)
      uni.showLoading({ title })
    }

    const hideLoading = () => {
      setLoading(false)
      uni.hideLoading()
    }

    const showModal = (title: string, content: string): Promise<UniApp.ShowModalRes> => {
      return new Promise(resolve => {
        uni.showModal({
          title,
          content,
          success: resolve,
          fail: resolve,
        })
      })
    }

    const showActionSheet = (itemList: string[]): Promise<UniApp.ShowActionSheetRes> => {
      return new Promise((resolve, reject) => {
        uni.showActionSheet({
          itemList,
          success: resolve,
          fail: reject,
        })
      })
    }

    const setNavigationBarTitle = (title: string) => {
      uni.setNavigationBarTitle({ title })
    }

    const setNavigationBarColor = (frontColor: '#000000' | '#ffffff', backgroundColor: string) => {
      uni.setNavigationBarColor({
        frontColor,
        backgroundColor,
      })
    }

    // 错误上报
    const reportError = (error: unknown) => {
      if (config.value.debug) {
        console.error('应用错误:', error)
      }

      // 这里可以集成错误上报服务
      // 例如：Sentry、Bugsnag等
    }

    // 性能监控
    const reportPerformance = (name: string, duration: number) => {
      if (config.value.debug) {
        console.log(`性能监控 - ${name}: ${duration}ms`)
      }

      // 这里可以集成性能监控服务
    }

    // 存储管理
    const clearStorage = () => {
      try {
        uni.clearStorageSync()
        showToast('缓存清理成功')
      } catch (error) {
        console.error('清理缓存失败:', error)
        showToast('清理缓存失败', 'error')
      }
    }

    const getStorageInfo = (): Promise<UniNamespace.GetStorageInfoSuccess> => {
      return new Promise((resolve, reject) => {
        uni.getStorageInfo({
          success: resolve,
          fail: reject,
        })
      })
    }

    return {
      // 状态
      loading,
      systemInfo,
      config,
      networkType,
      isOnline,
      isDarkMode,
      primaryColor,

      // 计算属性
      safeAreaTop,
      safeAreaBottom,
      statusBarHeight,

      // 方法
      initApp,
      setLoading,
      setSystemInfo,
      updateConfig,
      setTheme,
      setPrimaryColor,
      showToast,
      showLoading,
      hideLoading,
      showModal,
      showActionSheet,
      setNavigationBarTitle,
      setNavigationBarColor,
      reportError,
      reportPerformance,
      clearStorage,
      getStorageInfo,
    }
  },
  {
  persist: {
    key: 'app-store',
    storage: {
      getItem: (key: string) => uni.getStorageSync(key),
      setItem: (key: string, value: string) => uni.setStorageSync(key, value),
    },
    paths: ['config', 'isDarkMode', 'primaryColor'],
  },
  }
)
