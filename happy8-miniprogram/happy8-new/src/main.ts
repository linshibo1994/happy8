import { createSSRApp } from 'vue'
import { createPinia } from 'pinia'
import { createPersistedState } from 'pinia-plugin-persistedstate'
import App from './App.vue'

// 导入 Wot Design 配置
import wotConfig from '@/config/wot-design'

export function createApp() {
  const app = createSSRApp(App)

  // 创建 Pinia 实例
  const pinia = createPinia()

  // 配置持久化插件
  pinia.use(
    createPersistedState({
      storage: {
        getItem: (key: string) => {
          return uni.getStorageSync(key)
        },
        setItem: (key: string, value: string) => {
          uni.setStorageSync(key, value)
        },
      },
    })
  )

  app.use(pinia)

  // 配置 Wot Design（如果支持全局配置）
  if (typeof app.config !== 'undefined') {
    app.config.globalProperties.$wotConfig = wotConfig
  }

  return {
    app,
    Pinia: pinia,
  }
}