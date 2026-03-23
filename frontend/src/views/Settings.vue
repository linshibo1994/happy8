<template>
  <div class="settings-page">
    <DataVBorder :type="1">
      <div class="settings-content">
        <h2 class="page-title">系统设置</h2>

        <div class="settings-section">
          <h3 class="section-title">性能配置</h3>
          <div class="settings-group">
            <div class="setting-item">
              <div class="setting-info">
                <span class="setting-label">GPU 加速</span>
                <span class="setting-desc">启用 GPU 进行深度学习计算</span>
              </div>
              <label class="switch">
                <input type="checkbox" v-model="settings.gpuEnabled" />
                <span class="slider"></span>
              </label>
            </div>

            <div class="setting-item">
              <div class="setting-info">
                <span class="setting-label">并行处理</span>
                <span class="setting-desc">使用多线程并行计算</span>
              </div>
              <label class="switch">
                <input type="checkbox" v-model="settings.parallelEnabled" />
                <span class="slider"></span>
              </label>
            </div>

            <div class="setting-item">
              <div class="setting-info">
                <span class="setting-label">工作线程数</span>
                <span class="setting-desc">并行计算的线程数量</span>
              </div>
              <input
                type="number"
                v-model.number="settings.workerCount"
                min="1"
                max="16"
                class="setting-input"
              />
            </div>
          </div>
        </div>

        <div class="settings-section">
          <h3 class="section-title">显示设置</h3>
          <div class="settings-group">
            <div class="setting-item">
              <div class="setting-info">
                <span class="setting-label">深色主题</span>
                <span class="setting-desc">切换界面主题风格</span>
              </div>
              <label class="switch">
                <input type="checkbox" v-model="settings.darkTheme" />
                <span class="slider"></span>
              </label>
            </div>
          </div>
        </div>

        <div class="settings-section">
          <h3 class="section-title">API 配置</h3>
          <div class="settings-group">
            <div class="setting-item">
              <div class="setting-info">
                <span class="setting-label">后端地址</span>
                <span class="setting-desc">API 服务器地址</span>
              </div>
              <input
                type="text"
                v-model="settings.apiUrl"
                class="setting-input wide"
                placeholder="http://localhost:7000"
              />
            </div>
          </div>
        </div>

        <div class="settings-section">
          <h3 class="section-title">缓存管理</h3>
          <div class="settings-group">
            <div class="setting-item">
              <div class="setting-info">
                <span class="setting-label">清除缓存</span>
                <span class="setting-desc">清除本地存储的数据缓存</span>
              </div>
              <button class="btn-danger" @click="clearCache">清除</button>
            </div>
          </div>
        </div>

        <div class="settings-section">
          <h3 class="section-title">关于系统</h3>
          <div class="about-info">
            <div class="about-item">
              <span class="about-label">版本号</span>
              <span class="about-value">{{ systemInfo.version }}</span>
            </div>
            <div class="about-item">
              <span class="about-label">构建日期</span>
              <span class="about-value">{{ systemInfo.buildTime }}</span>
            </div>
            <div class="about-item">
              <span class="about-label">应用名称</span>
              <span class="about-value">{{ systemInfo.appName }}</span>
            </div>
            <div class="about-item">
              <span class="about-label">GPU 支持</span>
              <span class="about-value">{{ systemConfig.gpuEnabled ? '已启用' : '未启用' }}</span>
            </div>
            <div class="about-item">
              <span class="about-label">并行处理</span>
              <span class="about-value">{{ systemConfig.parallelEnabled ? '已启用' : '未启用' }}</span>
            </div>
            <div class="about-item">
              <span class="about-label">最大工作线程</span>
              <span class="about-value">{{ systemConfig.maxWorkers }}</span>
            </div>
          </div>
        </div>

        <div class="settings-actions">
          <button class="btn-primary" @click="saveSettings">保存设置</button>
          <button class="btn-secondary" @click="resetSettings">重置默认</button>
        </div>
      </div>
    </DataVBorder>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import DataVBorder from '@/components/common/DataVBorder.vue'
import { getSystemConfig, getVersion } from '@/api/predict'

const appStore = useAppStore()

const settings = ref({
  gpuEnabled: true,
  parallelEnabled: true,
  workerCount: 4,
  darkTheme: true,
  apiUrl: appStore.apiUrl
})

// 系统信息
const systemInfo = ref({
  version: 'v2.0.0',
  buildTime: '2024-12-09',
  appName: '快乐8智能预测系统',
  features: [] as string[]
})

// 系统配置
const systemConfig = ref({
  gpuEnabled: false,
  parallelEnabled: false,
  maxWorkers: 4,
  maxPeriods: 1000,
  maxPredictions: 10
})

const loadingConfig = ref(false)

// 加载系统配置
const loadSystemConfig = async () => {
  loadingConfig.value = true
  try {
    const [configRes, versionRes] = await Promise.all([
      getSystemConfig(),
      getVersion()
    ])

    if (configRes.success && configRes.data) {
      const config = configRes.data
      systemConfig.value = {
        gpuEnabled: config.features?.gpu_enabled || false,
        parallelEnabled: config.features?.parallel_enabled || false,
        maxWorkers: config.features?.max_workers || 4,
        maxPeriods: config.limits?.max_periods || 1000,
        maxPredictions: config.limits?.max_predictions || 10
      }
      // 更新设置默认值
      settings.value.gpuEnabled = systemConfig.value.gpuEnabled
      settings.value.parallelEnabled = systemConfig.value.parallelEnabled
      settings.value.workerCount = systemConfig.value.maxWorkers
    }

    if (versionRes.success && versionRes.data) {
      const version = versionRes.data
      systemInfo.value = {
        version: `v${version.version}`,
        buildTime: version.build_time ? new Date(version.build_time).toLocaleDateString() : '未知',
        appName: version.app_name || '快乐8智能预测系统',
        features: version.features || []
      }
    }
  } catch (error) {
    console.error('加载系统配置失败:', error)
    // 保持默认值
  } finally {
    loadingConfig.value = false
  }
}

const loadSettings = () => {
  settings.value.gpuEnabled = appStore.gpuEnabled
  settings.value.parallelEnabled = appStore.parallelEnabled
  settings.value.workerCount = appStore.workerCount
  settings.value.darkTheme = appStore.isDarkTheme
  settings.value.apiUrl = appStore.apiUrl
}

const saveSettings = () => {
  appStore.setGpuEnabled(settings.value.gpuEnabled)
  appStore.setParallelEnabled(settings.value.parallelEnabled)
  appStore.setWorkerCount(settings.value.workerCount)
  appStore.setTheme(settings.value.darkTheme ? 'dark' : 'light')
  appStore.setApiUrl(settings.value.apiUrl)
  appStore.notify.success('设置已保存')
}

const resetSettings = () => {
  settings.value = {
    gpuEnabled: systemConfig.value.gpuEnabled,
    parallelEnabled: systemConfig.value.parallelEnabled,
    workerCount: systemConfig.value.maxWorkers,
    darkTheme: appStore.isDarkTheme,
    apiUrl: appStore.apiUrl
  }
}

const clearCache = () => {
  if (confirm('确定要清除所有缓存吗?')) {
    localStorage.clear()
    appStore.notify.success('缓存已清除')
  }
}

onMounted(() => {
  loadSettings()
  loadSystemConfig()
})
</script>

<style scoped>
.settings-page {
  padding: 24px;
  max-width: 800px;
  margin: 0 auto;
}

.settings-content {
  padding: 24px;
}

.page-title {
  font-size: 24px;
  color: var(--text-primary);
  margin-bottom: 32px;
}

.settings-section {
  margin-bottom: 32px;
}

.section-title {
  font-size: 16px;
  color: var(--color-primary);
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-color);
}

.settings-group {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.setting-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.setting-label {
  font-size: 14px;
  color: var(--text-primary);
}

.setting-desc {
  font-size: 12px;
  color: var(--text-muted);
}

.setting-input {
  padding: 8px 12px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  width: 100px;
}

.setting-input.wide {
  width: 250px;
}

/* Switch 开关样式 */
.switch {
  position: relative;
  display: inline-block;
  width: 48px;
  height: 24px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: var(--bg-hover);
  transition: 0.3s;
  border-radius: 24px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.3s;
  border-radius: 50%;
}

input:checked + .slider {
  background-color: var(--color-primary);
}

input:checked + .slider:before {
  transform: translateX(24px);
}

.btn-danger {
  padding: 8px 16px;
  background: var(--color-error);
  border: none;
  border-radius: 6px;
  color: white;
  cursor: pointer;
}

.about-info {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.about-item {
  display: flex;
  justify-content: space-between;
  padding: 12px;
  background: var(--bg-secondary);
  border-radius: 6px;
}

.about-label {
  color: var(--text-secondary);
}

.about-value {
  color: var(--text-primary);
  font-weight: 500;
}

.settings-actions {
  display: flex;
  gap: 16px;
  margin-top: 32px;
}

.btn-primary,
.btn-secondary {
  padding: 12px 32px;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  background: var(--gradient-primary);
  border: none;
  color: white;
}

.btn-primary:hover {
  box-shadow: var(--shadow-glow);
}

.btn-secondary {
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
}

.btn-secondary:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
}
</style>
