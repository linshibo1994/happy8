<template>
  <div class="predict-page">
    <div class="predict-layout">
      <!-- 左侧: 算法选择和参数配置 -->
      <div class="predict-sidebar">
        <DataVBorder :type="2">
          <div class="sidebar-content">
            <h3 class="section-title">算法选择</h3>
            <div class="algorithm-list">
              <div
                v-for="(methods, category) in algorithmGroups"
                :key="category"
                class="algorithm-group"
              >
                <div class="group-title">{{ getCategoryName(category) }}</div>
                <div
                  v-for="method in methods"
                  :key="method.id"
                  class="algorithm-item"
                  :class="{ active: selectedMethod === method.id }"
                  @click="selectMethod(method.id)"
                >
                  {{ method.name }}
                </div>
              </div>
            </div>

            <h3 class="section-title">参数配置</h3>
            <div class="params-form">
              <div class="form-item">
                <label>历史期数</label>
                <input type="number" v-model.number="params.periods" min="10" max="1000" />
              </div>
              <div class="form-item">
                <label>预测数量</label>
                <input type="number" v-model.number="params.count" min="1" max="10" />
              </div>
              <div class="form-item checkbox">
                <label>
                  <input type="checkbox" v-model="params.useGpu" />
                  启用 GPU 加速
                </label>
              </div>
              <div class="form-item checkbox">
                <label>
                  <input type="checkbox" v-model="params.parallel" />
                  并行处理
                </label>
              </div>
              <div class="form-item checkbox">
                <label>
                  <input type="checkbox" v-model="params.duplex" />
                  复式投注
                </label>
              </div>
              <template v-if="params.duplex">
                <div class="form-item">
                  <label>复式红球数</label>
                  <input type="number" v-model.number="params.redCount" min="7" max="20" />
                </div>
                <div class="form-item">
                  <label>复式蓝球数</label>
                  <input type="number" v-model.number="params.blueCount" min="1" max="16" />
                </div>
                <div class="duplex-cost">
                  <div>预计注数：{{ duplexStakeCount }}</div>
                  <div>预计金额：¥{{ duplexTotalCost }}</div>
                </div>
              </template>
            </div>

            <div class="action-buttons">
              <button class="btn-primary" @click="startPredict" :disabled="isPredicting">
                {{ isPredicting ? '预测中...' : '开始预测' }}
              </button>
              <button class="btn-secondary" @click="cancelPredict" v-if="isPredicting">
                取消
              </button>
            </div>
          </div>
        </DataVBorder>
      </div>

      <!-- 右侧: 进度和结果 -->
      <div class="predict-main">
        <!-- 进度显示 -->
        <DataVBorder :type="1">
          <div class="progress-section">
            <h3 class="section-title">预测进度</h3>
            <ProgressSteps :currentStep="currentStep" :progress="progress" />
          </div>
        </DataVBorder>

        <!-- 实时日志 -->
        <DataVBorder :type="1">
          <div class="log-section">
            <h3 class="section-title">实时日志</h3>
            <div class="log-container" ref="logContainer">
              <div v-for="(log, index) in logs" :key="index" class="log-line">
                {{ log }}
              </div>
              <div v-if="logs.length === 0" class="log-empty">
                等待开始预测...
              </div>
            </div>
          </div>
        </DataVBorder>

        <!-- 预测结果 -->
        <DataVBorder :type="1" v-if="results.length > 0">
          <div class="result-section">
            <div class="result-header-row">
              <h3 class="section-title">预测结果</h3>
              <button class="btn-copy" @click="copyAllResults">
                <span class="icon-copy" v-html="'&#128203;'"></span>
                一键复制
              </button>
            </div>
            <div v-for="(result, index) in results" :key="index" class="result-item">
              <div class="result-header">
                <span class="result-index">#{{ index + 1 }}</span>
                <span class="result-confidence" :style="{ color: getConfidenceColor(result.confidence || 0) }">
                  置信度: {{ ((result.confidence || 0) * 100).toFixed(1) }}%
                </span>
              </div>
              <div v-if="result.duplex" class="result-meta">
                <span>复式：红{{ result.red_count || result.red_balls.length }} 蓝{{ result.blue_count || (result.blue_balls?.length || 1) }}</span>
                <span>注数：{{ getResultStakeCount(result) }}</span>
                <span>金额：¥{{ getResultTotalCost(result) }}</span>
              </div>
              <div class="result-balls">
                <LotteryBall
                  v-for="(num, i) in result.red_balls"
                  :key="'r-' + i"
                  :number="num"
                  type="red"
                  size="md"
                  :animate="true"
                  :delay="i * 100"
                />
                <template v-if="result.duplex && result.blue_balls && result.blue_balls.length > 0">
                  <LotteryBall
                    v-for="(blue, bi) in result.blue_balls"
                    :key="'b-' + bi"
                    :number="blue"
                    type="blue"
                    size="md"
                    :animate="true"
                    :delay="600 + bi * 60"
                  />
                </template>
                <LotteryBall
                  v-else-if="result.blue_ball !== undefined"
                  :number="result.blue_ball"
                  type="blue"
                  size="md"
                  :animate="true"
                  :delay="600"
                />
              </div>
            </div>
          </div>
        </DataVBorder>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { usePredictStore } from '@/stores/predict'
import { useAppStore } from '@/stores/app'
import { ALGORITHM_CATEGORIES } from '@/utils/constants'
import { getConfidenceColor } from '@/utils/format'
import { getMethods } from '@/api/predict'
import DataVBorder from '@/components/common/DataVBorder.vue'
import ProgressSteps from '@/components/common/ProgressSteps.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'
import type { PredictResult } from '@/types/predict'

const predictStore = usePredictStore()
const appStore = useAppStore()

const logContainer = ref<HTMLElement | null>(null)

const selectedMethod = computed(() => predictStore.selectedMethod)
const params = computed(() => predictStore.params)
const isPredicting = computed(() => predictStore.isPredicting)
const currentStep = computed(() => predictStore.currentStep)
const progress = computed(() => predictStore.progress)
const logs = computed(() => predictStore.logs)
const results = computed(() => predictStore.results)

const combination = (n: number, k: number): number => {
  if (k < 0 || n < k) return 0
  if (k === 0 || n === k) return 1
  const kk = Math.min(k, n - k)
  let result = 1
  for (let i = 1; i <= kk; i++) {
    result = (result * (n - kk + i)) / i
  }
  return Math.round(result)
}

const duplexStakeCount = computed(() => {
  if (!params.value.duplex) {
    return 1
  }
  const redCount = Math.max(7, Math.min(20, Math.floor(params.value.redCount || 7)))
  const blueCount = Math.max(1, Math.min(16, Math.floor(params.value.blueCount || 1)))
  return combination(redCount, 6) * blueCount
})

const duplexTotalCost = computed(() => duplexStakeCount.value * 2)

// 算法方法类型定义
interface AlgorithmMethod {
  id: string
  name: string
}

type AlgorithmCategory = 'intelligent' | 'deep_learning' | 'machine_learning' | 'statistical' | 'markov'

// 默认算法分组数据（后备方案）
const defaultAlgorithmGroups: Record<AlgorithmCategory, AlgorithmMethod[]> = {
  intelligent: [
    { id: 'super', name: '超级预测器' },
    { id: 'super_predictor', name: '超级预测器(别名)' },
    { id: 'high_confidence', name: '高置信度预测' },
    { id: 'high_confidence_full', name: '高置信度(完整模式)' },
    { id: 'high_confidence_advanced', name: '高置信度(高级模式)' },
    { id: 'high_confidence_lite', name: '高置信度(精简模式)' },
    { id: 'high_confidence_complete', name: '高置信度(完整版)' },
    { id: 'hybrid', name: '混合分析' },
    { id: 'hybrid_v2', name: '混合分析V2' }
  ],
  deep_learning: [
    { id: 'lstm', name: 'LSTM神经网络' },
    { id: 'transformer', name: 'Transformer' },
    { id: 'graph_nn', name: '图神经网络' }
  ],
  machine_learning: [
    { id: 'ensemble', name: '集成学习' },
    { id: 'adaptive_ensemble', name: '自适应集成学习' },
    { id: 'clustering', name: '聚类分析' }
  ],
  statistical: [
    { id: 'markov', name: '马尔可夫链' },
    { id: 'markov_2nd', name: '二阶马尔可夫链' },
    { id: 'markov_3rd', name: '三阶马尔可夫链' },
    { id: 'adaptive_markov', name: '自适应马尔可夫链' },
    { id: 'monte_carlo', name: '蒙特卡洛' },
    { id: 'dynamic_bayes', name: '动态贝叶斯网络' },
    { id: 'stats', name: '统计学预测' },
    { id: 'probability', name: '概率论预测' },
    { id: 'decision_tree', name: '决策树' },
    { id: 'patterns', name: '模式识别' },
    { id: 'frequency', name: '频率分析' },
    { id: 'frequency_cons2', name: '连对规则频率分析(cons_2)' },
    { id: 'hot_cold', name: '冷热号分析' },
    { id: 'consensus_halving', name: '交集递减融合' }
  ],
  markov: []
}

// 响应式算法分组
const algorithmGroups = ref<Record<string, AlgorithmMethod[]>>(defaultAlgorithmGroups)
const loadingMethods = ref(false)

// 方法名称映射
const methodNameMap: Record<string, string> = {
  'super': '超级预测器',
  'super_predictor': '超级预测器(别名)',
  'high_confidence': '高置信度预测',
  'high_confidence_full': '高置信度(完整模式)',
  'high_confidence_advanced': '高置信度(高级模式)',
  'high_confidence_lite': '高置信度(精简模式)',
  'high_confidence_complete': '高置信度(完整版)',
  'hybrid': '混合分析',
  'hybrid_v2': '混合分析V2',
  'lstm': 'LSTM神经网络',
  'transformer': 'Transformer',
  'graph_nn': '图神经网络',
  'ensemble': '集成学习',
  'adaptive_ensemble': '自适应集成学习',
  'clustering': '聚类分析',
  'markov': '马尔可夫链',
  'markov_2nd': '二阶马尔可夫链',
  'markov_3rd': '三阶马尔可夫链',
  'adaptive_markov': '自适应马尔可夫链',
  'monte_carlo': '蒙特卡洛',
  'dynamic_bayes': '动态贝叶斯网络',
  'stats': '统计学预测',
  'probability': '概率论预测',
  'decision_tree': '决策树',
  'patterns': '模式识别',
  'frequency': '频率分析',
  'frequency_cons2': '连对规则频率分析(cons_2)',
  'hot_cold': '冷热号分析',
  'consensus_halving': '交集递减融合'
}

const getMethodDisplayName = (method: string): string => {
  return methodNameMap[method] || method
}

// 从 API 加载算法列表
const loadMethods = async () => {
  loadingMethods.value = true
  try {
    const res = await getMethods()
    if (res.success && res.data && Array.isArray(res.data)) {
      // 按分类组织算法
      const groups: Record<string, AlgorithmMethod[]> = {
        intelligent: [],
        deep_learning: [],
        machine_learning: [],
        statistical: [],
        markov: []
      }

      res.data.forEach((method) => {
        const category = method.category || 'statistical'
        if (!groups[category]) {
          groups[category] = []
        }
        // 生成友好的显示名称
        const displayName = method.name || getMethodDisplayName(method.id)
        groups[category].push({
          id: method.id,
          name: displayName
        })
      })

      // 过滤掉空分类
      Object.keys(groups).forEach(key => {
        if (groups[key].length === 0) {
          delete groups[key]
        }
      })

      if (Object.keys(groups).length > 0) {
        algorithmGroups.value = groups
      }
    }
  } catch (error) {
    console.error('加载算法列表失败，使用默认列表:', error)
    // 保持使用默认列表
  } finally {
    loadingMethods.value = false
  }
}

const getCategoryName = (category: string) => {
  return ALGORITHM_CATEGORIES[category] || category
}

const selectMethod = (methodId: string) => {
  predictStore.selectedMethod = methodId
}

const startPredict = () => {
  predictStore.startPredict()
}

const cancelPredict = () => {
  predictStore.cancelPredict()
}

const getResultStakeCount = (result: PredictResult): number => {
  if (result.stake_count && result.stake_count > 0) {
    return result.stake_count
  }
  if (!result.duplex) {
    return 1
  }
  const redCount = result.red_count || result.red_balls.length
  const blueCount = result.blue_count || result.blue_balls?.length || (result.blue_ball ? 1 : 0)
  return combination(redCount, 6) * Math.max(blueCount, 1)
}

const getResultTotalCost = (result: PredictResult): number => {
  if (result.total_cost && result.total_cost > 0) {
    return result.total_cost
  }
  return getResultStakeCount(result) * 2
}

// 格式化预测结果为文本
const formatResultsText = (): string => {
  return results.value.map((result: PredictResult, index: number) => {
    const redBalls = result.red_balls
      .map((n: number) => n.toString().padStart(2, '0'))
      .join(' ')
    const confidence = ((result.confidence || 0) * 100).toFixed(1)
    if (result.duplex) {
      const blueBalls = (result.blue_balls || [])
        .map((n: number) => n.toString().padStart(2, '0'))
        .join(' ')
      return `#${index + 1} (置信度${confidence}%) 复式红球: ${redBalls} 复式蓝球: ${blueBalls} 注数: ${getResultStakeCount(result)} 金额: ¥${getResultTotalCost(result)}`
    }
    const blueBall = (result.blue_ball || 0).toString().padStart(2, '0')
    return `#${index + 1} (置信度${confidence}%) 红球: ${redBalls} 蓝球: ${blueBall}`
  }).join('\n')
}

// 降级复制方案（使用document.execCommand）
const fallbackCopy = (text: string): boolean => {
  try {
    const textArea = document.createElement('textarea')
    textArea.value = text
    textArea.style.cssText = 'position:fixed;opacity:0;pointer-events:none;'
    document.body.appendChild(textArea)
    textArea.select()
    const success = document.execCommand('copy')
    document.body.removeChild(textArea)
    return success
  } catch {
    return false
  }
}

// 一键复制所有预测结果
const copyAllResults = async () => {
  if (!results.value || results.value.length === 0) {
    appStore.notify.warning('暂无预测结果可复制')
    return
  }

  const text = formatResultsText()

  try {
    // 使用现代 Clipboard API
    await navigator.clipboard.writeText(text)
    appStore.notify.success('已成功复制所有预测号码到剪贴板！')
  } catch (error) {
    console.error('Clipboard API 失败:', error)
    // 降级到旧方案
    if (fallbackCopy(text)) {
      appStore.notify.success('已成功复制所有预测号码到剪贴板！')
    } else {
      appStore.notify.error('复制失败，请手动选择复制')
    }
  }
}

// 日志自动滚动
watch(logs, () => {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
})

// 组件挂载时加载算法列表
onMounted(() => {
  loadMethods()
})
</script>

<style scoped>
.predict-page {
  padding: 24px;
}

.predict-layout {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 24px;
}

.sidebar-content,
.progress-section,
.log-section,
.result-section {
  padding: 20px;
}

.section-title {
  font-size: 16px;
  color: var(--text-primary);
  margin-bottom: 16px;
}

.algorithm-group {
  margin-bottom: 16px;
}

.group-title {
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 8px;
}

.algorithm-item {
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  color: var(--text-secondary);
  transition: all 0.3s;
  margin-bottom: 4px;
}

.algorithm-item:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.algorithm-item.active {
  background: rgba(0, 212, 255, 0.15);
  color: var(--color-primary);
  border-left: 3px solid var(--color-primary);
}

.params-form {
  margin-bottom: 20px;
}

.duplex-cost {
  margin-top: 8px;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-secondary);
  color: var(--text-secondary);
  font-size: 13px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.form-item {
  margin-bottom: 12px;
}

.form-item label {
  display: block;
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.form-item input[type="number"] {
  width: 100%;
  padding: 10px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
}

.form-item.checkbox label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.action-buttons {
  display: flex;
  gap: 12px;
}

.btn-primary,
.btn-secondary {
  flex: 1;
  padding: 12px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  background: var(--gradient-primary);
  border: none;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  box-shadow: var(--shadow-glow);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
}

.btn-secondary:hover {
  border-color: var(--color-error);
  color: var(--color-error);
}

.predict-main {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.log-container {
  height: 200px;
  overflow-y: auto;
  background: #000;
  border-radius: 6px;
  padding: 12px;
  font-family: monospace;
  font-size: 13px;
}

.log-line {
  color: var(--color-success);
  margin-bottom: 4px;
}

.log-empty {
  color: var(--text-muted);
  text-align: center;
  padding: 20px;
}

.result-item {
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
  margin-bottom: 12px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
}

.result-meta {
  display: flex;
  gap: 14px;
  color: var(--text-secondary);
  font-size: 13px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}

.result-index {
  font-weight: bold;
  color: var(--text-primary);
}

.result-confidence {
  font-size: 14px;
}

.result-balls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.result-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.result-header-row .section-title {
  margin-bottom: 0;
}

.btn-copy {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: var(--gradient-primary);
  border: none;
  border-radius: 6px;
  color: white;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-copy:hover {
  box-shadow: var(--shadow-glow);
  transform: translateY(-2px);
}

.icon-copy {
  font-size: 16px;
  line-height: 1;
}

@media (max-width: 1024px) {
  .predict-layout {
    grid-template-columns: 1fr;
  }
}
</style>
