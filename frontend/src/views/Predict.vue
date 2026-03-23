<template>
  <div class="predict-page">
    <div class="predict-layout">
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
                <label>号码个数</label>
                <input type="number" v-model.number="params.count" min="1" max="30" />
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

      <div class="predict-main">
        <DataVBorder :type="1">
          <div class="progress-section">
            <h3 class="section-title">预测进度</h3>
            <ProgressSteps :currentStep="currentStep" :progress="progress" />
          </div>
        </DataVBorder>

        <DataVBorder :type="1">
          <div class="log-section">
            <h3 class="section-title">实时日志</h3>
            <div class="log-container" ref="logContainer">
              <div v-for="(log, index) in logs" :key="index" class="log-line">
                {{ log }}
              </div>
              <div v-if="logs.length === 0" class="log-empty">等待开始预测...</div>
            </div>
          </div>
        </DataVBorder>

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
              <div class="result-meta">
                <span v-if="result.hit_count !== undefined">命中数：{{ result.hit_count }}</span>
                <span v-if="result.hit_rate !== undefined">命中率：{{ ((result.hit_rate || 0) * 100).toFixed(1) }}%</span>
                <span v-if="result.execution_time">耗时：{{ result.execution_time.toFixed(3) }}s</span>
              </div>
              <div class="result-balls">
                <LotteryBall
                  v-for="(num, i) in result.numbers"
                  :key="'n-' + i"
                  :number="num"
                  size="md"
                  :animate="true"
                  :delay="i * 80"
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

interface AlgorithmMethod {
  id: string
  name: string
}

type AlgorithmCategory = 'intelligent' | 'deep_learning' | 'machine_learning' | 'statistical' | 'markov'

const defaultAlgorithmGroups: Record<AlgorithmCategory, AlgorithmMethod[]> = {
  intelligent: [
    { id: 'super_predictor', name: '超级预测器' },
    { id: 'high_confidence', name: '高置信度预测' },
  ],
  deep_learning: [
    { id: 'lstm', name: 'LSTM神经网络' },
    { id: 'transformer', name: 'Transformer' },
    { id: 'gnn', name: '图神经网络' },
  ],
  machine_learning: [
    { id: 'ensemble', name: '集成学习' },
    { id: 'advanced_ensemble', name: '高级集成学习' },
    { id: 'clustering', name: '聚类分析' },
    { id: 'monte_carlo', name: '蒙特卡洛模拟' },
    { id: 'bayesian', name: '贝叶斯推理' },
  ],
  statistical: [
    { id: 'frequency', name: '频率分析' },
    { id: 'hot_cold', name: '冷热号分析' },
    { id: 'missing', name: '遗漏分析' },
  ],
  markov: [
    { id: 'markov', name: '马尔可夫链(一阶)' },
    { id: 'markov_2nd', name: '马尔可夫链(二阶)' },
    { id: 'markov_3rd', name: '马尔可夫链(三阶)' },
    { id: 'adaptive_markov', name: '自适应马尔可夫链' },
  ],
}

const algorithmGroups = ref<Record<string, AlgorithmMethod[]>>(defaultAlgorithmGroups)

const loadMethods = async () => {
  try {
    const res = await getMethods()
    if (res.success && Array.isArray(res.data) && res.data.length > 0) {
      const groups: Record<string, AlgorithmMethod[]> = {}
      res.data.forEach((method) => {
        const category = method.category || 'statistical'
        if (!groups[category]) {
          groups[category] = []
        }
        groups[category].push({
          id: method.id,
          name: method.name || method.id,
        })
      })
      algorithmGroups.value = groups
    }
  } catch (error) {
    console.error('加载算法列表失败，使用默认列表:', error)
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

const formatResultsText = (): string => {
  return results.value
    .map((result: PredictResult, index: number) => {
      const numbers = result.numbers.map((num) => num.toString().padStart(2, '0')).join(' ')
      const confidence = ((result.confidence || 0) * 100).toFixed(1)
      return `#${index + 1} (置信度${confidence}%) 号码: ${numbers}`
    })
    .join('\n')
}

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

const copyAllResults = async () => {
  if (!results.value || results.value.length === 0) {
    appStore.notify.warning('暂无预测结果可复制')
    return
  }

  const text = formatResultsText()
  try {
    await navigator.clipboard.writeText(text)
    appStore.notify.success('已成功复制所有预测号码到剪贴板')
  } catch (error) {
    console.error('Clipboard API 失败:', error)
    if (fallbackCopy(text)) {
      appStore.notify.success('已成功复制所有预测号码到剪贴板')
    } else {
      appStore.notify.error('复制失败，请手动选择复制')
    }
  }
}

watch(logs, () => {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
})

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

.form-item {
  margin-bottom: 12px;
}

.form-item label {
  display: block;
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.form-item input[type='number'] {
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
  gap: 10px;
}

.btn-primary,
.btn-secondary {
  flex: 1;
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  border: none;
}

.btn-primary {
  background: var(--gradient-primary);
  color: #fff;
}

.btn-secondary {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.log-container {
  max-height: 220px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 12px;
}

.log-line {
  margin-bottom: 4px;
  color: var(--text-secondary);
}

.log-empty {
  color: var(--text-muted);
}

.result-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.result-item {
  margin-bottom: 16px;
  padding: 14px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
}

.result-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 10px;
}

.result-balls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.btn-copy {
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  border-radius: 6px;
  padding: 6px 10px;
  cursor: pointer;
}

@media (max-width: 1024px) {
  .predict-layout {
    grid-template-columns: 1fr;
  }
}
</style>
