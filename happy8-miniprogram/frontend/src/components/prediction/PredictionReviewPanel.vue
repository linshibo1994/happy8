<script setup lang="ts">
import { computed } from 'vue'
import { AlertTriangle, BarChart3, BrainCircuit, History } from 'lucide-vue-next'

import type { Algorithm, PredictionResult } from '@/types'

import PredictionNumberBall from './PredictionNumberBall.vue'

const props = defineProps<{
  mode: 'single' | 'batch'
  selectedAlgorithm: Algorithm | null
  selectedAlgorithms: Algorithm[]
  recentPredictions: PredictionResult[]
}>()

const displayAlgorithm = computed(() => {
  if (props.mode === 'single') {
    return props.selectedAlgorithm
  }

  return props.selectedAlgorithms[0] ?? props.selectedAlgorithm
})

const historicalCards = computed(() => {
  const algorithm = displayAlgorithm.value

  if (!algorithm) {
    return []
  }

  return [
    { label: '历史命中表现', value: `${Math.round(algorithm.successRate * 100)}%` },
    { label: '平均耗时', value: `${algorithm.averageCostMs}ms` },
    { label: '复杂度', value: algorithm.complexity },
  ]
})
</script>

<template>
  <aside class="review-panel" aria-label="预测复盘区">
    <header class="review-panel__header">
      <span class="section-kicker">复盘区</span>
      <h2>{{ mode === 'single' ? '算法说明' : '批量参考' }}</h2>
      <p v-if="displayAlgorithm">{{ displayAlgorithm.recommendedScenario }}</p>
      <p v-else>选择算法后展示说明、历史表现与最近同算法预测。</p>
    </header>

    <section class="review-card" aria-labelledby="algorithm-guide-title">
      <div class="review-card__title">
        <BrainCircuit :size="18" aria-hidden="true" />
        <h3 id="algorithm-guide-title">{{ displayAlgorithm?.displayName ?? '待选择算法' }}</h3>
      </div>
      <p>
        {{ displayAlgorithm?.description ?? '当前没有可用算法说明。' }}
      </p>
      <div v-if="mode === 'batch'" class="selected-tags" aria-label="批量已选算法">
        <span v-for="algorithm in selectedAlgorithms" :key="algorithm.name">{{ algorithm.displayName }}</span>
      </div>
    </section>

    <section class="review-card" aria-labelledby="performance-title">
      <div class="review-card__title">
        <BarChart3 :size="18" aria-hidden="true" />
        <h3 id="performance-title">历史表现</h3>
      </div>
      <div class="performance-grid">
        <article v-for="item in historicalCards" :key="item.label">
          <strong>{{ item.value }}</strong>
          <span>{{ item.label }}</span>
        </article>
      </div>
    </section>

    <section class="review-card" aria-labelledby="recent-title">
      <div class="review-card__title">
        <History :size="18" aria-hidden="true" />
        <h3 id="recent-title">最近同算法预测</h3>
      </div>

      <ol v-if="recentPredictions.length" class="recent-list">
        <li v-for="prediction in recentPredictions" :key="prediction.id">
          <header>
            <strong>第 {{ prediction.targetIssue }} 期</strong>
            <span>{{ Math.round(prediction.confidence * 100) }}%</span>
          </header>
          <div class="recent-balls">
            <PredictionNumberBall
              v-for="number in prediction.numbers.slice(0, 8)"
              :key="`${prediction.id}-${number}`"
              :value="number"
              size="small"
            />
          </div>
        </li>
      </ol>

      <p v-else class="empty-copy">暂无同算法预测记录，完成一次预测后会进入这里。</p>
    </section>

    <section class="risk-card" aria-labelledby="risk-title">
      <div class="review-card__title">
        <AlertTriangle :size="18" aria-hidden="true" />
        <h3 id="risk-title">风险提示</h3>
      </div>
      <ul>
        <li>快乐8 开奖具有随机性，历史数据只能作为模型输入。</li>
        <li>置信度是算法内部评分，不等同于实际命中概率。</li>
        <li>批量预测用于观察共识和分歧，不应叠加为确定性结论。</li>
      </ul>
    </section>
  </aside>
</template>

<style scoped>
.review-panel {
  display: grid;
  gap: 16px;
  align-content: start;
}

.review-panel__header h2 {
  margin: 3px 0 0;
  font-family: var(--h8-font-title);
  font-size: 22px;
  line-height: 1.2;
}

.review-panel__header p,
.review-card p,
.risk-card li,
.empty-copy {
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.55;
}

.review-panel__header p {
  margin: 8px 0 0;
}

.review-card,
.risk-card {
  display: grid;
  gap: 12px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 15px;
}

.review-card__title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--h8-color-cinnabar);
}

.review-card__title h3 {
  margin: 0;
  color: var(--h8-color-text);
  font-size: 15px;
}

.review-card p,
.empty-copy {
  margin: 0;
}

.selected-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.selected-tags span {
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  color: var(--h8-color-text-muted);
  padding: 4px 8px;
  font-size: 12px;
}

.performance-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.performance-grid article {
  min-width: 0;
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  padding: 10px;
}

.performance-grid strong {
  display: block;
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-size: 18px;
  line-height: 1;
}

.performance-grid span {
  display: block;
  margin-top: 6px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
  line-height: 1.3;
}

.recent-list {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.recent-list li {
  display: grid;
  gap: 8px;
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 10px;
}

.recent-list li:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

.recent-list header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.recent-list strong {
  font-size: 13px;
}

.recent-list span {
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-size: 12px;
  font-weight: 700;
}

.recent-balls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.risk-card {
  border-color: color-mix(in srgb, var(--h8-color-risk-orange) 45%, var(--h8-color-line));
  background: color-mix(in srgb, var(--h8-color-risk-orange) 7%, var(--h8-color-surface-strong));
}

.risk-card .review-card__title {
  color: var(--h8-color-risk-orange);
}

.risk-card ul {
  display: grid;
  gap: 7px;
  margin: 0;
  padding-left: 18px;
}

@media (max-width: 1180px) {
  .performance-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .performance-grid {
    grid-template-columns: 1fr;
  }
}
</style>
