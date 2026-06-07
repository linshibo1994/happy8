<script setup lang="ts">
import { computed } from 'vue'
import { ArrowRight, CheckCircle2, Lock, ShieldCheck } from 'lucide-vue-next'

import type { Algorithm, MembershipLevel } from '@/types'

const props = defineProps<{
  algorithm: Algorithm
  locked: boolean
}>()

const permissionText: Record<MembershipLevel, string> = {
  free: '免费可用',
  vip: 'VIP 可用',
  premium: 'Premium 可用',
}

const principleText = computed(() => {
  const map: Partial<Record<string, string>> = {
    frequency: '统计历史窗口内每个号码出现频次，按近期活跃度、区间均衡和重复控制生成候选。',
    hot_cold: '将近期热号与长期冷号分层，避免候选集完全偏向单一热度信号。',
    missing: '追踪号码连续未出现期数，结合回补窗口过滤极端遗漏值。',
    markov: '把开奖序列抽象为状态转移，使用上一阶段状态估算下一期候选概率。',
    adaptive_markov: '在不同历史窗口中动态调整转移阶数，使短期波动和长期趋势同时参与。',
    ensemble: '融合统计、遗漏和概率模型输出，用权重分配降低单模型偏差。',
    clustering: '按号码区间、和值、奇偶结构建立聚类，选择接近中心的代表性候选。',
    monte_carlo: '通过多轮随机采样模拟候选稳定度，保留重复出现概率较高的号码。',
    lstm: '使用序列模型处理较长历史窗口，捕捉号码结构随时间变化的趋势。',
    transformer: '用注意力权重识别关键期数和号码关系，适合分析长窗口影响。',
    gnn: '将号码共现关系建成图结构，寻找关系网络中更稳定的候选节点。',
    bayesian: '从历史先验概率出发，结合近期证据更新后验候选概率。',
    high_confidence: '对候选结果进行多轮置信过滤，减少低稳定度号码进入最终集。',
    super_predictor: '综合统计、序列、图关系和概率模型，形成多信号融合候选。',
  }

  return map[props.algorithm.name] ?? props.algorithm.description
})

const parameterRows = computed(() => [
  { label: '默认分析期数', value: props.algorithm.complexity === '高' ? '160 期' : '120 期' },
  { label: '预测号码数', value: '10 个' },
  { label: '数据要求', value: props.algorithm.complexity === '低' ? '近 30 期以上' : '近 100 期以上' },
])
</script>

<template>
  <aside class="algorithm-detail" aria-label="算法详情">
    <div class="algorithm-detail__header">
      <span class="algorithm-detail__kicker">{{ algorithm.category }} / {{ permissionText[algorithm.permissionLevel] }}</span>
      <h3>{{ algorithm.displayName }}</h3>
      <p>{{ algorithm.description }}</p>
    </div>

    <div class="algorithm-detail__status" :class="{ 'algorithm-detail__status--locked': locked }">
      <component :is="locked ? Lock : ShieldCheck" :size="18" aria-hidden="true" />
      <span>{{ locked ? '当前等级不可执行，但可查看详情与表现。' : '当前等级可直接使用该算法。' }}</span>
    </div>

    <section class="algorithm-detail__section">
      <h4>算法档案</h4>
      <p>{{ principleText }}</p>
      <dl>
        <div>
          <dt>复杂度</dt>
          <dd>{{ algorithm.complexity }}</dd>
        </div>
        <div>
          <dt>平均耗时</dt>
          <dd>{{ algorithm.averageCostMs }}ms</dd>
        </div>
        <div>
          <dt>历史成功率</dt>
          <dd>{{ Math.round(algorithm.successRate * 100) }}%</dd>
        </div>
      </dl>
    </section>

    <section class="algorithm-detail__section">
      <h4>参数与适用场景</h4>
      <ul class="algorithm-detail__params">
        <li v-for="row in parameterRows" :key="row.label">
          <span>{{ row.label }}</span>
          <strong>{{ row.value }}</strong>
        </li>
      </ul>
      <p class="algorithm-detail__scenario">
        <CheckCircle2 :size="17" aria-hidden="true" />
        {{ algorithm.recommendedScenario }}
      </p>
    </section>

    <RouterLink
      class="algorithm-detail__action"
      :to="{ name: 'prediction', query: { algorithm: algorithm.name } }"
      :aria-disabled="locked"
    >
      <span>{{ locked ? '查看升级路径后再执行' : '带入该算法去预测' }}</span>
      <ArrowRight :size="17" aria-hidden="true" />
    </RouterLink>
  </aside>
</template>

<style scoped lang="scss">
.algorithm-detail {
  position: sticky;
  top: calc(var(--h8-topbar-height) + 24px);
  display: grid;
  align-content: start;
  gap: 18px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 20px;
}

.algorithm-detail__kicker {
  color: var(--h8-color-cinnabar);
  font-size: 12px;
  font-weight: 800;
}

.algorithm-detail h3 {
  margin: 5px 0 8px;
  font-family: var(--h8-font-title);
  font-size: 24px;
  line-height: 1.25;
}

.algorithm-detail p {
  margin: 0;
  color: var(--h8-color-text-muted);
  line-height: 1.7;
}

.algorithm-detail__status {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  border: 1px solid color-mix(in srgb, var(--h8-color-turquoise) 36%, var(--h8-color-line));
  border-radius: var(--h8-radius-control);
  background: color-mix(in srgb, var(--h8-color-turquoise) 8%, var(--h8-color-surface));
  color: var(--h8-color-turquoise);
  padding: 10px 12px;
  font-size: 14px;
  line-height: 1.5;
}

.algorithm-detail__status--locked {
  border-color: color-mix(in srgb, var(--h8-color-bronze) 42%, var(--h8-color-line));
  background: color-mix(in srgb, var(--h8-color-bronze) 9%, var(--h8-color-surface));
  color: var(--h8-color-bronze);
}

.algorithm-detail__section {
  display: grid;
  gap: 12px;
  border-top: 1px solid var(--h8-color-line);
  padding-top: 18px;
}

.algorithm-detail h4 {
  margin: 0;
  font-size: 15px;
}

.algorithm-detail dl {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.algorithm-detail dl div {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  padding: 10px;
}

.algorithm-detail dt {
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.algorithm-detail dd {
  margin: 5px 0 0;
  color: var(--h8-color-text);
  font-family: var(--h8-font-number);
  font-weight: 700;
}

.algorithm-detail__params {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.algorithm-detail__params li,
.algorithm-detail__scenario {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.algorithm-detail__params li {
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 8px;
}

.algorithm-detail__params span {
  color: var(--h8-color-text-muted);
}

.algorithm-detail__params strong {
  font-family: var(--h8-font-number);
}

.algorithm-detail__scenario {
  justify-content: flex-start;
  color: var(--h8-color-turquoise) !important;
}

.algorithm-detail__action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 42px;
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-cinnabar);
  color: #fff;
  font-weight: 800;
}

.algorithm-detail__action[aria-disabled='true'] {
  background: var(--h8-color-bronze);
}

@media (max-width: 980px) {
  .algorithm-detail {
    position: static;
  }
}

@media (max-width: 520px) {
  .algorithm-detail dl {
    grid-template-columns: 1fr;
  }
}
</style>
