<script setup lang="ts">
import { computed } from 'vue'
import { ArrowRight, Target } from 'lucide-vue-next'

import NumberBall from '@/components/balls/NumberBall.vue'
import type { PredictionRecordView } from './PredictionRecordRow.vue'

const props = defineProps<{
  record?: PredictionRecordView
}>()

const hitSummary = computed(() => {
  if (!props.record) {
    return '暂无记录'
  }

  return `命中 ${props.record.hitCount} 个，命中率 ${Math.round(props.record.hitRate * 100)}%`
})
</script>

<template>
  <aside class="compare-panel" aria-label="预测详情对比">
    <template v-if="record">
      <div class="compare-panel__header">
        <span>复盘详情</span>
        <h3>{{ record.targetIssue }}</h3>
        <p>{{ record.algorithmLabel }} / {{ hitSummary }}</p>
      </div>

      <div class="compare-panel__summary">
        <Target :size="19" aria-hidden="true" />
        <span>{{ hitSummary }}。命中号码已使用朱砂实心和青铜外环标记。</span>
      </div>

      <div class="compare-panel__columns">
        <section>
          <h4>预测号码</h4>
          <div class="compare-panel__balls">
            <NumberBall
              v-for="number in record.predictedNumbers"
              :key="number"
              :value="number"
              :variant="record.hitNumbers.includes(number) ? 'hit' : 'prediction'"
            />
          </div>
        </section>

        <ArrowRight class="compare-panel__arrow" :size="22" aria-hidden="true" />

        <section>
          <h4>实际开奖号码</h4>
          <div class="compare-panel__balls">
            <NumberBall
              v-for="number in record.actualNumbers"
              :key="number"
              :value="number"
              :variant="record.hitNumbers.includes(number) ? 'hit' : 'miss'"
            />
          </div>
        </section>
      </div>

      <dl class="compare-panel__meta">
        <div>
          <dt>置信度</dt>
          <dd>{{ Math.round(record.confidence * 100) }}%</dd>
        </div>
        <div>
          <dt>执行耗时</dt>
          <dd>{{ record.elapsedMs }}ms</dd>
        </div>
        <div>
          <dt>结果来源</dt>
          <dd>{{ record.cached ? '缓存复用' : '实时执行' }}</dd>
        </div>
      </dl>
    </template>

    <template v-else>
      <div class="compare-panel__empty">
        <h3>暂无可对比记录</h3>
        <p>调整筛选条件或前往预测执行页生成一条记录后，这里会显示并排复盘。</p>
        <RouterLink to="/prediction">去执行预测</RouterLink>
      </div>
    </template>
  </aside>
</template>

<style scoped lang="scss">
.compare-panel {
  display: grid;
  align-content: start;
  gap: 18px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 20px;
}

.compare-panel__header span {
  color: var(--h8-color-cinnabar);
  font-size: 12px;
  font-weight: 800;
}

.compare-panel h3,
.compare-panel h4 {
  margin: 0;
}

.compare-panel h3 {
  margin-top: 4px;
  font-family: var(--h8-font-title);
  font-size: 22px;
}

.compare-panel p {
  margin: 6px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.6;
}

.compare-panel__summary {
  display: flex;
  gap: 9px;
  border: 1px solid color-mix(in srgb, var(--h8-color-cinnabar) 30%, var(--h8-color-line));
  border-radius: var(--h8-radius-control);
  background: color-mix(in srgb, var(--h8-color-cinnabar) 8%, var(--h8-color-surface));
  color: var(--h8-color-cinnabar);
  padding: 11px 12px;
  line-height: 1.55;
}

.compare-panel__columns {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 14px;
  align-items: start;
}

.compare-panel__columns section {
  display: grid;
  gap: 12px;
}

.compare-panel__balls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.compare-panel__arrow {
  margin-top: 46px;
  color: var(--h8-color-data-blue);
}

.compare-panel__meta {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
}

.compare-panel__meta div {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  padding: 10px;
}

.compare-panel__meta dt {
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.compare-panel__meta dd {
  margin: 5px 0 0;
  font-family: var(--h8-font-number);
  font-weight: 800;
}

.compare-panel__empty {
  display: grid;
  gap: 10px;
}

.compare-panel__empty a {
  display: inline-flex;
  width: fit-content;
  min-height: 38px;
  align-items: center;
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-cinnabar);
  color: #fff;
  padding: 0 14px;
  font-weight: 800;
}

@media (max-width: 680px) {
  .compare-panel__columns,
  .compare-panel__meta {
    grid-template-columns: 1fr;
  }

  .compare-panel__arrow {
    display: none;
  }
}
</style>
