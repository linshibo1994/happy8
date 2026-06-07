<script setup lang="ts">
import { computed, ref, watchEffect } from 'vue'
import { BrainCircuit, Filter, ShieldCheck } from 'lucide-vue-next'

import AlgorithmCard from '@/components/algorithms/AlgorithmCard.vue'
import AlgorithmDetailPanel from '@/components/algorithms/AlgorithmDetailPanel.vue'
import { useAlgorithmStore } from '@/stores/algorithm'
import { useMembershipStore } from '@/stores/membership'
import type { Algorithm, AlgorithmCategory, MembershipLevel } from '@/types'

type CategoryFilter = '全部' | AlgorithmCategory
type PermissionFilter = '全部' | MembershipLevel

const algorithmStore = useAlgorithmStore()
const membershipStore = useMembershipStore()

const categoryFilter = ref<CategoryFilter>('全部')
const permissionFilter = ref<PermissionFilter>('全部')
const selectedAlgorithmName = ref('frequency')

const categoryOptions = computed<CategoryFilter[]>(() => [
  '全部',
  ...Array.from(new Set(algorithmStore.algorithms.map((algorithm) => algorithm.category))),
])

const permissionOptions: Array<{ label: string; value: PermissionFilter }> = [
  { label: '全部权限', value: '全部' },
  { label: '免费', value: 'free' },
  { label: 'VIP', value: 'vip' },
  { label: 'Premium', value: 'premium' },
]

const levelRank: Record<MembershipLevel, number> = {
  free: 0,
  vip: 1,
  premium: 2,
}

const currentRank = computed(() => levelRank[membershipStore.status.level])

const isLocked = (algorithm: Algorithm) => levelRank[algorithm.permissionLevel] > currentRank.value

const filteredAlgorithms = computed(() =>
  algorithmStore.algorithms.filter((algorithm) => {
    const categoryMatched = categoryFilter.value === '全部' || algorithm.category === categoryFilter.value
    const permissionMatched = permissionFilter.value === '全部' || algorithm.permissionLevel === permissionFilter.value

    return categoryMatched && permissionMatched
  }),
)

const selectedAlgorithm = computed(
  () =>
    algorithmStore.algorithms.find((algorithm) => algorithm.name === selectedAlgorithmName.value) ??
    filteredAlgorithms.value[0] ??
    algorithmStore.algorithms[0],
)

const unlockedCount = computed(() => algorithmStore.algorithms.filter((algorithm) => !isLocked(algorithm)).length)
const premiumCount = computed(() => algorithmStore.algorithms.filter((algorithm) => algorithm.permissionLevel === 'premium').length)
const averageSuccessRate = computed(() => {
  if (!algorithmStore.algorithms.length) {
    return '0%'
  }

  const total = algorithmStore.algorithms.reduce((sum, algorithm) => sum + algorithm.successRate, 0)
  return `${Math.round((total / algorithmStore.algorithms.length) * 100)}%`
})

watchEffect(() => {
  if (!filteredAlgorithms.value.some((algorithm) => algorithm.name === selectedAlgorithmName.value) && filteredAlgorithms.value[0]) {
    selectedAlgorithmName.value = filteredAlgorithms.value[0].name
  }
})
</script>

<template>
  <section class="algorithms-page" aria-labelledby="algorithms-title">
    <header class="algorithms-page__hero">
      <div>
        <span class="section-kicker">Algorithm Center</span>
        <h2 id="algorithms-title">核心算法档案</h2>
        <p>
          按分类、权限和复杂度查看算法能力，先理解适用场景，再带入预测执行页。锁定算法仍可阅读档案，避免升级路径打断浏览。
        </p>
      </div>
      <div class="algorithms-page__hero-status">
        <ShieldCheck :size="20" aria-hidden="true" />
        <span>{{ membershipStore.status.levelName }} / 可执行 {{ unlockedCount }} 个算法</span>
      </div>
    </header>

    <div class="algorithms-page__metrics" aria-label="算法中心指标">
      <div>
        <span>核心算法</span>
        <strong>{{ algorithmStore.algorithms.length }}</strong>
      </div>
      <div>
        <span>Premium 算法</span>
        <strong>{{ premiumCount }}</strong>
      </div>
      <div>
        <span>平均成功率</span>
        <strong>{{ averageSuccessRate }}</strong>
      </div>
      <div>
        <span>当前剩余次数</span>
        <strong>{{ membershipStore.status.remainingPredictions }}</strong>
      </div>
    </div>

    <section class="algorithms-page__filters" aria-label="算法筛选">
      <div class="algorithms-page__filter-title">
        <Filter :size="18" aria-hidden="true" />
        <strong>筛选算法</strong>
      </div>

      <div class="algorithms-page__segmented" role="group" aria-label="分类筛选">
        <button
          v-for="category in categoryOptions"
          :key="category"
          type="button"
          :aria-pressed="categoryFilter === category"
          @click="categoryFilter = category"
        >
          {{ category }}
        </button>
      </div>

      <div class="algorithms-page__segmented" role="group" aria-label="权限筛选">
        <button
          v-for="option in permissionOptions"
          :key="option.value"
          type="button"
          :aria-pressed="permissionFilter === option.value"
          @click="permissionFilter = option.value"
        >
          {{ option.label }}
        </button>
      </div>
    </section>

    <div class="algorithms-page__layout">
      <main class="algorithms-page__list" aria-label="算法列表">
        <AlgorithmCard
          v-for="algorithm in filteredAlgorithms"
          :key="algorithm.name"
          :algorithm="algorithm"
          :locked="isLocked(algorithm)"
          :selected="selectedAlgorithmName === algorithm.name"
          @select="selectedAlgorithmName = $event"
        />

        <div v-if="!filteredAlgorithms.length" class="algorithms-page__empty">
          <BrainCircuit :size="28" aria-hidden="true" />
          <h3>没有匹配的算法</h3>
          <p>放宽分类或权限筛选后继续查看，也可以回到全部算法。</p>
          <button type="button" @click="categoryFilter = '全部'; permissionFilter = '全部'">查看全部算法</button>
        </div>
      </main>

      <AlgorithmDetailPanel
        v-if="selectedAlgorithm"
        :algorithm="selectedAlgorithm"
        :locked="isLocked(selectedAlgorithm)"
      />
    </div>
  </section>
</template>

<style scoped lang="scss">
.algorithms-page {
  display: grid;
  gap: 22px;
}

.algorithms-page__hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
}

.algorithms-page__hero h2 {
  margin: 4px 0 0;
  font-family: var(--h8-font-title);
  font-size: 31px;
  line-height: 1.2;
}

.algorithms-page__hero p {
  max-width: 820px;
  margin: 9px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.7;
}

.algorithms-page__hero-status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 38px;
  border: 1px solid color-mix(in srgb, var(--h8-color-bronze) 48%, var(--h8-color-line));
  border-radius: 999px;
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-bronze);
  padding: 8px 12px;
  font-weight: 800;
  white-space: nowrap;
}

.algorithms-page__metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.algorithms-page__metrics div {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 16px;
}

.algorithms-page__metrics span {
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.algorithms-page__metrics strong {
  display: block;
  margin-top: 6px;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 28px;
  line-height: 1;
}

.algorithms-page__filters {
  display: grid;
  gap: 12px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  padding: 16px;
}

.algorithms-page__filter-title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--h8-color-cinnabar);
}

.algorithms-page__segmented {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.algorithms-page__segmented button,
.algorithms-page__empty button {
  min-height: 34px;
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  padding: 6px 12px;
  cursor: pointer;
}

.algorithms-page__segmented button:focus-visible,
.algorithms-page__empty button:focus-visible {
  outline: 0;
  box-shadow: var(--h8-focus-ring);
}

.algorithms-page__segmented button[aria-pressed='true'] {
  border-color: var(--h8-color-cinnabar);
  background: var(--h8-color-cinnabar);
  color: #fff;
}

.algorithms-page__layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
  gap: 18px;
  align-items: start;
}

.algorithms-page__list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.algorithms-page__empty {
  grid-column: 1 / -1;
  display: grid;
  justify-items: start;
  gap: 10px;
  border: 1px dashed var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-surface-strong) 78%, transparent);
  padding: 24px;
}

.algorithms-page__empty h3,
.algorithms-page__empty p {
  margin: 0;
}

.algorithms-page__empty p {
  color: var(--h8-color-text-muted);
}

.algorithms-page__empty button {
  border-color: var(--h8-color-cinnabar);
  background: var(--h8-color-cinnabar);
  color: #fff;
  font-weight: 800;
}

@media (max-width: 1180px) {
  .algorithms-page__layout,
  .algorithms-page__list {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .algorithms-page__hero {
    display: grid;
  }

  .algorithms-page__hero-status {
    width: fit-content;
    white-space: normal;
  }

  .algorithms-page__metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 460px) {
  .algorithms-page__metrics {
    grid-template-columns: 1fr;
  }
}
</style>
