<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { ArrowRight, Database, LineChart, PlayCircle } from 'lucide-vue-next'

import { useAlgorithmStore } from '@/stores/algorithm'
import { useLotteryStore } from '@/stores/lottery'
import { useMembershipStore } from '@/stores/membership'
import { usePredictionStore } from '@/stores/prediction'

const route = useRoute()
const algorithmStore = useAlgorithmStore()
const lotteryStore = useLotteryStore()
const membershipStore = useMembershipStore()
const predictionStore = usePredictionStore()

const pageTitle = computed(() => route.meta.title?.toString() ?? '工作台')
const pageSubtitle = computed(() => route.meta.subtitle?.toString() ?? '等待页面实现')
</script>

<template>
  <section class="placeholder-page" :aria-labelledby="`${String(route.name)}-title`">
    <div class="placeholder-page__intro">
      <span class="section-kicker">工程骨架</span>
      <h2 :id="`${String(route.name)}-title`">{{ pageTitle }}</h2>
      <p>{{ pageSubtitle }}</p>
    </div>

    <div class="dashboard-grid">
      <article class="panel panel--primary">
        <div class="panel__heading">
          <Database :size="20" aria-hidden="true" />
          <h3>最新开奖</h3>
        </div>
        <p class="panel__meta">第 {{ lotteryStore.latestResult.issue }} 期</p>
        <div class="number-balls" aria-label="最新开奖号码">
          <span
            v-for="number in lotteryStore.latestResult.numbers"
            :key="number"
            class="number-ball number-ball--draw"
          >
            {{ number }}
          </span>
        </div>
      </article>

      <article class="panel">
        <div class="panel__heading">
          <PlayCircle :size="20" aria-hidden="true" />
          <h3>预测执行状态</h3>
        </div>
        <p class="panel__meta">{{ predictionStore.executionState.message }}</p>
        <div class="progress-track" aria-label="预测执行进度">
          <span :style="{ width: `${predictionStore.executionState.progress}%` }" />
        </div>
        <strong class="metric">{{ predictionStore.executionState.progress }}%</strong>
      </article>

      <article class="panel">
        <div class="panel__heading">
          <LineChart :size="20" aria-hidden="true" />
          <h3>推荐算法</h3>
        </div>
        <ul class="algorithm-list">
          <li v-for="algorithm in algorithmStore.recommendedAlgorithms" :key="algorithm.name">
            <span>{{ algorithm.displayName }}</span>
            <small>{{ algorithm.permissionLevel }} / {{ algorithm.averageCostMs }}ms</small>
          </li>
        </ul>
      </article>

      <article class="panel">
        <div class="panel__heading">
          <ArrowRight :size="20" aria-hidden="true" />
          <h3>会员权益</h3>
        </div>
        <p class="panel__meta">{{ membershipStore.status.levelName }}</p>
        <strong class="metric">{{ membershipStore.status.remainingPredictions }}</strong>
        <span class="metric-label">今日剩余预测次数</span>
      </article>
    </div>
  </section>
</template>
