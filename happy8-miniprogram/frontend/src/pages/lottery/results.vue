<template>
  <view class="lottery-page">
    <view class="container">
      <view class="page-header">
        <text class="page-title">快乐8 开奖结果</text>
        <wd-button type="secondary" size="small" @click="refresh">
          刷新
        </wd-button>
      </view>

      <view v-if="loading" class="loading">
        <wd-loading size="28px" />
      </view>

      <view v-else>
        <view v-if="latest" class="latest card">
          <view class="latest-header">
            <text class="issue">第 {{ latest.issue }} 期</text>
            <text class="date">{{ formatDate(latest.draw_date) }}</text>
          </view>
          <view class="numbers-grid">
            <view
              v-for="number in latest.numbers"
              :key="number"
              class="number-ball"
              :class="getNumberZone(number)"
            >
              {{ number }}
            </view>
          </view>
          <view class="latest-stats">
            <text>和值 {{ latest.sum_value }}</text>
            <text>奇偶 {{ latest.odd_count }}:{{ latest.even_count }}</text>
            <text>大小 {{ latest.big_count }}:{{ latest.small_count }}</text>
          </view>
        </view>

        <view class="history-section">
          <text class="section-title">历史开奖</text>
          <view v-if="history.length === 0" class="empty">
            <EmptyState title="暂无数据" description="稍后再试" />
          </view>
          <view v-else class="history-list">
            <view v-for="item in history" :key="item.issue" class="history-card card">
              <view class="history-header">
                <text class="issue">第 {{ item.issue }} 期</text>
                <text class="date">{{ formatDate(item.draw_date) }}</text>
              </view>
              <view class="numbers-grid">
                <view
                  v-for="number in item.numbers"
                  :key="number"
                  class="number-ball"
                  :class="getNumberZone(number)"
                >
                  {{ number }}
                </view>
              </view>
            </view>
          </view>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { EmptyState } from '@/components'
import { lotteryApi } from '@/services/lottery'
import type { LotteryResult } from '@/types'
import { STATUS_CODES } from '@/constants'

const loading = ref(true)
const latest = ref<LotteryResult | null>(null)
const history = ref<LotteryResult[]>([])

const loadData = async () => {
  loading.value = true
  try {
    const latestResponse = await lotteryApi.getLatestResults(1)
    if (latestResponse.code === STATUS_CODES.SUCCESS && latestResponse.data.results.length > 0) {
      latest.value = latestResponse.data.results[0]
    }
    const historyResponse = await lotteryApi.getHistoricalResults({ limit: 20, offset: 0 })
    if (historyResponse.code === STATUS_CODES.SUCCESS) {
      history.value = historyResponse.data.results
    }
  } catch (error) {
    console.error(error)
  } finally {
    loading.value = false
  }
}

const refresh = async () => {
  await loadData()
}

const formatDate = (value: string) => {
  const date = new Date(value)
  return `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date
    .getDate()
    .toString()
    .padStart(2, '0')}`
}

const getNumberZone = (number: number) => {
  if (number <= 20) return 'zone-1'
  if (number <= 40) return 'zone-2'
  if (number <= 60) return 'zone-3'
  return 'zone-4'
}

onMounted(async () => {
  await loadData()
})
</script>

<style lang="scss" scoped>
.lottery-page {
  min-height: 100vh;
  background: $background-color;
}

.container {
  padding: $spacing-md;
}

.page-header {
  @include flex-between;
  margin-bottom: $spacing-md;

  .page-title {
    font-size: $font-size-lg;
    font-weight: $font-weight-bold;
  }
}

.loading {
  @include flex-center;
  padding: $spacing-xl;
}

.latest {
  padding: $spacing-md;
  margin-bottom: $spacing-md;

  .latest-header {
    @include flex-between;
    margin-bottom: $spacing-sm;
  }

  .numbers-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: $spacing-xs;
    margin-bottom: $spacing-sm;
  }

  .number-ball {
    @include flex-center;
    height: 70rpx;
    border-radius: 50%;
    font-size: $font-size-md;
    font-weight: $font-weight-bold;
    color: white;

    &.zone-1 { background: #f44336; }
    &.zone-2 { background: #ff9800; }
    &.zone-3 { background: #4caf50; }
    &.zone-4 { background: #2196f3; }
  }

  .latest-stats {
    display: flex;
    gap: $spacing-sm;
    font-size: $font-size-sm;
    color: $text-secondary;
  }
}

.history-section {
  .section-title {
    font-size: $font-size-md;
    font-weight: $font-weight-bold;
    margin-bottom: $spacing-sm;
  }

  .history-list {
    display: grid;
    gap: $spacing-sm;
  }

  .history-card {
    padding: $spacing-md;

    .history-header {
      @include flex-between;
      margin-bottom: $spacing-xs;
    }

    .numbers-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: $spacing-xs;
    }
  }
}

.empty {
  margin-top: $spacing-xl;
}
</style>
