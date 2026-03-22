<template>
  <view class="prediction-detail-page">
    <view class="container">
      <view v-if="loading" class="loading">
        <wd-loading size="28px" />
      </view>

      <EmptyState
        v-else-if="!record"
        title="未找到预测"
        description="请返回重新选择"
        button-text="返回预测"
        @action="goBack"
      />

      <view v-else class="detail-card card">
        <view class="header">
          <text class="algorithm">{{ record.algorithm }}</text>
          <text class="time">{{ formatTime(record.created_at) }}</text>
        </view>

        <view class="meta">
          <view class="meta-item">
            <text class="label">目标期号</text>
            <text class="value">{{ record.target_issue }}</text>
          </view>
          <view class="meta-item">
            <text class="label">分析期数</text>
            <text class="value">{{ record.periods }}</text>
          </view>
          <view class="meta-item">
            <text class="label">预测个数</text>
            <text class="value">{{ record.count }}</text>
          </view>
          <view class="meta-item">
            <text class="label">置信度</text>
            <text class="value">{{ predictStore.formatConfidenceScore(record.confidence_score) }}</text>
          </view>
          <view class="meta-item">
            <text class="label">命中情况</text>
            <text class="value">{{ statusText }}</text>
          </view>
        </view>

        <view class="numbers">
          <text class="section-title">预测号码</text>
          <view class="numbers-grid">
            <view
              v-for="number in record.predicted_numbers"
              :key="number"
              class="number-ball"
              :class="predictStore.getNumberZone(number)"
            >
              {{ number }}
            </view>
          </view>
        </view>

        <view v-if="actualResult" class="compare">
          <text class="section-title">开奖结果</text>
          <view class="numbers-grid">
            <view
              v-for="number in actualResult.numbers"
              :key="number"
              class="number-ball actual"
              :class="{
                hit: record.predicted_numbers.includes(number)
              }"
            >
              {{ number }}
            </view>
          </view>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { onLoad } from '@dcloudio/uni-app'
import { EmptyState } from '@/components'
import { usePredictStore } from '@/stores/predict'
import { lotteryApi } from '@/services/lottery'
import type { PredictionHistory, LotteryResult } from '@/types'
import { PAGE_PATHS } from '@/constants'

const predictStore = usePredictStore()
const record = ref<PredictionHistory | null>(null)
const actualResult = ref<LotteryResult | null>(null)
const loading = ref(true)
let recordId: number | null = null

onLoad(options => {
  recordId = options?.id ? Number(options.id) : null
})

const statusText = computed(() => {
  if (!record.value) return ''
  if (record.value.is_hit === true) return '已命中'
  if (record.value.is_hit === false) return '未命中'
  return '待开奖'
})

const findRecord = () => {
  if (recordId === null) return null
  return predictStore.sortedPredictionHistory.find(item => Number(item.id) === recordId) || null
}

const loadRecord = async () => {
  if (recordId === null) {
    loading.value = false
    return
  }
  record.value = findRecord()
  if (!record.value) {
    await predictStore.loadPredictionHistory(50, 0)
    record.value = findRecord()
  }
  if (record.value) {
    try {
      const response = await lotteryApi.getResultByIssue(record.value.target_issue)
      if (response.code === 200) {
        actualResult.value = response.data
      }
    } catch (error) {
      console.error(error)
    }
  }
  loading.value = false
}

onMounted(async () => {
  await loadRecord()
})

const goBack = () => {
  uni.switchTab({ url: PAGE_PATHS.PREDICT })
}

const formatTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date
    .getDate()
    .toString()
    .padStart(2, '0')} ${date
    .getHours()
    .toString()
    .padStart(2, '0')}:${date
    .getMinutes()
    .toString()
    .padStart(2, '0')}`
}
</script>

<style lang="scss" scoped>
.prediction-detail-page {
  min-height: 100vh;
  background: $background-color;
}

.container {
  padding: $spacing-md;
}

.loading {
  @include flex-center;
  padding: $spacing-xl;
}

.detail-card {
  padding: $spacing-md;

  .header {
    @include flex-between;
    margin-bottom: $spacing-md;

    .algorithm {
      font-size: $font-size-lg;
      font-weight: $font-weight-bold;
      color: $text-primary;
    }

    .time {
      font-size: $font-size-sm;
      color: $text-secondary;
    }
  }

  .meta {
    margin-bottom: $spacing-md;

    .meta-item {
      display: block;
      font-size: $font-size-sm;
      color: $text-secondary;
      margin-bottom: 8rpx;
    }
  }

  .numbers, .compare {
    margin-bottom: $spacing-md;

    .section-title {
      font-size: $font-size-md;
      font-weight: $font-weight-bold;
      color: $text-primary;
      margin-bottom: $spacing-sm;
    }

    .numbers-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: $spacing-xs;

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

        &.actual {
          background: rgba($primary-color, 0.15);
          color: $primary-color;

          &.hit {
            background: $primary-color;
            color: white;
          }
        }
      }
    }
  }
}
</style>
