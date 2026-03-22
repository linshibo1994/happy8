<template>
  <view class="history-detail-page">
    <view class="container">
      <view v-if="!record && loading" class="loading">
        <wd-loading size="28px" />
      </view>

      <EmptyState
        v-else-if="!record"
        title="未找到记录"
        description="请返回历史列表查看其他记录"
        button-text="返回历史"
        @action="goBack"
      />

      <view v-else class="history-card card">
        <view class="header">
          <text class="algorithm">{{ record.algorithm }}</text>
          <text class="time">{{ formatTime(record.created_at) }}</text>
        </view>
        <view class="meta">
          <text class="meta-item">目标期号：{{ record.target_issue }}</text>
          <text class="meta-item">分析期数：{{ record.periods }}</text>
          <text class="meta-item">预测数量：{{ record.count }}</text>
          <text class="meta-item">
            置信度：{{ predictStore.formatConfidenceScore(record.confidence_score) }}
          </text>
          <text class="meta-item">
            命中情况：{{ record.is_hit === true ? '已命中' : record.is_hit === false ? '未命中' : '待开奖' }}
          </text>
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
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { onLoad } from '@dcloudio/uni-app'
import { EmptyState } from '@/components'
import { usePredictStore } from '@/stores/predict'
import type { PredictionHistory } from '@/types'
import { PAGE_PATHS } from '@/constants'

const predictStore = usePredictStore()
const record = ref<PredictionHistory | null>(null)
const loading = ref(true)
let recordId: number | null = null

onLoad(options => {
  recordId = options?.id ? Number(options.id) : null
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
  loading.value = false
}

onMounted(async () => {
  await loadRecord()
})

const goBack = () => {
  uni.switchTab({ url: PAGE_PATHS.HISTORY })
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
.history-detail-page {
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

.history-card {
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

  .numbers {
    .section-title {
      font-size: $font-size-md;
      font-weight: $font-weight-bold;
      margin-bottom: $spacing-sm;
      color: $text-primary;
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
      }
    }
  }
}
</style>
