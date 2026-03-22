<template>
  <view class="history-page">
    <view class="filter-bar">
      <view class="container">
        <wd-tabs v-model="activeTab" @change="handleTabChange">
          <wd-tab title="全部" name="all" />
          <wd-tab title="已命中" name="hit" />
          <wd-tab title="未命中" name="miss" />
        </wd-tabs>
        <wd-button type="secondary" size="small" @click="refreshHistory">
          刷新
        </wd-button>
      </view>
    </view>

    <view class="stats-overview" v-if="stats">
      <view class="container">
        <view class="stats-grid">
          <view class="stat-item">
            <text class="stat-number">{{ stats.total_predictions }}</text>
            <text class="stat-label">总预测</text>
          </view>
          <view class="stat-item">
            <text class="stat-number">{{ formatPercentage(stats.overall_hit_rate) }}</text>
            <text class="stat-label">命中率</text>
          </view>
          <view class="stat-item">
            <text class="stat-number">{{ stats.today_predictions }}</text>
            <text class="stat-label">今日预测</text>
          </view>
          <view class="stat-item">
            <text class="stat-number">{{ stats.hit_predictions }}</text>
            <text class="stat-label">命中次数</text>
          </view>
        </view>
      </view>
    </view>

    <view class="history-list">
      <view class="container">
        <view
          v-for="history in filteredHistory"
          :key="history.id"
          class="history-item card"
        >
          <view class="item-header">
            <view class="algorithm-info">
              <text class="algorithm-name">{{ history.algorithm }}</text>
              <text class="target-issue">第{{ history.target_issue }}期</text>
            </view>
            <view class="status-info">
              <text class="status-text" :class="getStatusClass(history)">
                {{ getStatusText(history) }}
              </text>
              <text class="create-time">{{ formatTime(history.created_at) }}</text>
            </view>
          </view>

          <view class="numbers-display">
            <view
              v-for="number in history.predicted_numbers"
              :key="number"
              class="number-ball"
              :class="predictStore.getNumberZone(number)"
            >
              {{ number }}
            </view>
          </view>

          <view class="item-footer">
            <view class="confidence-info">
              <text class="confidence-label">置信度</text>
              <text class="confidence-value">
                {{ predictStore.formatConfidenceScore(history.confidence_score) }}
              </text>
            </view>
            <view v-if="history.hit_rate" class="hit-info">
              <text class="hit-label">命中率</text>
              <text class="hit-value">
                {{ formatPercentage(history.hit_rate) }}
              </text>
            </view>
          </view>
        </view>

        <EmptyState
          v-if="!loading && filteredHistory.length === 0"
          title="暂无预测记录"
          description="快去尝试第一条预测吧"
          button-text="前往预测"
          @action="goToPredict"
        />

        <view v-if="hasMore" class="load-more">
          <wd-button type="secondary" size="small" :loading="loadingMore" @click="loadMore">
            {{ loadingMore ? '加载中...' : '加载更多' }}
          </wd-button>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { onPullDownRefresh } from '@dcloudio/uni-app'
import { usePredictStore } from '@/stores/predict'
import { EmptyState } from '@/components'
import { PAGE_PATHS } from '@/constants'

const predictStore = usePredictStore()

const activeTab = ref<'all' | 'hit' | 'miss'>('all')
const loading = ref(false)
const loadingMore = ref(false)

const stats = computed(() => predictStore.stats)
const historyMeta = computed(() => predictStore.historyMeta)

const filteredHistory = computed(() => {
  const list = predictStore.sortedPredictionHistory
  if (activeTab.value === 'hit') {
    return list.filter(item => item.is_hit)
  }
  if (activeTab.value === 'miss') {
    return list.filter(item => item.is_hit === false)
  }
  return list
})

const hasMore = computed(() => {
  const meta = historyMeta.value
  return meta.offset + meta.limit < meta.total
})

const formatPercentage = (value?: number | null) => {
  if (value === undefined || value === null) return '0%'
  return `${value.toFixed(2)}%`
}

const formatTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return `${date.getMonth() + 1}/${date.getDate()} ${date
    .getHours()
    .toString()
    .padStart(2, '0')}:${date
    .getMinutes()
    .toString()
    .padStart(2, '0')}`
}

const getStatusText = (history: ReturnType<typeof filteredHistory> extends Array<infer T> ? T : never) => {
  if (history.is_hit === true) return '已命中'
  if (history.is_hit === false) return '未命中'
  return '待开奖'
}

const getStatusClass = (history: ReturnType<typeof filteredHistory> extends Array<infer T> ? T : never) => {
  if (history.is_hit === true) return 'hit'
  if (history.is_hit === false) return 'miss'
  return 'pending'
}

const loadHistory = async (limit = 20, offset = 0) => {
  loading.value = true
  try {
    await predictStore.loadPredictionHistory(limit, offset)
    await predictStore.loadPredictionStats()
  } finally {
    loading.value = false
  }
}

const loadMore = async () => {
  if (loadingMore.value || !hasMore.value) return
  loadingMore.value = true
  try {
    const meta = historyMeta.value
    await predictStore.loadPredictionHistory(meta.limit, meta.offset + meta.limit)
  } finally {
    loadingMore.value = false
  }
}

const refreshHistory = async () => {
  await loadHistory(historyMeta.value.limit, 0)
}

const handleTabChange = ({ detail }: { detail: { name: 'all' | 'hit' | 'miss' } }) => {
  activeTab.value = detail.name
}

const goToPredict = () => {
  uni.switchTab({ url: PAGE_PATHS.PREDICT })
}

onMounted(async () => {
  await loadHistory()
})

onPullDownRefresh(async () => {
  await loadHistory()
  uni.stopPullDownRefresh()
})
</script>

<style lang="scss" scoped>
.history-page {
  min-height: 100vh;
  background: $background-color;
}

.filter-bar {
  background: white;
  padding: $spacing-sm $spacing-md;
  margin-bottom: $spacing-sm;
  @include border-1px($border-color-light);

  .container {
    @include flex-between;
    align-items: center;
  }
}

.stats-overview {
  background: white;
  margin-bottom: $spacing-sm;

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);

    .stat-item {
      @include flex-center;
      flex-direction: column;
      padding: $spacing-md;
      @include border-1px($border-color-light);

      .stat-number {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $primary-color;
        margin-bottom: 8rpx;
      }

      .stat-label {
        font-size: $font-size-xs;
        color: $text-secondary;
      }
    }
  }
}

.history-list {
  .history-item {
    margin-bottom: $spacing-sm;
    padding: $spacing-md;

    .item-header {
      @include flex-between;
      margin-bottom: $spacing-sm;

      .algorithm-info {
        .algorithm-name {
          font-size: $font-size-md;
          font-weight: $font-weight-bold;
          color: $text-primary;
        }

        .target-issue {
          font-size: $font-size-sm;
          color: $text-secondary;
        }
      }

      .status-info {
        text-align: right;

        .status-text {
          font-size: $font-size-sm;
          font-weight: $font-weight-bold;

          &.hit { color: $success-color; }
          &.miss { color: $error-color; }
          &.pending { color: $warning-color; }
        }

        .create-time {
          font-size: $font-size-xs;
          color: $text-secondary;
        }
      }
    }

    .numbers-display {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: $spacing-xs;
      margin-bottom: $spacing-sm;

      .number-ball {
        @include flex-center;
        height: 60rpx;
        border-radius: 50%;
        font-size: $font-size-sm;
        font-weight: $font-weight-bold;
        color: white;

        &.zone-1 { background: #f44336; }
        &.zone-2 { background: #ff9800; }
        &.zone-3 { background: #4caf50; }
        &.zone-4 { background: #2196f3; }
      }
    }

    .item-footer {
      @include flex-between;
      align-items: center;

      .confidence-info, .hit-info {
        @include flex-align-center;

        .confidence-label,
        .hit-label {
          font-size: $font-size-xs;
          color: $text-secondary;
          margin-right: 8rpx;
        }

        .confidence-value,
        .hit-value {
          font-size: $font-size-sm;
          color: $text-primary;
        }
      }
    }
  }
}

.load-more {
  @include flex-center;
  padding: $spacing-md;
}
</style>
