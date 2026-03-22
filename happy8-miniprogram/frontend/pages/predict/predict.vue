<template>
  <view class="predict-page">
    <view class="status-bar">
      <view class="container">
        <view class="status-info">
          <view class="user-level">
            <text class="level-badge" :class="currentLevel">
              {{ membershipStore.getLevelName(currentLevel) }}
            </text>
            <text v-if="remainingPredictionsText !== '无限'" class="remaining-count">剩余 {{ remainingPredictionsText }} 次</text>
            <text v-else class="remaining-count">无限次数</text>
          </view>
          <view class="next-issue">
            <text class="issue-label">下期:</text>
            <text class="issue-number">{{ predictionParams.target_issue || '待更新' }}</text>
          </view>
        </view>
      </view>
    </view>

    <view class="algorithm-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">选择算法</text>
          <text v-if="predictStore.hasSelectedAlgorithm" class="selected-hint">
            已选择: {{ predictStore.selectedAlgorithm?.display_name }}
          </text>
        </view>

        <view class="algorithm-tabs">
          <view
            v-for="tab in levelTabs"
            :key="tab.key"
            class="tab-item"
            :class="{ active: activeTab === tab.key }"
            @click="activeTab = tab.key"
          >
            <text class="tab-text">{{ tab.label }}</text>
            <text class="tab-count">({{ algorithmsByLevel[tab.key].length }})</text>
          </view>
        </view>

        <scroll-view class="algorithm-list" scroll-y>
          <view
            v-for="algorithm in algorithmsByLevel[activeTab]"
            :key="algorithm.algorithm_name"
            class="algorithm-item"
            :class="{
              selected: predictStore.selectedAlgorithm?.algorithm_name === algorithm.algorithm_name,
              disabled: !algorithm.has_permission
            }"
            @click="handleSelectAlgorithm(algorithm)"
          >
            <view class="algorithm-header">
              <text class="algorithm-name">{{ algorithm.display_name }}</text>
              <view class="algorithm-badges">
                <view class="level-badge" :class="algorithm.required_level">
                  {{ membershipStore.getLevelName(algorithm.required_level) }}
                </view>
              </view>
            </view>
            <text class="algorithm-desc">{{ algorithm.description }}</text>
            <view class="algorithm-stats">
              <view class="stat-item">
                <text class="stat-label">成功率</text>
                <text class="stat-value">{{ formatPercentage(algorithm.success_rate) }}</text>
              </view>
              <view class="stat-item">
                <text class="stat-label">使用次数</text>
                <text class="stat-value">{{ algorithm.usage_count }}</text>
              </view>
            </view>
            <view v-if="!algorithm.has_permission" class="permission-overlay">
              <text class="permission-text">需要升级</text>
            </view>
          </view>
        </scroll-view>
      </view>
    </view>

    <view v-if="predictStore.hasSelectedAlgorithm" class="params-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">预测参数</text>
        </view>
        <view class="params-form card">
          <view class="form-row">
            <text class="form-label">目标期号</text>
            <input
              class="form-input"
              type="number"
              :value="predictionParams.target_issue"
              @input="onIssueInput"
              placeholder="请输入期号"
            />
          </view>
          <view class="form-row">
            <text class="form-label">分析期数</text>
            <wd-stepper
              v-model="predictionParams.periods"
              :min="10"
              :max="200"
              :step="5"
            />
          </view>
          <view class="form-row">
            <text class="form-label">预测个数</text>
            <wd-stepper v-model="predictionParams.count" :min="1" :max="20" />
          </view>
        </view>
      </view>
    </view>

    <view class="predict-actions">
      <view class="container">
        <wd-button
          type="primary"
          size="large"
          :loading="predictStore.predicting"
          :disabled="!predictStore.canPredict"
          @click="triggerPrediction"
          custom-class="predict-button"
        >
          {{ predictStore.predicting ? '预测中...' : '开始预测' }}
        </wd-button>
        <view v-if="!predictStore.canPredict" class="tips">
          <text class="tips-text">请选择算法并填写目标期号</text>
        </view>
      </view>
    </view>

    <view v-if="predictStore.currentPrediction" class="result-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">预测结果</text>
        </view>
        <view class="result-card card">
          <view class="result-header">
            <text class="algorithm-name">{{ predictStore.selectedAlgorithm?.display_name }}</text>
            <text class="confidence-score">
              置信度: {{ predictStore.formatConfidenceScore(predictStore.currentPrediction.confidence_score) }}
            </text>
          </view>
          <view class="numbers-container">
            <view
              v-for="number in predictStore.currentPrediction.predicted_numbers"
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

    <view class="quick-history">
      <view class="container">
        <view class="section-header">
          <text class="section-title">最近预测</text>
          <text class="more-link" @click="viewHistory">全部 ></text>
        </view>
        <view v-if="historyList.length" class="history-wrapper">
          <view v-for="item in historyList" :key="item.id" class="history-item card">
            <view class="history-header">
              <text class="algorithm-name">{{ item.algorithm }}</text>
              <text class="prediction-time">{{ formatTime(item.created_at) }}</text>
            </view>
            <view class="mini-numbers">
              <view
                v-for="number in item.predicted_numbers.slice(0, 8)"
                :key="number"
                class="mini-number"
              >
                {{ number }}
              </view>
            </view>
            <view class="history-status">
              <text class="status-text">置信度 {{ predictStore.formatConfidenceScore(item.confidence_score) }}</text>
              <text v-if="item.actual_numbers?.length" class="hit-rate">
                命中率 {{ predictStore.calculateHitRate(item.predicted_numbers, item.actual_numbers) }}%
              </text>
            </view>
          </view>
        </view>
        <EmptyState
          v-else
          title="暂无预测记录"
          description="快去尝试第一条预测吧"
          button-text="开始预测"
          @action="triggerPrediction"
        />
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { onPullDownRefresh } from '@dcloudio/uni-app'
import { useMembershipStore } from '@/stores/member'
import { usePredictStore } from '@/stores/predict'
import type { AlgorithmInfo } from '@/types'
import { PAGE_PATHS } from '@/constants'
import { EmptyState } from '@/components'

const membershipStore = useMembershipStore()
const predictStore = usePredictStore()

const predictionParams = computed({
  get: () => predictStore.predictionParams,
  set: value => predictStore.updatePredictionParams(value)
})

const currentLevel = computed(() => membershipStore.currentLevel)
const remainingPredictionsText = computed(() => {
  if (membershipStore.remainingPredictions === -1) return '无限'
  return membershipStore.remainingPredictions
})

const levelTabs = [
  { key: 'free', label: '免费算法' },
  { key: 'vip', label: 'VIP算法' },
  { key: 'premium', label: '高级算法' }
]

const activeTab = ref<'free' | 'vip' | 'premium'>('free')
const algorithmsByLevel = computed(() => predictStore.algorithmsByLevel)
const historyList = computed(() => predictStore.sortedPredictionHistory.slice(0, 5))

const loadPageData = async () => {
  await Promise.all([
    predictStore.loadAvailableAlgorithms(),
    predictStore.loadPredictionHistory(),
    predictStore.loadLatestLotteryResults()
  ])
}

onMounted(async () => {
  await loadPageData()
})

onPullDownRefresh(async () => {
  await loadPageData()
  uni.stopPullDownRefresh()
})

const handleSelectAlgorithm = (algorithm: AlgorithmInfo) => {
  if (!algorithm.has_permission) {
    uni.showModal({
      title: '权限不足',
      content: '该算法需要更高会员等级，是否前往会员中心？',
      success: res => {
        if (res.confirm) {
          uni.switchTab({ url: PAGE_PATHS.MEMBER })
        }
      }
    })
    return
  }
  predictStore.setSelectedAlgorithm(algorithm)
  if (!predictionParams.value.target_issue && predictStore.latestResults.length) {
    predictStore.updatePredictionParams({
      target_issue: predictStore.latestResults[0].issue
    })
  }
}

const onIssueInput = (event: any) => {
  predictStore.updatePredictionParams({ target_issue: event.detail.value })
}

const triggerPrediction = async () => {
  if (!membershipStore.canUsePrediction()) {
    uni.showModal({
      title: '预测次数不足',
      content: '今日预测次数已用完，是否前往会员中心？',
      success: res => {
        if (res.confirm) {
          uni.switchTab({ url: PAGE_PATHS.MEMBER })
        }
      }
    })
    return
  }

  try {
    await predictStore.generatePrediction()
    membershipStore.updateDailyPredictionCount()
  } catch (error) {
    console.error(error)
  }
}

const formatPercentage = (value?: number | null) => {
  if (value === undefined || value === null) return '0%'
  return `${value.toFixed(2)}%`
}

const formatTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`
}

const viewHistory = () => {
  uni.switchTab({ url: PAGE_PATHS.HISTORY })
}
</script>

<style lang="scss" scoped>
.predict-page {
  min-height: 100vh;
  background: $background-color;
  padding-bottom: 200rpx;
}

.status-bar {
  background: white;
  padding: $spacing-md;
  margin-bottom: $spacing-sm;
  @include border-1px($border-color-light);

  .status-info {
    @include flex-between;

    .user-level {
      @include flex-align-center;

      .level-badge {
        padding: 8rpx $spacing-sm;
        border-radius: $border-radius-sm;
        font-size: $font-size-xs;
        color: white;
        margin-right: $spacing-sm;

        &.free { background: $secondary-color; }
        &.vip { background: $warning-color; }
        &.premium { background: $primary-color; }
      }

      .remaining-count {
        font-size: $font-size-sm;
        color: $text-secondary;
      }
    }

    .next-issue {
      @include flex-align-center;

      .issue-label {
        font-size: $font-size-sm;
        color: $text-secondary;
        margin-right: 8rpx;
      }

      .issue-number {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
        color: $primary-color;
      }
    }
  }
}

.algorithm-section {
  background: white;
  margin-bottom: $spacing-sm;

  .algorithm-tabs {
    @include flex-align-center;
    padding: 0 $spacing-md;
    margin-bottom: $spacing-md;
    @include border-1px($border-color-light);

    .tab-item {
      flex: 1;
      @include flex-center;
      padding: $spacing-sm 0;
      position: relative;

      .tab-text {
        font-size: $font-size-sm;
        color: $text-secondary;
      }

      .tab-count {
        font-size: $font-size-xs;
        color: $text-disabled;
        margin-left: 8rpx;
      }

      &.active {
        .tab-text {
          color: $primary-color;
          font-weight: $font-weight-bold;
        }

        &::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 50%;
          transform: translateX(-50%);
          width: 60rpx;
          height: 4rpx;
          background: $primary-color;
          border-radius: 2rpx;
        }
      }
    }
  }

  .algorithm-list {
    max-height: 600rpx;
    padding: 0 $spacing-md;
  }

  .algorithm-item {
    position: relative;
    padding: $spacing-md;
    margin-bottom: $spacing-sm;
    @include card;
    transition: all 0.3s ease;

    &.selected {
      border: 2rpx solid $primary-color;
      background: rgba($primary-color, 0.05);
    }

    &.disabled {
      opacity: 0.6;

      .permission-overlay {
        display: flex;
      }
    }

    .algorithm-header {
      @include flex-between;
      margin-bottom: $spacing-sm;

      .algorithm-name {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
        color: $text-primary;
      }

      .algorithm-badges {
        @include flex-align-center;

        .level-badge {
          padding: 4rpx 8rpx;
          border-radius: $border-radius-sm;
          font-size: $font-size-xs;
          color: white;

          &.free { background: $secondary-color; }
          &.vip { background: $warning-color; }
          &.premium { background: $primary-color; }
        }
      }
    }

    .algorithm-desc {
      font-size: $font-size-sm;
      color: $text-secondary;
      line-height: $line-height-md;
      margin-bottom: $spacing-sm;
      @include text-ellipsis-multiline(2);
    }

    .algorithm-stats {
      @include flex-between;

      .stat-item {
        @include flex-center;
        flex-direction: column;

        .stat-label {
          font-size: $font-size-xs;
          color: $text-secondary;
          margin-bottom: 4rpx;
        }

        .stat-value {
          font-size: $font-size-sm;
          font-weight: $font-weight-bold;
          color: $primary-color;
        }
      }
    }

    .permission-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      @include flex-center;
      background: rgba(0, 0, 0, 0.7);
      border-radius: $border-radius-md;
      display: none;

      .permission-text {
        color: white;
        font-size: $font-size-sm;
        font-weight: $font-weight-bold;
      }
    }
  }
}

.params-section {
  margin-bottom: $spacing-md;

  .params-form {
    padding: $spacing-md;

    .form-row {
      @include flex-between;
      padding: $spacing-md 0;
      @include border-1px($border-color-light);

      &:last-child {
        border-bottom: none;
      }

      .form-label {
        font-size: $font-size-md;
        color: $text-primary;
      }

      .form-input {
        flex: 1;
        max-width: 220rpx;
        text-align: right;
        font-size: $font-size-md;
        color: $text-primary;
      }
    }
  }
}

.predict-actions {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: white;
  padding: $spacing-md;
  @include safe-area-padding-bottom;
  box-shadow: 0 -4rpx 16rpx rgba(0, 0, 0, 0.1);

  .predict-button {
    width: 100%;
    margin-bottom: $spacing-sm;
  }

  .tips {
    text-align: center;

    .tips-text {
      font-size: $font-size-xs;
      color: $warning-color;
    }
  }
}

.result-section {
  margin-bottom: $spacing-md;

  .result-card {
    padding: $spacing-md;

    .result-header {
      @include flex-between;
      margin-bottom: $spacing-md;

      .algorithm-name {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
        color: $text-primary;
      }

      .confidence-score {
        font-size: $font-size-sm;
        color: $primary-color;
      }
    }

    .numbers-container {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: $spacing-sm;

      .number-ball {
        @include flex-center;
        height: 80rpx;
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

.quick-history {
  margin-bottom: $spacing-xl;

  .history-wrapper {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: $spacing-sm;
  }

  .history-item {
    padding: $spacing-sm;

    .history-header {
      @include flex-between;
      margin-bottom: $spacing-xs;

      .algorithm-name {
        font-size: $font-size-sm;
        font-weight: $font-weight-bold;
        color: $text-primary;
      }

      .prediction-time {
        font-size: $font-size-xs;
        color: $text-secondary;
      }
    }

    .mini-numbers {
      display: flex;
      flex-wrap: wrap;
      gap: 4rpx;
      margin-bottom: $spacing-xs;

      .mini-number {
        @include flex-center;
        width: 40rpx;
        height: 40rpx;
        border-radius: 50%;
        font-size: $font-size-xs;
        background: rgba($primary-color, 0.1);
        color: $primary-color;
      }
    }

    .history-status {
      @include flex-between;

      .status-text {
        font-size: $font-size-xs;
        color: $text-secondary;
      }

      .hit-rate {
        font-size: $font-size-xs;
        color: $text-secondary;
      }
    }
  }
}
</style>
