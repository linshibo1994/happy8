<template>
  <view class="index-page">
    <view class="hero-section">
      <swiper class="hero-swiper" autoplay indicator-dots circular>
        <swiper-item v-for="(banner, index) in banners" :key="index">
          <view class="banner-item" :style="{ backgroundColor: banner.color }">
            <text class="banner-title">{{ banner.title }}</text>
            <text class="banner-subtitle">{{ banner.subtitle }}</text>
          </view>
        </swiper-item>
      </swiper>
    </view>

    <view class="quick-actions">
      <view class="container">
        <view class="action-grid">
          <view
            v-for="action in quickActions"
            :key="action.name"
            class="action-item"
            @click="handleQuickAction(action)"
          >
            <view class="action-icon">{{ action.icon }}</view>
            <text class="action-name">{{ action.name }}</text>
          </view>
        </view>
      </view>
    </view>

    <view class="lottery-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">最新开奖</text>
          <text class="more-link" @click="viewMoreResults">更多 ></text>
        </view>

        <view class="lottery-card card">
          <view v-if="latestResult" class="lottery-content">
            <view class="lottery-header">
              <text class="issue-text">第{{ latestResult.issue }}期</text>
              <text class="date-text">{{ formatDate(latestResult.draw_date) }}</text>
            </view>

            <view class="numbers-container">
              <view
                v-for="number in latestResult.numbers"
                :key="number"
                class="number-ball"
                :class="getNumberType(number)"
              >
                {{ number }}
              </view>
            </view>

            <view class="lottery-stats">
              <view class="stat-item">
                <text class="stat-label">和值</text>
                <text class="stat-value">{{ latestResult.sum_value }}</text>
              </view>
              <view class="stat-item">
                <text class="stat-label">奇偶</text>
                <text class="stat-value">{{ latestResult.odd_count }}:{{ latestResult.even_count }}</text>
              </view>
              <view class="stat-item">
                <text class="stat-label">大小</text>
                <text class="stat-value">{{ latestResult.big_count }}:{{ latestResult.small_count }}</text>
              </view>
            </view>
          </view>
          <view v-else class="loading-state">
            <wd-loading :loading="true" type="outline" />
            <text class="loading-text">加载中...</text>
          </view>
        </view>
      </view>
    </view>

    <view class="algorithm-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">推荐算法</text>
          <text class="more-link" @click="viewAllAlgorithms">全部 ></text>
        </view>

        <view class="algorithm-list">
          <view
            v-for="algorithm in recommendedAlgorithms"
            :key="algorithm.algorithm_name"
            class="algorithm-card card"
            @click="selectAlgorithm(algorithm)"
          >
            <view class="algorithm-header">
              <text class="algorithm-name">{{ algorithm.display_name }}</text>
              <view class="algorithm-badge" :class="algorithm.required_level">
                {{ membershipStore.getLevelName(algorithm.required_level) }}
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
              <text class="permission-text">
                需要{{ membershipStore.getLevelName(algorithm.required_level) }}
              </text>
            </view>
          </view>
        </view>
      </view>
    </view>

    <view v-if="userStore.isLoggedIn && userStats" class="stats-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">我的统计</text>
        </view>
        <view class="stats-card card">
          <view class="stats-grid">
            <view class="stat-item">
              <text class="stat-number">{{ userStats.total_predictions }}</text>
              <text class="stat-label">总预测次数</text>
            </view>
            <view class="stat-item">
              <text class="stat-number">{{ formatPercentage(userStats.overall_hit_rate) }}</text>
              <text class="stat-label">命中率</text>
            </view>
            <view class="stat-item">
              <text class="stat-number">{{ userStats.today_predictions }}</text>
              <text class="stat-label">今日预测</text>
            </view>
            <view class="stat-item">
              <text class="stat-number">{{ remainingPredictionsText }}</text>
              <text class="stat-label">剩余次数</text>
            </view>
          </view>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { onPullDownRefresh } from '@dcloudio/uni-app'
import { useUserStore } from '@/stores/user'
import { useMembershipStore } from '@/stores/member'
import { usePredictStore } from '@/stores/predict'
import type { AlgorithmInfo } from '@/types'
import { PAGE_PATHS } from '@/constants'

const userStore = useUserStore()
const membershipStore = useMembershipStore()
const predictStore = usePredictStore()

const banners = ref([
  { title: 'Happy8智能预测', subtitle: '让AI为您预测幸运号码', color: '#d32f2f' },
  { title: '17种算法', subtitle: '从基础到AI深度学习', color: '#2196f3' },
  { title: '专业会员服务', subtitle: '更多权益等您体验', color: '#4caf50' }
])

const quickActions = ref([
  { name: '智能预测', icon: '🎯', action: 'predict' },
  { name: '预测历史', icon: '🗂️', action: 'history' },
  { name: '会员升级', icon: '💎', action: 'member' },
  { name: '我的账户', icon: '👤', action: 'profile' }
])

const latestResult = computed(() => predictStore.latestResults[0] ?? null)
const recommendedAlgorithms = computed<AlgorithmInfo[]>(() => predictStore.availableAlgorithms.slice(0, 4))
const userStats = computed(() => predictStore.stats)
const remainingPredictionsText = computed(() => {
  if (membershipStore.remainingPredictions === -1) return '无限'
  return membershipStore.remainingPredictions
})

const loadPageData = async () => {
  await Promise.all([
    predictStore.loadAvailableAlgorithms(),
    predictStore.loadLatestLotteryResults(),
    predictStore.loadPredictionStats(),
    userStore.isLoggedIn ? predictStore.loadPredictionHistory() : Promise.resolve()
  ])
}

onMounted(async () => {
  await loadPageData()
})

onPullDownRefresh(async () => {
  await loadPageData()
  uni.stopPullDownRefresh()
})

const handleQuickAction = (action: { action: string }) => {
  switch (action.action) {
    case 'predict':
      uni.switchTab({ url: PAGE_PATHS.PREDICT })
      break
    case 'history':
      uni.switchTab({ url: PAGE_PATHS.HISTORY })
      break
    case 'member':
      uni.switchTab({ url: PAGE_PATHS.MEMBER })
      break
    case 'profile':
      uni.switchTab({ url: PAGE_PATHS.PROFILE })
      break
  }
}

const selectAlgorithm = (algorithm: AlgorithmInfo) => {
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
  uni.switchTab({ url: PAGE_PATHS.PREDICT })
}

const viewMoreResults = () => {
  uni.navigateTo({ url: PAGE_PATHS.LOTTERY_RESULTS })
}

const viewAllAlgorithms = () => {
  uni.switchTab({ url: PAGE_PATHS.PREDICT })
}

const formatDate = (dateStr: string) => {
  const date = new Date(dateStr)
  return `${date.getMonth() + 1}月${date.getDate()}日`
}

const getNumberType = (number: number) => predictStore.getNumberZone(number)

const formatPercentage = (value?: number | null) => {
  if (value === undefined || value === null) return '0%'
  return `${value.toFixed(2)}%`
}
</script>

<style lang="scss" scoped>
.index-page {
  min-height: 100vh;
  background: $background-color;
}

.hero-section {
  height: 400rpx;

  .hero-swiper {
    height: 100%;

    .banner-item {
      @include flex-center;
      flex-direction: column;
      height: 100%;
      color: white;

      .banner-title {
        font-size: $font-size-xxl;
        font-weight: $font-weight-bold;
        margin-bottom: $spacing-sm;
      }

      .banner-subtitle {
        font-size: $font-size-md;
        opacity: 0.9;
      }
    }
  }
}

.quick-actions {
  background: white;
  padding: $spacing-md 0;
  margin-bottom: $spacing-sm;

  .action-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: $spacing-md;
  }

  .action-item {
    @include flex-center;
    flex-direction: column;

    .action-icon {
      width: 80rpx;
      height: 80rpx;
      @include flex-center;
      background: rgba($primary-color, 0.1);
      border-radius: 50%;
      margin-bottom: $spacing-xs;
      font-size: 40rpx;
    }

    .action-name {
      font-size: $font-size-sm;
      color: $text-secondary;
    }
  }
}

.section-header {
  @include flex-between;
  margin-bottom: $spacing-md;

  .section-title {
    font-size: $font-size-lg;
    font-weight: $font-weight-bold;
    color: $text-primary;
  }

  .more-link {
    font-size: $font-size-sm;
    color: $primary-color;
  }
}

.lottery-section {
  margin-bottom: $spacing-md;

  .lottery-card {
    .lottery-content {
      padding: $spacing-md;
    }

    .lottery-header {
      @include flex-between;
      margin-bottom: $spacing-md;

      .issue-text {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $primary-color;
      }

      .date-text {
        font-size: $font-size-sm;
        color: $text-secondary;
      }
    }

    .numbers-container {
      display: grid;
      grid-template-columns: repeat(10, 1fr);
      gap: $spacing-xs;
      margin-bottom: $spacing-md;

      .number-ball {
        @include flex-center;
        width: 60rpx;
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

    .lottery-stats {
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
          font-size: $font-size-md;
          font-weight: $font-weight-bold;
          color: $text-primary;
        }
      }
    }
  }

  .loading-state {
    @include flex-center;
    flex-direction: column;
    padding: $spacing-xl;

    .loading-text {
      margin-top: $spacing-sm;
      color: $text-secondary;
    }
  }
}

.algorithm-section {
  margin-bottom: $spacing-md;

  .algorithm-list {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: $spacing-sm;
  }

  .algorithm-card {
    position: relative;
    padding: $spacing-md;

    .algorithm-header {
      @include flex-between;
      margin-bottom: $spacing-sm;

      .algorithm-name {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
        color: $text-primary;
      }

      .algorithm-badge {
        padding: 4rpx $spacing-xs;
        border-radius: $border-radius-sm;
        font-size: $font-size-xs;
        color: white;

        &.free { background: $secondary-color; }
        &.vip { background: $warning-color; }
        &.premium { background: $primary-color; }
      }
    }

    .algorithm-desc {
      font-size: $font-size-sm;
      color: $text-secondary;
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

      .permission-text {
        color: white;
        font-size: $font-size-sm;
        font-weight: $font-weight-bold;
      }
    }
  }
}

.stats-section {
  .stats-card {
    padding: $spacing-md;

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: $spacing-md;

      .stat-item {
        @include flex-center;
        flex-direction: column;

        .stat-number {
          font-size: $font-size-xl;
          font-weight: $font-weight-bold;
          color: $primary-color;
          margin-bottom: 4rpx;
        }

        .stat-label {
          font-size: $font-size-sm;
          color: $text-secondary;
        }
      }
    }
  }
}
</style>
