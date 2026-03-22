<template>
  <view class="member-page">
    <view class="status-section">
      <view class="container">
        <view class="status-card card" v-if="membership">
          <view class="status-header">
            <view class="level-badge" :class="currentLevel">
              {{ membershipStore.getLevelName(currentLevel) }}
            </view>
            <view class="status-info">
              <text class="user-name">{{ userStore.getDisplayName }}</text>
              <text class="status-text">
                {{ membership.is_valid ? '会员有效' : '尚未开通会员' }}
              </text>
            </view>
          </view>
          <view class="status-meta">
            <view class="meta-item">
              <text class="meta-label">剩余天数</text>
              <text class="meta-value">{{ membershipStore.remainingDays }}</text>
            </view>
            <view class="meta-item">
              <text class="meta-label">每日上限</text>
              <text class="meta-value">
                {{ membershipStore.dailyPredictionLimit === -1 ? '无限' : membershipStore.dailyPredictionLimit }}
              </text>
            </view>
            <view class="meta-item">
              <text class="meta-label">今日已用</text>
              <text class="meta-value">{{ membershipStore.membershipStatus?.membership.predictions_today ?? 0 }}</text>
            </view>
          </view>
        </view>
        <view v-else class="status-card card">
          <text class="user-name">{{ userStore.getDisplayName }}</text>
          <text class="status-text">尚未开通会员</text>
        </view>
      </view>
    </view>

    <view class="plans-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">会员套餐</text>
        </view>
        <view class="plans-grid">
          <view
            v-for="plan in membershipStore.membershipPlans"
            :key="plan.id"
            class="plan-card card"
            :class="{ current: currentLevel === plan.level }"
          >
            <view class="plan-header">
              <text class="plan-name">{{ membershipStore.getLevelName(plan.level) }}</text>
              <text class="plan-duration">{{ membershipStore.formatDuration(plan.duration_days) }}</text>
            </view>
            <view class="plan-price">{{ membershipStore.formatPrice(plan.price) }}</view>
            <view class="plan-features">
              <view
                v-for="feature in membershipStore.getLevelFeatures(plan.level)"
                :key="feature"
                class="feature-item"
              >
                • {{ feature }}
              </view>
            </view>
            <wd-button
              v-if="currentLevel !== plan.level"
              type="primary"
              size="small"
              :loading="loadingPlanId === plan.id"
              @click="purchasePlan(plan)"
            >
              立即开通
            </wd-button>
            <wd-button v-else type="secondary" size="small" disabled>
              当前套餐
            </wd-button>
          </view>
        </view>
      </view>
    </view>

    <view class="orders-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">最近订单</text>
        </view>
        <EmptyState
          v-if="orders.length === 0"
          title="暂无订单记录"
          description="开通会员即可查看订单"
        />
        <view v-else class="orders-list">
          <view v-for="order in orders" :key="order.order_no" class="order-card card">
            <view class="order-header">
              <text class="order-plan">{{ order.plan_name || '会员套餐' }}</text>
              <text class="order-status" :class="order.status">
                {{ membershipStore.getOrderStatusText(order.status) }}
              </text>
            </view>
            <view class="order-meta">
              <text class="order-amount">{{ membershipStore.formatPrice(order.amount) }}</text>
              <text class="order-time">{{ formatTime(order.created_at) }}</text>
            </view>
            <wd-button
              v-if="order.status === 'pending'"
              type="primary"
              size="mini"
              @click="continuePay(order)"
            >
              继续支付
            </wd-button>
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
import { EmptyState } from '@/components'
import type { MembershipPlan, MembershipOrder } from '@/types'

const userStore = useUserStore()
const membershipStore = useMembershipStore()

const loadingPlanId = ref<number | null>(null)
const membership = computed(() => membershipStore.membershipStatus?.membership ?? null)
const currentLevel = computed(() => membershipStore.currentLevel)
const orders = computed<MembershipOrder[]>(() => membershipStore.orders.slice(0, 5))

const loadData = async () => {
  await membershipStore.loadAll()
  await membershipStore.loadOrders(5, 0)
}

onMounted(async () => {
  await loadData()
})

onPullDownRefresh(async () => {
  await loadData()
  uni.stopPullDownRefresh()
})

const purchasePlan = async (plan: MembershipPlan) => {
  try {
    loadingPlanId.value = plan.id
    const order = await membershipStore.createOrder(plan.id)
    if (order?.order_no) {
      await membershipStore.payOrder(order.order_no)
    }
  } catch (error) {
    console.error(error)
  } finally {
    loadingPlanId.value = null
  }
}

const continuePay = async (order: MembershipOrder) => {
  try {
    await membershipStore.payOrder(order.order_no)
  } catch (error) {
    console.error(error)
  }
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
.member-page {
  min-height: 100vh;
  background: $background-color;
}

.status-section {
  padding: $spacing-md 0;

  .status-card {
    padding: $spacing-md;

    .status-header {
      @include flex-align-center;
      margin-bottom: $spacing-md;

      .level-badge {
        padding: 8rpx $spacing-sm;
        border-radius: $border-radius-sm;
        color: white;
        margin-right: $spacing-md;

        &.free { background: $secondary-color; }
        &.vip { background: $warning-color; }
        &.premium { background: $primary-color; }
      }

      .user-name {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $text-primary;
      }

      .status-text {
        font-size: $font-size-sm;
        color: $text-secondary;
      }
    }

    .status-meta {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: $spacing-sm;

      .meta-item {
        @include flex-center;
        flex-direction: column;

        .meta-label {
          font-size: $font-size-xs;
          color: $text-secondary;
        }

        .meta-value {
          font-size: $font-size-lg;
          font-weight: $font-weight-bold;
          color: $primary-color;
        }
      }
    }
  }
}

.section-header {
  @include flex-between;
  align-items: center;
  margin-bottom: $spacing-md;

  .section-title {
    font-size: $font-size-lg;
    font-weight: $font-weight-bold;
    color: $text-primary;
  }
}

.plans-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280rpx, 1fr));
  gap: $spacing-md;

  .plan-card {
    padding: $spacing-md;
    transition: border 0.3s ease;

    &.current {
      border: 2rpx solid $primary-color;
    }

    .plan-header {
      @include flex-between;
      margin-bottom: $spacing-sm;

      .plan-name {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
      }
    }

    .plan-price {
      font-size: $font-size-xl;
      font-weight: $font-weight-bold;
      color: $primary-color;
      margin-bottom: $spacing-sm;
    }

    .plan-features {
      min-height: 120rpx;
      margin-bottom: $spacing-sm;
      color: $text-secondary;
      font-size: $font-size-sm;
    }
  }
}

.orders-section {
  margin-top: $spacing-md;

  .orders-list {
    display: grid;
    gap: $spacing-sm;
  }

  .order-card {
    padding: $spacing-md;

    .order-header {
      @include flex-between;
      margin-bottom: $spacing-sm;

      .order-plan {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
      }

      .order-status {
        font-size: $font-size-sm;
        font-weight: $font-weight-bold;
        text-transform: capitalize;

        &.pending { color: $warning-color; }
        &.paid { color: $success-color; }
        &.cancelled { color: $text-secondary; }
        &.refunded { color: $info-color; }
      }
    }

    .order-meta {
      @include flex-between;
      margin-bottom: $spacing-sm;
      color: $text-secondary;
    }
  }
}
</style>
