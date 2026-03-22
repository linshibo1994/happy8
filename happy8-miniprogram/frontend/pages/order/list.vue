<template>
  <view class="order-list-page">
    <view class="container">
      <view class="page-header">
        <text class="page-title">我的订单</text>
        <wd-button type="secondary" size="small" @click="refresh">
          刷新
        </wd-button>
      </view>

      <EmptyState
        v-if="orders.length === 0 && !loading"
        title="暂无订单"
        description="开通会员后可在此查看订单记录"
        button-text="前往会员中心"
        @action="goToMember"
      />

      <view v-else class="order-list">
        <view v-for="order in orders" :key="order.order_no" class="order-card card">
          <view class="order-header">
            <view>
              <text class="order-no">订单号 {{ order.order_no }}</text>
              <text class="order-time">{{ formatTime(order.created_at) }}</text>
            </view>
            <text class="order-status" :class="order.status">
              {{ membershipStore.getOrderStatusText(order.status) }}
            </text>
          </view>
          <view class="order-body">
            <view class="order-info">
              <text class="order-plan">{{ order.plan_name || '会员套餐' }}</text>
              <text class="order-amount">{{ membershipStore.formatPrice(order.amount) }}</text>
            </view>
            <view class="order-actions">
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

      <view v-if="loading" class="loading">
        <wd-loading size="24px" />
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { onPullDownRefresh } from '@dcloudio/uni-app'
import { useMembershipStore } from '@/stores/member'
import { EmptyState } from '@/components'
import type { MembershipOrder } from '@/types'
import { PAGE_PATHS } from '@/constants'

const membershipStore = useMembershipStore()
const loading = ref(false)
const orders = computed<MembershipOrder[]>(() => membershipStore.orders)

const loadOrders = async () => {
  loading.value = true
  try {
    await membershipStore.loadOrders(50, 0)
  } finally {
    loading.value = false
  }
}

const refresh = async () => {
  await loadOrders()
}

const continuePay = async (order: MembershipOrder) => {
  try {
    await membershipStore.payOrder(order.order_no)
  } catch (error) {
    console.error(error)
  }
}

const goToMember = () => {
  uni.switchTab({ url: PAGE_PATHS.MEMBER })
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

onMounted(async () => {
  await loadOrders()
})

onPullDownRefresh(async () => {
  await loadOrders()
  uni.stopPullDownRefresh()
})
</script>

<style lang="scss" scoped>
.order-list-page {
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
    color: $text-primary;
  }
}

.order-list {
  display: grid;
  gap: $spacing-sm;
}

.order-card {
  padding: $spacing-md;

  .order-header {
    @include flex-between;
    margin-bottom: $spacing-sm;

    .order-no {
      font-size: $font-size-md;
      font-weight: $font-weight-bold;
      color: $text-primary;
    }

    .order-time {
      display: block;
      font-size: $font-size-xs;
      color: $text-secondary;
    }

    .order-status {
      font-size: $font-size-sm;
      font-weight: $font-weight-bold;

      &.pending { color: $warning-color; }
      &.paid { color: $success-color; }
      &.cancelled { color: $text-secondary; }
      &.refunded { color: $info-color; }
    }
  }

  .order-body {
    @include flex-between;
    align-items: center;

    .order-plan {
      font-size: $font-size-md;
      color: $text-primary;
    }

    .order-amount {
      font-size: $font-size-lg;
      font-weight: $font-weight-bold;
      color: $primary-color;
    }
  }
}

.loading {
  @include flex-center;
  padding: $spacing-md;
}
</style>
