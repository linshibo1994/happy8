<template>
  <view class="order-detail-page">
    <view class="container">
      <view v-if="loading" class="loading">
        <wd-loading size="28px" />
      </view>

      <EmptyState
        v-else-if="!order"
        title="未找到订单"
        description="请返回订单列表重试"
        button-text="返回订单列表"
        @action="goBack"
      />

      <view v-else class="order-card card">
        <view class="order-header">
          <text class="order-title">订单号 {{ order.order_no }}</text>
          <text class="order-status" :class="order.status">
            {{ statusText }}
          </text>
        </view>
        <view class="order-meta">
          <view class="meta-item">
            <text class="label">创建时间</text>
            <text class="value">{{ formatTime(order.created_at) }}</text>
          </view>
          <view class="meta-item">
            <text class="label">金额</text>
            <text class="value">{{ membershipStore.formatPrice(order.amount) }}</text>
          </view>
          <view class="meta-item">
            <text class="label">套餐</text>
            <text class="value">{{ order.plan_name || '会员套餐' }}</text>
          </view>
          <view class="meta-item" v-if="order.paid_at">
            <text class="label">支付时间</text>
            <text class="value">{{ formatTime(order.paid_at) }}</text>
          </view>
        </view>
        <wd-button
          v-if="order.status === 'pending'"
          type="primary"
          size="large"
          @click="continuePay"
        >
          继续支付
        </wd-button>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { onLoad } from '@dcloudio/uni-app'
import { EmptyState } from '@/components'
import { membershipApi } from '@/services/membership'
import { useMembershipStore } from '@/stores/member'
import type { MembershipOrder } from '@/types'
import { PAGE_PATHS } from '@/constants'

const membershipStore = useMembershipStore()
const order = ref<MembershipOrder | null>(null)
const loading = ref(true)
const orderNo = ref('')

onLoad((options) => {
  orderNo.value = (options?.orderNo as string) || ''
})

const statusText = computed(() => {
  if (!order.value) return ''
  return membershipStore.getOrderStatusText(order.value.status)
})

const fetchOrder = async () => {
  if (!orderNo.value) {
    loading.value = false
    return
  }
  try {
    const response = await membershipApi.getOrderDetail(orderNo.value)
    if (response.code === 200) {
      order.value = response.data
    }
  } catch (error) {
    console.error(error)
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await fetchOrder()
})

const continuePay = async () => {
  if (!order.value) return
  try {
    await membershipStore.payOrder(order.value.order_no)
    await fetchOrder()
  } catch (error) {
    console.error(error)
  }
}

const formatTime = (dateStr?: string | null) => {
  if (!dateStr) return '-'
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

const goBack = () => {
  uni.navigateTo({ url: PAGE_PATHS.ORDER_LIST })
}
</script>

<style lang="scss" scoped>
.order-detail-page {
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

.order-card {
  padding: $spacing-md;

  .order-header {
    @include flex-between;
    margin-bottom: $spacing-md;

    .order-title {
      font-size: $font-size-lg;
      font-weight: $font-weight-bold;
      color: $text-primary;
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

  .order-meta {
    margin-bottom: $spacing-md;

    .meta-item {
      @include flex-between;
      margin-bottom: $spacing-sm;

      .label {
        font-size: $font-size-sm;
        color: $text-secondary;
      }

      .value {
        font-size: $font-size-sm;
        color: $text-primary;
      }
    }
  }
}
</style>
