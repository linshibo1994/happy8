<template>
  <view class="member-order">
    <!-- 订单确认 -->
    <view class="order-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">确认订单</text>
        </view>
        
        <view v-if="selectedPlan" class="order-card card">
          <view class="plan-info">
            <view class="plan-header">
              <text class="plan-name">{{ selectedPlan.name }}</text>
              <text class="plan-price">¥{{ selectedPlan.price }}</text>
            </view>
            
            <text class="plan-desc">{{ selectedPlan.description }}</text>
            
            <view class="plan-duration">
              <text class="duration-label">有效期:</text>
              <text class="duration-value">{{ selectedPlan.duration_days }}天</text>
            </view>
            
            <view class="plan-features">
              <view 
                v-for="feature in selectedPlan.features" 
                :key="feature"
                class="feature-item"
              >
                <text class="feature-icon">✓</text>
                <text class="feature-text">{{ feature }}</text>
              </view>
            </view>
          </view>
        </view>
      </view>
    </view>

    <!-- 优惠券 -->
    <view class="coupon-section">
      <view class="container">
        <view class="coupon-card card" @click="selectCoupon">
          <view class="coupon-info">
            <text class="coupon-label">优惠券</text>
            <text class="coupon-value">
              {{ selectedCoupon ? `${selectedCoupon.name} -¥${selectedCoupon.discount}` : '选择优惠券' }}
            </text>
          </view>
          <text class="arrow-icon">></text>
        </view>
      </view>
    </view>

    <!-- 费用明细 -->
    <view class="cost-section">
      <view class="container">
        <view class="cost-card card">
          <view class="cost-item">
            <text class="cost-label">商品金额</text>
            <text class="cost-value">¥{{ selectedPlan?.price || 0 }}</text>
          </view>
          
          <view v-if="selectedCoupon" class="cost-item discount">
            <text class="cost-label">优惠券</text>
            <text class="cost-value">-¥{{ selectedCoupon.discount }}</text>
          </view>
          
          <view class="cost-divider"></view>
          
          <view class="cost-item total">
            <text class="cost-label">实付金额</text>
            <text class="cost-value">¥{{ finalAmount }}</text>
          </view>
        </view>
      </view>
    </view>

    <!-- 支付方式 -->
    <view class="payment-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">支付方式</text>
        </view>
        
        <view class="payment-methods">
          <view 
            v-for="method in paymentMethods" 
            :key="method.type"
            class="payment-item card"
            :class="{ selected: selectedPayment === method.type }"
            @click="selectedPayment = method.type"
          >
            <view class="payment-info">
              <text class="payment-icon">{{ method.icon }}</text>
              <text class="payment-name">{{ method.name }}</text>
            </view>
            <view class="radio-icon">
              <text v-if="selectedPayment === method.type" class="selected-icon">✓</text>
            </view>
          </view>
        </view>
      </view>
    </view>

    <!-- 提交订单 -->
    <view class="submit-actions">
      <view class="container">
        <view class="total-info">
          <text class="total-label">实付金额:</text>
          <text class="total-amount">¥{{ finalAmount }}</text>
        </view>
        
        <wd-button 
          type="primary"
          size="large"
          :disabled="!selectedPlan || !selectedPayment || submitting"
          :loading="submitting"
          @click="submitOrder"
          custom-class="submit-btn"
        >
          {{ submitting ? '创建中...' : '立即支付' }}
        </wd-button>
      </view>
    </view>

    <!-- 优惠券选择弹窗 -->
    <wd-popup
      v-model="showCouponPopup"
      position="bottom"
      :close-on-click-modal="true"
    >
      <view class="coupon-popup">
        <view class="popup-header">
          <text class="popup-title">选择优惠券</text>
          <text class="popup-close" @click="showCouponPopup = false">×</text>
        </view>
        
        <scroll-view class="coupons-list" scroll-y>
          <view 
            v-for="coupon in availableCoupons" 
            :key="coupon.id"
            class="coupon-option"
            :class="{ 
              selected: selectedCoupon?.id === coupon.id,
              disabled: !coupon.can_use
            }"
            @click="chooseCoupon(coupon)"
          >
            <view class="coupon-content">
              <text class="coupon-name">{{ coupon.name }}</text>
              <text class="coupon-desc">{{ coupon.description }}</text>
              <text class="coupon-expire">有效期至: {{ formatTime(coupon.expires_at) }}</text>
            </view>
            <view class="coupon-amount">
              <text class="amount-text">¥{{ coupon.discount }}</text>
            </view>
          </view>
          
          <view 
            class="coupon-option no-coupon"
            :class="{ selected: !selectedCoupon }"
            @click="chooseCoupon(null)"
          >
            <text class="no-coupon-text">不使用优惠券</text>
          </view>
        </scroll-view>
      </view>
    </wd-popup>
  </view>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMembershipStore } from '@/stores/member'
import type { MembershipPlan, Coupon } from '@/types'
import { PAGE_PATHS } from '@/constants'

const membershipStore = useMembershipStore()

const selectedPlan = ref<MembershipPlan | null>(null)
const selectedCoupon = ref<Coupon | null>(null)
const selectedPayment = ref<string>('WECHAT')
const availableCoupons = ref<Coupon[]>([])
const showCouponPopup = ref(false)
const submitting = ref(false)

const paymentMethods = [
  { type: 'WECHAT', name: '微信支付', icon: '💳' },
  // { type: 'ALIPAY', name: '支付宝', icon: '🅰️' }
]

const finalAmount = computed(() => {
  const baseAmount = selectedPlan.value?.price || 0
  const discount = selectedCoupon.value?.discount || 0
  return Math.max(0, baseAmount - discount)
})

const formatTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return `${date.getFullYear()}/${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}`
}

const selectCoupon = () => {
  showCouponPopup.value = true
}

const chooseCoupon = (coupon: Coupon | null) => {
  if (coupon && !coupon.can_use) {
    uni.showToast({
      title: '优惠券不可用',
      icon: 'none'
    })
    return
  }
  
  selectedCoupon.value = coupon
  showCouponPopup.value = false
}

const submitOrder = async () => {
  if (!selectedPlan.value || !selectedPayment.value) return
  
  submitting.value = true
  try {
    const orderData = {
      plan_id: selectedPlan.value.id,
      payment_method: selectedPayment.value,
      coupon_id: selectedCoupon.value?.id
    }
    
    const order = await membershipStore.createOrder(orderData)
    
    // 直接发起支付
    await membershipStore.payOrder(order.id)
    
    uni.showToast({
      title: '支付成功',
      icon: 'success'
    })
    
    // 跳转到会员中心
    setTimeout(() => {
      uni.switchTab({ url: PAGE_PATHS.MEMBER })
    }, 1500)
  } catch (error) {
    console.error('支付失败:', error)
    uni.showToast({
      title: '支付失败',
      icon: 'error'
    })
  } finally {
    submitting.value = false
  }
}

const loadData = async () => {
  try {
    // 获取选中的套餐ID
    const pages = getCurrentPages()
    const currentPage = pages[pages.length - 1]
    const options = currentPage.options
    const planId = options.planId
    
    if (!planId) {
      uni.showToast({
        title: '参数错误',
        icon: 'error'
      })
      return
    }
    
    // 加载套餐和优惠券信息
    const [plans, coupons] = await Promise.all([
      membershipStore.getMembershipPlans(),
      membershipStore.getAvailableCoupons()
    ])
    
    selectedPlan.value = plans.find(plan => plan.id === planId) || null
    availableCoupons.value = coupons
    
    if (!selectedPlan.value) {
      uni.showToast({
        title: '套餐不存在',
        icon: 'error'
      })
    }
  } catch (error) {
    console.error('加载数据失败:', error)
    uni.showToast({
      title: '加载失败',
      icon: 'error'
    })
  }
}

onMounted(() => {
  loadData()
})
</script>

<style lang="scss" scoped>
.member-order {
  min-height: 100vh;
  background: $background-color;
  padding-bottom: 200rpx;
}

// 订单信息
.order-section {
  margin-bottom: $spacing-md;
  
  .order-card {
    padding: $spacing-md;
    
    .plan-header {
      @include flex-between;
      margin-bottom: $spacing-sm;
      
      .plan-name {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $text-primary;
      }
      
      .plan-price {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $primary-color;
      }
    }
    
    .plan-desc {
      font-size: $font-size-sm;
      color: $text-secondary;
      margin-bottom: $spacing-md;
    }
    
    .plan-duration {
      @include flex-align-center;
      margin-bottom: $spacing-md;
      
      .duration-label {
        font-size: $font-size-sm;
        color: $text-secondary;
        margin-right: $spacing-sm;
      }
      
      .duration-value {
        font-size: $font-size-sm;
        color: $primary-color;
        font-weight: $font-weight-bold;
      }
    }
    
    .plan-features {
      .feature-item {
        @include flex-align-center;
        margin-bottom: $spacing-xs;
        
        .feature-icon {
          color: $success-color;
          margin-right: $spacing-sm;
          font-weight: $font-weight-bold;
        }
        
        .feature-text {
          font-size: $font-size-sm;
          color: $text-primary;
        }
      }
    }
  }
}

// 优惠券
.coupon-section {
  margin-bottom: $spacing-md;
  
  .coupon-card {
    @include flex-between;
    padding: $spacing-md;
    
    .coupon-info {
      .coupon-label {
        font-size: $font-size-md;
        color: $text-primary;
        display: block;
        margin-bottom: 4rpx;
      }
      
      .coupon-value {
        font-size: $font-size-sm;
        color: $text-secondary;
      }
    }
    
    .arrow-icon {
      font-size: $font-size-lg;
      color: $text-disabled;
    }
  }
}

// 费用明细
.cost-section {
  margin-bottom: $spacing-md;
  
  .cost-card {
    padding: $spacing-md;
    
    .cost-item {
      @include flex-between;
      margin-bottom: $spacing-sm;
      
      &.discount {
        .cost-value {
          color: $success-color;
        }
      }
      
      &.total {
        margin-top: $spacing-sm;
        
        .cost-label, .cost-value {
          font-size: $font-size-md;
          font-weight: $font-weight-bold;
        }
        
        .cost-value {
          color: $primary-color;
        }
      }
      
      .cost-label {
        font-size: $font-size-sm;
        color: $text-secondary;
      }
      
      .cost-value {
        font-size: $font-size-sm;
        color: $text-primary;
      }
    }
    
    .cost-divider {
      height: 2rpx;
      background: $border-color-light;
      margin: $spacing-sm 0;
    }
  }
}

// 支付方式
.payment-section {
  margin-bottom: $spacing-md;
  
  .payment-methods {
    .payment-item {
      @include flex-between;
      padding: $spacing-md;
      margin-bottom: $spacing-sm;
      transition: all 0.3s ease;
      
      &.selected {
        border: 2rpx solid $primary-color;
        background: rgba($primary-color, 0.05);
      }
      
      .payment-info {
        @include flex-align-center;
        
        .payment-icon {
          font-size: $font-size-lg;
          margin-right: $spacing-sm;
        }
        
        .payment-name {
          font-size: $font-size-md;
          color: $text-primary;
        }
      }
      
      .radio-icon {
        .selected-icon {
          color: $primary-color;
          font-weight: $font-weight-bold;
        }
      }
    }
  }
}

// 提交订单
.submit-actions {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: white;
  padding: $spacing-md;
  @include safe-area-padding-bottom;
  box-shadow: 0 -4rpx 16rpx rgba(0, 0, 0, 0.1);
  
  .total-info {
    @include flex-between;
    margin-bottom: $spacing-sm;
    
    .total-label {
      font-size: $font-size-md;
      color: $text-secondary;
    }
    
    .total-amount {
      font-size: $font-size-lg;
      font-weight: $font-weight-bold;
      color: $primary-color;
    }
  }
  
  .submit-btn {
    width: 100%;
  }
}

// 优惠券弹窗
.coupon-popup {
  background: white;
  border-radius: $border-radius-lg $border-radius-lg 0 0;
  
  .popup-header {
    @include flex-between;
    padding: $spacing-md;
    @include border-1px($border-color-light);
    
    .popup-title {
      font-size: $font-size-md;
      font-weight: $font-weight-bold;
      color: $text-primary;
    }
    
    .popup-close {
      font-size: $font-size-xl;
      color: $text-disabled;
    }
  }
  
  .coupons-list {
    max-height: 600rpx;
    padding: 0 $spacing-md;
    
    .coupon-option {
      @include flex-between;
      padding: $spacing-md;
      margin-bottom: $spacing-sm;
      @include card;
      transition: all 0.3s ease;
      
      &.selected {
        border: 2rpx solid $primary-color;
        background: rgba($primary-color, 0.05);
      }
      
      &.disabled {
        opacity: 0.5;
      }
      
      &.no-coupon {
        @include flex-center;
        
        .no-coupon-text {
          font-size: $font-size-md;
          color: $text-secondary;
        }
      }
      
      .coupon-content {
        flex: 1;
        
        .coupon-name {
          font-size: $font-size-md;
          font-weight: $font-weight-bold;
          color: $text-primary;
          display: block;
          margin-bottom: 4rpx;
        }
        
        .coupon-desc {
          font-size: $font-size-sm;
          color: $text-secondary;
          display: block;
          margin-bottom: 4rpx;
        }
        
        .coupon-expire {
          font-size: $font-size-xs;
          color: $text-disabled;
        }
      }
      
      .coupon-amount {
        .amount-text {
          font-size: $font-size-lg;
          font-weight: $font-weight-bold;
          color: $primary-color;
        }
      }
    }
  }
}
</style>