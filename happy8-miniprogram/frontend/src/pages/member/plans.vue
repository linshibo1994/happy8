<template>
  <view class="member-plans">
    <!-- 当前会员状态 -->
    <view class="current-status">
      <view class="container">
        <view class="status-card card">
          <view class="status-header">
            <text class="current-level">当前等级</text>
            <text class="level-badge" :class="membershipStore.currentLevel.toLowerCase()">
              {{ membershipStore.getLevelName(membershipStore.currentLevel) }}
            </text>
          </view>
          
          <view class="status-info">
            <view class="info-item">
              <text class="label">到期时间:</text>
              <text class="value">{{ formatTime(membershipStore.expiresAt) }}</text>
            </view>
            <view class="info-item">
              <text class="label">剩余预测:</text>
              <text class="value">{{ getRemainingText() }}</text>
            </view>
          </view>
        </view>
      </view>
    </view>

    <!-- 套餐选择 -->
    <view class="plans-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">选择套餐</text>
          <text class="section-subtitle">解锁更多预测算法和功能</text>
        </view>
        
        <view class="plans-grid">
          <view 
            v-for="plan in plans" 
            :key="plan.id"
            class="plan-card"
            :class="{ 
              recommended: plan.is_recommended,
              selected: selectedPlan?.id === plan.id,
              current: plan.level === membershipStore.currentLevel
            }"
            @click="selectPlan(plan)"
          >
            <view v-if="plan.is_recommended" class="recommend-badge">
              <text class="recommend-text">推荐</text>
            </view>
            
            <view class="plan-header">
              <text class="plan-name">{{ plan.name }}</text>
              <view class="plan-price">
                <text class="price-currency">¥</text>
                <text class="price-amount">{{ plan.price }}</text>
                <text class="price-unit">/{{ plan.duration_days }}天</text>
              </view>
            </view>
            
            <text class="plan-desc">{{ plan.description }}</text>
            
            <view class="plan-features">
              <view 
                v-for="feature in plan.features" 
                :key="feature"
                class="feature-item"
              >
                <text class="feature-icon">✓</text>
                <text class="feature-text">{{ feature }}</text>
              </view>
            </view>
            
            <view v-if="plan.level === membershipStore.currentLevel" class="current-mark">
              <text class="current-text">当前套餐</text>
            </view>
          </view>
        </view>
      </view>
    </view>

    <!-- 购买按钮 -->
    <view class="purchase-actions">
      <view class="container">
        <view v-if="selectedPlan" class="selected-info">
          <text class="selected-name">已选择: {{ selectedPlan.name }}</text>
          <text class="selected-price">¥{{ selectedPlan.price }}</text>
        </view>
        
        <wd-button 
          type="primary"
          size="large"
          :disabled="!selectedPlan || purchasing"
          :loading="purchasing"
          @click="purchasePlan"
          custom-class="purchase-btn"
        >
          {{ purchasing ? '处理中...' : '立即购买' }}
        </wd-button>
        
        <view class="payment-methods">
          <text class="methods-text">支持微信支付</text>
        </view>
      </view>
    </view>

    <!-- 会员权益说明 -->
    <view class="benefits-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">会员权益对比</text>
        </view>
        
        <view class="benefits-table">
          <view class="table-header">
            <text class="feature-col">功能特权</text>
            <text class="level-col">免费版</text>
            <text class="level-col">VIP</text>
            <text class="level-col">高级版</text>
          </view>
          
          <view 
            v-for="benefit in memberBenefits" 
            :key="benefit.feature"
            class="table-row"
          >
            <text class="feature-name">{{ benefit.feature }}</text>
            <text class="benefit-value" :class="{ active: benefit.free }">
              {{ benefit.free ? '✓' : '✕' }}
            </text>
            <text class="benefit-value" :class="{ active: benefit.vip }">
              {{ benefit.vip ? '✓' : '✕' }}
            </text>
            <text class="benefit-value" :class="{ active: benefit.premium }">
              {{ benefit.premium ? '✓' : '✕' }}
            </text>
          </view>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMembershipStore } from '@/stores/member'
import type { MembershipPlan } from '@/types'
import { PAGE_PATHS } from '@/constants'

const membershipStore = useMembershipStore()

const plans = ref<MembershipPlan[]>([])
const selectedPlan = ref<MembershipPlan | null>(null)
const purchasing = ref(false)
const loading = ref(true)

// 会员权益对比
const memberBenefits = [
  {
    feature: '基础算法',
    free: true,
    vip: true,
    premium: true
  },
  {
    feature: '高级算法',
    free: false,
    vip: true,
    premium: true
  },
  {
    feature: '深度学习算法',
    free: false,
    vip: false,
    premium: true
  },
  {
    feature: '每日预测次数',
    free: '3次',
    vip: '20次',
    premium: '无限'
  },
  {
    feature: '历史记录查看',
    free: '7天',
    vip: '90天',
    premium: '无限'
  },
  {
    feature: '预测结果分析',
    free: false,
    vip: true,
    premium: true
  },
  {
    feature: '专属客服',
    free: false,
    vip: false,
    premium: true
  }
]

const getRemainingText = () => {
  if (membershipStore.remainingPredictions === -1) {
    return '无限制'
  }
  return `${membershipStore.remainingPredictions} 次`
}

const formatTime = (dateStr?: string) => {
  if (!dateStr) return '永久'
  const date = new Date(dateStr)
  return `${date.getFullYear()}/${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}`
}

const selectPlan = (plan: MembershipPlan) => {
  if (plan.level === membershipStore.currentLevel) {
    uni.showToast({
      title: '您已是该套餐用户',
      icon: 'none'
    })
    return
  }
  
  selectedPlan.value = plan
}

const purchasePlan = async () => {
  if (!selectedPlan.value) return
  
  purchasing.value = true
  try {
    const order = await membershipStore.createOrder(selectedPlan.value.id)
    
    // 跳转到订单详情页面进行支付
    uni.navigateTo({
      url: `${PAGE_PATHS.ORDER_DETAIL}?id=${order.id}`
    })
  } catch (error) {
    console.error('创建订单失败:', error)
    uni.showToast({
      title: '创建订单失败',
      icon: 'error'
    })
  } finally {
    purchasing.value = false
  }
}

const loadPlans = async () => {
  try {
    plans.value = await membershipStore.getMembershipPlans()
    
    // 自动选择推荐套餐（如果不是当前套餐）
    const recommendedPlan = plans.value.find(plan => 
      plan.is_recommended && plan.level !== membershipStore.currentLevel
    )
    if (recommendedPlan) {
      selectedPlan.value = recommendedPlan
    }
  } catch (error) {
    console.error('加载套餐失败:', error)
    uni.showToast({
      title: '加载套餐失败',
      icon: 'error'
    })
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadPlans()
})
</script>

<style lang="scss" scoped>
.member-plans {
  min-height: 100vh;
  background: $background-color;
  padding-bottom: 200rpx;
}

// 当前状态
.current-status {
  margin-bottom: $spacing-md;
  
  .status-card {
    padding: $spacing-md;
    
    .status-header {
      @include flex-between;
      margin-bottom: $spacing-md;
      
      .current-level {
        font-size: $font-size-md;
        color: $text-secondary;
      }
      
      .level-badge {
        padding: 8rpx $spacing-sm;
        border-radius: $border-radius-sm;
        font-size: $font-size-sm;
        color: white;
        font-weight: $font-weight-bold;
        
        &.free { background: $secondary-color; }
        &.vip { background: $warning-color; }
        &.premium { background: $primary-color; }
      }
    }
    
    .status-info {
      .info-item {
        @include flex-between;
        margin-bottom: $spacing-sm;
        
        &:last-child {
          margin-bottom: 0;
        }
        
        .label {
          font-size: $font-size-sm;
          color: $text-secondary;
        }
        
        .value {
          font-size: $font-size-sm;
          color: $text-primary;
          font-weight: $font-weight-medium;
        }
      }
    }
  }
}

// 套餐选择
.plans-section {
  margin-bottom: $spacing-md;
  
  .section-header {
    text-align: center;
    margin-bottom: $spacing-lg;
    
    .section-title {
      font-size: $font-size-lg;
      font-weight: $font-weight-bold;
      color: $text-primary;
      display: block;
      margin-bottom: $spacing-sm;
    }
    
    .section-subtitle {
      font-size: $font-size-sm;
      color: $text-secondary;
    }
  }
  
  .plans-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: $spacing-md;
  }
  
  .plan-card {
    position: relative;
    padding: $spacing-lg;
    @include card;
    transition: all 0.3s ease;
    
    &.recommended {
      border: 2rpx solid $warning-color;
      
      .recommend-badge {
        position: absolute;
        top: -2rpx;
        right: 40rpx;
        background: $warning-color;
        color: white;
        padding: 8rpx $spacing-sm;
        border-radius: 0 0 $border-radius-sm $border-radius-sm;
        
        .recommend-text {
          font-size: $font-size-xs;
          font-weight: $font-weight-bold;
        }
      }
    }
    
    &.selected {
      border: 2rpx solid $primary-color;
      background: rgba($primary-color, 0.05);
    }
    
    &.current {
      opacity: 0.6;
      
      .current-mark {
        position: absolute;
        top: $spacing-sm;
        right: $spacing-sm;
        
        .current-text {
          background: $success-color;
          color: white;
          padding: 4rpx 8rpx;
          border-radius: $border-radius-sm;
          font-size: $font-size-xs;
        }
      }
    }
    
    .plan-header {
      text-align: center;
      margin-bottom: $spacing-md;
      
      .plan-name {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $text-primary;
        display: block;
        margin-bottom: $spacing-sm;
      }
      
      .plan-price {
        @include flex-center;
        
        .price-currency {
          font-size: $font-size-md;
          color: $primary-color;
        }
        
        .price-amount {
          font-size: 60rpx;
          font-weight: $font-weight-bold;
          color: $primary-color;
          margin: 0 4rpx;
        }
        
        .price-unit {
          font-size: $font-size-sm;
          color: $text-secondary;
        }
      }
    }
    
    .plan-desc {
      font-size: $font-size-sm;
      color: $text-secondary;
      text-align: center;
      margin-bottom: $spacing-lg;
    }
    
    .plan-features {
      .feature-item {
        @include flex-align-center;
        margin-bottom: $spacing-sm;
        
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

// 购买按钮
.purchase-actions {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: white;
  padding: $spacing-md;
  @include safe-area-padding-bottom;
  box-shadow: 0 -4rpx 16rpx rgba(0, 0, 0, 0.1);
  
  .selected-info {
    @include flex-between;
    margin-bottom: $spacing-sm;
    
    .selected-name {
      font-size: $font-size-sm;
      color: $text-secondary;
    }
    
    .selected-price {
      font-size: $font-size-md;
      font-weight: $font-weight-bold;
      color: $primary-color;
    }
  }
  
  .purchase-btn {
    width: 100%;
    margin-bottom: $spacing-sm;
  }
  
  .payment-methods {
    text-align: center;
    
    .methods-text {
      font-size: $font-size-xs;
      color: $text-disabled;
    }
  }
}

// 权益对比
.benefits-section {
  .benefits-table {
    background: white;
    border-radius: $border-radius-md;
    overflow: hidden;
    
    .table-header {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr;
      background: rgba($primary-color, 0.1);
      padding: $spacing-sm;
      
      .feature-col, .level-col {
        font-size: $font-size-sm;
        font-weight: $font-weight-bold;
        color: $text-primary;
        text-align: center;
      }
      
      .feature-col {
        text-align: left;
      }
    }
    
    .table-row {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr;
      padding: $spacing-sm;
      @include border-1px($border-color-light);
      
      &:last-child {
        border-bottom: none;
      }
      
      .feature-name {
        font-size: $font-size-sm;
        color: $text-primary;
      }
      
      .benefit-value {
        font-size: $font-size-sm;
        color: $text-disabled;
        text-align: center;
        
        &.active {
          color: $success-color;
          font-weight: $font-weight-bold;
        }
      }
    }
  }
}
</style>