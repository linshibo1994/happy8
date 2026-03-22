<template>
  <view class="stat-card" :class="cardClass">
    <view v-if="title || $slots.header" class="card-header">
      <slot name="header">
        <text class="card-title">
          {{ title }}
        </text>
        <text v-if="subtitle" class="card-subtitle">
          {{ subtitle }}
        </text>
      </slot>
    </view>

    <view class="card-content">
      <view v-if="showMainStat" class="main-stat">
        <text class="stat-value" :style="{ color: valueColor }">
          {{ formattedValue }}
        </text>
        <text v-if="unit" class="stat-unit">
          {{ unit }}
        </text>
      </view>

      <view v-if="trend !== undefined" class="trend-indicator">
        <text class="trend-icon" :class="trendClass">
          {{ trendIcon }}
        </text>
        <text class="trend-text" :class="trendClass">
          {{ formattedTrend }}
        </text>
      </view>

      <view v-if="description" class="stat-description">
        <text class="description-text">
          {{ description }}
        </text>
      </view>

      <slot />
    </view>

    <view v-if="$slots.footer" class="card-footer">
      <slot name="footer" />
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  title?: string
  subtitle?: string
  value?: number | string
  unit?: string
  trend?: number
  description?: string
  type?: 'default' | 'primary' | 'success' | 'warning' | 'error'
  valueColor?: string
  showMainStat?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  subtitle: '',
  value: '',
  unit: '',
  trend: undefined,
  description: '',
  type: 'default',
  valueColor: '',
  showMainStat: true,
})

// 计算卡片类名
const cardClass = computed(() => `card-${props.type}`)

// 格式化数值
const formattedValue = computed(() => {
  if (typeof props.value === 'number') {
    return props.value.toLocaleString()
  }
  return props.value
})

// 趋势图标和样式
const trendIcon = computed(() => {
  if (props.trend === undefined) return ''
  if (props.trend > 0) return '↗'
  if (props.trend < 0) return '↘'
  return '→'
})

const trendClass = computed(() => {
  if (props.trend === undefined) return ''
  if (props.trend > 0) return 'trend-up'
  if (props.trend < 0) return 'trend-down'
  return 'trend-neutral'
})

const formattedTrend = computed(() => {
  if (props.trend === undefined) return ''
  const absValue = Math.abs(props.trend)
  if (absValue < 1) {
    return `${(absValue * 100).toFixed(1)}%`
  }
  return absValue.toFixed(1)
})
</script>

<style lang="scss" scoped>
.stat-card {
  @include card;
  padding: $spacing-md;

  &.card-primary {
    border-left: 8rpx solid $primary-color;
  }

  &.card-success {
    border-left: 8rpx solid $success-color;
  }

  &.card-warning {
    border-left: 8rpx solid $warning-color;
  }

  &.card-error {
    border-left: 8rpx solid $error-color;
  }
}

.card-header {
  margin-bottom: $spacing-sm;

  .card-title {
    font-size: $font-size-md;
    font-weight: $font-weight-bold;
    color: $text-primary;
    display: block;
    margin-bottom: 4rpx;
  }

  .card-subtitle {
    font-size: $font-size-sm;
    color: $text-secondary;
  }
}

.card-content {
  .main-stat {
    @include flex-align-center;
    margin-bottom: $spacing-sm;

    .stat-value {
      font-size: $font-size-xxl;
      font-weight: $font-weight-bold;
      color: $primary-color;
      margin-right: 8rpx;
    }

    .stat-unit {
      font-size: $font-size-md;
      color: $text-secondary;
    }
  }

  .trend-indicator {
    @include flex-align-center;
    margin-bottom: $spacing-sm;

    .trend-icon {
      font-size: $font-size-md;
      margin-right: 8rpx;
    }

    .trend-text {
      font-size: $font-size-sm;
      font-weight: $font-weight-medium;
    }

    .trend-up {
      color: $success-color;
    }

    .trend-down {
      color: $error-color;
    }

    .trend-neutral {
      color: $text-secondary;
    }
  }

  .stat-description {
    .description-text {
      font-size: $font-size-sm;
      color: $text-secondary;
      line-height: $line-height-md;
    }
  }
}

.card-footer {
  margin-top: $spacing-sm;
  padding-top: $spacing-sm;
  @include border-1px($border-color-light);
}
</style>
