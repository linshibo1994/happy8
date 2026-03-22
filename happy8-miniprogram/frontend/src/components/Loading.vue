<template>
  <view v-if="visible" class="loading-overlay" :class="{ fullscreen }">
    <view class="loading-content">
      <view class="loading-spinner" :class="type">
        <view v-if="type === 'dots'" class="dots-spinner">
          <view class="dot" />
          <view class="dot" />
          <view class="dot" />
        </view>
        <view v-else-if="type === 'circle'" class="circle-spinner">
          <view class="circle" />
        </view>
        <view v-else class="default-spinner">
          <text class="spinner-icon">🔄</text>
        </view>
      </view>

      <text v-if="text" class="loading-text">
        {{ text }}
      </text>
    </view>
  </view>
</template>

<script setup lang="ts">
interface Props {
  visible?: boolean
  type?: 'default' | 'dots' | 'circle'
  text?: string
  fullscreen?: boolean
}

withDefaults(defineProps<Props>(), {
  visible: false,
  type: 'default',
  text: '',
  fullscreen: false,
})
</script>

<style lang="scss" scoped>
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  @include flex-center;
  background: rgba(255, 255, 255, 0.8);
  z-index: $z-index-loading;

  &.fullscreen {
    position: fixed;
    background: rgba(0, 0, 0, 0.5);
  }
}

.loading-content {
  @include flex-center;
  flex-direction: column;
  padding: $spacing-md;
  background: white;
  border-radius: $border-radius-md;
  box-shadow: $shadow-lg;
}

.loading-spinner {
  margin-bottom: $spacing-sm;

  &.dots {
    .dots-spinner {
      @include flex-center;

      .dot {
        width: 12rpx;
        height: 12rpx;
        background: $primary-color;
        border-radius: 50%;
        margin: 0 4rpx;
        animation: dots-bounce 1.4s infinite ease-in-out both;

        &:nth-child(1) {
          animation-delay: -0.32s;
        }
        &:nth-child(2) {
          animation-delay: -0.16s;
        }
        &:nth-child(3) {
          animation-delay: 0s;
        }
      }
    }
  }

  &.circle {
    .circle-spinner {
      width: 60rpx;
      height: 60rpx;

      .circle {
        width: 100%;
        height: 100%;
        border: 4rpx solid $border-color-light;
        border-top-color: $primary-color;
        border-radius: 50%;
        animation: circle-spin 1s linear infinite;
      }
    }
  }

  &.default {
    .default-spinner {
      .spinner-icon {
        font-size: 60rpx;
        animation: default-spin 1s linear infinite;
      }
    }
  }
}

.loading-text {
  font-size: $font-size-sm;
  color: $text-secondary;
  text-align: center;
}

@keyframes dots-bounce {
  0%,
  80%,
  100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1);
  }
}

@keyframes circle-spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

@keyframes default-spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>
