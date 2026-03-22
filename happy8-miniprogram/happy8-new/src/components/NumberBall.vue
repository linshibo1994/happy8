<template>
  <view
    class="number-ball"
    :class="[ballType, sizeClass, { selected, disabled }]"
    :style="customStyle"
    @click="handleClick"
  >
    <text class="number-text">
      {{ number }}
    </text>
    <view v-if="confidence !== undefined" class="confidence-indicator">
      <view class="confidence-bar" :style="{ width: `${confidence * 100}%` }" />
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  number: number
  size?: 'small' | 'medium' | 'large'
  type?: 'zone' | 'hot' | 'cold' | 'custom'
  selected?: boolean
  disabled?: boolean
  confidence?: number
  customColor?: string
}

interface Emits {
  (e: 'click', number: number): void
}

const props = withDefaults(defineProps<Props>(), {
  size: 'medium',
  type: 'zone',
  selected: false,
  disabled: false,
  confidence: undefined,
  customColor: '',
})

const emit = defineEmits<Emits>()

// 计算号码区域类型
const ballType = computed(() => {
  if (props.type === 'zone') {
    if (props.number >= 1 && props.number <= 20) return 'zone-1'
    if (props.number >= 21 && props.number <= 40) return 'zone-2'
    if (props.number >= 41 && props.number <= 60) return 'zone-3'
    if (props.number >= 61 && props.number <= 80) return 'zone-4'
  }
  return props.type
})

// 计算尺寸类名
const sizeClass = computed(() => `size-${props.size}`)

// 自定义样式
const customStyle = computed(() => {
  const style: Record<string, string> = {}
  if (props.customColor) {
    style.backgroundColor = props.customColor
  }
  return style
})

const handleClick = () => {
  if (!props.disabled) {
    emit('click', props.number)
  }
}
</script>

<style lang="scss" scoped>
.number-ball {
  position: relative;
  @include flex-center;
  border-radius: 50%;
  font-weight: $font-weight-bold;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  overflow: hidden;

  &.size-small {
    width: 50rpx;
    height: 50rpx;
    font-size: $font-size-xs;
  }

  &.size-medium {
    width: 70rpx;
    height: 70rpx;
    font-size: $font-size-sm;
  }

  &.size-large {
    width: 90rpx;
    height: 90rpx;
    font-size: $font-size-md;
  }

  // 区域颜色
  &.zone-1 {
    background: #f44336;
  }
  &.zone-2 {
    background: #ff9800;
  }
  &.zone-3 {
    background: #4caf50;
  }
  &.zone-4 {
    background: #2196f3;
  }

  // 热冷号颜色
  &.hot {
    background: linear-gradient(135deg, #ff5722, #ff9800);
  }
  &.cold {
    background: linear-gradient(135deg, #2196f3, #03a9f4);
  }

  // 状态
  &.selected {
    transform: scale(1.1);
    box-shadow: 0 0 20rpx rgba($primary-color, 0.6);
    border: 4rpx solid white;
  }

  &.disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  &:active:not(.disabled) {
    transform: scale(0.95);
  }
}

.number-text {
  position: relative;
  z-index: 2;
}

.confidence-indicator {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 6rpx;
  background: rgba(255, 255, 255, 0.3);

  .confidence-bar {
    height: 100%;
    background: rgba(255, 255, 255, 0.8);
    transition: width 0.3s ease;
  }
}
</style>
