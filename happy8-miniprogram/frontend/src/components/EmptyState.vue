<template>
  <view class="empty-state">
    <view class="empty-icon">
      <image v-if="image" :src="image" class="empty-image" mode="aspectFit" />
      <text v-else class="empty-emoji">
        {{ emoji }}
      </text>
    </view>

    <text class="empty-title">
      {{ title }}
    </text>
    <text v-if="description" class="empty-description">
      {{ description }}
    </text>

    <wd-button
      v-if="buttonText"
      :type="buttonType"
      size="small"
      custom-class="empty-button"
      @click="handleAction"
    >
      {{ buttonText }}
    </wd-button>
  </view>
</template>

<script setup lang="ts">
interface Props {
  title: string
  description?: string
  image?: string
  emoji?: string
  buttonText?: string
  buttonType?: 'primary' | 'secondary' | 'error'
}

interface Emits {
  (e: 'action'): void
}

withDefaults(defineProps<Props>(), {
  description: '',
  image: '',
  emoji: '📭',
  buttonText: '',
  buttonType: 'primary',
})

const emit = defineEmits<Emits>()

const handleAction = () => {
  emit('action')
}
</script>

<style lang="scss" scoped>
.empty-state {
  @include flex-center;
  flex-direction: column;
  padding: $spacing-xl;
  text-align: center;
}

.empty-icon {
  margin-bottom: $spacing-md;

  .empty-image {
    width: 200rpx;
    height: 200rpx;
    opacity: 0.6;
  }

  .empty-emoji {
    font-size: 120rpx;
    opacity: 0.6;
  }
}

.empty-title {
  font-size: $font-size-lg;
  font-weight: $font-weight-bold;
  color: $text-primary;
  margin-bottom: $spacing-sm;
}

.empty-description {
  font-size: $font-size-md;
  color: $text-secondary;
  line-height: $line-height-md;
  margin-bottom: $spacing-md;
}

.empty-button {
  margin-top: $spacing-sm;
}
</style>
