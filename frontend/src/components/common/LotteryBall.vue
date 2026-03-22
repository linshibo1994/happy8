<template>
  <div
    class="lottery-ball"
    :class="[`ball-${size}`, { 'ball-animate': animate }]"
    :style="{ animationDelay: `${delay}ms` }"
  >
    <span class="ball-highlight"></span>
    <span class="ball-number">{{ formatBallNumber(number) }}</span>
  </div>
</template>

<script setup lang="ts">
import { formatBallNumber } from '@/utils/format'

interface Props {
  number: number
  size?: 'sm' | 'md' | 'lg'
  animate?: boolean
  delay?: number
}

withDefaults(defineProps<Props>(), {
  size: 'md',
  animate: false,
  delay: 0
})
</script>

<style scoped>
.lottery-ball {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-weight: 700;
  color: #fff;
  cursor: default;
  user-select: none;
  background: radial-gradient(circle at 30% 28%, #ffd1d1 0%, #ff8a8a 28%, #ff6b6b 62%, #cc4343 100%);
  box-shadow:
    inset 0 8px 10px rgba(255, 255, 255, 0.2),
    inset 0 -8px 14px rgba(0, 0, 0, 0.35),
    0 8px 16px rgba(0, 0, 0, 0.45),
    0 0 14px rgba(255, 107, 107, 0.35);
  transition: transform 0.24s ease, box-shadow 0.24s ease;
}

.lottery-ball:hover {
  transform: translateY(-2px) scale(1.03);
  box-shadow:
    inset 0 8px 12px rgba(255, 255, 255, 0.24),
    inset 0 -8px 16px rgba(0, 0, 0, 0.35),
    0 10px 20px rgba(0, 0, 0, 0.48),
    0 0 18px rgba(255, 107, 107, 0.42);
}

.ball-sm {
  width: 32px;
  height: 32px;
  font-size: 13px;
}

.ball-md {
  width: 48px;
  height: 48px;
  font-size: 18px;
}

.ball-lg {
  width: 60px;
  height: 60px;
  font-size: 22px;
}

.ball-highlight {
  position: absolute;
  top: 9%;
  left: 16%;
  width: 32%;
  height: 24%;
  border-radius: 50%;
  background: radial-gradient(circle at center, rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0));
  transform: rotate(-18deg);
  pointer-events: none;
}

.ball-number {
  position: relative;
  z-index: 1;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.35);
}

.ball-animate {
  animation: dropIn 0.55s ease both;
}

@keyframes dropIn {
  from {
    opacity: 0;
    transform: translateY(-16px) scale(0.9);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}
</style>
