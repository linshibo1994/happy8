<template>
  <div class="stat-card">
    <DataVBorder :type="2">
      <div class="card-content">
        <div class="card-header">
          <span class="card-icon" :style="{ color: iconColor }" v-html="icon"></span>
          <span class="card-trend" :class="trendClass" v-if="trend !== 0">
            {{ trend > 0 ? '+' : '' }}{{ trend }}%
          </span>
        </div>
        <div class="card-value">{{ displayValue }}</div>
        <div class="card-title">{{ title }}</div>
      </div>
    </DataVBorder>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import DataVBorder from './DataVBorder.vue'

interface Props {
  title: string
  value: number | string
  icon?: string
  trend?: number
  color?: string
  animate?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  icon: '&#128202;',
  trend: 0,
  color: 'var(--color-primary)',
  animate: true
})

const displayValue = ref<number | string>(0)

const iconColor = computed(() => props.color)
const trendClass = computed(() => (props.trend > 0 ? 'trend-up' : 'trend-down'))

const animateValue = (target: number) => {
  if (!props.animate || typeof props.value !== 'number') {
    displayValue.value = props.value
    return
  }

  const duration = 800
  const startTime = performance.now()

  const update = (now: number) => {
    const progress = Math.min((now - startTime) / duration, 1)
    const eased = 1 - Math.pow(1 - progress, 4)
    displayValue.value = Math.round(target * eased)
    if (progress < 1) {
      requestAnimationFrame(update)
    }
  }

  requestAnimationFrame(update)
}

watch(
  () => props.value,
  (newVal) => {
    if (typeof newVal === 'number') {
      animateValue(newVal)
    } else {
      displayValue.value = newVal
    }
  }
)

onMounted(() => {
  if (typeof props.value === 'number') {
    animateValue(props.value)
  } else {
    displayValue.value = props.value
  }
})
</script>

<style scoped>
.stat-card {
  transition: transform 0.3s, box-shadow 0.3s;
}

.stat-card:hover {
  transform: translateY(-4px);
}

.card-content {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.card-icon {
  font-size: 24px;
}

.card-trend {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 10px;
}

.trend-up {
  background: rgba(16, 185, 129, 0.2);
  color: var(--color-success);
}

.trend-down {
  background: rgba(239, 68, 68, 0.2);
  color: var(--color-error);
}

.card-value {
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.card-title {
  font-size: 14px;
  color: var(--text-secondary);
}
</style>
