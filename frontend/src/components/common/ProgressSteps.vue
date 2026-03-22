<template>
  <div class="progress-steps">
    <div class="steps-container">
      <div
        v-for="(step, index) in steps"
        :key="index"
        class="step"
        :class="getStepClass(index)"
      >
        <div class="step-indicator">
          <span v-if="index < currentStep" class="check-icon" v-html="'&#10003;'"></span>
          <span v-else>{{ index + 1 }}</span>
        </div>
        <div class="step-label">{{ step.label }}</div>
      </div>
    </div>

    <div class="progress-bar">
      <div class="progress-fill" :style="{ width: `${progress}%` }">
        <div class="progress-shimmer"></div>
      </div>
    </div>

    <div class="progress-info">
      <span class="current-step">{{ currentStepLabel }}</span>
      <span class="progress-percent">{{ progress }}%</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { PREDICT_STEPS } from '@/utils/constants'

interface Props {
  currentStep?: number
  progress?: number
}

const props = withDefaults(defineProps<Props>(), {
  currentStep: 0,
  progress: 0
})

const steps = PREDICT_STEPS

const currentStepLabel = computed(() => {
  return steps[props.currentStep]?.label || ''
})

const getStepClass = (index: number) => {
  if (index < props.currentStep) return 'step-completed'
  if (index === props.currentStep) return 'step-active'
  return 'step-pending'
}
</script>

<style scoped>
.progress-steps {
  padding: 20px;
}

.steps-container {
  display: flex;
  justify-content: space-between;
  margin-bottom: 20px;
}

.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
}

.step-indicator {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 8px;
  transition: all 0.3s;
}

.step-pending .step-indicator {
  background: var(--bg-secondary);
  border: 2px solid var(--border-color);
  color: var(--text-muted);
}

.step-active .step-indicator {
  background: var(--color-primary);
  border: 2px solid var(--color-primary);
  color: white;
  animation: pulse 1.5s infinite;
}

.step-completed .step-indicator {
  background: var(--color-success);
  border: 2px solid var(--color-success);
  color: white;
}

.check-icon {
  font-size: 16px;
}

.step-label {
  font-size: 12px;
  color: var(--text-secondary);
  text-align: center;
  max-width: 80px;
}

.step-active .step-label {
  color: var(--color-primary);
  font-weight: 500;
}

.progress-bar {
  height: 8px;
  background: rgba(0, 212, 255, 0.2);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #00d4ff, #0099cc);
  border-radius: 4px;
  transition: width 0.3s ease;
  position: relative;
  overflow: hidden;
}

.progress-shimmer {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes pulse {
  0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.4); }
  50% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(0, 212, 255, 0); }
}

.progress-info {
  display: flex;
  justify-content: space-between;
  margin-top: 12px;
  font-size: 14px;
}

.current-step {
  color: var(--text-secondary);
}

.progress-percent {
  color: var(--color-primary);
  font-weight: bold;
}
</style>
