import { ref, computed } from 'vue'
import { PREDICT_STEPS } from '@/utils/constants'

export function useProgress() {
  const currentStep = ref(0)
  const progress = ref(0)

  // 计算当前步骤信息
  const currentStepInfo = computed(() => PREDICT_STEPS[currentStep.value])

  // 计算步骤状态
  const getStepStatus = (index: number): 'pending' | 'active' | 'completed' => {
    if (index < currentStep.value) return 'completed'
    if (index === currentStep.value) return 'active'
    return 'pending'
  }

  // 根据进度更新步骤
  const updateFromProgress = (progressValue: number) => {
    progress.value = progressValue

    // 找到对应的步骤
    for (let i = PREDICT_STEPS.length - 1; i >= 0; i--) {
      const [start] = PREDICT_STEPS[i].range
      if (progressValue >= start) {
        currentStep.value = i
        break
      }
    }
  }

  // 设置步骤
  const setStep = (step: number) => {
    currentStep.value = Math.max(0, Math.min(step, PREDICT_STEPS.length - 1))
    const stepInfo = PREDICT_STEPS[currentStep.value]
    progress.value = stepInfo.range[0]
  }

  // 下一步
  const nextStep = () => {
    if (currentStep.value < PREDICT_STEPS.length - 1) {
      setStep(currentStep.value + 1)
    }
  }

  // 重置
  const reset = () => {
    currentStep.value = 0
    progress.value = 0
  }

  return {
    currentStep,
    progress,
    currentStepInfo,
    getStepStatus,
    updateFromProgress,
    setStep,
    nextStep,
    reset
  }
}
