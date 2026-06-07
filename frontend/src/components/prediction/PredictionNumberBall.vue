<script setup lang="ts">
import { computed } from 'vue'

type PredictionNumberBallVariant = 'draw' | 'prediction' | 'hit' | 'muted' | 'intersection'

const props = withDefaults(
  defineProps<{
    value: number
    variant?: PredictionNumberBallVariant
    size?: 'normal' | 'small'
  }>(),
  {
    variant: 'prediction',
    size: 'normal',
  },
)

const displayValue = computed(() => String(props.value).padStart(2, '0'))
const variantLabelMap: Record<PredictionNumberBallVariant, string> = {
  draw: '开奖号码',
  prediction: '预测号码',
  hit: '命中号码',
  muted: '未命中号码',
  intersection: '交集号码',
}

const accessibleLabel = computed(() => `${variantLabelMap[props.variant]} ${displayValue.value}`)
</script>

<template>
  <span
    class="prediction-number-ball"
    :class="[`prediction-number-ball--${variant}`, `prediction-number-ball--${size}`]"
    role="img"
    :aria-label="accessibleLabel"
  >
    {{ displayValue }}
  </span>
</template>

<style scoped>
.prediction-number-ball {
  display: inline-grid;
  width: 36px;
  height: 36px;
  place-items: center;
  flex: 0 0 auto;
  border: 1px solid transparent;
  border-radius: 50%;
  font-family: var(--h8-font-number);
  font-size: 14px;
  font-weight: 700;
  line-height: 1;
  transition:
    border-color 180ms ease,
    background 180ms ease,
    color 180ms ease,
    transform 180ms ease;
}

.prediction-number-ball--small {
  width: 24px;
  height: 24px;
  font-size: 11px;
}

.prediction-number-ball--draw,
.prediction-number-ball--hit {
  background: var(--h8-color-cinnabar);
  color: #fff;
}

.prediction-number-ball--prediction {
  border-color: color-mix(in srgb, var(--h8-color-cinnabar) 62%, var(--h8-color-line));
  background: var(--h8-color-jade);
  color: var(--h8-color-cinnabar);
}

.prediction-number-ball--hit {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--h8-color-bronze) 42%, transparent);
}

.prediction-number-ball--muted {
  border-color: var(--h8-color-line);
  background: color-mix(in srgb, var(--h8-color-line) 34%, var(--h8-color-surface-strong));
  color: var(--h8-color-text-muted);
}

.prediction-number-ball--intersection {
  border-color: var(--h8-color-data-blue);
  background:
    radial-gradient(circle at center, color-mix(in srgb, var(--h8-color-data-blue) 22%, transparent) 1px, transparent 1px),
    var(--h8-color-surface-strong);
  background-size: 6px 6px;
  color: var(--h8-color-data-blue);
}

@media (max-width: 760px) {
  .prediction-number-ball {
    width: 32px;
    height: 32px;
    font-size: 13px;
  }

  .prediction-number-ball--small {
    width: 24px;
    height: 24px;
    font-size: 11px;
  }
}
</style>
