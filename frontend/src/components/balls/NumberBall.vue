<script setup lang="ts">
import { computed } from 'vue'

type NumberBallVariant =
  | 'draw'
  | 'prediction'
  | 'hit'
  | 'miss'
  | 'intersection'
  | 'neutral'
  | 'outline'
  | 'selected'
  | 'muted'
type NumberBallSize = 'normal' | 'small' | 'table' | 'tiny'

const props = withDefaults(
  defineProps<{
    value: number | string
    variant?: NumberBallVariant
    size?: NumberBallSize
    ariaLabel?: string
  }>(),
  {
    variant: 'draw',
    size: 'normal',
    ariaLabel: undefined,
  },
)

const displayValue = computed(() => {
  if (typeof props.value === 'number') {
    return String(props.value).padStart(2, '0')
  }

  return props.value
})

const variantLabelMap: Record<NumberBallVariant, string> = {
  draw: '开奖号码',
  prediction: '预测号码',
  hit: '命中号码',
  miss: '未命中号码',
  intersection: '交集号码',
  neutral: '号码',
  outline: '候选号码',
  selected: '重点号码',
  muted: '低频号码',
}

const accessibleLabel = computed(() => props.ariaLabel ?? `${variantLabelMap[props.variant]} ${displayValue.value}`)
</script>

<template>
  <span
    class="h8-number-ball"
    :class="[`h8-number-ball--${props.variant}`, `h8-number-ball--${props.size}`]"
    role="img"
    :aria-label="accessibleLabel"
  >
    {{ displayValue }}
  </span>
</template>

<style scoped>
.h8-number-ball {
  --h8-ball-size: 36px;

  display: inline-grid;
  flex: 0 0 var(--h8-ball-size);
  width: var(--h8-ball-size);
  height: var(--h8-ball-size);
  place-items: center;
  border: 1px solid transparent;
  border-radius: 50%;
  font-family: var(--h8-font-number);
  font-size: 15px;
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0;
  white-space: nowrap;
}

.h8-number-ball--small {
  --h8-ball-size: 32px;

  font-size: 13px;
}

.h8-number-ball--table {
  --h8-ball-size: 24px;

  font-size: 11px;
}

.h8-number-ball--tiny {
  --h8-ball-size: 24px;

  font-size: 11px;
}

.h8-number-ball--draw {
  background: var(--h8-color-cinnabar);
  color: #fff;
}

.h8-number-ball--prediction {
  border-color: color-mix(in srgb, var(--h8-color-cinnabar) 72%, var(--h8-color-line));
  background: color-mix(in srgb, var(--h8-color-surface-strong) 92%, var(--h8-color-cinnabar));
  color: var(--h8-color-cinnabar);
}

.h8-number-ball--outline {
  border-color: color-mix(in srgb, var(--h8-color-cinnabar) 72%, var(--h8-color-line));
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-cinnabar);
}

.h8-number-ball--hit {
  border-color: var(--h8-color-bronze);
  background: var(--h8-color-cinnabar);
  color: #fff;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--h8-color-bronze) 32%, transparent);
  animation: hit-pulse 760ms ease-out 1;
}

.h8-number-ball--miss {
  border-color: var(--h8-color-line);
  background: color-mix(in srgb, var(--h8-color-surface) 88%, var(--h8-color-gray));
  color: var(--h8-color-text-muted);
}

.h8-number-ball--muted {
  border-color: var(--h8-color-line);
  background: color-mix(in srgb, var(--h8-color-surface) 88%, var(--h8-color-gray));
  color: var(--h8-color-text-muted);
}

.h8-number-ball--intersection {
  border-color: var(--h8-color-data-blue);
  background:
    radial-gradient(circle at 50% 50%, color-mix(in srgb, var(--h8-color-data-blue) 16%, transparent) 1px, transparent 1px),
    var(--h8-color-surface-strong);
  background-size: 6px 6px;
  color: var(--h8-color-data-blue);
}

.h8-number-ball--neutral {
  border-color: var(--h8-color-line);
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-text);
}

.h8-number-ball--selected {
  border-color: var(--h8-color-risk-orange);
  background: color-mix(in srgb, var(--h8-color-risk-orange) 12%, var(--h8-color-surface-strong));
  color: var(--h8-color-risk-orange);
  box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--h8-color-risk-orange) 20%, transparent);
}

@keyframes hit-pulse {
  0% {
    transform: scale(0.94);
  }

  45% {
    transform: scale(1.06);
  }

  100% {
    transform: scale(1);
  }
}

@media (max-width: 760px) {
  .h8-number-ball--normal {
    --h8-ball-size: 32px;

    font-size: 13px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .h8-number-ball--hit {
    animation: none;
  }
}
</style>
