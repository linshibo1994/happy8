<script setup lang="ts">
import NumberBall from '@/components/balls/NumberBall.vue'

export interface PredictionRecordView {
  id: string
  createdAt: string
  targetIssue: string
  algorithmName: string
  algorithmLabel: string
  predictedNumbers: number[]
  actualNumbers: number[]
  hitNumbers: number[]
  hitCount: number
  hitRate: number
  confidence: number
  elapsedMs: number
  cached: boolean
}

defineProps<{
  record: PredictionRecordView
  selected: boolean
}>()

defineEmits<{
  select: [id: string]
}>()

const formatPercent = (value: number) => `${Math.round(value * 100)}%`
</script>

<template>
  <tr class="prediction-row" :class="{ 'prediction-row--selected': selected }">
    <td>
      <button class="prediction-row__issue" type="button" @click="$emit('select', record.id)">
        <strong>{{ record.targetIssue }}</strong>
        <small>{{ record.createdAt }}</small>
      </button>
    </td>
    <td>
      <span class="prediction-row__algorithm">{{ record.algorithmLabel }}</span>
    </td>
    <td>
      <div class="prediction-row__balls" aria-label="预测号码">
        <NumberBall
          v-for="number in record.predictedNumbers"
          :key="number"
          :value="number"
          :variant="record.hitNumbers.includes(number) ? 'hit' : 'prediction'"
          size="table"
        />
      </div>
    </td>
    <td>
      <div class="prediction-row__balls" aria-label="实际开奖号码">
        <NumberBall
          v-for="number in record.actualNumbers.slice(0, 10)"
          :key="number"
          :value="number"
          :variant="record.hitNumbers.includes(number) ? 'hit' : 'miss'"
          size="table"
        />
      </div>
    </td>
    <td>
      <strong class="prediction-row__hit">{{ record.hitCount }}</strong>
    </td>
    <td>
      <strong class="prediction-row__rate">{{ formatPercent(record.hitRate) }}</strong>
    </td>
    <td>{{ formatPercent(record.confidence) }}</td>
    <td>{{ record.elapsedMs }}ms</td>
    <td>
      <span class="prediction-row__cache" :class="{ 'prediction-row__cache--yes': record.cached }">
        {{ record.cached ? '缓存' : '实时' }}
      </span>
    </td>
  </tr>
</template>

<style scoped lang="scss">
.prediction-row {
  transition: background 160ms ease;
}

.prediction-row:hover,
.prediction-row--selected {
  background: color-mix(in srgb, var(--h8-color-cinnabar) 7%, transparent);
}

.prediction-row td {
  border-bottom: 1px solid var(--h8-color-line);
  padding: 12px 10px;
  vertical-align: middle;
}

.prediction-row__issue {
  display: grid;
  gap: 3px;
  border: 0;
  background: transparent;
  color: var(--h8-color-text);
  padding: 0;
  text-align: left;
  cursor: pointer;
}

.prediction-row__issue strong,
.prediction-row__algorithm,
.prediction-row__hit,
.prediction-row__rate {
  font-family: var(--h8-font-number);
}

.prediction-row__issue small {
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.prediction-row__balls {
  display: flex;
  min-width: 150px;
  flex-wrap: wrap;
  gap: 4px;
}

.prediction-row__algorithm {
  display: inline-flex;
  min-height: 28px;
  align-items: center;
  border-radius: 999px;
  background: color-mix(in srgb, var(--h8-color-data-blue) 10%, transparent);
  color: var(--h8-color-data-blue);
  padding: 4px 8px;
  white-space: nowrap;
}

.prediction-row__hit {
  color: var(--h8-color-cinnabar);
}

.prediction-row__rate {
  color: var(--h8-color-text);
}

.prediction-row__cache {
  display: inline-flex;
  min-height: 26px;
  align-items: center;
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  color: var(--h8-color-text-muted);
  padding: 3px 8px;
  font-size: 12px;
  white-space: nowrap;
}

.prediction-row__cache--yes {
  border-color: color-mix(in srgb, var(--h8-color-bronze) 50%, var(--h8-color-line));
  color: var(--h8-color-bronze);
}
</style>
