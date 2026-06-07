<script setup lang="ts">
type DashboardPanelTone = 'default' | 'accent' | 'blue' | 'success' | 'warning'

const props = withDefaults(
  defineProps<{
    title: string
    kicker?: string
    description?: string
    tone?: DashboardPanelTone
  }>(),
  {
    kicker: undefined,
    description: undefined,
    tone: 'default',
  },
)
</script>

<template>
  <article class="dashboard-panel" :class="`dashboard-panel--${props.tone}`">
    <header class="dashboard-panel__header">
      <div class="dashboard-panel__title-group">
        <span v-if="props.kicker" class="dashboard-panel__kicker">{{ props.kicker }}</span>
        <div class="dashboard-panel__title-line">
          <span class="dashboard-panel__icon" aria-hidden="true">
            <slot name="icon" />
          </span>
          <h2>{{ props.title }}</h2>
        </div>
        <p v-if="props.description">{{ props.description }}</p>
      </div>

      <div v-if="$slots.actions" class="dashboard-panel__actions">
        <slot name="actions" />
      </div>
    </header>

    <div class="dashboard-panel__body">
      <slot />
    </div>
  </article>
</template>

<style scoped>
.dashboard-panel {
  min-width: 0;
  height: 100%;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: color-mix(in srgb, var(--h8-color-surface-strong) 96%, var(--h8-color-jade));
  box-shadow: 0 14px 34px var(--h8-color-shadow);
  padding: 18px;
}

.dashboard-panel--accent {
  border-color: color-mix(in srgb, var(--h8-color-cinnabar) 35%, var(--h8-color-line));
}

.dashboard-panel--blue {
  border-color: color-mix(in srgb, var(--h8-color-data-blue) 34%, var(--h8-color-line));
}

.dashboard-panel--success {
  border-color: color-mix(in srgb, var(--h8-color-turquoise) 34%, var(--h8-color-line));
}

.dashboard-panel--warning {
  border-color: color-mix(in srgb, var(--h8-color-risk-orange) 38%, var(--h8-color-line));
}

.dashboard-panel__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.dashboard-panel__title-group {
  min-width: 0;
}

.dashboard-panel__kicker {
  display: block;
  margin-bottom: 6px;
  color: var(--h8-color-cinnabar);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
}

.dashboard-panel__title-line {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.dashboard-panel__icon {
  display: inline-grid;
  flex: 0 0 28px;
  width: 28px;
  height: 28px;
  place-items: center;
  border-radius: 50%;
  background: color-mix(in srgb, var(--h8-color-cinnabar) 10%, transparent);
  color: var(--h8-color-cinnabar);
}

.dashboard-panel__title-line h2 {
  overflow-wrap: anywhere;
  margin: 0;
  color: var(--h8-color-text);
  font-size: 17px;
  line-height: 1.25;
  letter-spacing: 0;
}

.dashboard-panel__title-group p {
  margin: 6px 0 0;
  color: var(--h8-color-text-muted);
  font-size: 13px;
  line-height: 1.5;
}

.dashboard-panel__actions {
  flex: 0 0 auto;
}

.dashboard-panel__body {
  margin-top: 16px;
}
</style>
