<template>
  <div class="datav-border" :class="`border-type-${type}`">
    <div class="border-corner corner-tl"></div>
    <div class="border-corner corner-tr"></div>
    <div class="border-corner corner-bl"></div>
    <div class="border-corner corner-br"></div>
    <div class="border-content">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  type?: number
}

withDefaults(defineProps<Props>(), {
  type: 1
})
</script>

<style scoped>
.datav-border {
  position: relative;
  background: var(--gradient-card);
  border: 1px solid var(--border-color);
  padding: 2px;
}

.border-corner {
  position: absolute;
  width: 20px;
  height: 20px;
  border: 2px solid var(--color-primary);
}

.corner-tl {
  top: -2px;
  left: -2px;
  border-right: none;
  border-bottom: none;
}

.corner-tr {
  top: -2px;
  right: -2px;
  border-left: none;
  border-bottom: none;
}

.corner-bl {
  bottom: -2px;
  left: -2px;
  border-right: none;
  border-top: none;
}

.corner-br {
  bottom: -2px;
  right: -2px;
  border-left: none;
  border-top: none;
}

.border-content {
  position: relative;
  z-index: 1;
}

.border-type-1 {
  border-radius: 0;
}

.border-type-2 {
  border-radius: 8px;
}

.border-type-2 .border-corner {
  border-radius: 4px;
}

.border-type-3 {
  background: linear-gradient(135deg, var(--bg-card), var(--bg-secondary));
}

.datav-border::before {
  content: '';
  position: absolute;
  inset: 0;
  border: 1px solid transparent;
  border-image: linear-gradient(135deg, var(--color-primary), transparent) 1;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.3s;
}

.datav-border:hover::before {
  opacity: 1;
}
</style>
