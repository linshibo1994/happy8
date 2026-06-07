<script setup lang="ts">
import { computed } from 'vue'
import { darkTheme, lightTheme, NConfigProvider, NDialogProvider, NMessageProvider } from 'naive-ui'

import AppShell from '@/components/layout/AppShell.vue'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()

const naiveTheme = computed(() => (uiStore.themeMode === 'dark' ? darkTheme : lightTheme))
</script>

<template>
  <NConfigProvider :theme="naiveTheme" abstract>
    <NMessageProvider>
      <NDialogProvider>
        <AppShell />
      </NDialogProvider>
    </NMessageProvider>
  </NConfigProvider>
</template>
