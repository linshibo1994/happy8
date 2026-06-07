<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { darkTheme, lightTheme, NConfigProvider, NDialogProvider, NMessageProvider } from 'naive-ui'

import AppShell from '@/components/layout/AppShell.vue'
import { useLotteryStore } from '@/stores/lottery'
import { useUiStore } from '@/stores/ui'

const lotteryStore = useLotteryStore()
const uiStore = useUiStore()

const naiveTheme = computed(() => (uiStore.themeMode === 'dark' ? darkTheme : lightTheme))

onMounted(() => {
  void lotteryStore.refreshLatestResults()
})
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
