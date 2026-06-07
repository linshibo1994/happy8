<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { Bell, Moon, Sun } from 'lucide-vue-next'

import { appRoutes } from '@/router'
import { useLotteryStore } from '@/stores/lottery'
import { useMembershipStore } from '@/stores/membership'
import { useUiStore } from '@/stores/ui'
import { useUserStore } from '@/stores/user'

const route = useRoute()
const lotteryStore = useLotteryStore()
const membershipStore = useMembershipStore()
const uiStore = useUiStore()
const userStore = useUserStore()

const activeRouteTitle = computed(() => route.meta.title?.toString() ?? '工作台')
const activeRouteSubtitle = computed(() => route.meta.subtitle?.toString() ?? 'Happy8 数据指挥台')
const lastUpdatedText = computed(() => {
  if (!lotteryStore.lastUpdatedAt) {
    return '待同步'
  }

  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(lotteryStore.lastUpdatedAt))
})
</script>

<template>
  <div class="app-shell" :data-theme="uiStore.themeMode">
    <aside class="app-shell__sidebar" aria-label="桌面主导航">
      <RouterLink class="brand" to="/" aria-label="返回工作台">
        <span class="brand__mark">8</span>
        <span>
          <strong>Happy8</strong>
          <small>数据指挥台</small>
        </span>
      </RouterLink>

      <nav class="side-nav">
        <RouterLink
          v-for="item in appRoutes"
          :key="item.name"
          class="side-nav__item"
          :to="item.path"
          :aria-current="route.name === item.name ? 'page' : undefined"
        >
          <component :is="item.meta.icon" :size="19" aria-hidden="true" />
          <span>{{ item.meta.navigationLabel }}</span>
        </RouterLink>
      </nav>
    </aside>

    <div class="app-shell__main">
      <header class="top-status">
        <div class="top-status__title">
          <span class="top-status__eyebrow">Happy8</span>
          <h1>{{ activeRouteTitle }}</h1>
          <p>{{ activeRouteSubtitle }}</p>
        </div>

        <div class="top-status__meta" aria-label="账户和数据状态">
          <span class="status-pill status-pill--user">{{ userStore.profile.nickname }}</span>
          <span class="status-pill status-pill--member">{{ membershipStore.status.levelName }}</span>
          <span class="status-pill">剩余 {{ membershipStore.status.remainingPredictions }} 次</span>
          <span class="status-pill">数据 {{ lastUpdatedText }}</span>
          <button class="icon-button" type="button" aria-label="通知">
            <Bell :size="18" aria-hidden="true" />
          </button>
          <button
            class="icon-button"
            type="button"
            :aria-label="uiStore.themeMode === 'dark' ? '切换浅色主题' : '切换深色主题'"
            @click="uiStore.toggleTheme()"
          >
            <Moon v-if="uiStore.themeMode === 'light'" :size="18" aria-hidden="true" />
            <Sun v-else :size="18" aria-hidden="true" />
          </button>
        </div>
      </header>

      <main class="app-shell__content">
        <RouterView />
      </main>
    </div>

    <nav class="bottom-nav" aria-label="移动主导航">
      <RouterLink
        v-for="item in appRoutes"
        :key="item.name"
        class="bottom-nav__item"
        :to="item.path"
        :aria-current="route.name === item.name ? 'page' : undefined"
      >
        <component :is="item.meta.icon" :size="20" aria-hidden="true" />
        <span>{{ item.meta.navigationLabel }}</span>
      </RouterLink>
    </nav>
  </div>
</template>
