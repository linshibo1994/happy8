import { flushPromises, mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { describe, expect, it, vi } from 'vitest'

import App from '@/App.vue'
import { appRoutes, router } from '@/router'
import { autoSyncLatestLottery, fetchLatestLotteryResults } from '@/api'

vi.mock('@/api', () => ({
  autoSyncLatestLottery: vi.fn().mockResolvedValue({
    updated_count: 0,
    latest_result: null,
    synced_at: '2026-06-07T10:00:00+08:00',
    skipped: true,
    reason: '测试环境跳过',
  }),
  fetchLatestLotteryResults: vi.fn().mockResolvedValue({
    results: [],
    total: 0,
  }),
}))

describe('Happy8 前端骨架', () => {
  it('注册所有主导航路由', () => {
    expect(appRoutes.map((route) => route.path)).toEqual([
      '/',
      '/prediction',
      '/data',
      '/algorithms',
      '/history',
      '/membership',
      '/profile',
      '/system',
    ])
  })

  it('渲染 AppShell 和顶部状态栏', async () => {
    router.push('/')
    await router.isReady()

    const wrapper = mount(App, {
      global: {
        plugins: [createPinia(), router],
      },
    })

    expect(wrapper.text()).toContain('Happy8')
    expect(wrapper.text()).toContain('工作台')
    expect(wrapper.text()).toContain('剩余 18 次')
    await flushPromises()
    expect(autoSyncLatestLottery).toHaveBeenCalledTimes(1)
    expect(fetchLatestLotteryResults).toHaveBeenCalledWith(20)
  })
})
