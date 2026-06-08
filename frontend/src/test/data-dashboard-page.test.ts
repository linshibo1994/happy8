import { flushPromises, mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'

import DataDashboardPage from '@/pages/DataDashboardPage.vue'
import { router } from '@/router'
import { fetchLotteryHistory, fetchSandboxAnalysis } from '@/api'

vi.mock('@/api', () => ({
  autoSyncLatestLottery: vi.fn().mockResolvedValue({
    updated_count: 0,
    latest_result: null,
    synced_at: '2026-06-08T10:00:00+08:00',
  }),
  fetchLatestLotteryResults: vi.fn().mockResolvedValue({
    results: [],
    total: 0,
  }),
  fetchLotteryHistory: vi.fn().mockResolvedValue({
    results: [
      {
        issue: '20260608001',
        draw_date: '2026-06-08T10:00:00+08:00',
        numbers: [1, 2, 3, 8, 10, 12, 17, 19, 24, 28, 31, 35, 41, 45, 50, 56, 61, 66, 72, 80],
        sum_value: 621,
        odd_count: 10,
        even_count: 10,
        big_count: 8,
        small_count: 12,
      },
    ],
    total: 1,
  }),
  fetchSandboxAnalysis: vi.fn().mockResolvedValue({
    window_size: 100,
    actual_periods: 1,
    total: 1,
    events: [
      {
        issue: '20260608001',
        openedAt: '2026-06-08T10:00:00+08:00',
        numbers: [1, 2, 3, 8, 10, 12, 17, 19, 24, 28, 31, 35, 41, 45, 50, 56, 61, 66, 72, 80],
        event_type: 'consecutive',
        scope: 'global',
        zones: [1],
        groups: [[1, 2, 3]],
        longest_length: 3,
        group_count: 1,
        label: '三连号',
      },
    ],
    intervals: [],
    summary: {
      sample_periods: 1,
      event_level: 3,
      hit_periods: 1,
      hit_rate: 1,
      total_groups: 1,
      avg_gap: null,
      median_gap: null,
      max_gap: null,
      current_missing: 0,
      latest_issue: '20260608001',
      top_zones: [{ zone: 1, count: 1 }],
      baseline_delta: null,
      updated_at: '2026-06-08T10:00:00+08:00',
    },
  }),
}))

describe('数据沙盘页面', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('渲染数据沙盘核心区域并绑定分析接口', async () => {
    router.push('/data')
    await router.isReady()

    const wrapper = mount(DataDashboardPage, {
      global: {
        plugins: [createPinia(), router],
        stubs: {
          ChartPanel: {
            props: ['title'],
            template: '<section class="chart-stub">{{ title }}</section>',
          },
        },
      },
    })

    await flushPromises()

    expect(wrapper.text()).toContain('数据沙盘')
    expect(wrapper.text()).toContain('规则命中')
    expect(wrapper.text()).toContain('联网状态')
    expect(wrapper.text()).toContain('20260608001')
    expect(fetchLotteryHistory).toHaveBeenCalledWith(
      expect.objectContaining({
        page: 1,
        page_size: 20,
      }),
    )
    expect(fetchSandboxAnalysis).toHaveBeenCalledWith(
      expect.objectContaining({
        recent_periods: 100,
        event_type: 'consecutive',
        level: 3,
        scope: 'global',
      }),
    )
  })

  it('切换到八区筛选时提交 zones 参数', async () => {
    router.push('/data')
    await router.isReady()

    const wrapper = mount(DataDashboardPage, {
      global: {
        plugins: [createPinia(), router],
        stubs: {
          ChartPanel: {
            props: ['title'],
            template: '<section class="chart-stub">{{ title }}</section>',
          },
        },
      },
    })

    await flushPromises()

    const scopeButtons = wrapper.findAll('button').filter((button) => button.text() === '八区')
    await scopeButtons[0].trigger('click')
    const zoneButton = wrapper.findAll('button').find((button) => button.text().includes('3区'))
    await zoneButton?.trigger('click')
    await flushPromises()

    expect(fetchSandboxAnalysis).toHaveBeenLastCalledWith(
      expect.objectContaining({
        scope: 'zone',
        zones: [3],
      }),
    )
  })
})
