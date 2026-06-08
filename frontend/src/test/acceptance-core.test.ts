import { mount, type DOMWrapper, type VueWrapper } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { defineComponent, nextTick, type Component } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { appRoutes } from '@/router'
import DashboardPage from '@/pages/DashboardPage.vue'
import DataDashboardPage from '@/pages/DataDashboardPage.vue'
import PredictionHistoryPage from '@/pages/PredictionHistoryPage.vue'
import PredictionPage from '@/pages/PredictionPage.vue'

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
    results: [],
    total: 0,
  }),
  fetchSandboxAnalysis: vi.fn().mockResolvedValue({
    window_size: 100,
    actual_periods: 100,
    total: 0,
    events: [],
    intervals: [],
    summary: {
      sample_periods: 100,
      event_level: 3,
      hit_periods: 0,
      hit_rate: 0,
      total_groups: 0,
      avg_gap: null,
      median_gap: null,
      max_gap: null,
      current_missing: null,
      latest_issue: null,
      top_zones: [],
      baseline_delta: null,
      updated_at: '2026-06-08T10:00:00+08:00',
    },
  }),
}))

const ChartPanelStub = defineComponent({
  name: 'ChartPanel',
  props: {
    title: { type: String, required: true },
    subtitle: { type: String, default: '' },
    summary: { type: String, required: true },
    loading: { type: Boolean, default: false },
    empty: { type: Boolean, default: false },
    error: { type: String, default: '' },
  },
  template: `
    <article class="chart-panel-stub">
      <h3>{{ title }}</h3>
      <p v-if="subtitle">{{ subtitle }}</p>
      <p>{{ summary }}</p>
      <span v-if="loading">正在读取历史开奖数据</span>
      <span v-else-if="error">{{ error }}</span>
      <span v-else-if="empty">当前筛选范围暂无开奖记录</span>
    </article>
  `,
})

afterEach(() => {
  vi.useRealTimers()
  document.body.innerHTML = ''
})

async function mountWithContext(
  component: Component,
  route = '/',
  stubs: Record<string, Component | boolean> = {},
) {
  const pinia = createPinia()
  const router = createRouter({
    history: createMemoryHistory(),
    routes: appRoutes,
    scrollBehavior: () => ({ top: 0 }),
  })

  await router.push(route)
  await router.isReady()

  const wrapper = mount(component, {
    attachTo: document.body,
    global: {
      plugins: [pinia, router],
      stubs,
    },
  })

  return { wrapper, router, pinia }
}

function getButton(wrapper: VueWrapper, label: string) {
  const button = wrapper.findAll('button').find((item) => item.text().trim() === label)

  expect(button, `未找到按钮：${label}`).toBeTruthy()

  return button as DOMWrapper<HTMLButtonElement>
}

async function flushDom() {
  await nextTick()
  await Promise.resolve()
}

describe('前端重构核心验收路径', () => {
  it('首页加载并展示最新开奖和下期快捷入口', async () => {
    const { wrapper } = await mountWithContext(DashboardPage)

    expect(wrapper.text()).toContain('开奖、预测与复盘总览')
    expect(wrapper.text()).toContain('最新开奖')
    expect(wrapper.text()).toContain('第 20260607001 期')
    expect(wrapper.get('[aria-label="最新开奖号码"]').text()).toContain('01')
    expect(wrapper.get('[aria-label="最新开奖号码"]').text()).toContain('79')
    expect(wrapper.text()).toContain('下期期号')
    expect(wrapper.text()).toContain('20260607002')
    expect(getButton(wrapper, '一键预测').attributes('disabled')).toBeUndefined()
  })

  it('数据沙盘可加载并按期数、口径筛选', async () => {
    const { wrapper } = await mountWithContext(DataDashboardPage, '/data', {
      ChartPanel: ChartPanelStub,
    })
    await flushDom()

    expect(wrapper.text()).toContain('数据沙盘')
    expect(wrapper.text()).toContain('开奖期数')
    expect(wrapper.text()).toContain('窗口 100 期')
    expect(wrapper.text()).toContain('八区命中分布')
    expect(wrapper.text()).toContain('规则命中')

    await getButton(wrapper, '30').trigger('click')
    await flushDom()

    expect(wrapper.text()).toContain('窗口 30 期')

    await getButton(wrapper, '八区').trigger('click')
    await flushDom()

    expect(wrapper.text()).toContain('八区')
  })

  it('单算法预测入口能进入执行中并展示结果状态', async () => {
    vi.useFakeTimers()

    const { wrapper } = await mountWithContext(PredictionPage, '/prediction')

    expect(wrapper.text()).toContain('算法预测')
    expect(wrapper.text()).toContain('单算法')
    expect(wrapper.text()).toContain('频率分析')

    await getButton(wrapper, '开始预测').trigger('click')
    await flushDom()

    expect(wrapper.text()).toContain('预测中')
    expect(getButton(wrapper, '预测中').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('等待权限校验')

    await vi.advanceTimersByTimeAsync(2_500)
    await flushDom()

    expect(wrapper.text()).toContain('执行完成')
    expect(wrapper.text()).toContain('结果号码')
    expect(wrapper.text()).toContain('100%')
    expect(wrapper.text()).toContain('置信度')
    expect(wrapper.text()).toContain('未命中缓存')
  })

  it('批量预测入口逐个展示状态并生成汇总', async () => {
    vi.useFakeTimers()

    const { wrapper } = await mountWithContext(PredictionPage, '/prediction')

    await getButton(wrapper, '批量预测').trigger('click')
    await flushDom()

    expect(wrapper.text()).toContain('批量模式逐个算法落位')
    expect(wrapper.text()).toContain('频率分析')
    expect(wrapper.text()).toContain('冷热分析')
    expect(wrapper.text()).toContain('遗漏分析')

    await getButton(wrapper, '开始批量预测').trigger('click')
    await flushDom()

    expect(wrapper.text()).toContain('批量执行中')
    expect(wrapper.text()).toContain('总进度')

    await vi.advanceTimersByTimeAsync(3_000)
    await flushDom()

    expect(wrapper.text()).toContain('执行完成')
    expect(wrapper.text()).toContain('多算法共识')
    expect(wrapper.text()).toContain('成功算法')
    expect(wrapper.text()).toContain('3 / 3')
    expect(wrapper.text()).toContain('平均置信度')
  })

  it('预测历史复盘保留并显示 0% 命中率', async () => {
    const { wrapper } = await mountWithContext(PredictionHistoryPage, '/history')

    expect(wrapper.text()).toContain('预测历史与命中复盘')
    expect(wrapper.text()).toContain('0% 复盘')

    await wrapper.get('select').setValue('markov')
    await flushDom()

    expect(wrapper.text()).toContain('20260606002')
    expect(wrapper.text()).toContain('平均命中率0%')
    expect(wrapper.text()).toContain('命中 0 个，命中率 0%')
    expect(wrapper.text()).toContain('02')
    expect(wrapper.text()).toContain('80')
  })
})
