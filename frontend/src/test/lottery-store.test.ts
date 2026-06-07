import { describe, expect, it } from 'vitest'

import { mapLotteryResult } from '@/stores/lottery'

describe('开奖数据 Store', () => {
  it('将后端开奖结果字段映射为前端展示模型', () => {
    const result = mapLotteryResult({
      issue: '2026001',
      draw_date: '2026-06-07T21:30:00',
      numbers: Array.from({ length: 20 }, (_, index) => index + 1),
      sum_value: 210,
      odd_count: 10,
      even_count: 10,
      big_count: 0,
      small_count: 20,
      zone_distribution: {
        zone_1: 20,
        zone_2: 0,
        zone_3: 0,
        zone_4: 0,
      },
    })

    expect(result.issue).toBe('2026001')
    expect(result.openedAt).toBe('2026-06-07T21:30:00')
    expect(result.sum).toBe(210)
    expect(result.oddCount).toBe(10)
    expect(result.zoneDistribution).toEqual({
      '1-20': 20,
      '21-40': 0,
      '41-60': 0,
      '61-80': 0,
    })
  })

  it('缺少区间分布时可根据号码自动计算', () => {
    const result = mapLotteryResult({
      issue: '2026002',
      numbers: [1, 5, 20, 21, 30, 40, 41, 50, 60, 61, 70, 80],
    })

    expect(result.zoneDistribution).toEqual({
      '1-20': 3,
      '21-40': 3,
      '41-60': 3,
      '61-80': 3,
    })
  })
})
