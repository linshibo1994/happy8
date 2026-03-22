import type { PredictStep } from '@/types/predict'

export const BALL_MIN = 1
export const BALL_MAX = 80
export const BALL_COUNT = 20

// API配置
// 生产环境使用空字符串（相对路径），开发环境使用 localhost:7000
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''
export const API_PREFIX = '/api'
export const WS_BASE_URL = import.meta.env.VITE_WS_URL ?? ''
export const API_TIMEOUT = 30000

// 预测步骤
export const PREDICT_STEPS: PredictStep[] = [
  { key: 'INIT', label: '初始化算法', range: [0, 5] },
  { key: 'FETCH_DATA', label: '加载数据', range: [5, 15] },
  { key: 'PREPROCESS', label: '数据预处理', range: [15, 25] },
  { key: 'FEATURE_ENG', label: '特征工程', range: [25, 35] },
  { key: 'INFERENCE', label: '模型计算', range: [35, 85] },
  { key: 'VALIDATE', label: '结果验证', range: [85, 95] },
  { key: 'DONE', label: '生成预测', range: [95, 100] }
]

// 算法分类
export const ALGORITHM_CATEGORIES: Record<string, string> = {
  intelligent: '智能预测',
  deep_learning: '深度学习',
  machine_learning: '机器学习',
  statistical: '统计分析',
  markov: '马尔可夫'
}

// 球尺寸
export const BALL_SIZES = {
  sm: 32,
  md: 48,
  lg: 60
}
