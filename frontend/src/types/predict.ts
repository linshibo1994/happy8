// 预测方法
export interface PredictMethod {
  id: string
  name: string
  description: string
  category: 'intelligent' | 'deep_learning' | 'machine_learning' | 'statistical' | 'markov'
  supportsGpu?: boolean
  supportsParallel?: boolean
}

// 预测参数
export interface PredictParams {
  method: string
  periods?: number
  count?: number
  duplex?: boolean        // 是否复式预测
  redCount?: number       // 复式红球数量
  blueCount?: number      // 复式蓝球数量
  useGpu?: boolean
  parallel?: boolean
  explain?: boolean
}

// 预测结果
export interface PredictResult {
  id?: number
  method: string
  red_balls: number[]
  blue_ball?: number
  blue_balls?: number[]
  duplex?: boolean
  red_count?: number
  blue_count?: number
  stake_count?: number
  total_cost?: number
  confidence?: number
  tags?: string[]
  explanation?: string
  created_at?: string
}

// 预测进度消息
export interface ProgressMessage {
  type: 'progress' | 'log' | 'result' | 'error' | 'complete'
  step?: string
  percentage: number
  details?: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  timestamp?: string
}

// 预测任务
export interface PredictTask {
  taskId: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  results?: PredictResult[]
  startTime: number
  endTime?: number
}

// 预测步骤定义 (实际数据在 utils/constants.ts 中)
export interface PredictStep {
  key: string
  label: string
  range: [number, number]
}

// 预测响应 (匹配后端 PredictionResponse)
export interface PredictionResponse {
  success: boolean
  data?: PredictResult | PredictResult[] | Record<string, unknown>
  message: string
  confidence?: number
  execution_time?: number
  method?: string
  timestamp: string
}
