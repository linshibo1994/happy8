/**
 * API相关常量
 */
const resolveBaseUrl = () => {
  const envBase = (import.meta.env.VITE_API_BASE_URL || '').trim()
  if (envBase) {
    return envBase.replace(/\/+$/, '')
  }

  // H5 生产环境默认走同域，避免硬编码到不可达域名导致全站请求失败
  // #ifdef H5
  if (import.meta.env.PROD && typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin.replace(/\/+$/, '')
  }
  // #endif

  const fallback = 'http://localhost:8000'
  const base = fallback
  return base.replace(/\/+$/, '')
}

export const API_CONFIG = {
  // 根据环境设置API基础URL
  BASE_URL: resolveBaseUrl(),

  // API版本
  VERSION: import.meta.env.VITE_API_VERSION || 'v1',

  // 请求超时时间（毫秒）
  TIMEOUT: Number(import.meta.env.VITE_API_TIMEOUT) || 10000,

  // API前缀
  get API_PREFIX() {
    return `/api/${this.VERSION}`
  },

  // 完整的API根路径
  get FULL_BASE_URL() {
    return `${this.BASE_URL}${this.API_PREFIX}`
  },

  // 重试次数
  MAX_RETRY: 3,

  // 缓存过期时间（毫秒）
  CACHE_DURATION: 5 * 60 * 1000, // 5分钟
}

/**
 * 存储键名
 */
export const STORAGE_KEYS = {
  // 用户相关
  ACCESS_TOKEN: 'access_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_INFO: 'user_info',

  // 应用配置
  APP_CONFIG: 'app_config',
  SYSTEM_INFO: 'system_info',

  // 预测相关
  PREDICTION_CACHE: 'prediction_cache',
  ALGORITHM_CONFIGS: 'algorithm_configs',

  // 主题和设置
  THEME_MODE: 'theme_mode',
  LANGUAGE: 'language',
  NOTIFICATION_SETTINGS: 'notification_settings',
}

/**
 * 会员等级相关常量
 */
export const MEMBERSHIP = {
  LEVELS: {
    FREE: 'free',
    VIP: 'vip',
    PREMIUM: 'premium',
  },

  LEVEL_NAMES: {
    free: '免费用户',
    vip: 'VIP会员',
    premium: '高级会员',
  },

  LEVEL_COLORS: {
    free: '#999999',
    vip: '#ffd700',
    premium: '#9c27b0',
  },

  DAILY_LIMITS: {
    free: 5,
    vip: 50,
    premium: -1, // 无限制
  },

  FEATURES: {
    free: ['基础算法', '每日5次预测', '基础历史记录'],
    vip: ['所有基础功能', '中级算法', '每日50次预测', '详细统计分析', '客服支持'],
    premium: ['所有VIP功能', '高级算法', '无限预测', 'AI深度学习', '专属算法', '优先支持'],
  },
}

/**
 * 算法相关常量
 */
export const ALGORITHMS = {
  CATEGORIES: {
    BASIC: '基础算法',
    INTERMEDIATE: '中级算法',
    ADVANCED: '高级算法',
    AI: 'AI算法',
  },

  COMPLEXITY_LEVELS: {
    low: '简单',
    medium: '中等',
    high: '复杂',
    very_high: '极复杂',
  },

  COMPLEXITY_COLORS: {
    low: '#4caf50',
    medium: '#ff9800',
    high: '#f44336',
    very_high: '#9c27b0',
  },
}

/**
 * 状态码
 */
export const STATUS_CODES = {
  SUCCESS: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  VALIDATION_ERROR: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
}

/**
 * 错误消息
 */
export const ERROR_MESSAGES = {
  NETWORK_ERROR: '网络连接失败，请检查网络设置',
  TIMEOUT_ERROR: '请求超时，请稍后重试',
  AUTH_EXPIRED: '登录已过期，请重新登录',
  PERMISSION_DENIED: '权限不足，请升级会员',
  PREDICTION_LIMIT_EXCEEDED: '今日预测次数已用完',
  ALGORITHM_NOT_AVAILABLE: '算法暂不可用',
  INSUFFICIENT_DATA: '历史数据不足',
  UNKNOWN_ERROR: '未知错误，请稍后重试',
}

/**
 * 页面路径
 */
export const PAGE_PATHS = {
  // 主要页面
  INDEX: '/pages/index/index',
  PREDICT: '/pages/predict/predict',
  HISTORY: '/pages/history/history',
  MEMBER: '/pages/member/member',
  PROFILE: '/pages/profile/profile',

  // 子页面/补充页面
  PREDICTION_DETAIL: '/pages/prediction/detail',
  LOTTERY_RESULTS: '/pages/lottery/results',
  ORDER_DETAIL: '/pages/order/detail',
  ORDER_LIST: '/pages/order/list',
  HELP: '/pages/help/index',
  FEEDBACK: '/pages/feedback/index',
  FAVORITES: '/pages/favorites/index',
  POINTS: '/pages/points/index',
  SETTINGS_NOTIFICATIONS: '/pages/settings/notifications',
  SETTINGS_PRIVACY: '/pages/settings/privacy',
  ABOUT: '/pages/about/index',
  LOGIN: '/pages/login/login',
}

/**
 * 事件名称
 */
export const EVENTS = {
  // 用户相关事件
  USER_LOGIN: 'user:login',
  USER_LOGOUT: 'user:logout',
  USER_INFO_UPDATED: 'user:info_updated',

  // 会员相关事件
  MEMBERSHIP_UPDATED: 'membership:updated',
  ORDER_CREATED: 'order:created',
  PAYMENT_SUCCESS: 'payment:success',

  // 预测相关事件
  PREDICTION_COMPLETED: 'prediction:completed',
  ALGORITHM_SELECTED: 'algorithm:selected',

  // 应用相关事件
  APP_UPDATED: 'app:updated',
  THEME_CHANGED: 'theme:changed',
}

/**
 * 动画配置
 */
export const ANIMATIONS = {
  DURATION: {
    FAST: 150,
    NORMAL: 300,
    SLOW: 500,
  },

  EASING: {
    EASE_IN: 'ease-in',
    EASE_OUT: 'ease-out',
    EASE_IN_OUT: 'ease-in-out',
  },
}

/**
 * 主题配置
 */
export const THEME = {
  COLORS: {
    PRIMARY: '#d32f2f',
    SECONDARY: '#666666',
    SUCCESS: '#4caf50',
    WARNING: '#ff9800',
    ERROR: '#f44336',
    INFO: '#2196f3',

    BACKGROUND: '#f5f5f5',
    SURFACE: '#ffffff',

    TEXT_PRIMARY: '#333333',
    TEXT_SECONDARY: '#666666',
    TEXT_DISABLED: '#999999',
  },

  SPACING: {
    XS: '10rpx',
    SM: '20rpx',
    MD: '30rpx',
    LG: '40rpx',
    XL: '50rpx',
  },

  BORDER_RADIUS: {
    SM: '8rpx',
    MD: '16rpx',
    LG: '24rpx',
    ROUND: '50%',
  },
}

/**
 * 正则表达式
 */
export const REGEX = {
  PHONE: /^1[3-9]\d{9}$/,
  EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  ISSUE: /^\d{7}$/,
  NUMBER_RANGE: /^[1-9]\d*$/,
}
