// 导出所有自定义组件
export { default as Loading } from './Loading.vue'
export { default as EmptyState } from './EmptyState.vue'
export { default as NumberBall } from './NumberBall.vue'
export { default as StatCard } from './StatCard.vue'

// 组件注册配置（用于全局注册）
export const componentConfig = {
  Loading: () => import('./Loading.vue'),
  EmptyState: () => import('./EmptyState.vue'),
  NumberBall: () => import('./NumberBall.vue'),
  StatCard: () => import('./StatCard.vue'),
}
