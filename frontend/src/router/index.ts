import type { Component } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'

import {
  Activity,
  BarChart3,
  BrainCircuit,
  Crown,
  History,
  LayoutDashboard,
  ShieldCheck,
  UserRound,
} from 'lucide-vue-next'

import AlgorithmsPage from '@/pages/AlgorithmsPage.vue'
import DashboardPage from '@/pages/DashboardPage.vue'
import DataDashboardPage from '@/pages/DataDashboardPage.vue'
import MembershipPage from '@/pages/MembershipPage.vue'
import PredictionHistoryPage from '@/pages/PredictionHistoryPage.vue'
import PredictionPage from '@/pages/PredictionPage.vue'
import ProfilePage from '@/pages/ProfilePage.vue'
import SystemDiagnosticsPage from '@/pages/SystemDiagnosticsPage.vue'

export type AppRouteKey =
  | 'dashboard'
  | 'prediction'
  | 'data'
  | 'algorithms'
  | 'history'
  | 'membership'
  | 'profile'
  | 'system'

export interface AppRouteMeta {
  key: AppRouteKey
  title: string
  subtitle: string
  navigationLabel: string
  icon: Component
  adminOnly?: boolean
}

export interface AppRouteItem {
  path: string
  name: AppRouteKey
  component: Component
  meta: AppRouteMeta
}

export const appRoutes: AppRouteItem[] = [
  {
    path: '/',
    name: 'dashboard',
    component: DashboardPage,
    meta: {
      key: 'dashboard',
      title: '工作台',
      subtitle: '聚合最新开奖、预测入口、会员次数和系统状态。',
      navigationLabel: '工作台',
      icon: LayoutDashboard,
    },
  },
  {
    path: '/prediction',
    name: 'prediction',
    component: PredictionPage,
    meta: {
      key: 'prediction',
      title: '预测执行',
      subtitle: '承载单算法预测、批量预测、阶段进度和结果输出。',
      navigationLabel: '预测',
      icon: BrainCircuit,
    },
  },
  {
    path: '/data',
    name: 'data',
    component: DataDashboardPage,
    meta: {
      key: 'data',
      title: '历史数据看板',
      subtitle: '展示开奖历史、走势图、热冷号、遗漏和区间分布。',
      navigationLabel: '数据',
      icon: BarChart3,
    },
  },
  {
    path: '/algorithms',
    name: 'algorithms',
    component: AlgorithmsPage,
    meta: {
      key: 'algorithms',
      title: '算法中心',
      subtitle: '管理算法档案、权限等级、适用场景和历史表现。',
      navigationLabel: '算法',
      icon: Activity,
    },
  },
  {
    path: '/history',
    name: 'history',
    component: PredictionHistoryPage,
    meta: {
      key: 'history',
      title: '预测历史',
      subtitle: '复盘预测记录、命中号码、置信度和执行耗时。',
      navigationLabel: '历史',
      icon: History,
    },
  },
  {
    path: '/membership',
    name: 'membership',
    component: MembershipPage,
    meta: {
      key: 'membership',
      title: '会员中心',
      subtitle: '展示会员等级、权益套餐、剩余次数和订单记录。',
      navigationLabel: '会员',
      icon: Crown,
    },
  },
  {
    path: '/profile',
    name: 'profile',
    component: ProfilePage,
    meta: {
      key: 'profile',
      title: '个人中心',
      subtitle: '维护个人资料、偏好设置和登录状态。',
      navigationLabel: '我的',
      icon: UserRound,
    },
  },
  {
    path: '/system',
    name: 'system',
    component: SystemDiagnosticsPage,
    meta: {
      key: 'system',
      title: '系统诊断',
      subtitle: '后续接入数据同步、算法诊断和管理员入口。',
      navigationLabel: '系统',
      icon: ShieldCheck,
      adminOnly: true,
    },
  },
]

export const router = createRouter({
  history: createWebHistory(),
  routes: appRoutes,
  scrollBehavior: () => ({ top: 0 }),
})
