<template>
  <view class="app">
    <!-- 全局Loading -->
    <wd-loading v-if="appStore.loading" :loading="true" type="outline" />

    <!-- 全局Toast -->
    <wd-toast />

    <!-- 全局MessageBox -->
    <wd-message-box />
  </view>
</template>

<script setup lang="ts">
import { onLaunch, onShow, onHide, onError } from '@dcloudio/uni-app'
import { useAppStore } from '@/stores/app'
import { useUserStore } from '@/stores/user'

const appStore = useAppStore()
const userStore = useUserStore()

onLaunch(options => {
  console.log('App Launch', options)

  // 初始化应用
  appStore.initApp()

  // 检查登录状态
  userStore.checkLoginStatus()

  // 获取系统信息
  uni.getSystemInfo({
    success: res => {
      appStore.setSystemInfo(res)
    },
  })
})

onShow(options => {
  console.log('App Show', options)

  // 应用从后台进入前台时的逻辑
  if (userStore.isLoggedIn) {
    // 刷新用户信息
    userStore.refreshUserInfo()
  }
})

onHide(() => {
  console.log('App Hide')
})

onError(error => {
  console.error('App Error:', error)
  // 错误上报
  appStore.reportError(error)
})
</script>

<style lang="scss">
@import 'wot-design-uni/components/common/abstracts/variable';

// 全局样式
page {
  background-color: #f5f5f5;
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB',
    'Microsoft YaHei', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

.app {
  min-height: 100vh;
}

// 通用样式类
.container {
  padding: 20rpx;
}

.card {
  background: #fff;
  border-radius: 16rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.08);
  margin-bottom: 20rpx;
  overflow: hidden;
}

.text-primary {
  color: #d32f2f;
}

.text-secondary {
  color: #666;
}

.text-success {
  color: #4caf50;
}

.text-warning {
  color: #ff9800;
}

.text-danger {
  color: #f44336;
}

.text-center {
  text-align: center;
}

.flex {
  display: flex;
}

.flex-column {
  flex-direction: column;
}

.justify-center {
  justify-content: center;
}

.align-center {
  align-items: center;
}

.flex-1 {
  flex: 1;
}

.mt-10 {
  margin-top: 10rpx;
}
.mt-20 {
  margin-top: 20rpx;
}
.mb-10 {
  margin-bottom: 10rpx;
}
.mb-20 {
  margin-bottom: 20rpx;
}
.ml-10 {
  margin-left: 10rpx;
}
.mr-10 {
  margin-right: 10rpx;
}

.p-10 {
  padding: 10rpx;
}
.p-20 {
  padding: 20rpx;
}
.px-10 {
  padding-left: 10rpx;
  padding-right: 10rpx;
}
.py-10 {
  padding-top: 10rpx;
  padding-bottom: 10rpx;
}
</style>
