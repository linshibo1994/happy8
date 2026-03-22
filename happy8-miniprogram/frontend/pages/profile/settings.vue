<template>
  <view class="settings">
    <!-- 账户设置 -->
    <view class="account-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">账户设置</text>
        </view>
        
        <view class="settings-list">
          <view class="setting-item card" @click="updateProfile">
            <view class="item-content">
              <text class="item-icon">👤</text>
              <text class="item-title">个人信息</text>
            </view>
            <text class="item-arrow">></text>
          </view>
          
          <view class="setting-item card" @click="changePassword">
            <view class="item-content">
              <text class="item-icon">🔒</text>
              <text class="item-title">修改密码</text>
            </view>
            <text class="item-arrow">></text>
          </view>
          
          <view class="setting-item card" @click="bindWeChat">
            <view class="item-content">
              <text class="item-icon">💬</text>
              <view class="item-info">
                <text class="item-title">微信绑定</text>
                <text class="item-desc">{{ userStore.user?.wechat_bound ? '已绑定' : '未绑定' }}</text>
              </view>
            </view>
            <text class="item-arrow">></text>
          </view>
        </view>
      </view>
    </view>

    <!-- 应用设置 -->
    <view class="app-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">应用设置</text>
        </view>
        
        <view class="settings-list">
          <view class="setting-item card">
            <view class="item-content">
              <text class="item-icon">🔔</text>
              <text class="item-title">开奖提醒</text>
            </view>
            <wd-switch v-model="settings.notificationEnabled" @change="updateNotificationSetting" />
          </view>
          
          <view class="setting-item card">
            <view class="item-content">
              <text class="item-icon">🌙</text>
              <text class="item-title">深色模式</text>
            </view>
            <wd-switch v-model="settings.darkMode" @change="updateDarkMode" />
          </view>
          
          <view class="setting-item card">
            <view class="item-content">
              <text class="item-icon">📊</text>
              <text class="item-title">数据统计</text>
            </view>
            <wd-switch v-model="settings.analyticsEnabled" @change="updateAnalyticsSetting" />
          </view>
          
          <view class="setting-item card" @click="clearCache">
            <view class="item-content">
              <text class="item-icon">🗑️</text>
              <view class="item-info">
                <text class="item-title">清除缓存</text>
                <text class="item-desc">{{ cacheSize }}</text>
              </view>
            </view>
            <text class="item-arrow">></text>
          </view>
        </view>
      </view>
    </view>

    <!-- 隐私设置 -->
    <view class="privacy-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">隐私设置</text>
        </view>
        
        <view class="settings-list">
          <view class="setting-item card" @click="viewPrivacyPolicy">
            <view class="item-content">
              <text class="item-icon">📋</text>
              <text class="item-title">隐私政策</text>
            </view>
            <text class="item-arrow">></text>
          </view>
          
          <view class="setting-item card" @click="viewUserAgreement">
            <view class="item-content">
              <text class="item-icon">📄</text>
              <text class="item-title">用户协议</text>
            </view>
            <text class="item-arrow">></text>
          </view>
          
          <view class="setting-item card" @click="manageData">
            <view class="item-content">
              <text class="item-icon">💾</text>
              <text class="item-title">数据管理</text>
            </view>
            <text class="item-arrow">></text>
          </view>
        </view>
      </view>
    </view>

    <!-- 其他设置 -->
    <view class="other-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">其他</text>
        </view>
        
        <view class="settings-list">
          <view class="setting-item card" @click="checkUpdate">
            <view class="item-content">
              <text class="item-icon">📱</text>
              <view class="item-info">
                <text class="item-title">检查更新</text>
                <text class="item-desc">当前版本 v{{ appVersion }}</text>
              </view>
            </view>
            <text class="item-arrow">></text>
          </view>
          
          <view class="setting-item card" @click="feedback">
            <view class="item-content">
              <text class="item-icon">💬</text>
              <text class="item-title">意见反馈</text>
            </view>
            <text class="item-arrow">></text>
          </view>
          
          <view class="setting-item card" @click="contactService">
            <view class="item-content">
              <text class="item-icon">🎧</text>
              <text class="item-title">联系客服</text>
            </view>
            <text class="item-arrow">></text>
          </view>
        </view>
      </view>
    </view>

    <!-- 退出登录 -->
    <view class="logout-section">
      <view class="container">
        <wd-button 
          type="danger"
          size="large"
          @click="logout"
          custom-class="logout-btn"
        >
          退出登录
        </wd-button>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useUserStore } from '@/stores/user'
import { PAGE_PATHS } from '@/constants'

const userStore = useUserStore()

const settings = ref({
  notificationEnabled: true,
  darkMode: false,
  analyticsEnabled: true
})

const cacheSize = ref('12.5MB')
const appVersion = ref('1.0.0')

const updateProfile = () => {
  uni.showModal({
    title: '功能开发中',
    content: '个人信息修改功能正在开发中',
    showCancel: false
  })
}

const changePassword = () => {
  uni.showModal({
    title: '功能开发中',
    content: '密码修改功能正在开发中',
    showCancel: false
  })
}

const bindWeChat = () => {
  if (userStore.user?.wechat_bound) {
    uni.showModal({
      title: '解绑微信',
      content: '确定要解绑微信账号吗？',
      success: (res) => {
        if (res.confirm) {
          // 解绑微信逻辑
          uni.showToast({
            title: '功能开发中',
            icon: 'none'
          })
        }
      }
    })
  } else {
    // 绑定微信逻辑
    uni.showToast({
      title: '功能开发中',
      icon: 'none'
    })
  }
}

const updateNotificationSetting = (value: boolean) => {
  // 更新通知设置
  console.log('通知设置:', value)
  uni.showToast({
    title: value ? '已开启通知' : '已关闭通知',
    icon: 'success'
  })
}

const updateDarkMode = (value: boolean) => {
  // 更新深色模式
  console.log('深色模式:', value)
  uni.showToast({
    title: value ? '已开启深色模式' : '已关闭深色模式',
    icon: 'success'
  })
}

const updateAnalyticsSetting = (value: boolean) => {
  // 更新数据统计设置
  console.log('数据统计:', value)
  uni.showToast({
    title: value ? '已开启数据统计' : '已关闭数据统计',
    icon: 'success'
  })
}

const clearCache = () => {
  uni.showModal({
    title: '清除缓存',
    content: '确定要清除所有缓存数据吗？',
    success: (res) => {
      if (res.confirm) {
        // 清除缓存逻辑
        cacheSize.value = '0MB'
        uni.showToast({
          title: '缓存已清除',
          icon: 'success'
        })
      }
    }
  })
}

const viewPrivacyPolicy = () => {
  uni.showModal({
    title: '隐私政策',
    content: '隐私政策页面正在开发中',
    showCancel: false
  })
}

const viewUserAgreement = () => {
  uni.showModal({
    title: '用户协议',
    content: '用户协议页面正在开发中',
    showCancel: false
  })
}

const manageData = () => {
  uni.showModal({
    title: '数据管理',
    content: '数据管理功能正在开发中',
    showCancel: false
  })
}

const checkUpdate = () => {
  uni.showLoading({ title: '检查更新中...' })
  
  setTimeout(() => {
    uni.hideLoading()
    uni.showToast({
      title: '已是最新版本',
      icon: 'success'
    })
  }, 1500)
}

const feedback = () => {
  uni.showModal({
    title: '意见反馈',
    content: '意见反馈功能正在开发中',
    showCancel: false
  })
}

const contactService = () => {
  uni.showModal({
    title: '联系客服',
    content: '客服微信：happy8service\n工作时间：9:00-18:00',
    confirmText: '复制微信号',
    success: (res) => {
      if (res.confirm) {
        uni.setClipboardData({
          data: 'happy8service',
          success: () => {
            uni.showToast({
              title: '已复制到剪贴板',
              icon: 'success'
            })
          }
        })
      }
    }
  })
}

const logout = () => {
  uni.showModal({
    title: '退出登录',
    content: '确定要退出登录吗？',
    success: async (res) => {
      if (res.confirm) {
        try {
          await userStore.logout()
          uni.showToast({
            title: '已退出登录',
            icon: 'success'
          })
          
          // 跳转到登录页
          setTimeout(() => {
            uni.reLaunch({ url: PAGE_PATHS.INDEX })
          }, 1500)
        } catch (error) {
          console.error('退出登录失败:', error)
          uni.showToast({
            title: '退出失败',
            icon: 'error'
          })
        }
      }
    }
  })
}

onMounted(() => {
  // 加载设置
  const savedSettings = uni.getStorageSync('app_settings')
  if (savedSettings) {
    settings.value = { ...settings.value, ...savedSettings }
  }
})
</script>

<style lang="scss" scoped>
.settings {
  min-height: 100vh;
  background: $background-color;
  padding-bottom: 200rpx;
}

.section-header {
  padding: $spacing-md 0 $spacing-sm;
  
  .section-title {
    font-size: $font-size-md;
    font-weight: $font-weight-bold;
    color: $text-primary;
  }
}

.settings-list {
  .setting-item {
    @include flex-between;
    padding: $spacing-md;
    margin-bottom: $spacing-sm;
    
    .item-content {
      @include flex-align-center;
      flex: 1;
      
      .item-icon {
        font-size: $font-size-lg;
        margin-right: $spacing-md;
      }
      
      .item-title {
        font-size: $font-size-md;
        color: $text-primary;
      }
      
      .item-info {
        .item-title {
          font-size: $font-size-md;
          color: $text-primary;
          display: block;
          margin-bottom: 4rpx;
        }
        
        .item-desc {
          font-size: $font-size-sm;
          color: $text-secondary;
        }
      }
    }
    
    .item-arrow {
      font-size: $font-size-lg;
      color: $text-disabled;
      margin-left: $spacing-sm;
    }
  }
}

.logout-section {
  margin-top: $spacing-lg;
  
  .logout-btn {
    width: 100%;
  }
}

// 各个设置分组的间距
.account-section,
.app-section,
.privacy-section,
.other-section {
  margin-bottom: $spacing-lg;
}
</style>