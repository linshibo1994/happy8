<template>
  <view class="profile-page">
    <view class="header">
      <view class="container">
        <view v-if="userStore.isLoggedIn" class="profile-card card">
          <view class="user-row">
            <image class="avatar" :src="userStore.getAvatarUrl" mode="aspectFill" />
            <view class="info">
              <text class="name">{{ userStore.getDisplayName }}</text>
              <text class="meta">{{ levelText }}</text>
            </view>
            <wd-button type="secondary" size="small" @click="openEdit">
              编辑
            </wd-button>
          </view>
        </view>
        <view v-else class="profile-card card">
          <text class="name">欢迎使用 Happy8</text>
          <wd-button type="primary" size="small" @click="login">微信登录</wd-button>
        </view>
      </view>
    </view>

    <view v-if="userStore.isLoggedIn" class="stats">
      <view class="container">
        <view class="stats-grid">
          <view class="stat-item">
            <text class="value">{{ userStats?.total_predictions ?? 0 }}</text>
            <text class="label">总预测</text>
          </view>
          <view class="stat-item">
            <text class="value">{{ formatPercentage(userStats?.overall_hit_rate) }}</text>
            <text class="label">命中率</text>
          </view>
          <view class="stat-item">
            <text class="value">{{ membershipStore.remainingDays }}</text>
            <text class="label">剩余天数</text>
          </view>
          <view class="stat-item">
            <text class="value">{{ remainingPredictionsText }}</text>
            <text class="label">今日次数</text>
          </view>
        </view>
      </view>
    </view>

    <view class="menu">
      <view class="container">
        <view class="menu-section card">
          <view class="menu-item" v-for="item in menus" :key="item.id" @click="handleMenu(item)">
            <text class="menu-title">{{ item.title }}</text>
            <text class="menu-desc">{{ item.desc }}</text>
            <text class="menu-arrow">></text>
          </view>
        </view>
      </view>
    </view>

    <view v-if="userStore.isLoggedIn" class="logout">
      <view class="container">
        <wd-button type="error" size="large" @click="logout">退出登录</wd-button>
      </view>
    </view>

    <wd-popup v-model="showEdit" position="bottom" :safe-area-inset-bottom="true">
      <view class="edit-modal">
        <view class="modal-header">
          <text class="title">编辑资料</text>
          <wd-button type="text" @click="showEdit = false">关闭</wd-button>
        </view>
        <view class="modal-body">
          <wd-input v-model="editForm.nickname" label="昵称" placeholder="请输入昵称" />
          <wd-input v-model="editForm.real_name" label="真实姓名" placeholder="请输入真实姓名" />
        </view>
        <view class="modal-footer">
          <wd-button type="primary" :loading="saving" @click="saveProfile">保存</wd-button>
        </view>
      </view>
    </wd-popup>
  </view>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useUserStore } from '@/stores/user'
import { usePredictStore } from '@/stores/predict'
import { useMembershipStore } from '@/stores/member'
import type { PredictionStats } from '@/types'
import { PAGE_PATHS } from '@/constants'

const userStore = useUserStore()
const predictStore = usePredictStore()
const membershipStore = useMembershipStore()

const userStats = ref<PredictionStats | null>(null)
const showEdit = ref(false)
const saving = ref(false)
const editForm = ref({ nickname: '', real_name: '' })

const levelText = computed(() => membershipStore.membershipStatus?.membership.level ? membershipStore.getLevelName(membershipStore.membershipStatus.membership.level) : '未开通会员')
const remainingPredictionsText = computed(() => {
  const remaining = membershipStore.remainingPredictions
  if (remaining === -1) return '无限'
  return remaining
})

const menus = computed(() => [
  { id: 1, title: '预测历史', desc: '查看历史记录', path: PAGE_PATHS.HISTORY, requireLogin: true },
  { id: 2, title: '会员中心', desc: '查看会员权益', path: PAGE_PATHS.MEMBER, requireLogin: true },
  { id: 3, title: '订单记录', desc: '查看近期订单', path: PAGE_PATHS.ORDER_LIST, requireLogin: true }
])

const loadData = async () => {
  if (!userStore.isLoggedIn) return
  await Promise.all([
    predictStore.loadPredictionStats(),
    membershipStore.loadAll()
  ])
  userStats.value = predictStore.stats
}

onMounted(async () => {
  await loadData()
})

const login = async () => {
  try {
    await userStore.wechatLogin()
    await loadData()
  } catch (error) {
    console.error(error)
  }
}

const logout = async () => {
  await userStore.logout()
}

const openEdit = () => {
  if (!userStore.userInfo) return
  editForm.value = {
    nickname: userStore.userInfo.nickname ?? '',
    real_name: userStore.userInfo.real_name ?? ''
  }
  showEdit.value = true
}

const saveProfile = async () => {
  try {
    saving.value = true
    await userStore.updateProfile({
      nickname: editForm.value.nickname,
      real_name: editForm.value.real_name
    })
    showEdit.value = false
  } catch (error) {
    console.error(error)
  } finally {
    saving.value = false
  }
}

const handleMenu = (item: { path: string; requireLogin: boolean }) => {
  if (item.requireLogin && !userStore.isLoggedIn) {
    uni.showToast({ title: '请先登录', icon: 'none' })
    return
  }
  uni.navigateTo({ url: item.path })
}

const formatPercentage = (value?: number | null) => {
  if (value === undefined || value === null) return '0%'
  return `${value.toFixed(2)}%`
}
</script>

<style lang="scss" scoped>
.profile-page {
  min-height: 100vh;
  background: $background-color;
}

.container {
  padding: $spacing-md;
}

.profile-card {
  padding: $spacing-md;

  .user-row {
    @include flex-between;
    align-items: center;

    .avatar {
      width: 120rpx;
      height: 120rpx;
      border-radius: 50%;
      margin-right: $spacing-md;
    }

    .info {
      flex: 1;

      .name {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $text-primary;
      }

      .meta {
        font-size: $font-size-sm;
        color: $text-secondary;
      }
    }
  }
}

.stats {
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: $spacing-sm;

    .stat-item {
      @include flex-center;
      flex-direction: column;
      padding: $spacing-md;
      background: white;
      border-radius: $border-radius-md;

      .value {
        font-size: $font-size-lg;
        font-weight: $font-weight-bold;
        color: $primary-color;
      }

      .label {
        font-size: $font-size-xs;
        color: $text-secondary;
      }
    }
  }
}

.menu-section {
  margin-top: $spacing-md;

  .menu-item {
    @include flex-between;
    align-items: center;
    padding: $spacing-md 0;
    border-bottom: 1rpx solid $border-color-light;

    &:last-child {
      border-bottom: none;
    }

    .menu-title {
      font-size: $font-size-md;
      color: $text-primary;
    }

    .menu-desc {
      font-size: $font-size-sm;
      color: $text-secondary;
    }

    .menu-arrow {
      font-size: $font-size-md;
      color: $text-secondary;
    }
  }
}

.logout {
  margin-top: $spacing-lg;
}

.edit-modal {
  padding: $spacing-md;

  .modal-header {
    @include flex-between;
    margin-bottom: $spacing-md;
  }

  .modal-footer {
    margin-top: $spacing-md;
  }
}
</style>
