import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { UserInfo, UserProfile, LoginResponse } from '@/types'
import { STORAGE_KEYS, STATUS_CODES, PAGE_PATHS } from '@/constants'
import { userApi } from '@/services/user'
import { useMembershipStore } from '@/stores/member'

export const useUserStore = defineStore(
  'user',
  () => {
    // 状态
    const accessToken = ref<string>('')
    const refreshToken = ref<string>('')
    const userInfo = ref<UserInfo | null>(null)
    const userProfile = ref<UserProfile | null>(null)
    const membershipStore = useMembershipStore()

    const extractProfile = (info: UserInfo | null): UserProfile | null => {
      if (!info) return null
      return {
        real_name: info.real_name ?? undefined,
        gender: (info.gender as UserProfile['gender']) ?? 'unknown',
        birthday: info.birthday ?? undefined,
        address: info.address ?? undefined,
        preferences: (info.preferences as Record<string, unknown> | undefined) ?? undefined,
      }
    }

    // 登录状态
    const isLoggedIn = computed(() => {
      return !!accessToken.value && !!userInfo.value
    })

    // 方法
    const login = async (code: string) => {
      try {
        const response = await userApi.wechatLogin({ code })

        if (response.code === STATUS_CODES.SUCCESS) {
          // 保存token
          accessToken.value = response.data.access_token
          refreshToken.value = response.data.refresh_token

          // 保存用户信息
          userInfo.value = response.data.user_info
          userProfile.value = extractProfile(userInfo.value)

          // 存储到本地
          saveTokens()

          // 同步会员状态
          try {
            await membershipStore.loadMembershipStatus()
          } catch (error) {
            console.error('加载会员状态失败:', error)
          }

          console.log('登录成功')
          return response.data
        } else {
          throw new Error(response.message)
        }
      } catch (error) {
        console.error('登录失败:', error)
        throw error
      }
    }

    const wechatLogin = async () => {
      return new Promise<LoginResponse>((resolve, reject) => {
        uni.login({
          provider: 'weixin',
          success: async loginRes => {
            try {
              const result = await login(loginRes.code)
              resolve(result)
            } catch (error) {
              reject(error)
            }
          },
          fail: error => {
            console.error('微信登录失败:', error)
            reject(new Error('微信登录失败'))
          },
        })
      })
    }

    const logout = async () => {
      try {
        // 调用后端登出接口
        if (accessToken.value) {
          await userApi.logout()
        }
      } catch (error) {
        console.error('登出请求失败:', error)
      } finally {
        // 清除本地数据
        clearUserData()
        try {
          membershipStore.resetStore()
        } catch (error) {
          console.error('重置会员状态失败:', error)
        }

        // 跳转到首页
        uni.switchTab({
          url: PAGE_PATHS.INDEX,
        })

        uni.showToast({
          title: '已退出登录',
          icon: 'success',
        })
      }
    }

    const refreshUserInfo = async () => {
      if (!isLoggedIn.value) return

      try {
        const response = await userApi.getProfile()

        if (response.code === STATUS_CODES.SUCCESS) {
          userInfo.value = response.data
          userProfile.value = extractProfile(response.data)
          uni.setStorageSync(STORAGE_KEYS.USER_INFO, response.data)
        }
      } catch (error) {
        console.error('刷新用户信息失败:', error)

        // 如果是token过期，尝试刷新token
        if (
          typeof error === 'object' &&
          error !== null &&
          'code' in error &&
          (error as { code?: number }).code === STATUS_CODES.UNAUTHORIZED
        ) {
          await tryRefreshToken()
        }
      }
    }

    const updateProfile = async (profileData: Partial<UserProfile>) => {
      try {
        const response = await userApi.updateProfile(profileData)

        if (response.code === STATUS_CODES.SUCCESS) {
          userInfo.value = response.data
          userProfile.value = extractProfile(response.data)
          if (userInfo.value) {
            uni.setStorageSync(STORAGE_KEYS.USER_INFO, userInfo.value)
          }

          uni.showToast({
            title: '更新成功',
            icon: 'success',
          })

          return response.data
        } else {
          throw new Error(response.message)
        }
      } catch (error) {
        console.error('更新个人资料失败:', error)
        uni.showToast({
          title: '更新失败',
          icon: 'error',
        })
        throw error
      }
    }

    const tryRefreshToken = async () => {
      if (!refreshToken.value) {
        // 没有刷新token，需要重新登录
        await logout()
        return false
      }

      try {
        const response = await userApi.refreshToken({
          refresh_token: refreshToken.value,
        })

        if (response.code === STATUS_CODES.SUCCESS) {
          accessToken.value = response.data.access_token
          refreshToken.value = response.data.refresh_token

          saveTokens()
          return true
        } else {
          throw new Error(response.message)
        }
      } catch (error) {
        console.error('刷新token失败:', error)
        await logout()
        return false
      }
    }

    const checkLoginStatus = () => {
      try {
        // 从本地存储恢复token
        const savedAccessToken = uni.getStorageSync(STORAGE_KEYS.ACCESS_TOKEN)
        const savedRefreshToken = uni.getStorageSync(STORAGE_KEYS.REFRESH_TOKEN)
        const savedUserInfo = uni.getStorageSync(STORAGE_KEYS.USER_INFO)

        if (savedAccessToken && savedUserInfo) {
          accessToken.value = savedAccessToken
          refreshToken.value = savedRefreshToken
          userInfo.value = savedUserInfo

          // 验证token有效性
          refreshUserInfo()
          membershipStore.loadMembershipStatus().catch(error =>
            console.error('恢复会员状态失败:', error)
          )
        }
      } catch (error) {
        console.error('检查登录状态失败:', error)
      }
    }

    const saveTokens = () => {
      try {
        uni.setStorageSync(STORAGE_KEYS.ACCESS_TOKEN, accessToken.value)
        uni.setStorageSync(STORAGE_KEYS.REFRESH_TOKEN, refreshToken.value)

        if (userInfo.value) {
          uni.setStorageSync(STORAGE_KEYS.USER_INFO, userInfo.value)
        }
      } catch (error) {
        console.error('保存token失败:', error)
      }
    }

    const clearUserData = () => {
      // 清除状态
      accessToken.value = ''
      refreshToken.value = ''
      userInfo.value = null
      userProfile.value = null

      // 清除本地存储
      try {
        uni.removeStorageSync(STORAGE_KEYS.ACCESS_TOKEN)
        uni.removeStorageSync(STORAGE_KEYS.REFRESH_TOKEN)
        uni.removeStorageSync(STORAGE_KEYS.USER_INFO)
      } catch (error) {
        console.error('清除用户数据失败:', error)
      }
    }

    // 获取用户头像（带默认值）
    const getAvatarUrl = computed(() => {
      return userInfo.value?.avatar_url || '/static/images/default-avatar.png'
    })

    // 获取用户昵称（带默认值）
    const getDisplayName = computed(() => {
      return userInfo.value?.nickname || userProfile.value?.real_name || '未设置昵称'
    })

    // 检查是否需要完善信息
    const needCompleteProfile = computed(() => {
      if (!userInfo.value) return false

      return !userInfo.value.phone || !userProfile.value?.real_name
    })

    // 权限检查相关方法
    const hasPhone = computed(() => {
      return !!userInfo.value?.phone
    })

    const hasEmail = computed(() => {
      return !!userInfo.value?.email
    })

    return {
      // 状态
      accessToken,
      refreshToken,
      userInfo,
      userProfile,

      // 计算属性
      isLoggedIn,
      getAvatarUrl,
      getDisplayName,
      needCompleteProfile,
      hasPhone,
      hasEmail,

      // 方法
      login,
      wechatLogin,
      logout,
      refreshUserInfo,
      updateProfile,
      tryRefreshToken,
      checkLoginStatus,
      clearUserData,
    }
  },
  {
    persist: {
      key: 'user-store',
      storage: {
        getItem: (key: string) => uni.getStorageSync(key),
        setItem: (key: string, value: string) => uni.setStorageSync(key, value),
      },
      paths: ['accessToken', 'refreshToken', 'userInfo', 'userProfile'],
    },
  }
)
