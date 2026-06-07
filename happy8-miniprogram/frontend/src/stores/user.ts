import { defineStore } from 'pinia'

import type { UserProfile } from '@/types'

interface UserState {
  token: string
  profile: UserProfile
}

export const useUserStore = defineStore('user', {
  state: (): UserState => ({
    token: '',
    profile: {
      id: 'mock-user-001',
      nickname: '体验用户',
      role: 'user',
    },
  }),
  getters: {
    isLoggedIn: (state) => Boolean(state.token || state.profile.id),
    isAdmin: (state) => state.profile.role === 'admin',
  },
  actions: {
    setToken(token: string) {
      this.token = token
    },
    updateProfile(profile: Partial<UserProfile>) {
      this.profile = {
        ...this.profile,
        ...profile,
      }
    },
  },
})
