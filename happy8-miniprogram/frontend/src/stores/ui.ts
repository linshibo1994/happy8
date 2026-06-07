import { defineStore } from 'pinia'

type ThemeMode = 'light' | 'dark'
type LayoutDensity = 'comfortable' | 'compact'

interface UiState {
  themeMode: ThemeMode
  density: LayoutDensity
  mobileDrawerOpen: boolean
  globalMessage: string
}

export const useUiStore = defineStore('ui', {
  state: (): UiState => ({
    themeMode: 'light',
    density: 'comfortable',
    mobileDrawerOpen: false,
    globalMessage: '',
  }),
  actions: {
    toggleTheme() {
      this.themeMode = this.themeMode === 'light' ? 'dark' : 'light'
    },
    setDensity(density: LayoutDensity) {
      this.density = density
    },
    setMobileDrawerOpen(open: boolean) {
      this.mobileDrawerOpen = open
    },
    setGlobalMessage(message: string) {
      this.globalMessage = message
    },
  },
})
