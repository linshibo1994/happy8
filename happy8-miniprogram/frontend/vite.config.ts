import { defineConfig } from 'vite'
import uni from '@dcloudio/vite-plugin-uni'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [uni()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@/components': resolve(__dirname, 'src/components'),
      '@/pages': resolve(__dirname, 'src/pages'),
      '@/stores': resolve(__dirname, 'src/stores'),
      '@/services': resolve(__dirname, 'src/services'),
      '@/utils': resolve(__dirname, 'src/utils'),
      '@/types': resolve(__dirname, 'src/types'),
      '@/constants': resolve(__dirname, 'src/constants'),
      '@/config': resolve(__dirname, 'src/config'),
    }
  },
  server: {
    port: 8080,
    host: '0.0.0.0',
    open: false
  },
  build: {
    target: 'es6',
    cssTarget: 'chrome61',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  optimizeDeps: {
    include: [
      'vue',
      'pinia',
      'pinia-plugin-persistedstate',
      'dayjs',
      '@vueuse/core'
    ]
  }
})