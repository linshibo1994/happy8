import { ref, onUnmounted } from 'vue'
import { WebSocketClient } from '@/api/websocket'
import type { ProgressMessage } from '@/types/predict'

// Vue 组合式函数封装 WebSocket
export function useWebSocket(path: string = '/ws/predict') {
  const client = new WebSocketClient(path)
  const isConnected = ref(false)
  const error = ref<string | null>(null)
  const lastMessage = ref<ProgressMessage | null>(null)

  const connect = async () => {
    try {
      await client.connect()
      isConnected.value = true
      error.value = null
    } catch (e) {
      isConnected.value = false
      error.value = '无法连接到服务器'
      throw e
    }
  }

  const disconnect = () => {
    client.disconnect()
    isConnected.value = false
  }

  const send = (data: unknown) => {
    client.send(data)
  }

  // 监听消息
  const onMessage = (callback: (message: ProgressMessage) => void) => {
    const handler = (message: ProgressMessage) => {
      lastMessage.value = message
      callback(message)
    }
    client.on('message', handler)
    return () => client.off('message', handler)
  }

  // 监听特定类型的消息
  const onProgress = (callback: (message: ProgressMessage) => void) => {
    client.on('progress', callback)
    return () => client.off('progress', callback)
  }

  const onError = (callback: (message: ProgressMessage) => void) => {
    client.on('error', callback)
    return () => client.off('error', callback)
  }

  const onResult = (callback: (message: ProgressMessage) => void) => {
    client.on('result', callback)
    return () => client.off('result', callback)
  }

  // 组件卸载时自动断开连接
  onUnmounted(() => {
    disconnect()
  })

  return {
    client,
    isConnected,
    error,
    lastMessage,
    connect,
    disconnect,
    send,
    onMessage,
    onProgress,
    onError,
    onResult
  }
}
