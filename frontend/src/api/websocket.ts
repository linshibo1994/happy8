import { WS_BASE_URL } from '@/utils/constants'
import type { ProgressMessage } from '@/types/predict'

// WebSocket 消息回调类型
type MessageCallback = (data: ProgressMessage) => void

// WebSocket 客户端类
export class WebSocketClient {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 3000
  private listeners: Map<string, MessageCallback[]> = new Map()

  constructor(path: string = '/ws/predict') {
    this.url = `${WS_BASE_URL}${path}`
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)

        this.ws.onopen = () => {
          console.log('WebSocket已连接')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: ProgressMessage = JSON.parse(event.data)
            this.emit(message.type, message)
            // 同时触发 'message' 事件，方便统一监听
            this.emit('message', message)
          } catch (e) {
            console.error('解析WebSocket消息失败:', e)
          }
        }

        this.ws.onclose = () => {
          console.log('WebSocket已断开')
          this.emit('close', { type: 'log', percentage: 0, details: '连接已断开', status: 'error' })
          this.tryReconnect()
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket错误:', error)
          reject(error)
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  private tryReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`)
      setTimeout(() => this.connect(), this.reconnectDelay)
    }
  }

  on(event: string, callback: MessageCallback): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event)!.push(callback)
  }

  off(event: string, callback: MessageCallback): void {
    const callbacks = this.listeners.get(event)
    if (callbacks) {
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
    }
  }

  private emit(event: string, data: ProgressMessage): void {
    const callbacks = this.listeners.get(event)
    if (callbacks) {
      callbacks.forEach(cb => cb(data))
    }
  }

  send(data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.error('WebSocket未连接')
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.listeners.clear()
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}
