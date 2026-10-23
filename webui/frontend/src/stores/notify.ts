import { defineStore } from 'pinia'
import { ref } from 'vue'

export type NotifyType = 'success' | 'info' | 'warning' | 'error'

export interface NotifyItem {
  id: number
  message: string
  type: NotifyType
  timeout: number
}

let nextId = 0

export const useNotifyStore = defineStore('notify', () => {
  const current = ref<NotifyItem | null>(null)
  let queue: NotifyItem[] = []

  function show(message: string, type: NotifyType = 'info', timeout?: number) {
    // Errors stay until dismissed (timeout 0); transient types auto-close.
    const effective = timeout ?? (type === 'error' ? 0 : 3000)
    const item: NotifyItem = { id: nextId++, message, type, timeout: effective }
    if (!current.value) {
      current.value = item
    } else {
      queue.push(item)
    }
  }

  function dismiss() {
    if (queue.length > 0) {
      current.value = queue.shift()!
    } else {
      current.value = null
    }
  }

  return { current, show, dismiss }
})
