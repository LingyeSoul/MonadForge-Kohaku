/** A paused job still owns its task and can continue without a checkpoint. */
export function isTaskActive(state: string): boolean {
  return ['queued', 'pending', 'running', 'paused', 'stopping'].includes(state)
}

export function isTaskStoppable(state: string): boolean {
  return ['queued', 'pending', 'running', 'paused'].includes(state)
}

export function isTaskTerminal(state: string): boolean {
  return ['success', 'failed', 'cancelled'].includes(state)
}

export function canResumeTask(task: { state: string; resumable?: boolean }): boolean {
  return task.state === 'paused' || Boolean(task.resumable && ['failed', 'cancelled'].includes(task.state))
}

/** Vuetify color name for a task state chip/card accent. */
export function taskStateColor(state: string): string | undefined {
  if (state === 'paused') return 'warning'
  if (state === 'running' || state === 'stopping') return 'primary'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

/** MDI icon name for a task state. */
export function taskStateIcon(state: string): string {
  if (state === 'paused') return 'mdi-pause-circle'
  if (state === 'running' || state === 'stopping') return 'mdi-progress-clock'
  if (state === 'success') return 'mdi-check-circle'
  if (state === 'failed') return 'mdi-alert-circle'
  if (state === 'cancelled') return 'mdi-cancel'
  return 'mdi-clock-outline'
}
