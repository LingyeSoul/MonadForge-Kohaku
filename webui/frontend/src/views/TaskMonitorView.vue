<template>
  <v-container fluid class="task-monitor pa-4 d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow: hidden;">
    <WorkspaceHeading class="task-toolbar" :title="t('taskTitle')" :subtitle="t('taskSubtitle')">
      <v-spacer />
      <div class="task-toolbar-actions d-flex align-center">
        <v-btn
          variant="text"
          size="small"
          :disabled="!taskStore.daemonUp"
          :prepend-icon="taskStore.daemonPaused ? 'mdi-play' : 'mdi-pause'"
          :color="taskStore.daemonPaused ? 'success' : undefined"
          @click="taskStore.daemonPaused ? taskStore.resumeQueue() : taskStore.pauseQueue()"
        >
          {{ taskStore.daemonPaused ? t('taskResumeQueue') : t('taskPauseQueue') }}
        </v-btn>
        <v-btn
          v-if="!isWindows"
          variant="text"
          size="small"
          color="error"
          :disabled="!taskStore.daemonUp"
          prepend-icon="mdi-power"
          @click="showShutdownDlg = true"
        >
          {{ t('taskShutdownDaemon') }}
        </v-btn>
        <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="refreshView">
          {{ t('ppRefresh') }}
        </v-btn>
      </div>
    </WorkspaceHeading>
    <v-btn-toggle v-model="taskFilter" density="compact" mandatory divided
      :aria-label="t('taskFilterLabel')" class="task-filter mb-3 align-self-start flex-shrink-0">
      <v-btn value="all" size="small">{{ t('taskFilterAll') }}</v-btn>
      <v-btn value="active" size="small">{{ t('taskFilterActive') }}</v-btn>
      <v-btn value="success" size="small">{{ t('taskFilterSuccess') }}</v-btn>
      <v-btn value="failed" size="small">{{ t('taskFilterFailed') }}</v-btn>
      <v-btn value="cancelled" size="small">{{ t('taskFilterCancelled') }}</v-btn>
    </v-btn-toggle>

    <section v-if="historyError" class="workspace-empty workspace-empty--error flex-grow-1" role="alert">
      <div class="workspace-empty__mark"><v-icon icon="mdi-cloud-alert-outline" size="32" /></div>
      <h2>{{ t('taskLoadFailedTitle') }}</h2>
      <p>{{ t('taskLoadFailed') }}</p>
      <v-btn variant="outlined" prepend-icon="mdi-refresh" @click="refreshView">{{ t('ppRefresh') }}</v-btn>
    </section>

    <section v-else-if="historyLoading && historyTasks.length === 0" class="workspace-empty workspace-loading" role="status" aria-live="polite">
      <v-progress-circular indeterminate size="28" width="2" />
      <h2>{{ t('taskLoading') }}</h2>
    </section>

    <div v-else-if="visibleTasks.length === 0" class="workspace-empty flex-grow-1">
      <div class="workspace-empty__mark"><v-icon :icon="taskFilter === 'all' ? 'mdi-format-list-checks' : 'mdi-filter-off-outline'" size="32" /></div>
      <h2>{{ taskFilter === 'all' ? t('taskNoHistory') : t('taskNoMatches') }}</h2>
      <p>{{ t(taskFilter === 'all' ? 'taskNoHistoryDescription' : 'taskNoMatchesDescription') }}</p>
      <v-btn v-if="taskFilter !== 'all'" variant="outlined" prepend-icon="mdi-filter-off-outline" @click="taskFilter = 'all'">
        {{ t('taskFilterAll') }}
      </v-btn>
    </div>

    <div
      v-else
      class="d-flex flex-column flex-grow-1"
      :aria-busy="historyLoading"
      :style="selectedTask ? 'min-height: 0; overflow: hidden;' : 'min-height: 0; overflow-y: auto;'"
    >
      <v-progress-linear v-if="historyLoading" indeterminate color="primary"
        height="2" class="task-refresh-progress" :aria-label="t('taskLoading')" />
      <template v-for="task in displayedTasks" :key="task.task_id">
        <!-- Compact card for non-selected tasks -->
        <v-card
          v-if="selectedTask !== task.task_id"
          class="task-card mb-2 flex-shrink-0"
          :class="{ 'task-complete': task.state === 'success', 'task-failed': task.state === 'failed', 'task-running': task.state === 'running' || task.state === 'stopping' }"
          variant="tonal"
          :color="stateColor(task.state)"
        >
          <v-card-title class="task-card-heading text-body-2 d-flex align-center">
            <v-icon :icon="stateIcon(task.state)" size="small" class="task-card-state-icon mr-2" />
            <span class="task-card-command text-truncate">{{ task.command }}</span>
            <div class="task-card-badges">
              <v-chip v-if="task.command.startsWith('moss-') || task.command.startsWith('qwen21-') || task.category === 'training'" size="x-small" variant="tonal">{{ task.command.startsWith('moss-') ? 'MOSS' : task.command.startsWith('qwen21-') ? 'Qwen-Image 2.1' : 'Anima' }}</v-chip>
              <v-chip
                v-if="task.state === 'pending' && task.queue_position != null"
                size="x-small"
                variant="tonal"
                color="info"
              >
                #{{ task.queue_position }}
              </v-chip>
              <v-chip size="x-small" variant="outlined">{{ stateLabel(task.state) }}</v-chip>
              <v-chip v-if="task.legacy" size="x-small" color="warning" variant="tonal">
                {{ t('taskLegacy') }}
              </v-chip>
            </div>
          </v-card-title>
          <v-card-subtitle class="task-card-subtitle text-caption">
            {{ task.task_id }} &middot; PID {{ task.pid ?? '—' }}
            <span v-if="task.recovery_step != null"> &middot; step {{ task.recovery_step }}</span>
            <span v-if="task.last_progress?.step != null"> &middot; {{ t('taskLastProgress') }} {{ task.last_progress.step }}<span v-if="task.last_progress.total_steps != null">/{{ task.last_progress.total_steps }}</span></span>
          </v-card-subtitle>
          <v-card-text v-if="task.terminal_reason" class="text-caption pt-0 text-medium-emphasis">
            {{ t('taskTerminalReason') }}: {{ task.terminal_reason }}
          </v-card-text>
          <v-card-text v-if="task.legacy" class="text-caption pt-0 text-warning">
            {{ t('taskLegacyDescription') }}
          </v-card-text>
          <TaskAttemptHistory :task="task" />
          <v-card-text v-if="task.state === 'running' || task.state === 'stopping'" class="pt-0">
            <v-progress-linear indeterminate color="primary" height="2" />
          </v-card-text>
          <v-card-actions class="task-card-actions">
            <v-btn
              v-if="canResumeTask(task)"
              size="x-small"
              color="primary"
              variant="text"
              prepend-icon="mdi-play-circle"
              @click="resumeTask(task)"
            >{{ t(task.state === 'paused' ? 'taskUnpause' : 'taskResume') }}</v-btn>
            <v-btn
              v-if="isTaskTerminal(task.state)"
              size="x-small"
              variant="text"
              prepend-icon="mdi-delete-outline"
              @click="deleteHistory(task)"
            >{{ t('taskDeleteHistory') }}</v-btn>
            <v-btn
              v-if="task.state === 'running'"
              size="x-small"
              color="primary"
              variant="text"
              prepend-icon="mdi-pause"
              @click="pauseTask(task)"
            >
              {{ t('taskPause') }}
            </v-btn>
            <v-btn
              v-if="isTaskStoppable(task.state)"
              size="x-small"
              color="error"
              variant="text"
              @click="cancelTask(task)"
            >
              {{ t('taskCancel') }}
            </v-btn>
            <v-spacer class="task-action-spacer" />
            <v-btn
              v-if="isTaskActive(task.state) || task.output_lines > 0 || ['qwen21-generate', 'moss-generate'].includes(task.command)"
              size="x-small"
              variant="text"
              @click="selectedTask = task.task_id"
            >
              {{ t('taskShowLogs') }}
            </v-btn>
          </v-card-actions>
        </v-card>

        <!-- Expanded card with full-height log stream -->
        <v-card
          v-else
          class="task-card mb-2 d-flex flex-column"
          :class="{ 'task-complete': task.state === 'success', 'task-failed': task.state === 'failed', 'task-running': task.state === 'running' || task.state === 'stopping' }"
          variant="tonal"
          :color="stateColor(task.state)"
          style="flex: 1 1 0; min-height: 0;"
        >
          <v-card-title class="task-card-heading text-body-2 d-flex align-center flex-shrink-0 py-2">
            <v-icon :icon="stateIcon(task.state)" size="small" class="task-card-state-icon mr-2" />
            <span class="task-card-command text-truncate">{{ task.command }}</span>
            <div class="task-card-badges">
              <v-chip v-if="task.command.startsWith('moss-') || task.command.startsWith('qwen21-') || task.category === 'training'" size="x-small" variant="tonal">{{ task.command.startsWith('moss-') ? 'MOSS' : task.command.startsWith('qwen21-') ? 'Qwen-Image 2.1' : 'Anima' }}</v-chip>
              <v-chip
                v-if="task.state === 'pending' && task.queue_position != null"
                size="x-small"
                variant="tonal"
                color="info"
              >
                #{{ task.queue_position }}
              </v-chip>
              <v-chip size="x-small" variant="outlined">{{ stateLabel(task.state) }}</v-chip>
              <v-chip v-if="task.legacy" size="x-small" color="warning" variant="tonal">
                {{ t('taskLegacy') }}
              </v-chip>
            </div>
          </v-card-title>
          <v-card-subtitle class="task-card-subtitle text-caption flex-shrink-0">
            {{ task.task_id }} &middot; PID {{ task.pid ?? '—' }}
            <span v-if="task.recovery_step != null"> &middot; step {{ task.recovery_step }}</span>
            <span v-if="task.last_progress?.step != null"> &middot; {{ t('taskLastProgress') }} {{ task.last_progress.step }}<span v-if="task.last_progress.total_steps != null">/{{ task.last_progress.total_steps }}</span></span>
          </v-card-subtitle>
          <div v-if="task.terminal_reason" class="text-caption text-medium-emphasis px-2">
            {{ t('taskTerminalReason') }}: {{ task.terminal_reason }}
          </div>
          <div v-if="task.legacy" class="text-caption text-warning px-2">
            {{ t('taskLegacyDescription') }}
          </div>
          <TaskAttemptHistory :task="task" />
          <v-progress-linear v-if="task.state === 'running' || task.state === 'stopping'" indeterminate color="primary" height="2" class="flex-shrink-0" />
          <v-card-text class="task-detail-content d-flex flex-column pa-2" style="flex: 1 1 0; min-height: 0;">
            <LogStream :task-id="task.task_id" :attempt-id="task.current_job_id" @done="onTaskDone" />
            <QwenTaskResults :family="task.command.startsWith('moss-') ? 'moss' : 'qwen21'" v-if="['qwen21-generate', 'moss-generate'].includes(task.command)" :task-id="task.task_id" :state="task.state" />
          </v-card-text>
          <v-card-actions class="task-card-actions flex-shrink-0 py-1">
            <v-btn
              v-if="canResumeTask(task)"
              size="x-small"
              color="primary"
              variant="text"
              prepend-icon="mdi-play-circle"
              @click="resumeTask(task)"
            >{{ t(task.state === 'paused' ? 'taskUnpause' : 'taskResume') }}</v-btn>
            <v-btn
              v-if="isTaskTerminal(task.state)"
              size="x-small"
              variant="text"
              prepend-icon="mdi-delete-outline"
              @click="deleteHistory(task)"
            >{{ t('taskDeleteHistory') }}</v-btn>
            <v-btn
              v-if="task.state === 'running'"
              size="x-small"
              color="primary"
              variant="text"
              prepend-icon="mdi-pause"
              @click="pauseTask(task)"
            >
              {{ t('taskPause') }}
            </v-btn>
            <v-btn
              v-if="isTaskStoppable(task.state)"
              size="x-small"
              color="error"
              variant="text"
              @click="cancelTask(task)"
            >
              {{ t('taskCancel') }}
            </v-btn>
            <v-spacer class="task-action-spacer" />
            <v-btn size="x-small" variant="text" @click="selectedTask = ''">
              {{ t('taskHideLogs') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </template>
    </div>

    <v-pagination
      v-if="historyTotal > historyPageSize"
      v-model="historyPage"
      :length="Math.ceil(historyTotal / historyPageSize)"
      density="compact"
      class="flex-shrink-0 mt-2"
      @update:model-value="loadPage"
    />

    <v-dialog :model-value="!!pendingAction" max-width="480" :persistent="actionLoading" @update:model-value="value => { if (!value && !actionLoading) pendingAction = null }">
      <v-card v-if="pendingAction">
        <v-card-title>{{ actionLabel }}</v-card-title>
        <v-card-text>
          <p class="mb-3">{{ t(pendingAction.kind === 'delete' ? 'taskDeleteHistoryConfirm' : pendingAction.kind === 'resume' ? 'taskLegacyResumeConfirm' : 'taskCancelConfirm') }}</p>
          <p class="text-caption text-medium-emphasis text-break">{{ pendingAction.task.command }} · {{ pendingAction.task.task_id }}</p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" :disabled="actionLoading" @click="pendingAction = null">{{ t('dsCancel') }}</v-btn>
          <v-btn :color="pendingAction.kind === 'resume' ? 'primary' : 'error'" :loading="actionLoading" @click="confirmAction">{{ actionLabel }}</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Shutdown daemon confirmation dialog -->
    <v-dialog v-model="showShutdownDlg" max-width="450" :persistent="shutdownLoading">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-power" color="error" class="mr-2" />
          {{ t('taskShutdownDaemon') }}
        </v-card-title>
        <v-card-text>
          <div class="text-body-2 mb-3">{{ t('taskShutdownDaemonDesc') }}</div>
          <v-select
            v-model="shutdownMode"
            :items="shutdownModes"
            item-title="title"
            item-value="value"
            :label="t('taskShutdownMode')"
            density="compact"
            hide-details
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" :disabled="shutdownLoading" @click="showShutdownDlg = false">
            {{ t('dsCancel') }}
          </v-btn>
          <v-btn
            color="error"
            :loading="shutdownLoading"
            @click="confirmShutdown"
          >
            {{ t('taskShutdownConfirm') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { canResumeTask, isTaskActive, isTaskStoppable, isTaskTerminal, taskStateColor as stateColor, taskStateIcon as stateIcon } from '../utils/taskState.ts'
import { useRoute } from 'vue-router'
import QwenTaskResults from '../components/QwenTaskResults.vue'
const route = useRoute()
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'
import { useTaskStore, mergeTaskList, type TaskFilter, type TaskInfo } from '../stores/task'
import { useI18n } from '../composables/useI18n'
import LogStream from '../components/LogStream.vue'
import TaskAttemptHistory from '../components/TaskAttemptHistory.vue'
import WorkspaceHeading from '../components/WorkspaceHeading.vue'

const taskStore = useTaskStore()
const { t } = useI18n()
const selectedTask = ref(typeof route.query.task === 'string' ? route.query.task : '')
watch(() => route.query.task, value => { selectedTask.value = typeof value === 'string' ? value : '' })
const taskFilter = ref<TaskFilter>('all')
const stateLabel = (state: TaskInfo['state']) => t(`taskState_${state}`)
const historyTasks = ref<TaskInfo[]>([])
const historyTotal = ref(0)
const historyPage = ref(1)
const historyPageSize = 25
const historyError = ref(false)
const historyLoading = ref(true)
let historyRequestId = 0
const visibleTasks = computed(() => historyTasks.value)
const displayedTasks = computed(() => {
  if (!selectedTask.value) return visibleTasks.value
  const selected = visibleTasks.value.find((task) => task.task_id === selectedTask.value)
  return selected ? [selected] : visibleTasks.value
})

async function loadPage() {
  const requestId = ++historyRequestId
  historyLoading.value = true
  try {
    const result = await taskStore.fetchTaskPage({
      state: taskFilter.value,
      page: historyPage.value,
      pageSize: historyPageSize,
    })
    if (requestId !== historyRequestId) return
    historyTasks.value = mergeTaskList(historyTasks.value, result.tasks)
    historyTotal.value = result.total
    if (selectedTask.value && !result.tasks.some(task => task.task_id === selectedTask.value)) selectedTask.value = ''
    historyError.value = false
    historyLoading.value = false
  } catch {
    if (requestId !== historyRequestId) return
    historyTasks.value = []
    historyTotal.value = 0
    historyError.value = true
    historyLoading.value = false
  }
}

watch(taskFilter, () => {
  selectedTask.value = ''
  historyPage.value = 1
  void loadPage()
})

watch(historyPage, () => { selectedTask.value = '' })

// On Windows the daemon hosts the WebUI as a sidecar and the system-tray app is
// the proper "complete exit" entry point (its Quit already shuts the daemon
// down). So the in-WebUI "Stop Daemon" button only makes sense on non-Windows
// setups where there is no tray.
const isWindows = navigator.userAgent.includes('Windows')

// ── Shutdown daemon dialog ─────────────────────────────────────
const showShutdownDlg = ref(false)
const shutdownMode = ref<'detach' | 'cooperative-stop' | 'force'>('cooperative-stop')
const shutdownModes = computed(() => [
  { title: t('taskShutdownDetach'), value: 'detach' },
  { title: t('taskShutdownCooperative'), value: 'cooperative-stop' },
  { title: t('taskShutdownForce'), value: 'force' },
])
const shutdownLoading = ref(false)

async function confirmShutdown() {
  shutdownLoading.value = true
  try {
    if (await taskStore.shutdownDaemon(shutdownMode.value)) showShutdownDlg.value = false
  } finally {
    shutdownLoading.value = false
  }
}

let pollTimer = 0

async function refreshView() {
  await taskStore.poll()
  await loadPage()
}

onMounted(() => {
  void refreshView()
  pollTimer = window.setInterval(() => void refreshView(), 5000)
})

onUnmounted(() => {
  clearInterval(pollTimer)
})

type TaskAction = { kind: 'delete' | 'cancel' | 'resume'; task: TaskInfo }
const pendingAction = ref<TaskAction | null>(null)
const actionLoading = ref(false)
const actionLabel = computed(() => t(pendingAction.value?.kind === 'delete' ? 'taskDeleteHistory' : pendingAction.value?.kind === 'resume' ? 'taskResume' : 'taskStopConfirm'))

async function resumeTask(task: TaskInfo) {
  if (task.legacy && task.state !== 'paused') pendingAction.value = { kind: 'resume', task }
  else if (await taskStore.resumeTask(task.task_id, task)) await loadPage()
}
function deleteHistory(task: TaskInfo) { pendingAction.value = { kind: 'delete', task } }
function cancelTask(task: TaskInfo) { pendingAction.value = { kind: 'cancel', task } }

async function pauseTask(task: TaskInfo) {
  if (await taskStore.pauseTask(task.task_id)) await loadPage()
}

async function confirmAction() {
  const action = pendingAction.value
  if (!action || actionLoading.value) return
  actionLoading.value = true
  try {
    const success = action.kind === 'delete'
      ? await taskStore.deleteHistory(action.task.task_id, true)
      : action.kind === 'resume'
        ? await taskStore.resumeTask(action.task.task_id, action.task, true)
        : await taskStore.cancelTask(action.task.task_id)
    if (!success) return
    pendingAction.value = null
    if (action.kind === 'delete') {
      if (selectedTask.value === action.task.task_id) selectedTask.value = ''
      const maxPage = Math.max(1, Math.ceil(Math.max(0, historyTotal.value - 1) / historyPageSize))
      historyPage.value = Math.min(historyPage.value, maxPage)
    }
    await loadPage()
  } finally {
    actionLoading.value = false
  }
}

function onTaskDone() {
  void refreshView()
}
</script>

<style scoped>
.task-toolbar { flex-shrink: 0; }
.task-detail-content { overflow-y: auto; }
.task-detail-content :deep(.log-stream) { min-height: 160px !important; }
.task-refresh-progress { flex: 0 0 2px; margin-bottom: 6px; }
.task-card { container: task-card / inline-size; }
.task-card-heading { gap: 0; }
.task-card-state-icon { flex: 0 0 auto; }
.task-card-command { flex: 1 1 auto; min-width: 0; }
.task-card-badges {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 4px 6px;
  min-width: 0;
  margin-left: auto;
  padding-left: 12px;
}
.task-card-badges :deep(.v-chip) { flex: 0 0 auto; margin-inline-start: 0 !important; }
.task-card-subtitle { color: var(--text-secondary); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.task-card-actions { min-height: 44px; }
.task-filter {
  padding: 2px;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md) !important;
  background: var(--bg-surface);
}
.task-filter :deep(.v-btn) {
  min-width: 64px;
  border-radius: var(--radius-sm) !important;
  color: var(--text-secondary);
}
.task-filter :deep(.v-btn--active) { color: rgb(var(--v-theme-on-primary)); }
@container task-card (max-width: 520px) {
  .task-card-heading {
    display: grid !important;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    row-gap: 8px;
  }
  .task-card-state-icon { margin-inline-end: 8px !important; }
  .task-card-command { min-width: 0; }
  .task-card-badges {
    grid-column: 1 / -1;
    justify-content: flex-start;
    margin-left: 0;
    padding-left: 26px;
  }
}
/* Task status left-border accents */
.task-complete {
  border-left: 3px solid var(--success) !important;
}
.task-failed {
  border-left: 3px solid var(--error) !important;
}
.task-running {
  border-left: 3px solid var(--forge-ember) !important;
}

/* Card hover */
:deep(.v-card) {
  transition: border-color var(--dur-slow) ease, box-shadow var(--dur-slow) ease;
}

@media (max-width: 600px) {
  .task-monitor {
    padding: 16px 12px !important;
  }

  .task-toolbar {
    align-items: stretch !important;
    flex-direction: column;
    flex-wrap: nowrap;
    gap: 8px;
  }

  .task-toolbar-actions :deep(.v-btn--disabled) {
    color: var(--text-muted) !important;
    opacity: 0.7;
  }

  .task-toolbar > .v-spacer {
    display: none;
  }

  .task-toolbar-actions {
    flex-wrap: wrap;
    gap: 2px 4px;
  }

  .task-toolbar-actions :deep(.v-btn) {
    margin-inline-start: 0 !important;
  }

  .task-filter {
    align-self: stretch !important;
    max-width: 100%;
    min-height: 36px;
    overflow-x: auto;
    overflow-y: hidden;
  }

  .task-card-subtitle {
    overflow: visible;
    text-overflow: clip;
    white-space: normal;
  }

  .task-card-actions {
    flex-wrap: wrap;
    gap: 2px 4px;
    justify-content: flex-start;
  }

  .task-card-actions :deep(.v-btn) {
    margin-inline-start: 0 !important;
  }

  .task-action-spacer {
    display: none;
  }
}
</style>
