<template>
  <div ref="logContainer" class="log-stream pa-2" tabindex="0" role="log" aria-live="off"
    :aria-label="t('dashLiveLog')" :aria-busy="!done && !connected" @scroll="trackScroll">
    <div class="log-status">
      <span class="log-status__dot" :class="{ 'is-live': connected && !done }" aria-hidden="true" />
      <span>{{ t(done ? 'taskLogComplete' : connected ? 'taskLogLive' : 'taskConnecting') }}</span>
    </div>
    <div v-if="truncatedNote" class="log-truncate-note">
      {{ t('taskLogTruncated', { total: historyTotal, kept: lines.length }) }}
    </div>
    <div v-if="lines.length === 0" class="log-empty">
      {{ t(done ? 'taskNoLogs' : 'taskConnecting') }}
    </div>
    <div
      v-for="ln in lines"
      :key="ln.seq"
      class="log-line"
      :class="ln.level"
    >{{ ln.text }}</div>
    <button v-if="!followTail && lines.length > 0" type="button" class="log-follow" @click="followLatest">
      <v-icon icon="mdi-arrow-down" size="14" />{{ t('taskLogFollow') }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed, watch, nextTick, ref, onUnmounted } from 'vue'
import { useTaskStream, type LogStreamHandle } from '../composables/useTask'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{
  taskId: string
  attemptId?: string | null
  /** Existing stream handle (e.g. the dashboard's useTrainingStream) —
   * when provided, no second WebSocket is opened for the same task. */
  external?: LogStreamHandle | null
}>()
const emit = defineEmits<{ done: [] }>()
const logContainer = ref<HTMLElement>()
const { t } = useI18n()

const own = props.external ? null : useTaskStream(() => props.taskId, () => props.attemptId ?? '')

const stream = computed<LogStreamHandle | null>(() => props.external ?? own)
const lines = computed(() => stream.value?.lines.value ?? [])
const connected = computed(() => stream.value?.connected.value ?? false)
const done = computed(() => stream.value?.done.value ?? false)
const historyTotal = computed(() => stream.value?.historyTotal.value ?? 0)
const truncatedNote = computed(() => historyTotal.value > lines.value.length && lines.value.length > 0)

// Autoscroll per coalesced flush — but only while the user is pinned to the
// bottom, so reading up through a busy log isn't a losing fight with the
// scroll position.
const followTail = ref(true)
function trackScroll() {
  const el = logContainer.value
  if (el) followTail.value = el.scrollHeight - el.scrollTop - el.clientHeight < 48
}
watch([() => props.taskId, () => props.attemptId], () => { followTail.value = true })
watch(() => stream.value?.activity.value ?? 0, async () => {
  await nextTick()
  const el = logContainer.value
  if (!el) return
  if (followTail.value) el.scrollTop = el.scrollHeight
}, { immediate: true })

function followLatest() {
  followTail.value = true
  const el = logContainer.value
  if (!el) return
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  el.scrollTo({ top: el.scrollHeight, behavior: reducedMotion ? 'auto' : 'smooth' })
}

watch(() => stream.value?.done.value ?? false, (val) => {
  if (val) emit('done')
})

onUnmounted(() => own?.disconnect())
</script>

<style scoped>
.log-stream {
  position: relative;
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  font-family: var(--font-mono);
  font-size: var(--fs-caption);
  background: var(--bg-deep);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
}
.log-status {
  position: sticky;
  top: -8px;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  margin: -8px -8px 8px;
  padding: 6px 8px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--bg-deep);
  color: var(--text-muted);
  font-size: var(--fs-micro);
}
.log-status__dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); }
.log-status__dot.is-live { background: var(--success); }
.log-empty { padding: 16px 4px; color: var(--text-secondary); line-height: 1.6; }
@media (pointer: coarse) {
  .log-follow { min-height: 44px; }
}
.log-follow {
  position: sticky;
  bottom: 0;
  z-index: 2;
  display: flex;
  align-items: center;
  gap: 4px;
  min-height: 32px;
  margin: 8px 0 0 auto;
  padding: 4px 10px;
  border: 1px solid var(--forge-ember);
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  color: var(--forge-ember);
  font-family: var(--font-sans);
  font-size: var(--fs-caption);
  cursor: pointer;
}
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
  line-height: 1.5;
  color: var(--text-secondary);
  font-size: var(--fs-caption);
  padding: 1px 0;
  animation: logEntry 0.2s ease-out both;
}

@keyframes logEntry {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}

.log-truncate-note {
  padding: 4px 6px;
  margin-bottom: 4px;
  font-size: var(--fs-micro);
  color: var(--text-muted);
  border-bottom: 1px dashed var(--border-subtle);
}

.log-info { color: var(--info); }
.log-warn { color: var(--warning); }
.log-error { color: var(--error); font-weight: 500; }
</style>
