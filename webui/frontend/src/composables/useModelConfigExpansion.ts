import { computed, ref, watch, type Ref } from 'vue'

/** Search opens matches once, then respects the user's collapse choices. */
export function useModelConfigExpansion(search: Ref<string | null>, groupIds: Ref<string[]>) {
  const normalOpen = ref<string[]>(['basic'])
  const searchOpen = ref<string[]>([])
  const query = computed(() => search.value?.trim() ?? '')
  watch([query, groupIds], ([text, ids], [previousText, previousIds]) => {
    if (!text) return
    if (text !== previousText) searchOpen.value = [...ids]
    else {
      const old = new Set(previousIds)
      searchOpen.value = [...searchOpen.value.filter(id => ids.includes(id)), ...ids.filter(id => !old.has(id))]
    }
  })
  const expandedGroups = computed(() => query.value ? searchOpen.value : normalOpen.value)
  function onExpandedGroups(value: unknown): void {
    if (!Array.isArray(value)) return
    const target = query.value ? searchOpen : normalOpen
    target.value = value.map(String)
  }
  return { expandedGroups, onExpandedGroups }
}
