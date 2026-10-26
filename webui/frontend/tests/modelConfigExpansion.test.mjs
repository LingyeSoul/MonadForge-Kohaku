import assert from 'node:assert/strict'
import test from 'node:test'
import { effectScope, nextTick, ref } from 'vue'
import { useModelConfigExpansion } from '../src/composables/useModelConfigExpansion.ts'

function setup(t) {
  const search = ref(''), groups = ref(['basic', 'training', 'generate'])
  const scope = effectScope()
  t.after(() => scope.stop())
  return { search, groups, ...scope.run(() => useModelConfigExpansion(search, groups)) }
}

test('basic settings start open and can be collapsed and reopened', t => {
  const ui = setup(t)
  assert.deepEqual(ui.expandedGroups.value, ['basic'])
  ui.onExpandedGroups([])
  assert.deepEqual(ui.expandedGroups.value, [])
  ui.onExpandedGroups(['basic', 'training'])
  assert.deepEqual(ui.expandedGroups.value, ['basic', 'training'])
})

test('search opens matches once and never undoes a manual collapse on field edits', async t => {
  const ui = setup(t)
  ui.search.value = 'seed'
  ui.groups.value = ['training', 'generate']
  await nextTick()
  assert.deepEqual(ui.expandedGroups.value, ['training', 'generate'])
  ui.onExpandedGroups(['generate'])
  ui.groups.value = ['training', 'generate']
  await nextTick()
  assert.deepEqual(ui.expandedGroups.value, ['generate'])
  ui.onExpandedGroups([])
  ui.search.value = 'seed '
  await nextTick()
  assert.deepEqual(ui.expandedGroups.value, [])
  ui.search.value = 'output'
  await nextTick()
  assert.deepEqual(ui.expandedGroups.value, ['training', 'generate'])
})

test('clearing search restores pre-search choices and new matches open independently', async t => {
  const ui = setup(t)
  ui.onExpandedGroups(['training'])
  ui.search.value = 'cache'
  ui.groups.value = ['basic']
  await nextTick()
  ui.onExpandedGroups([])
  ui.groups.value = ['basic', 'tune']
  await nextTick()
  assert.deepEqual(ui.expandedGroups.value, ['tune'])
  ui.search.value = null
  ui.groups.value = ['basic', 'training', 'generate']
  await nextTick()
  assert.deepEqual(ui.expandedGroups.value, ['training'])
})
