import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { setImmediate } from 'node:timers/promises'
import test from 'node:test'
import { parse, compileScript } from '@vue/compiler-sfc'
import * as vue from 'vue'
import ts from 'typescript'
import { modelPathMustExist } from '../src/utils/modelConfig.ts'

const source = readFileSync(new URL('../src/components/ModelPathField.vue', import.meta.url), 'utf8')
const compiled = compileScript(parse(source).descriptor, { id: 'output-path-regression' }).content
const code = ts.transpileModule(compiled, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText

function setup(t, section, name, value) {
  const cleanup = [], calls = [], exports = {}
  t.mock.timers.enable({ apis: ['setTimeout'] })
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url)
    return new Response(JSON.stringify({ exists: false }))
  })
  new Function('require', 'exports', code)(name => {
    if (name === 'vue') return { ...vue, onBeforeUnmount: fn => cleanup.push(fn) }
    if (name.endsWith('.vue')) return { default: {} }
    if (name === '../stores/config') return { useConfigStore: () => ({ getFieldValue: () => '' }) }
    if (name === '../composables/useI18n') return { useI18n: () => ({ t: key => key }) }
    throw new Error(`Unexpected import: ${name}`)
  }, exports)
  const props = vue.reactive({ modelValue: value, field: { key: `${section}.${name}`, path_kind: 'file', path_must_exist: modelPathMustExist(section, name), read_only: false } })
  const scope = vue.effectScope()
  const field = scope.run(() => exports.default.setup(props, { expose() {}, emit() {} }))
  t.after(() => { for (const fn of cleanup) fn(); scope.stop() })
  return { props, field, calls, flush: async () => { t.mock.timers.tick(1000); await setImmediate() } }
}

for (const [section, name] of [['train', 'output'], ['tune', 'output'], ['cache', 'out'], ['generate', 'out_dir']]) {
  test(`${section}.${name} does not warn or probe when a destination has not been created`, async t => {
    const { field, calls, flush } = setup(t, section, name, 'output/new-name.safetensors')
    await flush()
    assert.deepEqual(calls, [])
    assert.equal(field.exists.value, null)
    assert.equal(field.validating.value, false)
    assert.equal(field.hintText.value, 'cfOutputDestinationHint')
  })
}

test('missing input weights still trigger existence validation', async t => {
  const { field, calls, flush } = setup(t, 'train', 'dit', 'missing.safetensors')
  await flush()
  assert.equal(calls.length, 1)
  assert.match(calls[0], /\/api\/files\/validate\?path=missing/)
  assert.equal(field.exists.value, false)
  assert.equal(field.requiresExisting.value, true)
})

test('changing an input into an output clears its old warning', async t => {
  const { field, props, flush } = setup(t, 'train', 'dit', 'missing.safetensors')
  await flush()
  assert.equal(field.exists.value, false)
  props.field.path_must_exist = false
  assert.equal(field.exists.value, null)
  assert.equal(field.validating.value, false)
  assert.equal(field.requiresExisting.value, false)
})
