import assert from 'node:assert/strict'
import test from 'node:test'
import { createPinia, setActivePinia } from 'pinia'
import { initialDraft, parseSchema, readDraft, resolvedPaths, resultImages } from '../src/utils/qwen21.ts'
import { useQwenWorkspace, useMossWorkspace } from '../src/stores/qwenWorkspace.ts'
import { taskModelFamily, useModelWorkspace } from '../src/stores/modelWorkspace.ts'
import { modelFieldLabel, mossText } from '../src/i18n/modelWorkspace.ts'

function field(name, value) {
  return { name, kind: typeof value === 'number' ? 'int' : 'str', value, nullable: false, choices: [], help: name, advanced: false, multiline: false }
}
function schema(family) {
  const model = field('model_dir', `/models/${family}`)
  return { model_field_names: ['model_dir'], train: [model, field('rank', 32)], cache: [model, field('src', `/data/${family}`)], generate: [model, field('prompt', family)], ...(family === 'moss' ? { tune: [model, field('rank', 64), field('margin_gb', 2)] } : {}) }
}
function setup(t) {
  const entries = new Map()
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: key => entries.get(key) ?? null, setItem: (key, value) => entries.set(key, String(value)), removeItem: key => entries.delete(key) } })
  t.after(() => { if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor); else delete globalThis.localStorage })
  setActivePinia(createPinia())
  return entries
}
const response = value => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })

test('action applicability follows final overrides and never overwrites hidden profile values', async t => {
  setup(t)
  const calls = []
  const boolField = (name, value) => ({ ...field(name, value), kind: 'bool' })
  const spec = {
    model_field_names: ['model_dir'],
    train: [field('model_dir', '/models/qwen21'), field('algo', 'lora'), field('factor', -1), field('down_init', 'kaiming')],
    cache: [field('model_dir', '/models/qwen21'), boolField('skip_text', false), boolField('skip_latents', false), field('te_blocks_to_swap', 0)],
    generate: [field('model_dir', '/models/qwen21'), field('lora', ''), field('multipliers', '1.0,0.0'), field('true_cfg_scale', 1), field('negative_prompt', '')],
  }
  const stored = initialDraft(parseSchema(spec))
  stored.train.factor = -2 // old unused value must not block plain LoRA submission
  stored.generate.multipliers = '0.75,0.0'
  stored.generate.negative_prompt = 'saved negative'
  stored.cache.skip_text = true
  stored.cache.te_blocks_to_swap = 12
  t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
    calls.push({ url, init })
    if (url === '/api/system/runtimes') return response({ runtimes: [{ name: 'default', available: true, python: '/python/default', error: null }] })
    if (url.endsWith('/schema')) return response(spec)
    if (url.endsWith('/profiles')) return response(['default'])
    if (url.endsWith('/profiles/default')) return response(stored)
    if (url.includes('/jobs/')) return response({ task_id: 'isolated-job' })
    throw new Error(`Unexpected URL ${url}`)
  })
  const workspace = useQwenWorkspace()
  await workspace.ensureLoaded()
  const original = JSON.stringify(workspace.draft)
  await workspace.submit('train', {})
  await workspace.submit('generate', {})
  await workspace.submit('generate', { lora: 'selected.safetensors', true_cfg_scale: 2 })
  await workspace.submit('cache', { skip_text: false, skip_latents: true })
  const payloads = calls.filter(call => call.url.includes('/jobs/')).map(call => JSON.parse(call.init.body).values)
  assert.equal(payloads[0].factor, -1)
  assert.equal(payloads[1].multipliers, '0.0')
  assert.equal(payloads[1].negative_prompt, '')
  assert.equal(payloads[2].multipliers, '0.75,0.0')
  assert.equal(payloads[2].negative_prompt, 'saved negative')
  assert.equal(payloads[3].te_blocks_to_swap, 12)
  assert.equal(JSON.stringify(workspace.draft), original)
})

test('MOSS tuning values survive profile load, edit and serialization', () => {
  const parsed = parseSchema(schema('moss'))
  const draft = initialDraft(parsed)
  draft.tune.rank = 80
  draft.tune.margin_gb = 4
  const loaded = readDraft(parsed, JSON.stringify(draft))
  loaded.train.rank = 16
  assert.deepEqual(loaded.tune, { rank: 80, margin_gb: 4 })
  assert.deepEqual(readDraft(parsed, JSON.stringify(loaded)).tune, loaded.tune)
  assert.equal(initialDraft(parseSchema(schema('qwen21'))).tune, undefined)
})

test('each line splits model fields by its own schema rule', () => {
  const moss = parseSchema({
    model_field_names: ['model_dir'],
    train: [field('model_dir', '/m'), field('dit', '/m/dit'), field('rank', 32)],
    cache: [field('model_dir', '/m'), field('text_encoder', 'int8'), field('src', '/data')],
    generate: [field('model_dir', '/m'), field('text_encoder', 'int8'), field('prompt', 'sky')],
  })
  const draft = initialDraft(moss)
  assert.deepEqual(draft.models, { model_dir: '/m' })
  assert.equal(draft.train.dit, '/m/dit')
  assert.equal(draft.cache.text_encoder, 'int8')
  assert.equal(draft.generate.text_encoder, 'int8')
  assert.deepEqual(readDraft(moss, JSON.stringify(draft)), draft)
  const qwen = parseSchema({
    model_field_names: ['model_dir', 'dit'],
    train: [field('model_dir', '/q'), field('dit', '/q/dit'), field('rank', 32)],
    cache: [], generate: [],
  })
  assert.deepEqual(initialDraft(qwen).models, { model_dir: '/q', dit: '/q/dit' })
  assert.equal(initialDraft(qwen).train.dit, undefined)
})

test('a saved value missing from the form names the field instead of faking a number error', () => {
  const parsed = parseSchema({ model_field_names: [], train: [field('rank', 32)], cache: [], generate: [] })
  assert.throws(() => readDraft(parsed, JSON.stringify({ models: {}, train: {}, cache: {}, generate: {} })), /rank: missing from the form values/)
  assert.throws(() => parseSchema({ model_field_names: ['dit'], train: [field('rank', 32)], cache: [], generate: [] }), /Model fields missing from the train form: dit/)
})

test('model workspaces isolate profiles and route cache and training through their own API', async t => {
  setup(t)
  const calls = []
  const profiles = Object.fromEntries(['moss', 'qwen21'].map(family => [family, initialDraft(parseSchema(schema(family)))]))
  profiles.moss.tune.rank = 88
  t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
    calls.push({ url, init })
    if (url === '/api/system/runtimes') return response({ runtimes: [{ name: 'default', available: true, python: '/python/default', error: null }, { name: 'curation', available: true, python: '/python/curation', error: null }, { name: 'v100', available: false, python: null, error: 'V100 is not registered' }] })
    const family = url.split('/')[2]
    if (url.endsWith('/schema')) return response(schema(family))
    if (url.endsWith('/profiles')) return response(['default'])
    if (url.endsWith('/profiles/default')) {
      if (init.method === 'PUT') profiles[family] = JSON.parse(init.body)
      return response(profiles[family])
    }
    if (url.includes('/jobs/')) return response({ task_id: `${family}-job` })
    throw new Error(`Unexpected URL ${url}`)
  })
  const moss = useMossWorkspace(), qwen = useQwenWorkspace()
  await Promise.all([moss.ensureLoaded(), qwen.ensureLoaded()])
  moss.update('train', 'rank', 8)
  assert.equal(qwen.draft.train.rank, 32)
  await moss.save()
  assert.equal(profiles.moss.tune.rank, 88)
  assert.equal(profiles.qwen21.models.model_dir, '/models/qwen21')
  const workspace = useModelWorkspace()
  await workspace.loadRuntimes()
  workspace.selectRuntime('curation')
  assert.throws(() => workspace.selectRuntime('v100'), /not registered/)
  assert.equal(workspace.runtimeProfile, 'curation')
  assert.equal(await moss.submit('train', {}), 'moss-job')
  assert.equal(await qwen.submit('cache', {}), 'qwen21-job')
  const train = calls.find(call => call.url === '/api/moss/jobs/train')
  assert.deepEqual(JSON.parse(train.init.body), { values: { model_dir: '/models/moss', rank: 8 }, runtime_profile: 'curation' })
  const cache = calls.find(call => call.url === '/api/qwen21/jobs/cache')
  assert.deepEqual(JSON.parse(cache.init.body), { values: { model_dir: '/models/qwen21', src: '/data/qwen21' }, runtime_profile: 'curation' })
})

test('concurrent submit clicks are guarded while a model action is pending', async t => {
  setup(t)
  const moss = useMossWorkspace()
  let resolve, actions = 0
  const first = moss.perform(async () => { actions++; await new Promise(done => { resolve = done }) })
  await moss.perform(async () => { actions++ })
  assert.equal(actions, 1)
  assert.equal(moss.busy, true)
  resolve()
  await first
  assert.equal(moss.busy, false)
})

test('saved invalid model selection does not blank the app and task labels stay task-bound', t => {
  const storage = setup(t)
  storage.set('monadforge-model-family', 'removed-experiment')
  const workspace = useModelWorkspace()
  assert.equal(workspace.family, 'anima')
  workspace.select('moss')
  assert.equal(storage.get('monadforge-model-family'), 'moss')
  assert.equal(taskModelFamily('qwen21-generate'), 'qwen21')
  assert.equal(taskModelFamily('moss-cache-train'), 'moss')
  assert.equal(taskModelFamily('lora'), 'anima')
})

test('MOSS outputs retain their model identity without inventing Qwen multipliers', () => {
  const moss = resultImages({ images: [{ file: 'one.png', seed: 2, prompt: 'sky', model: 'edit' }] }, 'moss')[0]
  assert.equal(moss.multiplier, null)
  assert.equal(moss.model, 'edit')
  const qwen = resultImages({ images: [{ file: 'one.png', seed: 2, prompt: 'sky', multiplier: 0 }] })[0]
  assert.equal(qwen.multiplier, 0)
  assert.deepEqual(resolvedPaths({ dit: '/m/dit', text_encoder: '/m/te', vae: '/m/vae', tokenizer: '/m/tok' }, 'moss'), { dit: '/m/dit', text_encoder: '/m/te', vae: '/m/vae', tokenizer: '/m/tok' })
  assert.equal(modelFieldLabel('moss', 'cn', 'algo', 'qwenField_algo'), '适配器算法')
  assert.match(mossText('cn', 'cacheHint'), /MOSS/)
  assert.doesNotMatch(mossText('cn', 'cacheHint'), /RGBA/)
})

test('unavailable saved V100 runtime blocks submission without falling back', async t => {
  const storage = setup(t)
  storage.set('monadforge-runtime-profile', 'v100')
  const calls = []
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url)
    return response({ runtimes: [{ name: 'default', available: true, python: '/python/default', error: null }, { name: 'v100', available: false, python: null, error: 'V100 missing' }] })
  })
  const workspace = useModelWorkspace()
  const moss = useMossWorkspace()
  moss.schema = parseSchema(schema('moss'))
  moss.draft = initialDraft(moss.schema)
  await assert.rejects(moss.submit('generate', {}), /V100 missing/)
  assert.equal(workspace.runtimeProfile, 'v100')
  assert.deepEqual(calls, ['/api/system/runtimes'])
})

function configWorkspace(t) {
  const storage = setup(t)
  const schemas = Object.fromEntries(['qwen21', 'moss'].map(family => {
    const value = schema(family)
    value.train.push({ ...field('algo', 'lora'), choices: ['lora', 'lokr', 'loha'] })
    return [family, value]
  }))
  const documents = Object.fromEntries(Object.entries(schemas).map(([family, value]) => {
    const defaultDraft = initialDraft(parseSchema(value))
    const kr = structuredClone(defaultDraft)
    kr.train.algo = 'lokr'
    kr.train.rank = 48
    return [family, { default: defaultDraft, kr }]
  }))
  const calls = []
  const failures = new Set()
  t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
    calls.push({ url, init })
    if (failures.has(`${init.method ?? 'GET'} ${url}`)) return new Response(JSON.stringify({ detail: 'profile unavailable' }), { status: 503 })
    const [, , family, resource, name] = url.split('/')
    if (resource === 'schema') return response(schemas[family])
    if (resource === 'profiles' && !name) return response(Object.keys(documents[family]))
    if (resource === 'profiles' && name) {
      if (init.method === 'PUT') documents[family][name] = JSON.parse(init.body)
      return response(documents[family][name])
    }
    throw new Error(`Unexpected URL ${url}`)
  })
  return { storage, documents, calls, failures, qwen: useQwenWorkspace(), moss: useMossWorkspace() }
}

test('configuration catalog reads method ownership without selecting or replacing the working draft', async t => {
  const { qwen, calls } = configWorkspace(t)
  await qwen.ensureLoaded()
  qwen.update('train', 'rank', 99)
  const currentDraft = qwen.draft
  await qwen.loadProfileCatalog()
  assert.deepEqual(qwen.profileAlgorithms, { default: 'lora', kr: 'lokr' })
  assert.equal(qwen.configMethod, '')
  assert.equal(qwen.configProfile, '')
  assert.equal(qwen.draft, currentDraft)
  assert.equal(qwen.draft.train.rank, 99)
  assert.equal(qwen.dirty, true)
  qwen.selectConfigMethod('lokr')
  assert.equal(qwen.configMethod, 'lokr')
  assert.equal(qwen.configProfile, '')
  assert.equal(qwen.draft, currentDraft)
  assert.equal(calls.some(call => call.init.method === 'PUT'), false)
  assert.throws(() => qwen.selectConfigMethod('unknown'), /Unsupported/)
})

test('method selection accepts only matching profiles and isolates each model workspace', async t => {
  const { qwen, moss, calls } = configWorkspace(t)
  await Promise.all([qwen.loadProfileCatalog(), moss.loadProfileCatalog()])
  qwen.selectConfigMethod('lokr')
  const beforeMismatch = calls.length
  await assert.rejects(qwen.selectConfigProfile('default'), /does not belong/)
  assert.equal(calls.length, beforeMismatch)
  assert.equal(qwen.configProfile, '')
  await qwen.selectConfigProfile('kr')
  assert.equal(qwen.configProfile, 'kr')
  assert.equal(qwen.profile, 'kr')
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.dirty, false)
  assert.equal(moss.configMethod, '')
  assert.equal(moss.configProfile, '')
  assert.equal(moss.draft.train.algo, 'lora')
  assert.throws(() => qwen.update('train', 'algo', 'lora'), /Choose a method/)
})

test('a profile changed on disk cannot silently replace the selected method or draft', async t => {
  const { qwen, documents, storage, failures } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  const previous = qwen.draft
  documents.qwen21.kr.train.algo = 'lora'
  await assert.rejects(qwen.selectConfigProfile('kr'), /does not match/)
  assert.equal(qwen.draft, previous)
  assert.equal(qwen.profile, 'default')
  assert.equal(qwen.configMethod, 'lokr')
  assert.equal(qwen.configProfile, '')
  assert.equal(storage.get('monadforge-qwen-profile'), 'default')
  documents.qwen21.kr.train.algo = 'lokr'
  failures.add('GET /api/qwen21/profiles/kr')
  await assert.rejects(qwen.selectConfigProfile('kr'), /profile unavailable/)
  assert.equal(qwen.draft, previous)
  assert.equal(qwen.configProfile, '')
})

test('reset and discard preserve the selected method for saved and new configurations', async t => {
  const { qwen } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  await qwen.selectConfigProfile('kr')
  qwen.resetDefaults()
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.draft.train.rank, 32)
  assert.equal(qwen.dirty, true)
  qwen.discard()
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.draft.train.rank, 48)
  assert.equal(qwen.dirty, false)
  qwen.selectConfigMethod('loha')
  qwen.beginConfigProfile('new-ha')
  qwen.update('train', 'rank', 72)
  qwen.discard()
  assert.equal(qwen.draft.train.algo, 'loha')
  assert.equal(qwen.draft.train.rank, 32)
  assert.equal(qwen.configProfile, 'new-ha')
})

test('creating a profile uses chosen method defaults or copies an explicitly selected configuration', async t => {
  const { qwen, documents, calls } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.update('train', 'rank', 99)
  qwen.selectConfigMethod('lokr')
  await qwen.createConfigProfile('new-kr')
  assert.equal(qwen.draft.train.rank, 32)
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.configProfile, 'new-kr')
  assert.equal(qwen.profileAlgorithms['new-kr'], 'lokr')
  assert.equal(qwen.dirty, false)
  assert.equal(documents.qwen21.default.train.algo, 'lora')
  assert.equal(documents.qwen21.default.train.rank, 32)
  qwen.update('train', 'rank', 61)
  await qwen.createConfigProfile('copy-kr')
  assert.equal(documents.qwen21['copy-kr'].train.rank, 61)
  assert.equal(documents.qwen21['copy-kr'].train.algo, 'lokr')
  assert.equal(calls.filter(call => call.init.method === 'PUT').length, 2)
  await assert.rejects(qwen.createConfigProfile('default'), /already exists/)
  assert.equal(qwen.profile, 'copy-kr')
  await qwen.createConfigProfile('fresh-kr', false)
  assert.equal(documents.qwen21['fresh-kr'].train.rank, 32)
  assert.equal(documents.qwen21['fresh-kr'].train.algo, 'lokr')
})

test('failed new-profile saves restore unsaved edits and all prior selection state', async t => {
  const { qwen, failures, storage } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  await qwen.selectConfigProfile('kr')
  qwen.update('train', 'rank', 71)
  const previousDraft = qwen.draft
  const previousSaved = qwen.saved
  failures.add('PUT /api/qwen21/profiles/copy-kr')
  await assert.rejects(qwen.createConfigProfile('copy-kr'), /profile unavailable/)
  assert.equal(qwen.draft, previousDraft)
  assert.equal(qwen.draft.train.rank, 71)
  assert.equal(qwen.saved, previousSaved)
  assert.equal(qwen.profile, 'kr')
  assert.equal(qwen.configMethod, 'lokr')
  assert.equal(qwen.configProfile, 'kr')
  assert.equal(qwen.dirty, true)
  assert.equal(qwen.profiles.includes('copy-kr'), false)
  assert.equal(qwen.profileAlgorithms['copy-kr'], undefined)
  assert.equal(storage.get('monadforge-qwen-profile'), 'kr')
})

test('catalog failures remain visible and preserve the previous usable catalog and draft', async t => {
  const { qwen, failures } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  const previousCatalog = qwen.profileAlgorithms
  const previousDraft = qwen.draft
  failures.add('GET /api/qwen21/profiles/kr')
  await assert.rejects(qwen.loadProfileCatalog(), /profile unavailable/)
  assert.equal(qwen.profileAlgorithms, previousCatalog)
  assert.equal(qwen.draft, previousDraft)
})

test('switching methods while a profile loads prevents stale selection and draft replacement', async t => {
  const { qwen, documents } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  const previousDraft = qwen.draft
  const originalFetch = globalThis.fetch
  let finishLoad
  t.mock.method(globalThis, 'fetch', (url, init) => {
    if (url === '/api/qwen21/profiles/kr') return new Promise(resolve => { finishLoad = resolve })
    return originalFetch(url, init)
  })
  const loading = qwen.selectConfigProfile('kr')
  qwen.selectConfigMethod('loha')
  finishLoad(response(documents.qwen21.kr))
  await loading
  assert.equal(qwen.configMethod, 'loha')
  assert.equal(qwen.configProfile, '')
  assert.equal(qwen.draft, previousDraft)
  assert.equal(qwen.profile, 'default')
})

function deferred() {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

test('a late save of profile A cannot replace profile B or assign A to B in the catalog', async t => {
  const { moss, documents, storage } = configWorkspace(t)
  await moss.loadProfileCatalog()
  moss.selectConfigMethod('lora')
  await moss.selectConfigProfile('default')
  moss.update('train', 'rank', 19)
  const gate = deferred(), originalFetch = globalThis.fetch
  let sent
  t.mock.method(globalThis, 'fetch', (url, init) => {
    if (url === '/api/moss/profiles/default' && init?.method === 'PUT') {
      sent = JSON.parse(init.body)
      return gate.promise
    }
    return originalFetch(url, init)
  })
  const saving = moss.save()
  moss.selectConfigMethod('lokr')
  await moss.selectConfigProfile('kr')
  gate.resolve(response(sent))
  await saving
  assert.equal(sent.train.rank, 19)
  assert.equal(moss.profile, 'kr')
  assert.equal(moss.configProfile, 'kr')
  assert.deepEqual(moss.draft, documents.moss.kr)
  assert.equal(moss.dirty, false)
  assert.equal(moss.profileAlgorithms.kr, 'lokr')
  assert.equal(storage.get('monadforge-moss-profile'), 'kr')
})

test('editing while save is pending preserves the newer draft as unsaved', async t => {
  const { moss } = configWorkspace(t)
  await moss.loadProfileCatalog()
  const gate = deferred(), originalFetch = globalThis.fetch
  let sent
  t.mock.method(globalThis, 'fetch', (url, init) => {
    if (init?.method === 'PUT') { sent = JSON.parse(init.body); return gate.promise }
    return originalFetch(url, init)
  })
  moss.update('train', 'rank', 16)
  const saving = moss.save()
  moss.update('train', 'rank', 24)
  gate.resolve(response(sent))
  await saving
  assert.equal(moss.draft.train.rank, 24)
  assert.equal(JSON.parse(moss.saved).train.rank, 16)
  assert.equal(moss.dirty, true)
})

test('failed creation of A cannot roll back a newer selection of B', async t => {
  const { moss, documents } = configWorkspace(t)
  await moss.loadProfileCatalog()
  moss.selectConfigMethod('lora')
  const gate = deferred(), originalFetch = globalThis.fetch
  t.mock.method(globalThis, 'fetch', (url, init) => {
    if (url.endsWith('/new-profile') && init?.method === 'PUT') return gate.promise
    return originalFetch(url, init)
  })
  const creating = moss.createConfigProfile('new-profile')
  moss.selectConfigMethod('lokr')
  await moss.selectConfigProfile('kr')
  gate.reject(new Error('write failed'))
  await assert.rejects(creating, /write failed/)
  assert.equal(moss.profile, 'kr')
  assert.equal(moss.configMethod, 'lokr')
  assert.deepEqual(moss.draft, documents.moss.kr)
})

test('latest profile load wins when two responses arrive in reverse order', async t => {
  const { moss, documents } = configWorkspace(t)
  await moss.loadProfileCatalog()
  const a = deferred(), b = deferred()
  t.mock.method(globalThis, 'fetch', url => url.endsWith('/default') ? a.promise : b.promise)
  const first = moss.loadProfile('default'), second = moss.loadProfile('kr')
  assert.equal(moss.loading, true)
  b.resolve(response(documents.moss.kr))
  await second
  a.resolve(response(documents.moss.default))
  await first
  assert.equal(moss.profile, 'kr')
  assert.equal(moss.loading, false)
  assert.deepEqual(moss.draft, documents.moss.kr)
})

for (const phase of ['cache', 'train', 'generate', 'tune', 'chain']) {
  test(`${phase} submission snapshots its own stage and runtime before async work`, async t => {
    setup(t)
    const moss = useMossWorkspace(), workspace = useModelWorkspace()
    // The same keys intentionally carry different values in every stage.
    const spec = { model_field_names: ['model_dir'] }
    const stages = ['cache', 'train', 'generate', 'tune']
    stages.forEach((stage, i) => { spec[stage] = [field('model_dir', '/moss-A'), field('seed', i + 1), field('output', `${stage}-A`)] })
    moss.schema = parseSchema(spec)
    moss.draft = initialDraft(moss.schema)
    const original = structuredClone(JSON.parse(JSON.stringify(moss.draft)))
    const gate = deferred(), calls = []
    t.mock.method(globalThis, 'fetch', async (url, init) => {
      if (url === '/api/system/runtimes') return gate.promise
      calls.push({ url, body: JSON.parse(init.body) })
      return response({ task_id: 'test-job' })
    })
    const submission = phase === 'chain' ? moss.chain() : moss.submit(phase, {})
    moss.profile = 'profile-B'
    moss.update('models', 'model_dir', '/moss-B')
    for (const stage of stages) moss.update(stage, 'seed', 100)
    workspace.runtimeProfile = 'curation'
    workspace.select('qwen21')
    gate.resolve(response({ runtimes: [
      { name: 'default', available: true, python: '/python/default', error: null },
      { name: 'curation', available: true, python: '/python/curation', error: null },
    ] }))
    await submission
    assert.equal(calls.length, 1)
    assert.equal(calls[0].url, phase === 'chain' ? '/api/moss/chain' : `/api/moss/jobs/${phase}`)
    assert.equal(calls[0].body.runtime_profile, 'default')
    for (const stage of phase === 'chain' ? ['cache', 'train'] : [phase]) {
      const values = phase === 'chain' ? calls[0].body[stage].values : calls[0].body.values
      assert.deepEqual(values, { ...original[stage], ...original.models })
    }
    assert.equal(moss.draft.train.seed, 100)
  })
}

test('same-name MOSS labels and help are scoped to their request phase', async () => {
  const { modelFieldHelp } = await import('../src/i18n/modelWorkspace.ts')
  assert.equal(modelFieldLabel('moss', 'en', 'output', '', 'train'), 'LoRA output')
  assert.equal(modelFieldLabel('moss', 'en', 'output', '', 'tune'), 'Tuning report (JSON)')
  assert.equal(modelFieldLabel('moss', 'cn', 'steps', '', 'tune'), '计时训练步数')
  assert.equal(modelFieldLabel('moss', 'cn', 'steps', '', 'generate'), '采样步数')
  assert.match(modelFieldHelp('moss', 'cn', 'output', '', 'tune'), /JSON/)
  assert.doesNotMatch(modelFieldHelp('moss', 'cn', 'seed', '', 'train'), /同一张图/)
  assert.equal(modelFieldLabel('qwen21', 'cn', 'output', 'Qwen output', 'tune'), 'Qwen output')
})
