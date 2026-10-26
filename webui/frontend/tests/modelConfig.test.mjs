import assert from 'node:assert/strict'
import test from 'node:test'
import { BASIC_GROUP_ID, groupModelConfigFields, modelAlgorithmItems, modelActionValues, modelFieldActive, modelFieldType, modelPathKind, modelPathExtensions, modelPathMustExist, modelSubBlocks } from '../src/utils/modelConfig.ts'

// Mirrors one field of the backend schema: `group` comes from
// `library.lines.requests.field_group`, `advanced` from the request metadata.
function field(name, value = '', choices = [], options = {}) {
  return { name, value, choices, kind: 'str', nullable: false, help: name, advanced: false, multiline: false, group: null, ...options }
}
const names = (values, options = {}) => values.split(' ').map(name => field(name, '', [], options))
const entries = groups => groups.flatMap(group => group.sections.flatMap(({ section, fields }) =>
  fields.map(field => ({ group: group.id, section, field }))))

test('basic fields open the page while advanced fields use the backend section', () => {
  const schema = {
    model_field_names: ['model_dir', 'dit', 'text_encoder', 'vae', 'processor', 'scheduler'],
    train: [
      ...names('model_dir dit text_encoder vae processor scheduler', { group: 'Paths', advanced: true }),
      ...names('cache output', { group: 'Paths' }),
      ...names('epochs max_steps lr warmup_ratio', { group: 'Training' }),
      ...names('rank alpha', { group: 'Architecture' }),
      field('logit_mean', 0, [], { group: 'Training', advanced: true }),
      ...names('targets factor down_init', { group: 'Architecture', advanced: true }),
      ...names('blocks_to_swap grad_checkpointing compile', { group: 'Performance', advanced: true }),
      field('algo', 'lokr', ['lora', 'lokr', 'loha'], { group: 'Architecture' }),
    ],
    cache: names('src overwrite'),
    generate: [
      ...names('prompt steps seed out_dir', { group: 'Other' }),
      field('example', '', [], { group: 'Other', advanced: true }),
      ...names('spectrum spectrum_w', { group: 'Performance', advanced: true }),
    ],
  }
  const groups = groupModelConfigFields(schema, { train: { algo: 'lokr', compile: true, grad_checkpointing: false, blocks_to_swap: null } })
  // Panels follow the Anima order; `basic` always leads.
  assert.deepEqual(groups.map(group => group.id), [BASIC_GROUP_ID, 'training', 'performance', 'paths', 'architecture', 'generate'])
  const grouped = Object.fromEntries(groups.map(group => [group.id, group.sections.map(({ section, fields }) =>
    [section, fields.map(field => field.name)])]))
  // Basic *training* fields open the page; the generate section keeps one panel of
  // its own whether a knob is advanced or not, so sampling knobs never mix into
  // the training cards.
  assert.deepEqual(grouped.basic, [
    ['train', ['cache', 'output', 'epochs', 'max_steps', 'lr', 'warmup_ratio', 'rank', 'alpha']],
  ])
  assert.deepEqual(grouped.training, [['train', ['logit_mean']]])
  assert.deepEqual(grouped.performance, [['train', ['blocks_to_swap', 'grad_checkpointing', 'compile']]])
  assert.deepEqual(grouped.generate, [['generate', ['prompt', 'steps', 'seed', 'out_dir', 'example', 'spectrum', 'spectrum_w']]])
  // `down_init` is lora-only and the draft selects lokr, so it stays stored but unrendered.
  assert.deepEqual(grouped.architecture, [['train', ['targets', 'factor']]])
  assert.equal(grouped.other, undefined)
  // Model paths stay in their own panel even though the section is advanced.
  assert.deepEqual(grouped.paths, [
    ['models', ['model_dir', 'dit', 'text_encoder', 'vae', 'processor', 'scheduler']],
  ])
  assert.equal(entries(groups).length, 8 + 1 + 3 + 6 + 2 + 7)
})

test('a closed set of choices renders as a select, never as a path picker', () => {
  // MOSS's `text_encoder` is an int8 / fp8 / bf16 precision; Qwen's is a shard
  // folder. Rendering the MOSS one as a path let an empty value reach the API.
  const mossPrecision = field('text_encoder', 'int8', ['int8', 'fp8', 'bf16'])
  assert.equal(modelFieldType(mossPrecision), 'select')
  assert.equal(modelPathKind('moss', mossPrecision), undefined)
  // MOSS replaces the DiT / VAE folder with one file; Qwen accepts either.
  assert.equal(modelPathKind('moss', field('dit', '')), 'file')
  assert.equal(modelPathKind('moss', field('vae', '')), 'file')
  assert.equal(modelPathKind('qwen21', field('dit', '')), 'either')
  assert.equal(modelPathKind('qwen21', field('vae', '')), 'either')

  const qwenShards = field('text_encoder', '')
  assert.equal(modelFieldType(qwenShards), 'path')
  assert.equal(modelPathKind('qwen21', qwenShards), 'either')

  assert.equal(modelFieldType(field('src', '')), 'path')
  assert.equal(modelFieldType(field('prompt', '')), 'str')
  assert.equal(modelFieldType(field('rank', 32, [], { kind: 'int' })), 'int')
})

test('a section the backend adds later renders under its own name', () => {
  const schema = {
    model_field_names: [],
    train: [field('weird', 1, [], { group: 'Preview Sampling', advanced: true })],
    cache: [],
    generate: [],
  }
  const groups = groupModelConfigFields(schema)
  assert.deepEqual(groups.map(group => group.id), ['preview-sampling'])
  assert.equal(groups[0].titleKey, 'Preview Sampling')
})

test('advanced fields without a backend section fall back to the trailing panel', () => {
  const schema = { model_field_names: [], train: [field('orphan', 1, [], { advanced: true })], cache: [], generate: [] }
  const groups = groupModelConfigFields(schema)
  assert.deepEqual(groups.map(group => group.id), ['other'])
  assert.equal(groups[0].titleKey, 'cfgGroupOther')
})

test('future fields remain visible while irrelevant algorithm fields stay stored but unrendered', () => {
  const future = field('future_training_parameter', 'original')
  const factor = field('factor', -1, [], { kind: 'int', advanced: true, group: 'Architecture' })
  const downInit = field('down_init', 'weight_svd', ['kaiming', 'weight_svd'], { group: 'Architecture' })
  const generate = field('future_generator_parameter', 'generation value', [], { group: 'Other' })
  const schema = { model_field_names: [], train: [future, factor, downInit, field('algo', 'loha', [], { group: 'Architecture' })], cache: [], generate: [generate] }
  const before = structuredClone(schema)
  const groups = groupModelConfigFields(schema)
  const rows = entries(groups)
  // Only the future fields render: `factor` is gated on the lokr algorithm and
  // `down_init` on lora, so neither earns a panel in the loha fixture.
  assert.deepEqual(groups.map(group => group.id), [BASIC_GROUP_ID, 'generate'])
  assert.equal(rows.find(row => row.group === BASIC_GROUP_ID).field, future)
  assert.equal(rows.find(row => row.section === 'generate').field, generate)
  assert.equal(rows.some(row => row.field.name === 'factor'), false)
  assert.equal(rows.some(row => row.field.name === 'down_init'), false)
  assert.deepEqual(schema, before)
  assert.equal(new Set(rows.map(row => `${row.section}.${row.field.name}`)).size, rows.length)
})

for (const family of ['qwen21', 'moss']) {
  for (const algo of ['lora', 'lokr', 'loha']) {
    test(`${family}/${algo} exposes only consumed adapter parameters and retains hidden values`, () => {
      const schema = { model_field_names: [], train: [field('algo', 'lora', [], { group: 'Architecture' }), field('factor', -1, [], { group: 'Architecture' }), field('down_init', 'kaiming', [], { group: 'Architecture' }), ...names('rank alpha targets lora_dtype', { group: 'Architecture' })], cache: [], generate: [] }
      const draft = { train: { algo, factor: 7, down_init: 'weight_svd' } }
      const before = structuredClone(draft)
      const visible = entries(groupModelConfigFields(schema, draft, family)).map(row => row.field.name)
      assert.equal(visible.includes('factor'), algo === 'lokr')
      assert.equal(visible.includes('down_init'), algo === 'lora')
      for (const shared of ['rank', 'alpha', 'targets', 'lora_dtype']) assert.ok(visible.includes(shared))
      assert.deepEqual(draft, before)
      const payload = modelActionValues(family, 'train', schema.train, draft.train)
      assert.equal(payload.factor, algo === 'lokr' ? 7 : -1)
      assert.equal(payload.down_init, algo === 'lora' ? 'weight_svd' : 'kaiming')
      assert.deepEqual(draft, before)
    })
  }
}

test('performance controls track compile, activation-checkpointing, auto-swap and sigma mode', () => {
  for (const family of ['qwen21', 'moss']) {
    for (const compile of [true, false]) {
      assert.equal(modelFieldActive(family, 'train', 'compile_mode', { compile }), compile)
      assert.equal(modelFieldActive(family, 'train', 'compile_seq', { compile }), compile)
      for (const grad_checkpointing of [true, false]) {
        assert.equal(modelFieldActive(family, 'train', 'activation_memory_budget', { compile, grad_checkpointing }), compile && !grad_checkpointing)
      }
    }
  }
  for (const blocks_to_swap of [null, 0, 5]) assert.equal(modelFieldActive('qwen21', 'train', 'activation_reserve_gb', { blocks_to_swap }), blocks_to_swap === null)
  for (const sigma_uniform of [0, 0.5, 1]) for (const name of ['sigma_mu', 'sigma_s', 'sigma_cutoff']) {
    assert.equal(modelFieldActive('moss', 'train', name, { sigma_uniform }), sigma_uniform < 1)
  }
})

test('generation ignores no-adapter multipliers without changing the saved comparison strengths', () => {
  const values = { lora: '', multipliers: '0.75,0.0', true_cfg_scale: 1, negative_prompt: 'saved negative' }
  const fields = [field('multipliers', '1.0,0.0'), field('negative_prompt', '')]
  assert.equal(modelFieldActive('qwen21', 'generate', 'multipliers', values), false)
  assert.equal(modelFieldActive('qwen21', 'generate', 'negative_prompt', values), false)
  assert.deepEqual(modelActionValues('qwen21', 'generate', fields, values), { ...values, multipliers: '0.0', negative_prompt: '' })
  assert.equal(values.multipliers, '0.75,0.0')
  const enabled = { ...values, lora: 'adapter.safetensors', true_cfg_scale: 2 }
  assert.equal(modelFieldActive('qwen21', 'generate', 'multipliers', enabled), true)
  assert.equal(modelFieldActive('qwen21', 'generate', 'negative_prompt', enabled), true)
  assert.deepEqual(modelActionValues('qwen21', 'generate', fields, enabled), enabled)
})

test('MOSS generation keeps fallback CFG controls for examples but hides overwritten parameters', () => {
  for (const cfg of [0, 1, 3]) {
    const plain = { cfg, spectrum: false, lora: '' }
    for (const name of ['negative_prompt', 'uncond_skip', 'batched']) assert.equal(modelFieldActive('moss', 'generate', name, plain), cfg !== 1)
    const example = { ...plain, example: 'parameters.json' }
    for (const name of ['prompt', 'prompts_file', 'prompt_format', 'width', 'height', 'seed', 'negative_prompt']) assert.equal(modelFieldActive('moss', 'generate', name, example), false)
    for (const name of ['cfg', 'uncond_skip', 'steps', 'shift', 'batched']) assert.equal(modelFieldActive('moss', 'generate', name, example), true)
  }
  for (const sampler of ['euler', 'er_sde']) for (const spectrum of [true, false]) {
    assert.equal(modelFieldActive('moss', 'generate', 'spectrum_m', { sampler, spectrum }), spectrum)
  }
  for (const name of ['out_dir', 'model_dir', 'decode_only']) assert.equal(modelFieldActive('moss', 'generate', name, { decode_only: true }), true)
  for (const name of ['prompt', 'lora', 'model', 'steps', 'spectrum', 'example']) assert.equal(modelFieldActive('moss', 'generate', name, { decode_only: true }), false)
})

test('prompt sources, explicit dimensions and cache passes gate only their dependent fields', () => {
  for (const family of ['qwen21', 'moss']) {
    assert.equal(modelFieldActive(family, 'generate', 'prompt', { prompts_file: 'prompts.txt' }), false)
    assert.equal(modelFieldActive(family, 'generate', 'prompt', { prompts_file: '' }), true)
    assert.equal(modelFieldActive(family, 'cache', 'max_text_tokens', { skip_text: true }), false)
    assert.equal(modelFieldActive(family, 'cache', 'caption_format', { skip_text: true }), true)
    for (const name of ['resolution', 'vae_dtype', 'save_crops']) {
      assert.equal(modelFieldActive(family, 'cache', name, { skip_latents: true }), false)
      assert.equal(modelFieldActive(family, 'cache', name, { skip_latents: false }), true)
    }
  }
  assert.equal(modelFieldActive('qwen21', 'generate', 'resolution', { width: 512, height: 512 }), false)
  assert.equal(modelFieldActive('qwen21', 'generate', 'resolution', { width: null, height: null }), true)
})

test('same-name training and generation settings retain independent values and identities', () => {
  const trainSeed = field('seed', 1, [], { group: 'Training' }), generationSeed = field('seed', 42, [], { group: 'Other' })
  const trainModel = field('model', 'base', [], { group: 'Architecture' }), generationModel = field('model', 'sft', [], { group: 'Architecture' })
  const trainCompile = field('compile', false, [], { group: 'Performance', advanced: true }), generationCompile = field('compile', true, [], { group: 'Performance', advanced: true })
  const schema = { model_field_names: [], train: [trainSeed, trainModel, trainCompile], cache: [], generate: [generationSeed, generationModel, generationCompile] }
  const rows = entries(groupModelConfigFields(schema))
  const keyed = new Map(rows.map(row => [`${row.section}.${row.field.name}`, row.field]))
  assert.equal(keyed.get('train.seed'), trainSeed)
  assert.equal(keyed.get('generate.seed'), generationSeed)
  assert.equal(keyed.get('train.model'), trainModel)
  assert.equal(keyed.get('generate.model'), generationModel)
  assert.equal(keyed.get('train.compile'), trainCompile)
  assert.equal(keyed.get('generate.compile'), generationCompile)
  assert.equal(keyed.size, 6)
})

test('method choices come from the schema without replacing or normalizing API values', () => {
  const choices = ['loha', 'lora', 'lokr', 'custom_adapter', 'LoRA']
  const schema = { model_field_names: [], train: [field('algo', 'lora', choices, { group: 'Architecture' })], cache: [], generate: [] }
  assert.deepEqual(modelAlgorithmItems(schema), [
    { title: 'LoHa', value: 'loha' }, { title: 'LoRA', value: 'lora' }, { title: 'LoKr', value: 'lokr' },
    { title: 'custom_adapter', value: 'custom_adapter' }, { title: 'LoRA', value: 'LoRA' },
  ])
  assert.deepEqual(choices, ['loha', 'lora', 'lokr', 'custom_adapter', 'LoRA'])
  assert.deepEqual(modelAlgorithmItems({ model_field_names: [], train: [], cache: [], generate: [] }), [])
  assert.deepEqual(modelAlgorithmItems(null), [])
  assert.deepEqual(groupModelConfigFields(null), [])
})

test('the MOSS tune section renders in its own panel, not under training', () => {
  const swapGrid = field('swap_grid', '8,12', [], { group: 'Training', advanced: true })
  const modelDir = field('model_dir', '', [], { group: 'Paths', advanced: true })
  const schema = {
    model_field_names: ['model_dir'],
    train: [
      field('epochs', 8, [], { group: 'Training', advanced: true }),
      field('lr', 1e-4, [], { group: 'Training' }),
    ],
    cache: [],
    generate: [],
    tune: [swapGrid, modelDir],
  }
  const groups = groupModelConfigFields(schema, null, 'moss')
  assert.deepEqual(groups.map(group => group.id), [BASIC_GROUP_ID, 'training', 'tune'])
  assert.deepEqual(groups.find(group => group.id === 'tune').sections, [
    { section: 'tune', fields: [swapGrid] },
  ])
})

// Mirrors the live MOSS schema: only `model_dir` is a model field, so it is the
// sole train row routed to the models section; every other row stays in its phase.
function mossSchema() {
  const advanced = (values, group, options = {}) => names(values, { group, advanced: true, ...options })
  return {
    model_field_names: ['model_dir'],
    cache: names('src overwrite'),
    train: [
      ...names('cache output model dit rank alpha lr batch_size'),
      ...names('epochs', { value: '8' }),
      ...names('grad_checkpointing', { value: false }),
      ...names('algo', { value: 'lora' }),
      ...advanced('max_steps', 'Training'),
      ...advanced('factor down_init svd_slice svd_slice_count channel_scaling_alpha channel_stats dc_isolate channel_mean ref_timestep ref_isolate repa_weight repa_layer repa_pe_root lora_dtype targets', 'Architecture'),
      ...advanced('optimizer warmup_ratio grad_accum save_every_epochs caption_dropout plain_caption_rate caption_variant_rate sigma_mu sigma_s sigma_cutoff max_grad_norm log_every seed', 'Training'),
      ...advanced('sigma_uniform', 'Training', { value: 0.01 }),
      ...advanced('token_batch precision fp16_fix fp16_fix_c activation_memory_budget blocks_to_swap fit_report expandable_segments', 'Performance'),
      ...advanced('compile', 'Performance', { value: true }),
      ...advanced('model_dir resume', 'Paths'),
    ],
    generate: [
      ...names('prompt negative_prompt prompt_format width height steps sampler uncond_skip seed out_dir ref_image ref_scale ref_vision prompts_file example batched vae_dtype vae_compile vae shift decode_only dave dave_strength dave_tau lora_multiplier'),
      ...names('model dit quant precision fp16_fix fp16_fix_c text_encoder lora', { value: 'styles/moss.safetensors' }),
      ...names('cfg', { value: 3 }),
      ...names('spectrum', { value: true }),
      ...advanced('compile spectrum_window_size spectrum_flex_window spectrum_warmup spectrum_stop_caching_step spectrum_w spectrum_m spectrum_lam', 'Performance'),
      ...names('model_dir', { group: 'Paths', advanced: true }),
    ],
    tune: [
      ...names('cache sizes model dit precision fp16_fix fp16_fix_c rank compile batch_size token_batch swap_grid patience margin_gb tolerance reference budget_step warmup steps output seed'),
      ...names('model_dir', { group: 'Paths', advanced: true }),
    ],
  }
}
const enabledMoss = {
  train: { batch_size: 8, down_init: 'weight_svd', channel_scaling_alpha: 0.5, dc_isolate: true, repa_weight: 1, precision: 'fp16', fp16_fix: 'online' },
  generate: { ref_image: 'ref.png', dave: 'auto', precision: 'fp16', fp16_fix: 'online' },
  tune: { batch_size: 8, precision: 'fp16', fp16_fix: 'online' },
}
const panel = (groups, id, section = 'train') => groups.find(group => group.id === id).sections.find(entry => entry.section === section)
const blockIds = (panelId, fields) => modelSubBlocks(panelId, fields).map(block => [block.id, block.titleKey, block.fields.length])

test('the MOSS schema renders every field once and the generate wall split by meaning', () => {
  const groups = groupModelConfigFields(mossSchema(), enabledMoss, 'moss')
  const ids = [BASIC_GROUP_ID, 'training', 'performance', 'paths', 'architecture', 'generate', 'tune']
  assert.deepEqual(groups.map(group => group.id), ids)
  const section = { [BASIC_GROUP_ID]: 'train', generate: 'generate', tune: 'tune' }
  const count = id => panel(groups, id, section[id] ?? 'train').fields.length
  assert.deepEqual(ids.map(count), [10, 15, 9, 1, 14, 43, 21])
  assert.deepEqual(blockIds('generate', panel(groups, 'generate', 'generate').fields), [
    ['prompt', 'cfgSubPrompt', 5], ['sampling', 'cfgSubSampling', 9],
    ['reference', 'cfgSubReferenceImage', 3], ['dave', 'cfgSubDave', 3],
    ['model', 'cfgSubModel', 8], ['decode', 'cfgSubDecode', 4],
    ['spectrum', 'cfgSubSpectrum', 8], ['output', 'cfgSubOutput', 1], ['performance', 'cfgGroupPerformance', 2],
  ])
  // 118 schema rows -> 114 rendered: the algo selector owns `algo`, generate/tune
  // `model_dir` stay in the models section only, and `factor` is gated off for lora.
  const rendered = groups.flatMap(group => group.sections.flatMap(section => section.fields))
  assert.equal(rendered.length, 114)
  const keys = groups.flatMap(group => group.sections.flatMap(section => section.fields.map(field => `${section.section}.${field.name}`)))
  assert.equal(new Set(keys).size, keys.length)
  for (const group of groups.filter(entry => entry.id !== BASIC_GROUP_ID)) {
    for (const section of group.sections) {
      const blocks = modelSubBlocks(group.id, section.fields)
      const names = blocks.flatMap(block => block.fields.map(field => field.name))
      assert.equal(names.length, section.fields.length)
      assert.deepEqual([...names].sort(), section.fields.map(field => field.name).sort())
    }
  }
})

test('advanced panels render semantic sub-blocks in a deterministic order', () => {
  const groups = groupModelConfigFields(mossSchema(), enabledMoss, 'moss')
  const section = { generate: 'generate', paths: 'models', tune: 'tune' }
  const blocks = (id, name = section[id] ?? 'train') => blockIds(id, panel(groups, id, name).fields)
  assert.deepEqual(blocks('training'), [
    ['schedule', 'cfgSubSchedule', 8], ['caption', 'cfgSubCaption', 3], ['sigma', 'cfgSubSigma', 4],
  ])
  assert.deepEqual(blocks('performance'), [
    ['memory', 'cfgSubMemory', 5], ['compile', 'cfgSubCompile', 1], ['precision', 'cfgSubPrecision', 3],
  ])
  assert.deepEqual(blocks('architecture'), [
    ['adapter', 'cfgSubAdapter', 5], ['channel', 'cfgSubChannel', 4], ['reference', 'cfgSubReference', 5],
  ])
  assert.deepEqual(blocks('paths', 'models'), [['paths', '', 1]])
  assert.deepEqual(blocks('tune', 'tune'), [
    ['model', 'cfgSubModel', 4], ['workload', 'cfgSubWorkload', 4],
    ['performance', 'cfgGroupPerformance', 4], ['sweep', 'cfgSubSweep', 8], ['output', 'cfgSubOutput', 1],
  ])
  // Unmatched future knobs trail in the rest block instead of disappearing.
  const [future, prompt] = [field('future_knob'), field('prompt')]
  assert.deepEqual(blockIds('generate', [future, prompt]), [
    ['prompt', 'cfgSubPrompt', 1], ['performance', 'cfgGroupPerformance', 1],
  ])
})

test('MOSS adapter gates clear incompatible action values without changing the profile', () => {
  const fields = [field('svd_slice', 0), field('svd_slice_count', 0), field('channel_scaling_alpha', 0.5), field('channel_stats', 'default-stats'), field('dc_isolate', true), field('channel_mean', 'default-mean')]
  for (const algo of ['lokr', 'loha']) {
    const saved = { algo, down_init: 'weight_svd', svd_slice: 2, svd_slice_count: 4, channel_scaling_alpha: 1, channel_stats: 'custom-stats', dc_isolate: true, channel_mean: 'custom-mean' }
    const before = structuredClone(saved)
    for (const field of fields) assert.equal(modelFieldActive('moss', 'train', field.name, saved), false)
    const payload = modelActionValues('moss', 'train', fields, saved)
    assert.equal(payload.svd_slice, 0)
    assert.equal(payload.svd_slice_count, 0)
    assert.deepEqual(saved, before)
  }
  for (const name of ['svd_slice', 'svd_slice_count']) {
    assert.equal(modelFieldActive('moss', 'train', name, { algo: 'lora', down_init: 'kaiming' }), false)
    assert.equal(modelFieldActive('moss', 'train', name, { algo: 'lora', down_init: 'weight_svd' }), true)
  }
  assert.equal(modelFieldActive('moss', 'train', 'channel_stats', { channel_scaling_alpha: 0 }), false)
  assert.equal(modelFieldActive('moss', 'train', 'channel_mean', { dc_isolate: false }), false)
})

test('MOSS dependent controls follow their own phase values and decoding retains VAE overrides', () => {
  for (const phase of ['train', 'tune']) {
    assert.equal(modelFieldActive('moss', phase, 'token_batch', { batch_size: 1 }), false)
    assert.equal(modelFieldActive('moss', phase, 'token_batch', { batch_size: 4 }), true)
  }
  assert.equal(modelFieldActive('moss', 'generate', 'dave_tau', { dave: '' }), false)
  assert.equal(modelFieldActive('moss', 'generate', 'ref_scale', { ref_image: '' }), false)
  assert.equal(modelFieldActive('moss', 'generate', 'fp16_fix_c', { precision: 'bf16', fp16_fix: 'online' }), false)
  const values = { decode_only: true, vae: 'custom.safetensors', vae_dtype: 'fp32', vae_compile: true, prompt: 'unused' }
  const fields = [field('vae', ''), field('vae_dtype', 'auto'), field('vae_compile', false), field('prompt', '')]
  const payload = modelActionValues('moss', 'generate', fields, values)
  assert.deepEqual(payload, { ...values, prompt: '' })
  assert.equal(modelFieldActive('moss', 'generate', 'fp16_fix_c', { ...values, precision: 'fp16', fp16_fix: 'online' }), false)
})

test('file filters use the phase instead of sharing the training output extension', () => {
  assert.equal(modelPathExtensions('moss', 'train', 'output'), '.safetensors')
  assert.equal(modelPathExtensions('moss', 'tune', 'output'), '.json')
  assert.equal(modelPathExtensions('moss', 'train', 'resume'), '.pt')
  // The reference-image picker passes a list; file_service splits it.
  assert.equal(modelPathExtensions('moss', 'generate', 'ref_image'), '.png,.jpg,.jpeg,.webp,.bmp')
})

test('decode-only reads its output folder, so that one must already exist', () => {
  assert.equal(modelPathMustExist('generate', 'out_dir', {}), false)
  assert.equal(modelPathMustExist('generate', 'out_dir', { decode_only: true }), true)
  assert.equal(modelPathMustExist('generate', 'out_dir', { decode_only: false }), false)
  assert.equal(modelPathMustExist('train', 'output', { decode_only: true }), false)
})
