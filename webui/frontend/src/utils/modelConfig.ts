import { isModelField, type Phase, type QwenDraft, type QwenField, type QwenSchema, type Values } from './qwen21.ts'

export type ModelConfigSection = 'models' | 'train' | 'generate' | 'tune'
export interface ModelConfigGroup {
  id: string
  titleKey: string
  sections: { section: ModelConfigSection; fields: QwenField[] }[]
}

export type TypedModelFamily = 'qwen21' | 'moss'

/**
 * The everyday-knob card every model page opens with, mirroring Anima's
 * 「基本设置」. A field lands here when it is NOT advanced (Anima's `is_basic`,
 * the model lines' `advanced=False`), so the browser card, the desktop GUI's
 * Advanced fold and the request defaults all describe the same split.
 */
export const BASIC_GROUP_ID = 'basic'

// Panel order for the advanced knobs. The section of a field comes from the
// backend (`library.lines.requests.field_group`, shared with the desktop GUI);
// only the heading and the order are the browser's business. A section added
// later still renders — after these, under its own name.
const PANEL_ORDER = ['training', 'performance', 'paths', 'architecture', 'generate', 'tune', 'other'] as const
const PANEL_TITLE_KEYS: Record<string, string> = {
  training: 'cfgGroupTraining',
  performance: 'cfgGroupPerformance',
  paths: 'cfgGroupPaths',
  architecture: 'cfgGroupArchitecture',
  generate: 'qwTestParameters',
  tune: 'cfgGroupTune',
  other: 'cfgGroupOther',
}

/** Match the actual request consumers, not whether a field happens to be in the schema. */
/** Which picker a path-shaped field gets. Keyed by request field name. */
export const PATH_KINDS: Record<string, 'file' | 'directory' | 'either'> = {
  model_dir: 'directory', dit: 'either', text_encoder: 'either', vae: 'either', processor: 'directory', scheduler: 'either',
  src: 'directory', cache: 'directory', out: 'directory', out_dir: 'directory', output: 'file', lora: 'file', prompts_file: 'file',
  resume: 'file', channel_stats: 'file', channel_mean: 'file', repa_pe_root: 'directory', ref_image: 'either',
}

/**
 * The form widget a field needs. A closed set of `choices` always wins over the
 * path picker: the Qwen line's `text_encoder` is a shard folder, but MOSS's is
 * an `int8` / `fp8` / `bf16` precision — rendering that as a file picker let a
 * cleared box write an empty string, which the request layer refuses.
 */
export function modelFieldType(field: QwenField): 'path' | 'select' | QwenField['kind'] {
  if (field.choices.length > 0) return 'select'
  return PATH_KINDS[field.name] ? 'path' : field.kind
}

export function modelPathKind(field: QwenField): 'file' | 'directory' | 'either' | undefined {
  return field.choices.length > 0 ? undefined : PATH_KINDS[field.name]
}

export function modelPathExtensions(family: TypedModelFamily, section: string, name: string): string {
  if (name === 'scheduler' || (family === 'moss' && section === 'tune' && name === 'output')) return '.json'
  if (name === 'prompts_file') return '.txt'
  if (name === 'resume') return '.pt'
  if (name === 'ref_image') return '.png,.jpg,.jpeg,.webp,.bmp'
  return '.safetensors'
}

export function modelPathMustExist(section: string, name: string): boolean {
  return !['cache.out', 'train.output', 'generate.out_dir', 'tune.output'].includes(`${section}.${name}`)
}

export function modelFieldActive(family: TypedModelFamily, section: string, name: string, values: Values): boolean {
  const supplied = (key: string) => typeof values[key] === 'string' && String(values[key]).trim().length > 0
  if (family === 'moss') {
    if (section === 'generate' && values.decode_only === true) return ['out_dir', 'model_dir', 'decode_only', 'vae', 'vae_dtype', 'vae_compile'].includes(name)
    if (['train', 'generate', 'tune'].includes(section)) {
      if (name === 'fp16_fix') return values.precision !== 'bf16'
      if (name === 'fp16_fix_c') return values.precision !== 'bf16' && values.fp16_fix === 'online'
    }
    if (['train', 'tune'].includes(section) && name === 'token_batch') return Number(values.batch_size ?? 1) > 1
    if (section === 'train') {
      const lora = (values.algo ?? 'lora') === 'lora'
      if (['svd_slice', 'svd_slice_count'].includes(name)) return lora && values.down_init === 'weight_svd'
      if (['channel_scaling_alpha', 'dc_isolate'].includes(name)) return lora
      if (name === 'channel_stats') return lora && Number(values.channel_scaling_alpha ?? 0) > 0
      if (name === 'channel_mean') return lora && values.dc_isolate === true
      if (['repa_layer', 'repa_pe_root'].includes(name)) return Number(values.repa_weight ?? 0) > 0
    }
  }
  if (section === 'train') {
    const algorithm = String(values.algo ?? 'lora')
    if (name === 'factor') return algorithm === 'lokr'
    if (name === 'down_init') return algorithm === 'lora'
    if (name === 'compile_seq' || name === 'compile_mode') return values.compile === true
    if (name === 'activation_reserve_gb') return values.blocks_to_swap == null
    if (name === 'activation_memory_budget') return values.compile === true && values.grad_checkpointing !== true
    if (family === 'moss' && ['sigma_mu', 'sigma_s', 'sigma_cutoff'].includes(name)) return Number(values.sigma_uniform ?? 0) < 1
  }
  if (section === 'cache') {
    if (['max_text_tokens', 'te_blocks_to_swap', 'text_encoder'].includes(name)) return values.skip_text !== true
    if (['resolution', 'vae_dtype', 'save_crops', 'vae'].includes(name)) return values.skip_latents !== true
  }
  if (section === 'generate') {
    if (name === 'lora_multiplier' || name === 'multipliers') return supplied('lora')
    if (family === 'qwen21') {
      if (name === 'prompt') return !supplied('prompts_file')
      if (name === 'negative_prompt') return Number(values.true_cfg_scale ?? 1) > 1
      if (name === 'resolution') return values.width == null || values.height == null
    } else {
      const example = supplied('example')
      if (name === 'prompt') return !example && !supplied('prompts_file')
      if (['prompts_file', 'prompt_format', 'width', 'height', 'seed'].includes(name)) return !example
      if (name === 'negative_prompt') return !example && Number(values.cfg ?? 3) !== 1
      if (name === 'uncond_skip' || name === 'batched') return example || Number(values.cfg ?? 3) !== 1
      if (name.startsWith('spectrum_')) return values.spectrum === true
      if (['ref_scale', 'ref_vision'].includes(name)) return supplied('ref_image')
      if (['dave_strength', 'dave_tau'].includes(name)) return supplied('dave')
    }
  }
  return true
}

export function modelSectionValues(schema: QwenSchema | null, draft: Partial<QwenDraft> | null, section: ModelConfigSection | 'cache'): Values {
  if (section === 'models') return draft?.models ?? {}
  return { ...Object.fromEntries((schema?.[section] ?? []).map(field => [field.name, field.value])), ...draft?.[section] }
}

/** Ignore inactive values only in action payloads; saved profiles retain them for later reuse. */
export function modelActionValues(family: TypedModelFamily, phase: Phase, fields: QwenField[], values: Values): Values {
  const active = { ...values }
  for (const field of fields) {
    if (!modelFieldActive(family, phase, field.name, values)) active[field.name] = field.value
  }
  if (family === 'qwen21' && phase === 'generate' && !String(values.lora ?? '').trim()) {
    active.multipliers = '0.0'
  }
  return active
}

/** Organize the typed workspaces without merging their storage sections. */
export function groupModelConfigFields(schema: QwenSchema | null, draft: Partial<QwenDraft> | null = null, family: TypedModelFamily = 'qwen21'): ModelConfigGroup[] {
  if (!schema) return []
  const groups: ModelConfigGroup[] = [
    { id: BASIC_GROUP_ID, titleKey: 'cfgBasicSettings', sections: [] },
    ...PANEL_ORDER.map(id => ({ id, titleKey: PANEL_TITLE_KEYS[id], sections: [] })),
  ]
  const byId = new Map(groups.map(group => [group.id, group]))
  function append(id: string, section: ModelConfigSection, field: QwenField): void {
    let group = byId.get(id)
    if (!group) {
      // A section the backend introduced after this list was written: keep it,
      // under its own name until a translation key exists for it.
      group = { id, titleKey: field.group ?? id, sections: [] }
      byId.set(id, group)
      groups.push(group)
    }
    let entry = group.sections.find(entry => entry.section === section)
    if (!entry) {
      entry = { section, fields: [] }
      group.sections.push(entry)
    }
    entry.fields.push(field)
  }
  /** Basic fields open the page; advanced ones go to the section the backend named.
   *  Generate knobs keep their own panel whatever the backend marks, since they
   *  only apply to a sampling run and would otherwise mix into the training cards. */
  function target(field: QwenField, section: ModelConfigSection): string {
    if (section === 'generate') return 'generate'
    if (!field.advanced) return BASIC_GROUP_ID
    const name = (field.group ?? '').trim()
    return name ? name.toLowerCase().replace(/[^a-z0-9]+/g, '-') : 'other'
  }
  for (const field of schema.train) {
    if (isModelField(schema, field.name)) {
      append(target(field, 'models'), 'models', field)
      continue
    }
    // The selector owns algo; irrelevant knobs must not reappear through search/help.
    if (field.name === 'algo' || !modelFieldActive(family, 'train', field.name, modelSectionValues(schema, draft, 'train'))) continue
    append(target(field, 'train'), 'train', field)
  }
  for (const field of schema.generate) {
    if (!isModelField(schema, field.name) && modelFieldActive(family, 'generate', field.name, modelSectionValues(schema, draft, 'generate'))) append(target(field, 'generate'), 'generate', field)
  }
  // The MOSS swap-tuning form is its own panel and submit action; its fields
  // never belong to the training section even when the backend names a group.
  for (const field of schema.tune ?? []) {
    if (!isModelField(schema, field.name) && modelFieldActive(family, 'tune', field.name, modelSectionValues(schema, draft, 'tune'))) append('tune', 'tune', field)
  }
  return groups.filter(group => group.sections.length > 0)
}

/** One labelled block inside a panel; `titleKey: ''` renders the grid as-is. */
export interface ModelConfigSubBlock {
  id: string
  titleKey: string
  fields: QwenField[]
}

interface SubBlockRule {
  id: string
  titleKey: string
  fields?: string[]
}

/**
 * Second-level headings, in render order. A panel already is one backend
 * `field.group`, so the finer split is a field-name table: the request classes
 * reuse names across phases (`seed`, `precision`, `blocks_to_swap`), and one
 * table keeps a shared knob under the same heading everywhere. Anything the
 * table misses lands in the panel's trailing block, so a new backend field still
 * renders — and a panel without an entry keeps today's single grid.
 */
const SUB_BLOCK_RULES: Record<string, { blocks: SubBlockRule[]; rest: SubBlockRule }> = {
  generate: {
    blocks: [
      { id: 'prompt', titleKey: 'cfgSubPrompt', fields: ['prompt', 'negative_prompt', 'prompt_format', 'prompts_file', 'example'] },
      { id: 'sampling', titleKey: 'cfgSubSampling', fields: [
        'width', 'height', 'steps', 'cfg',
        'true_cfg_scale', 'sampler', 'uncond_skip', 'seed', 'shift', 'lora_multiplier', 'multipliers', 'resolution',
      ] },
      { id: 'reference', titleKey: 'cfgSubReferenceImage', fields: ['ref_image', 'ref_scale', 'ref_vision'] },
      { id: 'dave', titleKey: 'cfgSubDave', fields: ['dave', 'dave_strength', 'dave_tau'] },
      { id: 'model', titleKey: 'cfgSubModel', fields: ['model', 'dit', 'quant', 'precision', 'fp16_fix', 'fp16_fix_c', 'lora', 'text_encoder', 'model_dir'] },
      { id: 'decode', titleKey: 'cfgSubDecode', fields: ['vae', 'vae_dtype', 'vae_compile', 'decode_only'] },
      { id: 'spectrum', titleKey: 'cfgSubSpectrum', fields: ['spectrum', 'spectrum_window_size', 'spectrum_flex_window', 'spectrum_warmup', 'spectrum_stop_caching_step', 'spectrum_w', 'spectrum_m', 'spectrum_lam'] },
      { id: 'output', titleKey: 'cfgSubOutput', fields: ['out_dir'] },
    ],
    rest: { id: 'performance', titleKey: 'cfgGroupPerformance' },
  },
  training: {
    blocks: [
      { id: 'schedule', titleKey: 'cfgSubSchedule', fields: ['max_steps', 'warmup_ratio', 'grad_accum', 'save_every_epochs', 'optimizer', 'max_grad_norm', 'log_every', 'seed'] },
      { id: 'caption', titleKey: 'cfgSubCaption', fields: ['caption_dropout', 'plain_caption_rate', 'caption_variant_rate'] },
      { id: 'sigma', titleKey: 'cfgSubSigma', fields: ['sigma_mu', 'sigma_s', 'sigma_cutoff', 'sigma_uniform', 'logit_mean', 'logit_std'] },
    ],
    rest: { id: 'other', titleKey: 'cfgGroupOther' },
  },
  performance: {
    blocks: [
      { id: 'memory', titleKey: 'cfgSubMemory', fields: ['token_batch', 'activation_memory_budget', 'activation_reserve_gb', 'blocks_to_swap', 'fit_report', 'expandable_segments'] },
      { id: 'compile', titleKey: 'cfgSubCompile', fields: ['compile', 'compile_seq', 'compile_mode'] },
      { id: 'precision', titleKey: 'cfgSubPrecision', fields: ['precision', 'fp16_fix', 'fp16_fix_c', 'lora_dtype', 'vae_dtype'] },
    ],
    rest: { id: 'other', titleKey: 'cfgGroupOther' },
  },
  tune: {
    blocks: [
      { id: 'model', titleKey: 'cfgSubModel', fields: ['cache', 'dit', 'model', 'rank'] },
      { id: 'workload', titleKey: 'cfgSubWorkload', fields: ['sizes', 'batch_size', 'token_batch', 'seed'] },
      { id: 'performance', titleKey: 'cfgGroupPerformance', fields: ['precision', 'fp16_fix', 'fp16_fix_c', 'compile'] },
      { id: 'sweep', titleKey: 'cfgSubSweep', fields: ['swap_grid', 'steps', 'warmup', 'budget_step', 'patience', 'tolerance', 'margin_gb', 'reference'] },
      { id: 'output', titleKey: 'cfgSubOutput', fields: ['output'] },
    ],
    rest: { id: 'other', titleKey: 'cfgGroupOther' },
  },
  architecture: {
    blocks: [
      { id: 'adapter', titleKey: 'cfgSubAdapter', fields: ['rank', 'alpha', 'factor', 'down_init', 'svd_slice', 'svd_slice_count', 'targets', 'lora_dtype'] },
      { id: 'channel', titleKey: 'cfgSubChannel', fields: ['channel_scaling_alpha', 'channel_stats', 'channel_mean', 'dc_isolate'] },
      { id: 'reference', titleKey: 'cfgSubReference', fields: ['ref_timestep', 'ref_isolate', 'repa_weight', 'repa_layer', 'repa_pe_root'] },
    ],
    rest: { id: 'other', titleKey: 'cfgGroupOther' },
  },
}

/** Bucket one section's fields into its panel's sub-blocks: the rule order sets
 *  the block order, the section's own order survives inside each block, and a
 *  field is never dropped or shown twice (first matching rule wins). */
export function modelSubBlocks(panelId: string, fields: QwenField[]): ModelConfigSubBlock[] {
  const plan = SUB_BLOCK_RULES[panelId]
  if (!plan) return fields.length ? [{ id: panelId, titleKey: '', fields: [...fields] }] : []
  const owner = new Map<string, string>()
  for (const block of plan.blocks) for (const name of block.fields ?? []) if (!owner.has(name)) owner.set(name, block.id)
  const buckets = new Map<string, QwenField[]>()
  for (const field of fields) {
    const id = owner.get(field.name) ?? plan.rest.id
    const bucket = buckets.get(id)
    if (bucket) bucket.push(field)
    else buckets.set(id, [field])
  }
  return [...plan.blocks, plan.rest]
    .map(block => ({ id: block.id, titleKey: block.titleKey, fields: buckets.get(block.id) ?? [] }))
    .filter(block => block.fields.length > 0)
}

/** Preserve the API values and ordering; only familiar display names change. */
export function modelAlgorithmItems(schema: QwenSchema | null): { title: string; value: string }[] {
  const titles: Record<string, string> = { lora: 'LoRA', lokr: 'LoKr', loha: 'LoHa' }
  return (schema?.train.find(field => field.name === 'algo')?.choices ?? [])
    .map(value => ({ title: titles[value] ?? value, value }))
}
