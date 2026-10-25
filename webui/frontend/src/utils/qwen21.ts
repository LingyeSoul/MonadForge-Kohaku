/** Runtime-checked Qwen form transport; defaults are owned by the Python requests. */
export type Scalar = string | number | boolean | null
export type Values = Record<string, Scalar>
export type Phase = 'cache' | 'train' | 'generate' | 'tune'
type Json = Scalar | Json[] | { [key: string]: Json }
type JsonObject = { [key: string]: Json }
export interface QwenField {
  name: string
  kind: 'str' | 'int' | 'float' | 'bool'
  value: Scalar
  nullable: boolean
  choices: string[]
  help: string
  advanced: boolean
  multiline: boolean
  /** Form section, owned by the backend (`library.lines.requests.field_group`). */
  group: string | null
}
export interface QwenSchema {
  /** Field names the backend keeps in the profile's `models` section. */
  model_field_names: string[]
  cache: QwenField[]
  train: QwenField[]
  generate: QwenField[]
  tune?: QwenField[]
}
export interface QwenDraft {
  models: Values
  cache: Values
  train: Values
  generate: Values
  tune?: Values
}
export interface QwenResult {
  file: string
  multiplier: number | null
  model?: string
  seed: number
  prompt: string
}
export const draftKey = 'monadforge-qwen21-form-v1'

function object(value: Json, label: string): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label}: expected an object`)
  return value
}
function string(value: Json, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label}: expected a string`)
  return value
}
function bool(value: Json, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label}: expected a boolean`)
  return value
}
function number(value: Json, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label}: expected a finite number`)
  return value
}
function scalar(value: Json, label: string): Scalar {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  return number(value, label)
}
function names(value: Json, label: string): string[] {
  if (!Array.isArray(value)) throw new Error(`Missing ${label}`)
  const result = value.map(item => string(item, label))
  if (new Set(result).size !== result.length) throw new Error(`Duplicate ${label}`)
  return result
}
function field(value: Json): QwenField {
  const row = object(value, 'Qwen field')
  const kind = row.kind
  if (kind !== 'str' && kind !== 'int' && kind !== 'float' && kind !== 'bool') throw new Error('Invalid Qwen field kind')
  if (!Array.isArray(row.choices)) throw new Error('Missing Qwen choices')
  return {
    name: string(row.name, 'field.name'), kind,
    value: scalar(row.value, 'field.value'), nullable: bool(row.nullable, 'field.nullable'),
    choices: row.choices.map(item => string(item, 'field.choice')),
    help: string(row.help, 'field.help'), advanced: bool(row.advanced, 'field.advanced'),
    multiline: bool(row.multiline, 'field.multiline'),
    group: row.group == null ? null : string(row.group, 'field.group'),
  }
}
export function parseSchema(value: Json): QwenSchema {
  const row = object(value, 'Qwen schema')
  function fields(phase: Phase | 'tune'): QwenField[] {
    const entries = row[phase]
    if (!Array.isArray(entries)) throw new Error(`Missing ${phase} fields`)
    const result = entries.map(field)
    if (new Set(result.map(item => item.name)).size !== result.length) throw new Error(`Duplicate ${phase} fields`)
    return result
  }
  const modelFields = names(row.model_field_names, 'model fields')
  const train = fields('train')
  const trainNames = new Set(train.map(item => item.name))
  // A model field the train form does not offer would silently leave the draft.
  const orphans = modelFields.filter(name => !trainNames.has(name))
  if (orphans.length > 0) throw new Error(`Model fields missing from the train form: ${orphans.join(', ')}`)
  return { model_field_names: modelFields, cache: fields('cache'), train, generate: fields('generate'), ...(row.tune ? { tune: fields('tune') } : {}) }
}
/** The backend owns which fields live in the `models` section, and it differs per line. */
export function isModelField(schema: QwenSchema, name: string): boolean {
  return schema.model_field_names.includes(name)
}
export function initialDraft(schema: QwenSchema): QwenDraft {
  const defaults = (fields: QwenField[]): Values => Object.fromEntries(fields.map(item => [item.name, item.value]))
  return {
    models: defaults(schema.train.filter(item => isModelField(schema, item.name))),
    cache: defaults(schema.cache.filter(item => !isModelField(schema, item.name))),
    train: defaults(schema.train.filter(item => !isModelField(schema, item.name))),
    generate: defaults(schema.generate.filter(item => !isModelField(schema, item.name))),
    ...(schema.tune ? { tune: defaults(schema.tune.filter(item => !isModelField(schema, item.name))) } : {}),
  }
}
export function checkedValues(fields: QwenField[], value: Json): Values {
  const row = object(value, 'Qwen values')
  const entries = fields.map(item => {
    if (!(item.name in row)) throw new Error(`${item.name}: missing from the form values`)
    const val = scalar(row[item.name], item.name)
    const valid = val === null ? item.nullable
      : item.kind === 'bool' ? typeof val === 'boolean'
        : item.kind === 'str' ? typeof val === 'string'
          : typeof val === 'number' && Number.isFinite(val) && (item.kind !== 'int' || Number.isInteger(val))
    if (!valid || (item.choices.length > 0 && !item.choices.includes(String(val)))) throw new Error(`Invalid ${item.name}: ${String(val)}`)
    return [item.name, val] as const
  })
  return Object.fromEntries(entries)
}
export function inputValue(field: Pick<QwenField, 'kind' | 'nullable'>, value: Scalar): Scalar {
  if (value === '' || value === null) return field.nullable ? null : value
  return field.kind === 'int' || field.kind === 'float' ? Number(value) : value
}
export function readDraft(schema: QwenSchema, text: string | null): QwenDraft {
  if (text === null) return initialDraft(schema)
  const value: Json = JSON.parse(text)
  const row = object(value, 'Saved Qwen draft')
  return {
    models: checkedValues(schema.train.filter(item => isModelField(schema, item.name)), row.models),
    cache: checkedValues(schema.cache.filter(item => !isModelField(schema, item.name)), row.cache),
    train: checkedValues(schema.train.filter(item => !isModelField(schema, item.name)), row.train),
    generate: checkedValues(schema.generate.filter(item => !isModelField(schema, item.name)), row.generate),
    ...(schema.tune ? { tune: checkedValues(schema.tune.filter(item => !isModelField(schema, item.name)), row.tune ?? {}) } : {}),
  }
}
export async function requestJson(path: string, init: RequestInit): Promise<Json> {
  const response = await fetch(path, init)
  const text = await response.text()
  let body: Json
  try { body = JSON.parse(text) } catch {
    throw new Error(`${path}: HTTP ${response.status}; invalid JSON: ${text.slice(0, 300)}`)
  }
  if (!response.ok) {
    const row = object(body, `HTTP ${response.status}`)
    throw new Error(`${path}: HTTP ${response.status}; ${JSON.stringify(row.detail ?? body)}`)
  }
  return body
}
export function jobId(value: Json): string {
  const id = string(object(value, 'Qwen job').task_id, 'task_id')
  if (!id) throw new Error('Empty task_id in Qwen response')
  return id
}
export function resolvedPaths(value: Json, family: 'qwen21' | 'moss' = 'qwen21'): Record<string, string> {
  const row = object(value, 'Qwen model paths')
  return Object.fromEntries((family === 'moss' ? ['dit', 'text_encoder', 'vae', 'tokenizer'] : ['dit', 'text_encoder', 'vae', 'processor', 'scheduler']).map(name => [name, string(row[name], name)]))
}
export function resultImages(value: Json, family: 'qwen21' | 'moss' = 'qwen21'): QwenResult[] {
  const row = object(value, 'Qwen results')
  if (!Array.isArray(row.images)) throw new Error('Missing Qwen result images')
  return row.images.map(value => {
    const image = object(value, 'Qwen image')
    return { file: string(image.file, 'file'), prompt: string(image.prompt, 'prompt'), multiplier: family === 'moss' ? null : number(image.multiplier, 'multiplier'), seed: number(image.seed, 'seed'), ...(family === 'moss' ? { model: string(image.model, 'model') } : {}) }
  })
}

export function profileNames(value: Json): string[] {
  if (!Array.isArray(value)) throw new Error('Expected a Qwen profile list')
  return value.map(item => string(item, 'profile name'))
}
export interface CacheStatus {
  directory: string
  pairs: number
  missing_text: number
  missing_latents: number
  errors: string[]
  ready: boolean
}
export function cacheStatus(value: Json): CacheStatus {
  const row = object(value, 'Qwen cache status')
  if (!Array.isArray(row.errors)) throw new Error('Missing cache errors')
  return {
    directory: string(row.directory, 'directory'), pairs: number(row.pairs, 'pairs'),
    missing_text: number(row.missing_text, 'missing_text'), missing_latents: number(row.missing_latents, 'missing_latents'),
    errors: row.errors.map(item => string(item, 'cache error')), ready: bool(row.ready, 'ready'),
  }
}
