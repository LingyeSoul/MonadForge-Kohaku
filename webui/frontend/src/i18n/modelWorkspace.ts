/** Model-specific wording for the shared typed-request workbench. */
import { MOSS_FIELDS, MOSS_GENERATE_FIELDS, MOSS_TUNE_FIELDS, type MossFieldCopy } from './mossFields.ts'
import type { QwenField } from '../utils/qwen21.ts'
import type { FieldHelp } from '../utils/fieldHelp.ts'

const mossEn = {
  sourcesHint: 'Select a root containing MOSS-Image-Base/ and MOSS-Image-SFT/. Tokenizer, unquantized text encoder and default VAE come from Base; quantized encoders use the model catalog and single-file overrides take precedence.',
  fieldReference: 'Inputs retain the original English parameter keys. Click the information button for original English documentation and a Chinese explanation in the side reference.',
  cacheHint: 'Build MOSS text embeddings and packed latent caches before training. Existing pairs are reused unless overwrite is enabled.',
  generateHint: 'Choose a model variant, prompt and optional adapter. The LoRA strength controls this generation; 0 disables the adapter.',
  autoHint: 'MOSS uses its own swap and activation-budget defaults. A swap count of -1 sizes from free VRAM. Adapter alpha follows the selected algorithm; inspect each field’s help for its default.',
  unsupported: 'MOSS uses its own trainer and supports LoRA, LoKr and LoHa adapters. Anima presets and Qwen component overrides do not apply.',
  cacheIsolation: 'MOSS caches are separate from Anima and Qwen caches. Model sources are edited in Configuration.',
  cacheNotReady: 'MOSS cache is not ready', hardware: 'MOSS parameter defaults',
}
const mossCn: Record<keyof typeof mossEn, string> = {
  sourcesHint: '模型根目录包含 MOSS-Image-Base/ 与 MOSS-Image-SFT/。tokenizer、未量化文本编码器和默认 VAE 来自 Base；量化文本编码器从模型目录表定位，指定单文件时优先使用覆盖值。',
  fieldReference: '表单保留英文原始参数名。点击参数旁的信息按钮，在侧边查看英文原文和中文释义；基本设置和搜索结果分组均可收起。',
  cacheHint: '训练前生成 MOSS 文本嵌入与打包潜变量缓存；未启用覆盖时复用已有缓存对。',
  generateHint: '选择模型变体、提示词及可选适配器。LoRA 强度用于当前生成，填 0 可禁用适配器。',
  autoHint: 'MOSS 使用自己的换出层数与激活内存预算默认值；换出层数填 -1 时按可用显存估算。alpha 默认值随适配器算法变化，详见对应字段说明。',
  unsupported: 'MOSS 使用独立训练后端，支持 LoRA、LoKr 和 LoHa。Anima 硬件预设与 Qwen 单组件覆盖参数不适用于此模型。',
  cacheIsolation: 'MOSS 缓存与 Anima、Qwen 缓存独立；模型来源在配置页修改。',
  cacheNotReady: 'MOSS 缓存尚未就绪', hardware: 'MOSS 参数默认值',
}
export function mossText(language: string, key: keyof typeof mossEn): string {
  return language === 'cn' || language.startsWith('zh') ? mossCn[key] : mossEn[key]
}
/** Every MOSS parameter's English label, Chinese label and Chinese help. */
function mossField(name: string, section: string): MossFieldCopy | undefined {
  const phase = section === 'generate' ? MOSS_GENERATE_FIELDS : section === 'tune' ? MOSS_TUNE_FIELDS : {}
  return phase[name] ?? MOSS_FIELDS[name]
}

function isChinese(language: string): boolean {
  return language === 'cn' || language.startsWith('zh')
}

/** The label for a request field; MOSS carries its own 中文 copy. */
export function modelFieldLabel(family: string, language: string, name: string, fallback: string, section: string): string {
  const entry = family === 'moss' ? mossField(name, section) : undefined
  if (entry) return isChinese(language) ? entry[1] : entry[0]
  return fallback === `qwenField_${name}` ? name.replace(/_/g, ' ') : fallback
}

/**
 * The long help under a field. The request definitions stay the single source
 * of the English text; MOSS ships a 中文 mirror so the panel is readable in 中文.
 */
export function modelFieldHelp(family: string, language: string, name: string, fallback: string, section: string): string {
  const entry = family === 'moss' ? mossField(name, section) : undefined
  return entry && isChinese(language) ? entry[2] : fallback
}

/** The editable MOSS form uses the exact request key, independent of locale. */
export function modelInputLabel(family: string, language: string, name: string, fallback: string, section: string): string {
  return family === 'moss' ? name : modelFieldLabel(family, language, name, fallback, section)
}

export function modelFieldReference(family: string, language: string, section: string, field: Pick<QwenField, 'name' | 'help'>, fallbackLabel: string): FieldHelp {
  if (family !== 'moss') return `${fallbackLabel}\n${modelFieldHelp(family, language, field.name, field.help, section)}`
  const copy = mossField(field.name, section)
  return { original: field.help, translatedLabel: copy?.[1] ?? field.name, translation: copy?.[2] ?? '此参数暂无中文翻译，请参阅英文原文。' }
}
