<template>
  <v-row class="model-fields">
    <v-col v-for="field in visibleFields" :key="`${qwen.family}/${qwen.profile}/${section}/${field.name}`" cols="12" :sm="field.multiline || section === 'models' ? 12 : 6">
      <ConfigField :field="metadata(field)" :model-value="qwen.draft?.[section]?.[field.name]"
        @update="qwen.update(section, field.name, $event as Scalar)" @help-click="emit('help', `${section}.${field.name}`)" />
    </v-col>
  </v-row>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ConfigField from './ConfigField.vue'
import { useI18n } from '../composables/useI18n'
import { useActiveModelWorkspace, type QwenSection } from '../stores/qwenWorkspace'
import { useAppStore } from '../stores/app'
import { modelInputLabel } from '../i18n/modelWorkspace'
import type { FieldMeta } from '../stores/config'
import type { QwenField, Scalar } from '../utils/qwen21'
import { modelFieldActive, modelFieldType, modelPathKind, modelPathExtensions, modelPathMustExist, modelSectionValues } from '../utils/modelConfig'

const props = defineProps<{ section: QwenSection; fields: QwenField[] }>()
const emit = defineEmits<{ help: [key: string] }>()
const qwen = useActiveModelWorkspace()
const app = useAppStore()
const { t } = useI18n()
const visibleFields = computed(() => props.fields.filter(field => modelFieldActive(
  qwen.value.family, props.section, field.name, modelSectionValues(qwen.value.schema, qwen.value.draft, props.section),
)))
function metadata(field: QwenField): FieldMeta {
  const values = modelSectionValues(qwen.value.schema, qwen.value.draft, props.section)
  const path = modelPathKind(qwen.value.family, field)
  return {
    key: `${props.section}.${field.name}`, label: modelInputLabel(qwen.value.family, app.language, field.name, t(`qwenField_${field.name}`), props.section),
    value: qwen.value.draft?.[props.section]?.[field.name], default_value: field.value,
    field_type: modelFieldType(field),
    description: field.nullable ? t('qwenAuto') : '', origin: 'method', group: props.section,
    is_virtual: false, read_only: qwen.value.busy || qwen.value.loading, options: field.choices, nullable: field.nullable,
    path_kind: path, path_extensions: modelPathExtensions(qwen.value.family, props.section, field.name),
    path_must_exist: modelPathMustExist(props.section, field.name, values),
    multiline: field.multiline,
  }
}
</script>

<style scoped>
.model-fields { min-width: 0; }
</style>
