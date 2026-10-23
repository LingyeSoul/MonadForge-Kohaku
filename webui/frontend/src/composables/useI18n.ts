import { computed } from 'vue'
import { useAppStore } from '../stores/app'
import { getFrontendTranslations, type TranslationMessages } from '../i18n'

/** The Vue app owns every translation; there is no server catalog to fetch. */
const LANGUAGES = ['en', 'cn', 'ko', 'ja']

export async function initI18n(): Promise<void> {}

export function useI18n() {
  const appStore = useAppStore()

  const frontend = computed<TranslationMessages>(() => getFrontendTranslations(appStore.language))

  /**
   * Translate a key from the frontend catalog. Supports {param} interpolation.
   * A missing key returns the key itself so a missed translation is visible.
   */
  function t(key: string, params?: Record<string, string | number>): string {
    let text = frontend.value[key] ?? key
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        text = text.replace(`{${k}}`, String(v))
      }
    }
    return text
  }

  function setLanguage(lang: string): void {
    appStore.setLanguage(lang)
  }

  return { t, setLanguage, languages: LANGUAGES, loaded: true }
}
