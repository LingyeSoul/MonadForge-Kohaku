/** Keep original backend documentation alongside, rather than replacing it with, a translation. */
export interface BilingualFieldHelp {
  original: string
  translation: string
  translatedLabel: string
}
export type FieldHelp = string | BilingualFieldHelp
