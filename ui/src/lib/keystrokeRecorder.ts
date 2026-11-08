import type { InputEventRow, KeyEventRow, UiEventRow } from '../types/typingSession'

const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta', 'CapsLock'])

export class KeystrokeRecorder {
  readonly keys: KeyEventRow[] = []
  readonly inputs: InputEventRow[] = []
  readonly ui: UiEventRow[] = []
  private readonly start = performance.now()
  private listening = false

  private timestamp() {
    return Math.round((performance.now() - this.start) * 1000) / 1000
  }

  private onKey = (event: KeyboardEvent) => {
    const flags =
      (event.shiftKey ? 1 : 0) |
      (event.ctrlKey ? 2 : 0) |
      (event.altKey ? 4 : 0) |
      (event.metaKey ? 8 : 0) |
      (event.getModifierState?.('CapsLock') ? 16 : 0) |
      (event.repeat ? 32 : 0) |
      (event.isComposing ? 64 : 0) |
      (event.isTrusted ? 0 : 128)

    this.keys.push([
      0,
      0,
      event.type === 'keydown' ? 'd' : 'u',
      this.timestamp(),
      event.code,
      event.key,
      flags,
    ])
  }

  private onInput = (event: Event) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) return
    this.inputs.push([
      0,
      0,
      this.timestamp(),
      event instanceof InputEvent ? event.inputType || null : null,
      target.value.length,
      target.selectionStart,
    ])
  }

  startListening() {
    if (this.listening) return
    this.listening = true
    document.addEventListener('keydown', this.onKey, true)
    document.addEventListener('keyup', this.onKey, true)
    document.addEventListener('input', this.onInput, true)
  }

  stopListening() {
    if (!this.listening) return
    this.listening = false
    document.removeEventListener('keydown', this.onKey, true)
    document.removeEventListener('keyup', this.onKey, true)
    document.removeEventListener('input', this.onInput, true)
  }

  recordUi(event: string, detail: string | null = null) {
    this.ui.push([this.timestamp(), event, detail])
  }

  get firstKeyTime() {
    const first = this.keys.find(
      ([, , type, , , key]) => type === 'd' && !MODIFIER_KEYS.has(key) && key !== 'Enter',
    )
    return first?.[3] ?? null
  }

  get duration() {
    return this.keys.length ? this.keys[this.keys.length - 1][3] : 0
  }
}
