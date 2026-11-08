import type { TypingTask } from '../constants/typingTasks'

export type KeyEventRow = [number, number, 'd' | 'u', number, string, string, number]
export type InputEventRow = [number, number, number, string | null, number, number | null]
export type UiEventRow = [number, string, string | null]

export type TypingSession = {
  schema: 'typing-station'
  schema_version: 1
  app_version: string
  session_id: string
  participant_id: string
  status: 'final'
  completed: boolean
  recovered: false
  aborted: false
  clock: {
    time_origin_epoch_ms: number
    start_epoch_ms: number
    start_perf_ms: number
    start_local_iso: string
    tz_offset_min: number
  }
  tasks: Array<{
    idx: number
    type: 'transform'
    name: 'transform_typing'
    t_start: number
    t_end: number
    completed: boolean
    skipped: false
  }>
  units: Array<{
    task: number
    unit: number
    kind: 'transform'
    question_id: number
    question_type: string
    instruction: string
    source: string
    prompt: string
    expected: string
    typed: string
    t_shown: number
    t_first_key: number | null
    t_end: number
    skipped: false
  }>
  key_fields: ['task', 'unit', 'type', 't', 'code', 'key', 'flags']
  keys: KeyEventRow[]
  input_fields: ['task', 'unit', 't', 'input_type', 'len', 'caret']
  inputs: InputEventRow[]
  ui_fields: ['t', 'event', 'detail']
  ui: UiEventRow[]
  metadata: {
    username: string
  }
  config: {
    question_pool_id: string
    questions: TypingTask[]
  }
}
