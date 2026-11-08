import { QUESTION_POOL_ID, type TypingTask } from '../constants/typingTasks'
import type { KeystrokeRecorder } from './keystrokeRecorder'
import type { TypingSession } from '../types/typingSession'

function uuid() {
  return crypto.randomUUID()
}

export function buildTypingSession(
  task: TypingTask,
  recorder: KeystrokeRecorder,
  typed: string,
  username: string,
): TypingSession {
  const end = recorder.duration
  const startEpoch = Date.now() - end
  const taskRecord = {
    idx: 0 as const,
    type: 'transform' as const,
    name: 'transform_typing' as const,
    t_start: 0,
    t_end: end,
    completed: true,
    skipped: false as const,
  }
  return {
    schema: 'typing-station',
    schema_version: 1,
    app_version: 'typing-prediction-1.0.0',
    session_id: uuid(),
    participant_id: username,
    status: 'final',
    completed: true,
    recovered: false,
    aborted: false,
    clock: {
      time_origin_epoch_ms: performance.timeOrigin,
      start_epoch_ms: startEpoch,
      start_perf_ms: 0,
      start_local_iso: new Date(startEpoch).toISOString(),
      tz_offset_min: -new Date().getTimezoneOffset(),
    },
    tasks: [taskRecord],
    units: [{
      task: 0,
      unit: 0,
      kind: 'transform',
      question_id: task.id,
      question_type: task.type,
      instruction: task.instruction,
      source: task.source,
      prompt: task.prompt,
      expected: task.expected,
      typed,
      t_shown: 0,
      t_first_key: recorder.firstKeyTime,
      t_end: end,
      skipped: false,
    }],
    key_fields: ['task', 'unit', 'type', 't', 'code', 'key', 'flags'],
    keys: recorder.keys,
    input_fields: ['task', 'unit', 't', 'input_type', 'len', 'caret'],
    inputs: recorder.inputs,
    ui_fields: ['t', 'event', 'detail'],
    ui: recorder.ui,
    metadata: { username },
    config: { question_pool_id: QUESTION_POOL_ID, questions: [task] },
  }
}
