import type { TypingSession } from '../types/typingSession'

export type PredictionResponse = {
  score: number | null
  scale: string
  label: string | null
  level: string | null
  model: 'best-attempt model' | 'no-baseline model' | 'not enough data yet'
  note: string
  username: string
  rule: string
  features: Record<string, number | null>
  scaled_features: Record<string, number | null> | null
  model_inputs: Record<string, number | null>
  fatigue_index: {
    text: string
    percentile: number
    drivers: Array<{
      label: string
      strength: string
      pushes: string
    }>
  } | null
  best: {
    score: number
    attempts: number
    new_best: boolean
    best_at: string
  } | null
  saved_as_training_data?: boolean
}

const API_URL = import.meta.env.VITE_API_URL || 'https://keyboard-fatigue-api.vercel.app/api/score'

function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!value || typeof value !== 'object') return false
  const result = value as Record<string, unknown>
  return (
    (result.score === null || (
      typeof result.score === 'number' &&
      Number.isFinite(result.score) &&
      result.score >= 1 &&
      result.score <= 9
    )) &&
    typeof result.scale === 'string' &&
    typeof result.note === 'string' &&
    (result.model === 'best-attempt model' ||
      result.model === 'no-baseline model' ||
      result.model === 'not enough data yet') &&
    (result.model === 'not enough data yet'
      ? result.score === null && result.label === null && result.level === null
      : typeof result.score === 'number' &&
        typeof result.label === 'string' &&
        result.label.trim().length > 0 &&
        typeof result.level === 'string')
  )
}

export async function predictTiredness(session: TypingSession): Promise<PredictionResponse> {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(session),
  })
  const data: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const message = data && typeof data === 'object' && 'error' in data ? String(data.error) : `HTTP ${response.status}`
    throw new Error(message)
  }
  if (!isPredictionResponse(data)) throw new Error('Prediction service returned an invalid response')
  return data
}
