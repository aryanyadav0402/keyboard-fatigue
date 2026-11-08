import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { selectTypingTask, type TypingTask } from './constants/typingTasks'
import { buildTypingSession } from './lib/buildTypingSession'
import { KeystrokeRecorder } from './lib/keystrokeRecorder'
import { predictTiredness, type PredictionResponse } from './lib/predictionApi'
import './App.css'

type Screen = 'intro' | 'task' | 'submitting' | 'result' | 'error'

function App() {
  const [screen, setScreen] = useState<Screen>('intro')
  const [task, setTask] = useState<TypingTask>(() => selectTypingTask())
  const [answer, setAnswer] = useState('')
  const [answerError, setAnswerError] = useState('')
  const [username, setUsername] = useState('')
  const [error, setError] = useState('')
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const recorder = useRef<KeystrokeRecorder | null>(null)
  const taskStartedAt = useRef(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const usernameRef = useRef<HTMLInputElement>(null)
  const formatMetric = (value: number | null | undefined) =>
    value == null ? '—' : value.toFixed(1)
  const displayMetrics = [
    ['keys_per_second', 'keys / second'],
    ['gap_median_ms', 'median gap'],
    ['key_hold_ms', 'key hold'],
    ['backspaces_per_100_keys', 'backspaces / 100 keys'],
  ] as const
  const formatFeatureValue = (key: (typeof displayMetrics)[number][0], value: number | null | undefined) => {
    if (value == null) return '—'
    if (key === 'keys_per_second') return value.toFixed(2)
    if (key === 'gap_median_ms' || key === 'key_hold_ms') return `${value.toFixed(1)} ms`
    return value.toFixed(1)
  }
  const recommendationFor = (score: number) => {
    if (score >= 7.5) {
      return {
        tone: 'high',
        title: 'Time to rest soon',
        text: 'Your typing looks quite tired. Take a break soon and avoid driving, operating equipment, or other risky work until you feel rested.',
      }
    }
    if (score >= 5) {
      return {
        tone: 'moderate',
        title: 'A break may help',
        text: 'Your typing may be showing some tiredness. Consider taking a short break, stretching, and checking in with yourself before continuing.',
      }
    }
    return {
      tone: 'alert',
      title: 'Looking relatively steady',
      text: 'Your typing looks relatively steady right now. Keep listening to your body and take a break whenever you need one.',
    }
  }

  useEffect(() => {
    if (screen === 'task') inputRef.current?.focus()
    if (screen === 'intro') usernameRef.current?.focus()
  }, [screen])

  useEffect(() => {
    if (screen !== 'task') return
    const handleBlur = () => recorder.current?.recordUi('window_blur')
    const handleFocus = () => recorder.current?.recordUi('window_focus')
    const handleVisibility = () =>
      recorder.current?.recordUi('visibility', document.visibilityState)
    window.addEventListener('blur', handleBlur)
    window.addEventListener('focus', handleFocus)
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      window.removeEventListener('blur', handleBlur)
      window.removeEventListener('focus', handleFocus)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [screen])

  const begin = () => {
    const nextTask = selectTypingTask()
    const nextRecorder = new KeystrokeRecorder()
    nextRecorder.startListening()
    setTask(nextTask)
    setAnswer('')
    setAnswerError('')
    setError('')
    setPrediction(null)
    recorder.current = nextRecorder
    taskStartedAt.current = performance.now()
    setScreen('task')
  }

  const submit = async () => {
    const currentRecorder = recorder.current
    const trimmed = answer.trim()
    if (!currentRecorder || !trimmed || screen !== 'task') return
    const minimumLength = Math.ceil(task.expected.length * 0.7)
    if (trimmed.length < minimumLength) {
      setAnswerError('Please type the required sentence before submitting.')
      return
    }
    setAnswerError('')
    currentRecorder.stopListening()
    setScreen('submitting')
    try {
      const session = buildTypingSession(task, currentRecorder, trimmed, username.trim())
      const result = await predictTiredness(session)
      setPrediction(result)
      setScreen('result')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Unable to analyze this session.')
      setScreen('error')
    }
  }

  const restart = () => {
    recorder.current?.stopListening()
    recorder.current = null
    setScreen('intro')
    setUsername('')
    setAnswer('')
    setAnswerError('')
    setPrediction(null)
    setError('')
  }

  const switchUser = () => {
    recorder.current?.stopListening()
    recorder.current = null
    setUsername('')
    setAnswer('')
    setAnswerError('')
    setPrediction(null)
    setError('')
    setScreen('intro')
  }

  return (
    <main className="app-shell">
      <div className="orb orb-one" />
      <div className="orb orb-two" />
      <div className="orb orb-three" />
      <section className="app-content cozy-card">
        <div className="brand">
          <span className="brand-icon" aria-hidden="true">☾</span>
          <span className="brand-name">tiredtype</span>
          {screen === 'task' && (
            <button className="back-button" type="button" onClick={switchUser}>
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M7.75 4.75 3.5 9m0 0 4.25 4.25M3.5 9h13" />
              </svg>
              switch user
            </button>
          )}
        </div>
        {screen === 'intro' && (
          <section className="intro-panel">
            <p className="eyebrow">a gentle typing check-in</p>
            <h1>How tired is your<br /><em>typing pattern?</em></h1>
            <p className="lede">
              Complete one short typing challenge. We’ll measure rhythm, pauses, corrections,
              and speed to estimate your current tiredness.
            </p>
            <div className="signal-grid">
              <div><strong>01</strong><span>one focused<br />typing task</span></div>
              <div><strong>02</strong><span>raw keystrokes<br />never shown</span></div>
              <div><strong>03</strong><span>instant<br />recommendation</span></div>
            </div>
            <div className="setup-card">
              <form onSubmit={(event) => { event.preventDefault(); if (username.trim()) begin() }}>
                <label aria-label="Username">
                  <input
                    ref={usernameRef}
                    type="text"
                    value={username}
                    onChange={(event) => setUsername(event.target.value)}
                    autoComplete="username"
                    maxLength={64}
                    placeholder="enter your username"
                    required
                  />
                </label>
              </form>
              <p className="task-hint intro-hint">press <kbd>Enter</kbd> to continue</p>
            </div>
          </section>
        )}

        {screen === 'task' && (
          <section className="task-panel">
            <p className="hello-user">hello {username}</p>
            <h2>{task.instruction}</h2>
            <div className="target-box">
              <span className="target-text">{task.source}</span>
            </div>
            <form onSubmit={(event) => { event.preventDefault(); void submit() }}>
              <label className="typing-line">
                <input
                  ref={inputRef}
                  value={answer}
                  onChange={(event) => {
                    setAnswer(event.target.value)
                    setAnswerError('')
                  }}
                  onPaste={(event) => { event.preventDefault(); recorder.current?.recordUi('paste_blocked') }}
                  onDrop={(event) => { event.preventDefault(); recorder.current?.recordUi('drop_blocked') }}
                  onBlur={() => recorder.current?.recordUi('textarea_blur')}
                  onFocus={() => recorder.current?.recordUi('textarea_focus')}
                  autoComplete="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  aria-label="Type your transformed answer"
                  placeholder="type your answer here"
                />
              </label>
              {answerError && <p className="input-error" role="alert">{answerError}</p>}
              <p className="task-hint intro-hint task-input-hint">press <kbd>Enter</kbd> to submit</p>
            </form>
          </section>
        )}

        {screen === 'submitting' && (
          <section className="message-panel">
            <div className="loader" aria-hidden="true" />
            <p className="eyebrow">one moment</p>
            <h2>Reading your typing pattern<span className="blink">_</span></h2>
            <p className="muted">Comparing speed, rhythm, corrections, and pauses.</p>
          </section>
        )}

        {screen === 'result' && prediction && (
          <section className="result-panel">
            <p className="eyebrow">your check-in is complete</p>
            {prediction.model === 'no-baseline model' && prediction.fatigue_index ? (
              <>
                <h2>{prediction.fatigue_index.text}</h2>
              </>
            ) : prediction.model === 'best-attempt model' ? (
              <h2>Your tiredness signal: {formatMetric(prediction.score)} — {prediction.label}</h2>
            ) : (
              <h2>Your tiredness signal</h2>
            )}
            {prediction.model === 'not enough data yet' ? (
              <div className="baseline-message">
                <p>{prediction.note}</p>
              </div>
            ) : prediction.model === 'no-baseline model' ? (
              <div className={`score-card ${prediction.level?.toLowerCase() ?? ''}`}>
                <div
                  className="score-meter rough-score-meter"
                  style={{ '--score-position': `${Math.min(100, Math.max(0, ((prediction.score ?? 0) / 9) * 100))}%` } as CSSProperties}
                  aria-label={`Rough tiredness score ${formatMetric(prediction.score)} out of 9`}
                >
                  <div className="score-pointer">
                    <span className="score-label">{prediction.label}</span>
                    <strong>{formatMetric(prediction.score)}</strong>
                  </div>
                  <div className="score-track" />
                  <div className="score-scale"><span>0</span><span>3</span><span>6</span><span>9</span></div>
                </div>
                <div className="metrics">
                  <div className="metric-grid">
                    {displayMetrics.map(([key, label]) => (
                      <div key={key}><strong>{formatFeatureValue(key, prediction.features[key])}</strong><span>{label}</span></div>
                    ))}
                  </div>
                </div>
                <p className="baseline-note">This attempt establishes your baseline. Future check-ins will be compared with it.</p>
              </div>
            ) : (
              <div               className={`score-card ${prediction.level?.toLowerCase() ?? ''}`}>
                <div
                  className="score-meter"
                  style={{ '--score-position': `${Math.min(100, Math.max(0, ((prediction.score ?? 0) / 9) * 100))}%` } as CSSProperties}
                  aria-label={`Tiredness score ${formatMetric(prediction.score)} out of 9`}
                >
                  <div className="score-pointer">
                    <span className="score-label">{prediction.label}</span>
                    <strong>{formatMetric(prediction.score)}</strong>
                  </div>
                  <div className="score-track" />
                  <div className="score-scale"><span>0</span><span>3</span><span>6</span><span>9</span></div>
                </div>
                <div className="metrics">
                  <div className="metric-grid">
                    {displayMetrics.map(([key, label]) => (
                      <div key={key}><strong>{formatFeatureValue(key, prediction.features[key])}</strong><span>{label}</span></div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {prediction.score !== null && (() => {
              const recommendation = recommendationFor(prediction.score)
              return (
                <aside className={`info-banner recommendation-banner ${recommendation.tone}`} aria-label="Recommendation">
                  <svg className="recommendation-icon" viewBox="0 0 24 24" aria-hidden="true">
                    {recommendation.tone === 'high' ? (
                      <path d="M12 9v4m0 4h.01M10.3 3.8 2.9 17a2 2 0 0 0 1.75 3h14.7a2 2 0 0 0 1.75-3L13.7 3.8a2 2 0 0 0-3.4 0Z" />
                    ) : recommendation.tone === 'moderate' ? (
                      <path d="M8 11V5.5a1.5 1.5 0 0 1 3 0V10m0-1.5a1.5 1.5 0 0 1 3 0V10m0-1a1.5 1.5 0 0 1 3 0v5.5a5 5 0 0 1-5 5h-1.5A5.5 5.5 0 0 1 5 14l-2-3.5a1.5 1.5 0 0 1 2.6-1.5L8 12" />
                    ) : (
                      <path d="m5 12 4 4L19 6" />
                    )}
                  </svg>
                  <div className="recommendation-copy">
                    <strong>{recommendation.title}</strong>
                    <span>{recommendation.text}</span>
                  </div>
                </aside>
              )
            })()}
            <button className="secondary-button" onClick={begin}>Run another check-in <span>↻</span></button>
          </section>
        )}

        {screen === 'error' && (
          <section className="message-panel error-panel">
            <div className="error-icon">!</div>
            <p className="eyebrow">a small hiccup</p>
            <h2>We couldn’t reach the prediction service.</h2>
            <p className="muted">{error}</p>
            <div className="button-row">
              <button className="primary-button" onClick={() => { setError(''); setAnswer(''); setAnswerError(''); setScreen('task'); recorder.current = new KeystrokeRecorder(); recorder.current.startListening() }}>Retry task <span>↻</span></button>
              <button className="secondary-button" onClick={restart}>Start over</button>
            </div>
          </section>
        )}
      </section>

    </main>
  )
}

export default App
