import { useEffect, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import './App.css'

type EngineDefaults = {
  hotkey: string
  pause_media_while_dictating: boolean
  music_processes: string[]
  engine: string
}

type EngineStatus = {
  running: boolean
  pid?: number | null
}

type DictationStats = {
  lifetime_words: number
  average_words_per_minute: number
  total_audio_ms: number
  total_entries: number
}

type DictationHistoryEntry = {
  timestamp: string
  backend: string
  raw_text: string
  final_text: string
  word_count: number
  raw_word_count: number
  audio_duration_ms: number
  transcription_ms: number
  cleanup_ms: number
  words_per_minute: number
  submit_command: boolean
}

type DictationHistoryResponse = {
  stats: DictationStats
  entries: DictationHistoryEntry[]
}

const fallbackDefaults: EngineDefaults = {
  hotkey: 'Ctrl+Windows hold; Alt+Shift+Z repeat last',
  pause_media_while_dictating: true,
  music_processes: ['chrome', 'msedge', 'firefox', 'brave', 'opera', 'spotify', 'YouTube Music'],
  engine: 'Cohere Transcribe local ASR with deterministic cleanup',
}

const emptyHistory: DictationHistoryResponse = {
  stats: {
    lifetime_words: 0,
    average_words_per_minute: 0,
    total_audio_ms: 0,
    total_entries: 0,
  },
  entries: [],
}

function isTauriRuntime() {
  return '__TAURI_INTERNALS__' in window
}

function App() {
  const [defaults, setDefaults] = useState<EngineDefaults>(fallbackDefaults)
  const [status, setStatus] = useState<EngineStatus>({ running: false })
  const [history, setHistory] = useState<DictationHistoryResponse>(emptyHistory)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')

  async function refreshStatus() {
    if (!isTauriRuntime()) {
      setStatus({ running: false })
      return
    }
    const nextStatus = await invoke<EngineStatus>('engine_status')
    setStatus(nextStatus)
  }

  async function refreshHistory() {
    if (!isTauriRuntime()) {
      setHistory(emptyHistory)
      return
    }
    const nextHistory = await invoke<DictationHistoryResponse>('get_dictation_history')
    setHistory(nextHistory)
  }

  async function startEngine() {
    if (!isTauriRuntime()) {
      setMessage('Open the Tauri app to start the Windows dictation engine.')
      return
    }
    setBusy(true)
    setMessage('')
    try {
      const nextStatus = await invoke<EngineStatus>('start_engine')
      setStatus(nextStatus)
      setMessage('Engine is running in the tray.')
    } catch (error) {
      setMessage(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function stopEngine() {
    if (!isTauriRuntime()) {
      setMessage('Open the Tauri app to stop the Windows dictation engine.')
      return
    }
    setBusy(true)
    setMessage('')
    try {
      const nextStatus = await invoke<EngineStatus>('stop_engine')
      setStatus(nextStatus)
      setMessage('Engine stopped.')
    } catch (error) {
      setMessage(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function openConfig() {
    if (!isTauriRuntime()) {
      setMessage('Config folder is available from the Tauri app.')
      return
    }
    try {
      await invoke('open_config_folder')
    } catch (error) {
      setMessage(String(error))
    }
  }

  useEffect(() => {
    let cancelled = false

    async function initializeEngine() {
      if (!isTauriRuntime()) {
        setStatus({ running: false })
        return
      }

      try {
        const nextDefaults = await invoke<EngineDefaults>('get_engine_defaults')
        if (!cancelled) {
          setDefaults(nextDefaults)
        }
        refreshHistory().catch((error) => setMessage(String(error)))

        const currentStatus = await invoke<EngineStatus>('engine_status')
        if (cancelled) {
          return
        }
        if (currentStatus.running) {
          setStatus(currentStatus)
          setMessage('Engine is running in the background.')
          return
        }

        setStatus(currentStatus)
      } catch (error) {
        if (!cancelled) {
          setMessage(String(error))
        }
      } finally {
        if (!cancelled) {
          setBusy(false)
        }
      }
    }

    window.setTimeout(() => {
      initializeEngine()
    }, 0)

    const timer = window.setInterval(() => {
      refreshStatus().catch((error) => setMessage(String(error)))
      refreshHistory().catch((error) => setMessage(String(error)))
    }, 2000)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const latestEntry = history.entries[0]

  return (
    <main className="shell">
      <section className="statusPanel" aria-live="polite">
        <div>
          <p className="eyebrow">FDvoice dictation</p>
          <h1>Dictation console</h1>
          <p className="lede">
            {status.running
              ? 'Hold Ctrl+Windows anywhere in Windows while speaking, then release. Press Alt+Shift+Z to repeat the last transcript.'
              : 'The engine starts automatically when FDvoice opens.'}
          </p>
        </div>
        <div className={status.running ? 'indicator active' : 'indicator'}>
          <span></span>
          {status.running ? `PID ${status.pid}` : 'Idle'}
        </div>
      </section>

      <section className="overview" aria-label="Dictation overview">
        <Metric label="Lifetime words" value={formatInteger(history.stats.lifetime_words)} />
        <Metric label="Average WPM" value={formatDecimal(history.stats.average_words_per_minute)} />
        <Metric label="Dictations" value={formatInteger(history.stats.total_entries)} />
        <Metric label="Spoken time" value={formatDuration(history.stats.total_audio_ms)} />
      </section>

      <section className="workspace">
        <section className="panel controlsPanel" aria-label="Engine controls">
          <div>
            <p className="sectionLabel">Engine</p>
            <h2>{status.running ? 'Running in background' : 'Stopped'}</h2>
          </div>
          <div className="controls">
            <button type="button" onClick={startEngine} disabled={busy || status.running}>
              Start
            </button>
            <button type="button" onClick={stopEngine} disabled={busy || !status.running}>
              Stop
            </button>
            <button type="button" className="secondary" onClick={openConfig}>
              Config
            </button>
          </div>
          <div className="settingsList">
            <div>
              <span>Hotkey</span>
              <strong>{defaults?.hotkey ?? 'Ctrl+Windows'}</strong>
            </div>
            <div>
              <span>Media pause</span>
              <strong>{defaults?.pause_media_while_dictating ? 'On' : 'Off'}</strong>
            </div>
            <div>
              <span>Engine</span>
              <strong>{defaults?.engine ?? 'Loading'}</strong>
            </div>
          </div>
        </section>

        <section className="panel latestPanel" aria-label="Latest dictation">
          <div className="panelHeader">
            <div>
              <p className="sectionLabel">Latest</p>
              <h2>Prompt output</h2>
            </div>
            {latestEntry && <span className="timeBadge">{formatTime(latestEntry.timestamp)}</span>}
          </div>
          {latestEntry ? (
            <div className="latestGrid">
              <PromptBlock label="What you said" text={latestEntry.raw_text} />
              <PromptBlock label="Inserted text" text={latestEntry.final_text} />
            </div>
          ) : (
            <p className="emptyState">No dictations recorded yet.</p>
          )}
        </section>
      </section>

      {message && <p className="message">{message}</p>}

      <section className="panel historyPanel" aria-label="Previous prompts">
        <div className="panelHeader">
          <div>
            <p className="sectionLabel">History</p>
            <h2>Previous prompts</h2>
          </div>
          <span className="timeBadge">Last {history.entries.length}</span>
        </div>
        <div className="historyList">
          {history.entries.length ? (
            history.entries.map((entry) => <HistoryItem key={`${entry.timestamp}-${entry.final_text}`} entry={entry} />)
          ) : (
            <p className="emptyState">Recent prompts will appear here after dictation.</p>
          )}
        </div>
      </section>
    </main>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <article className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  )
}

function PromptBlock({ label, text }: { label: string; text: string }) {
  return (
    <div className="promptBlock">
      <span>{label}</span>
      <p>{text}</p>
    </div>
  )
}

function HistoryItem({ entry }: { entry: DictationHistoryEntry }) {
  return (
    <article className="historyItem">
      <div className="historyMeta">
        <strong>{formatTime(entry.timestamp)}</strong>
        <span>{entry.word_count} words</span>
        <span>{formatDuration(entry.audio_duration_ms)}</span>
        <span>{formatDecimal(entry.words_per_minute)} WPM</span>
        <span>STT {formatMilliseconds(entry.transcription_ms)}</span>
        <span>Cleanup {formatMilliseconds(entry.cleanup_ms)}</span>
      </div>
      <div className="historyPrompts">
        <PromptBlock label="Original" text={entry.raw_text} />
        <PromptBlock label="Inserted" text={entry.final_text} />
      </div>
    </article>
  )
}

function formatInteger(value: number) {
  return new Intl.NumberFormat().format(value)
}

function formatDecimal(value: number) {
  return value.toFixed(1)
}

function formatMilliseconds(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)}s`
  }
  return `${value}ms`
}

function formatDuration(value: number) {
  const totalSeconds = Math.round(value / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes === 0) {
    return `${seconds}s`
  }
  return `${minutes}m ${seconds}s`
}

function formatTime(value: string) {
  const normalized = value.replace(/([+-]\d{2})(\d{2})$/, '$1:$2')
  const date = new Date(normalized)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export default App
