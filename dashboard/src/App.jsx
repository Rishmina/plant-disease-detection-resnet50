import { useCallback, useEffect, useRef, useState } from 'react'
import UploadPanel from './components/UploadPanel.jsx'
import ResultCard from './components/ResultCard.jsx'
import ConfidenceBars from './components/ConfidenceBars.jsx'
import HistoryTable from './components/HistoryTable.jsx'
import StatusBanner from './components/StatusBanner.jsx'
import { predictImage, checkHealth, ApiError } from './api.js'
import { createThumbnail } from './utils/image.js'
import { parseClassName, formatPercent } from './utils/formatClassName.js'

const HEALTH_POLL_MS = 15000
const MAX_HISTORY_ENTRIES = 25

export default function App() {
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [result, setResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [history, setHistory] = useState([])
  const [apiStatus, setApiStatus] = useState('checking') // checking | ok | model_missing | offline
  const [liveMessage, setLiveMessage] = useState('')

  const abortRef = useRef(null)
  const previewUrlRef = useRef(null)
  const lastFileRef = useRef(null)

  const pollHealth = useCallback(async () => {
    const health = await checkHealth()
    if (health.status === 'unreachable') setApiStatus('offline')
    else if (!health.model_loaded) setApiStatus('model_missing')
    else setApiStatus('ok')
  }, [])

  useEffect(() => {
    pollHealth()
    const id = setInterval(() => {
      if (!document.hidden) pollHealth()
    }, HEALTH_POLL_MS)

    const onVisibilityChange = () => {
      if (!document.hidden) pollHealth()
    }
    document.addEventListener('visibilitychange', onVisibilityChange)

    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [pollHealth])

  // Revoke the object URL whenever it's replaced or the app unmounts, so
  // large image previews don't pile up in memory over a long session.
  useEffect(() => {
    previewUrlRef.current = previewUrl
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    }
  }, [previewUrl])

  useEffect(() => () => abortRef.current?.abort(), [])

  const runPrediction = useCallback(
    async (file) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      lastFileRef.current = file
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        return URL.createObjectURL(file)
      })
      setStatus('loading')
      setErrorMessage('')
      setLiveMessage('Running inference on the uploaded sample.')

      try {
        const [prediction, thumbnail] = await Promise.all([
          predictImage(file, { signal: controller.signal }),
          createThumbnail(file).catch(() => null),
        ])

        setResult(prediction)
        setStatus('success')
        pollHealth()

        const { crop, condition } = parseClassName(prediction.predicted_class)
        setLiveMessage(
          `Result: ${crop} ${condition}, ${formatPercent(prediction.confidence)} confidence.`
        )

        setHistory((prev) =>
          [
            {
              id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
              thumbnail,
              predicted_class: prediction.predicted_class,
              confidence: prediction.confidence,
              timestamp: Date.now(),
            },
            ...prev,
          ].slice(0, MAX_HISTORY_ENTRIES)
        )
      } catch (err) {
        if (err.name === 'AbortError') {
          // Only reset to idle if *this* request was the one the user cancelled —
          // if it was aborted because a newer upload superseded it, leave the
          // (already-updated) status alone.
          if (controller.signal.reason === 'user-cancelled') {
            setStatus('idle')
            setLiveMessage('Scan cancelled.')
          }
          return
        }
        setStatus('error')
        setResult(null)
        if (err instanceof ApiError) {
          setErrorMessage(err.message)
          setLiveMessage(`Scan failed: ${err.message}`)
          if (err.status === 0) setApiStatus('offline')
          if (err.status === 503) setApiStatus('model_missing')
        } else {
          setErrorMessage('Something went wrong reading that image.')
          setLiveMessage('Scan failed.')
        }
      }
    },
    [pollHealth]
  )

  const handleCancel = useCallback(() => {
    abortRef.current?.abort('user-cancelled')
  }, [])

  const handleRetry = useCallback(() => {
    if (lastFileRef.current) runPrediction(lastFileRef.current)
  }, [runPrediction])

  return (
    <div className="min-h-screen">
      <div aria-live="polite" className="sr-only">
        {liveMessage}
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage">
              Vertical Farm · Canopy Health Unit
            </p>
            <h1 className="font-heading text-3xl font-medium text-bone sm:text-4xl">
              Canopy Scan <span className="text-leaf">/</span> Disease Monitor
            </h1>
          </div>
        </header>

        <div className="mb-6">
          <StatusBanner apiStatus={apiStatus} />
        </div>

        <main className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <UploadPanel
            previewUrl={previewUrl}
            status={status}
            onFile={runPrediction}
            onCancel={handleCancel}
          />

          <div className="flex flex-col gap-6">
            <ResultCard
              status={status}
              result={result}
              errorMessage={errorMessage}
              onRetry={handleRetry}
            />
            <ConfidenceBars
              status={status}
              probabilities={result?.all_class_probabilities}
            />
          </div>
        </main>

        <div className="mt-6">
          <HistoryTable entries={history} />
        </div>

        <footer className="mt-8 pb-4 text-center font-mono text-[11px] uppercase tracking-widest2 text-sage-dim">
          Session-local history only · Not persisted between refreshes
        </footer>
      </div>
    </div>
  )
}
