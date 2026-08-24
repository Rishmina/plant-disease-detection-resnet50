import { API_BASE_URL } from '../api.js'

export default function StatusBanner({ apiStatus }) {
  const isOnline = apiStatus === 'ok'
  const isChecking = apiStatus === 'checking'

  return (
    <div
      className={`flex items-center justify-between gap-3 border px-4 py-2 font-mono text-[11px] uppercase tracking-widest2 ${
        isOnline
          ? 'border-leaf/30 bg-leaf/5 text-leaf'
          : isChecking
            ? 'border-panel-border bg-panel text-sage'
            : 'border-rust/40 bg-rust/10 text-rust'
      }`}
      role="status"
    >
      <span className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${
            isOnline ? 'bg-leaf animate-pulse' : isChecking ? 'bg-sage-dim' : 'bg-rust'
          }`}
          aria-hidden="true"
        />
        {isOnline && 'System online — model loaded'}
        {isChecking && 'Checking connection…'}
        {apiStatus === 'model_missing' && 'API reachable — model not loaded'}
        {apiStatus === 'offline' && `API unreachable at ${API_BASE_URL}`}
      </span>
      {!isOnline && !isChecking && (
        <span className="hidden text-sage-dim sm:inline">
          Start it with: uvicorn predict_api:app --reload
        </span>
      )}
    </div>
  )
}
