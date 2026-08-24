import { parseClassName, formatPercent } from '../utils/formatClassName.js'
import StatusBadges, { REVIEW_THRESHOLD } from './StatusBadges.jsx'

function Placeholder({ label }) {
  return (
    <div className="flex h-full min-h-[220px] flex-col items-center justify-center gap-2 border border-dashed border-panel-border px-6 text-center">
      <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage-dim">{label}</p>
    </div>
  )
}

export default function ResultCard({ status, result, errorMessage, onRetry }) {
  return (
    <section className="rounded-none border border-panel-border bg-panel shadow-panel">
      <header className="flex items-center justify-between border-b border-panel-border px-5 py-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage">Readout</p>
          <h2 className="font-heading text-xl text-bone">Diagnosis</h2>
        </div>
      </header>

      <div className="p-5">
        {status === 'idle' && <Placeholder label="Awaiting sample — load a leaf image to begin" />}

        {status === 'loading' && (
          <div className="flex h-full min-h-[220px] flex-col items-center justify-center gap-3 border border-panel-border bg-forest-950/30 px-6 text-center">
            <span className="font-mono text-[11px] uppercase tracking-widest2 text-leaf animate-flicker motion-reduce:animate-none">
              Running inference…
            </span>
            <div className="h-px w-2/3 overflow-hidden bg-panel-border">
              <div className="h-px w-1/3 animate-pulse bg-leaf" />
            </div>
          </div>
        )}

        {status === 'error' && (
          <div className="flex h-full min-h-[220px] flex-col items-center justify-center gap-3 border border-rust/40 bg-rust/5 px-6 text-center">
            <p className="font-mono text-[11px] uppercase tracking-widest2 text-rust">Scan failed</p>
            <p className="max-w-xs text-sm text-bone/90">{errorMessage}</p>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="border border-rust/50 bg-rust/10 px-4 py-2 font-mono text-xs uppercase tracking-widest2 text-rust transition-colors hover:bg-rust/20 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rust"
              >
                Retry scan
              </button>
            )}
          </div>
        )}

        {status === 'success' && result && (() => {
          const { crop, condition, isHealthy } = parseClassName(result.predicted_class)
          const needsReview = result.confidence < REVIEW_THRESHOLD

          return (
            <div className="animate-fade-up space-y-4">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage">{crop}</p>
                <h3
                  className={`font-heading text-2xl leading-tight ${
                    isHealthy ? 'text-leaf' : 'text-bone'
                  }`}
                >
                  {condition}
                </h3>
              </div>

              <StatusBadges
                predictedClass={result.predicted_class}
                confidence={result.confidence}
                size="md"
              />

              <div className="grid grid-cols-2 gap-3">
                <div className="border border-panel-border bg-forest-950/40 px-4 py-3">
                  <p className="font-mono text-[10px] uppercase tracking-widest2 text-sage-dim">Confidence</p>
                  <p
                    className={`font-mono text-3xl font-semibold ${
                      needsReview ? 'text-rust' : 'text-leaf'
                    }`}
                  >
                    {formatPercent(result.confidence)}
                  </p>
                </div>
                <div className="border border-panel-border bg-forest-950/40 px-4 py-3">
                  <p className="font-mono text-[10px] uppercase tracking-widest2 text-sage-dim">Inference time</p>
                  <p className="font-mono text-3xl font-semibold text-bone">
                    {result.inference_time_ms.toFixed(1)}
                    <span className="ml-1 text-base text-sage">ms</span>
                  </p>
                </div>
              </div>

              <p className="font-mono text-[11px] text-sage">
                Class ID: <span className="text-bone/90">{result.predicted_class}</span>
              </p>
            </div>
          )
        })()}
      </div>
    </section>
  )
}
