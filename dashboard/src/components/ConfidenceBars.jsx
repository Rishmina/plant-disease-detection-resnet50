import { parseClassName, formatPercent } from '../utils/formatClassName.js'

export default function ConfidenceBars({ status, probabilities }) {
  const rows = probabilities
    ? Object.entries(probabilities)
        .map(([className, value]) => ({ className, value, ...parseClassName(className) }))
        .sort((a, b) => b.value - a.value)
    : []

  return (
    <section className="rounded-none border border-panel-border bg-panel shadow-panel">
      <header className="flex items-center justify-between border-b border-panel-border px-5 py-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage">All classes</p>
          <h2 className="font-heading text-xl text-bone">Probability Breakdown</h2>
        </div>
        {rows.length > 0 && (
          <span className="font-mono text-[11px] text-sage-dim">{rows.length} classes</span>
        )}
      </header>

      <div className="p-5">
        {status === 'idle' && (
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage-dim">
            No sample scanned yet
          </p>
        )}

        {status === 'loading' && (
          <ul className="space-y-3" aria-hidden="true">
            {Array.from({ length: 6 }).map((_, i) => (
              <li key={i} className="h-4 w-full animate-pulse bg-panel-light" style={{ width: `${90 - i * 8}%` }} />
            ))}
          </ul>
        )}

        {status === 'error' && (
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage-dim">
            Breakdown unavailable
          </p>
        )}

        {status === 'success' && rows.length > 0 && (
          <ul className="space-y-2.5">
            {rows.map((row, i) => {
              const isTop = i === 0
              return (
                <li key={row.className}>
                  <div className="mb-1 flex items-baseline justify-between gap-2">
                    <span
                      className={`truncate text-sm ${isTop ? 'text-bone' : 'text-bone/75'}`}
                      title={row.label}
                    >
                      {row.crop} <span className="text-sage">·</span> {row.condition}
                    </span>
                    <span
                      className={`shrink-0 font-mono text-sm ${isTop ? 'text-leaf' : 'text-sage'}`}
                    >
                      {formatPercent(row.value)}
                    </span>
                  </div>
                  <div className="h-2 w-full origin-left bg-forest-950/60">
                    <div
                      className={`h-2 origin-left animate-bar-rise ${
                        isTop ? 'bg-leaf' : 'bg-sage-dim'
                      }`}
                      style={{ width: `${Math.max(row.value * 100, 1)}%` }}
                    />
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </section>
  )
}
