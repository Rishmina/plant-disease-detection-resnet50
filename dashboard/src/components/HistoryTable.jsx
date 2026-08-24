import { parseClassName, formatPercent } from '../utils/formatClassName.js'
import StatusBadges, { REVIEW_THRESHOLD } from './StatusBadges.jsx'

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export default function HistoryTable({ entries }) {
  return (
    <section className="rounded-none border border-panel-border bg-panel shadow-panel">
      <header className="flex items-center justify-between border-b border-panel-border px-5 py-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage">Log</p>
          <h2 className="font-heading text-xl text-bone">Session History</h2>
        </div>
        <span className="font-mono text-[11px] text-sage-dim">{entries.length} scans</span>
      </header>

      {entries.length === 0 ? (
        <div className="px-5 py-10 text-center">
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage-dim">
            Scans from this session will appear here
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] border-collapse text-left">
            <thead>
              <tr className="border-b border-panel-border">
                <th className="px-5 py-3 font-mono text-[11px] font-medium uppercase tracking-widest2 text-sage-dim">
                  Sample
                </th>
                <th className="px-5 py-3 font-mono text-[11px] font-medium uppercase tracking-widest2 text-sage-dim">
                  Predicted Class
                </th>
                <th className="px-5 py-3 font-mono text-[11px] font-medium uppercase tracking-widest2 text-sage-dim">
                  Confidence
                </th>
                <th className="px-5 py-3 font-mono text-[11px] font-medium uppercase tracking-widest2 text-sage-dim">
                  Status
                </th>
                <th className="px-5 py-3 font-mono text-[11px] font-medium uppercase tracking-widest2 text-sage-dim">
                  Time
                </th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => {
                const { crop, condition } = parseClassName(entry.predicted_class)
                const needsReview = entry.confidence < REVIEW_THRESHOLD
                return (
                  <tr key={entry.id} className="border-b border-panel-border/60 last:border-0">
                    <td className="px-5 py-3">
                      {entry.thumbnail ? (
                        <img
                          src={entry.thumbnail}
                          alt=""
                          className="h-12 w-12 border border-panel-border object-cover"
                        />
                      ) : (
                        <div
                          className="flex h-12 w-12 items-center justify-center border border-panel-border bg-forest-950/40 font-mono text-[10px] text-sage-dim"
                          aria-hidden="true"
                        >
                          N/A
                        </div>
                      )}
                    </td>
                    <td className="px-5 py-3">
                      <p className="text-sm text-bone">{condition}</p>
                      <p className="font-mono text-[11px] text-sage-dim">{crop}</p>
                    </td>
                    <td className="px-5 py-3">
                      <span
                        className={`font-mono text-sm ${
                          needsReview ? 'text-rust' : 'text-leaf'
                        }`}
                      >
                        {formatPercent(entry.confidence)}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <StatusBadges
                        predictedClass={entry.predicted_class}
                        confidence={entry.confidence}
                        size="sm"
                      />
                    </td>
                    <td className="px-5 py-3 font-mono text-[12px] text-sage">
                      {formatTime(entry.timestamp)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
