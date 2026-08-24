import { parseClassName } from '../utils/formatClassName.js'

export const REVIEW_THRESHOLD = 0.7

const SIZE_CLASSES = {
  md: 'px-3 py-1 text-[11px] gap-2',
  sm: 'px-2 py-0.5 text-[10px] gap-1.5',
}

// Disease status: purely a function of predicted_class. Independent of confidence.
export function DiseaseBadge({ predictedClass, size = 'md' }) {
  const { condition, isHealthy } = parseClassName(predictedClass)
  const sizing = SIZE_CLASSES[size]

  if (isHealthy) {
    return (
      <span
        className={`inline-flex items-center whitespace-nowrap border border-leaf/40 bg-leaf/10 font-mono uppercase tracking-widest2 text-leaf ${sizing}`}
      >
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-leaf" aria-hidden="true" />
        Healthy
      </span>
    )
  }

  return (
    <span
      className={`inline-flex items-center border border-rust/50 bg-rust/10 font-mono uppercase tracking-widest2 text-rust ${sizing}`}
    >
      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-rust" aria-hidden="true" />
      <span className="whitespace-nowrap">Disease Detected</span>
      <span className="text-rust/50" aria-hidden="true">
        &middot;
      </span>
      <span>{condition}</span>
    </span>
  )
}

// Confidence status: purely a function of the confidence score. Independent of disease status.
export function ConfidenceBadge({ confidence, size = 'md' }) {
  const sizing = SIZE_CLASSES[size]
  const needsReview = confidence < REVIEW_THRESHOLD

  if (needsReview) {
    return (
      <span
        className={`inline-flex items-center whitespace-nowrap border-2 border-rust bg-rust/20 font-mono font-semibold uppercase tracking-widest2 text-rust shadow-rustglow ${sizing}`}
      >
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-rust" aria-hidden="true" />
        Needs Review
      </span>
    )
  }

  return (
    <span
      className={`inline-flex items-center whitespace-nowrap border border-leaf/25 bg-leaf/5 font-mono uppercase tracking-widest2 text-leaf/80 ${sizing}`}
    >
      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-leaf/70" aria-hidden="true" />
      Confident
    </span>
  )
}

// Both badges together — disease status and confidence status are independent
// axes, so a result can be any combination of the two (e.g. a confident
// disease call, or a healthy call the model isn't sure about).
export default function StatusBadges({ predictedClass, confidence, size = 'md' }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <DiseaseBadge predictedClass={predictedClass} size={size} />
      <ConfidenceBadge confidence={confidence} size={size} />
    </div>
  )
}
