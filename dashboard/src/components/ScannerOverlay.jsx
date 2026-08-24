export default function ScannerOverlay({ active }) {
  if (!active) return null

  return (
    <div
      className="pointer-events-none absolute inset-0 z-20 overflow-hidden"
      aria-hidden="true"
    >
      {/* dim wash so the sweep reads clearly against the leaf photo */}
      <div className="absolute inset-0 bg-forest-950/30" />

      {/* sweeping scan line */}
      <div className="absolute inset-x-0 top-0 h-full">
        <div className="absolute left-0 right-0 h-16 animate-scanline motion-reduce:animate-none motion-reduce:opacity-60 motion-reduce:top-1/2">
          <div className="h-px w-full bg-leaf shadow-glow" />
          <div className="h-16 w-full bg-gradient-to-b from-leaf/25 via-leaf/5 to-transparent" />
        </div>
      </div>

      {/* corner reticle brackets */}
      {[
        'top-3 left-3 border-t-2 border-l-2',
        'top-3 right-3 border-t-2 border-r-2',
        'bottom-3 left-3 border-b-2 border-l-2',
        'bottom-3 right-3 border-b-2 border-r-2',
      ].map((pos) => (
        <div
          key={pos}
          className={`absolute h-7 w-7 ${pos} border-leaf animate-reticlePulse motion-reduce:animate-none`}
        />
      ))}

      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-none border border-leaf/40 bg-forest/80 px-3 py-1">
        <span className="font-mono text-[11px] uppercase tracking-widest2 text-leaf animate-flicker motion-reduce:animate-none">
          Analyzing sample…
        </span>
      </div>
    </div>
  )
}
