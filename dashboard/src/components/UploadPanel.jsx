import { useCallback, useRef, useState } from 'react'
import ScannerOverlay from './ScannerOverlay.jsx'
import { validateImageFile } from '../utils/image.js'

export default function UploadPanel({ previewUrl, status, onFile, onCancel }) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [localError, setLocalError] = useState('')
  const inputRef = useRef(null)

  const handleFiles = useCallback(
    (fileList) => {
      const file = fileList?.[0]
      if (!file) return

      const validationError = validateImageFile(file)
      if (validationError) {
        setLocalError(validationError)
        return
      }

      setLocalError('')
      onFile(file)
    },
    [onFile]
  )

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  const onDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const onDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const onInputChange = (e) => {
    handleFiles(e.target.files)
    e.target.value = '' // allow re-selecting the same file
  }

  const isScanning = status === 'loading'

  return (
    <section className="rounded-none border border-panel-border bg-panel shadow-panel">
      <header className="flex items-center justify-between border-b border-panel-border px-5 py-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage">Intake / Camera 01</p>
          <h2 className="font-heading text-xl text-bone">Sample Scanner</h2>
        </div>
        <span
          className={`h-2.5 w-2.5 rounded-full ${
            isScanning ? 'bg-leaf animate-pulse' : 'bg-sage-dim'
          }`}
          aria-hidden="true"
        />
      </header>

      <div className="p-5">
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload a leaf image by dragging a file here or pressing Enter to browse"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`relative flex aspect-square w-full cursor-pointer flex-col items-center justify-center overflow-hidden border-2 border-dashed transition-colors focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-leaf ${
            isDragOver ? 'border-leaf bg-leaf/5' : 'border-panel-border bg-forest-950/40'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/bmp"
            className="sr-only"
            onChange={onInputChange}
          />

          {previewUrl ? (
            <>
              <img
                src={previewUrl}
                alt="Uploaded leaf sample preview"
                className="h-full w-full object-cover"
              />
              <ScannerOverlay active={isScanning} />
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 px-6 text-center">
              <svg
                width="40"
                height="40"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="text-sage"
                aria-hidden="true"
              >
                <rect x="3" y="4" width="18" height="16" rx="0.5" strokeLinejoin="round" />
                <circle cx="8.5" cy="9.5" r="1.5" />
                <path
                  d="M3 16l5-5a1.5 1.5 0 012.12 0L15 15.9m-.5-.4l1.88-1.88a1.5 1.5 0 012.12 0L21 16.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <p className="font-body text-sm text-bone">
                Drop a leaf image here, or <span className="text-leaf underline underline-offset-2">click to browse</span>
              </p>
              <p className="font-mono text-[11px] uppercase tracking-widest2 text-sage-dim">
                JPG · PNG · WEBP · BMP, up to 10MB
              </p>
            </div>
          )}
        </div>

        {localError && (
          <p role="alert" className="mt-3 border border-rust/40 bg-rust/10 px-3 py-2 text-sm text-rust">
            {localError}
          </p>
        )}

        {isScanning && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="mt-4 w-full border border-rust/40 bg-rust/5 px-4 py-2 font-mono text-xs uppercase tracking-widest2 text-rust transition-colors hover:bg-rust/15 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rust"
          >
            Cancel scan
          </button>
        )}

        {previewUrl && !isScanning && (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="mt-4 w-full border border-panel-border bg-forest-950/40 px-4 py-2 font-mono text-xs uppercase tracking-widest2 text-sage transition-colors hover:border-leaf hover:text-leaf focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-leaf"
          >
            Load a different sample
          </button>
        )}
      </div>
    </section>
  )
}
