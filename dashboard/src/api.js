export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'https://plant-disease-detection-resnet50.onrender.com'

const PREDICT_TIMEOUT_MS = 20000
const HEALTH_TIMEOUT_MS = 5000

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

// Combines a caller-supplied AbortSignal (e.g. "cancel this upload") with an
// internal timeout, so a hung request can't strand the UI in a loading state.
function withTimeout(timeoutMs, externalSignal) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(new DOMException('Timed out', 'TimeoutError')), timeoutMs)

  if (externalSignal) {
    if (externalSignal.aborted) controller.abort(externalSignal.reason)
    else externalSignal.addEventListener('abort', () => controller.abort(externalSignal.reason), { once: true })
  }

  return { signal: controller.signal, cancel: () => clearTimeout(timeoutId) }
}

// POST /predict — multipart/form-data with a "file" field.
// Returns { predicted_class, confidence, inference_time_ms, all_class_probabilities }
export async function predictImage(file, { signal } = {}) {
  const formData = new FormData()
  formData.append('file', file)

  const { signal: combinedSignal, cancel } = withTimeout(PREDICT_TIMEOUT_MS, signal)

  let response
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: combinedSignal,
    })
  } catch (err) {
    if (signal?.aborted) throw err // caller-initiated cancel — let it propagate as AbortError
    if (err.name === 'TimeoutError' || combinedSignal.reason?.name === 'TimeoutError') {
      throw new ApiError('The request took too long. Is the model still loading?', 0)
    }
    throw new ApiError(`Could not reach the API at ${API_BASE_URL}. Is uvicorn running?`, 0)
  } finally {
    cancel()
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    throw new ApiError(detail, response.status)
  }

  return response.json()
}

export async function checkHealth({ signal } = {}) {
  const { signal: combinedSignal, cancel } = withTimeout(HEALTH_TIMEOUT_MS, signal)
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: combinedSignal })
    if (!res.ok) return { status: 'error', model_loaded: false }
    return await res.json()
  } catch {
    return { status: 'unreachable', model_loaded: false }
  } finally {
    cancel()
  }
}
