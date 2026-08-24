export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024 // 10MB
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

export function validateImageFile(file) {
  if (!file.type.startsWith('image/') || !ACCEPTED_TYPES.includes(file.type)) {
    return 'Unsupported file type. Use JPG, PNG, WEBP, or BMP.'
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `File is too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Max size is ${MAX_UPLOAD_BYTES / 1024 / 1024}MB.`
  }
  return null
}

// Downscales an image file to a small JPEG data URL so history entries stay
// cheap to keep in memory — the full-resolution file is never retained.
export function createThumbnail(file, maxDim = 160) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    const objectUrl = URL.createObjectURL(file)

    img.onload = () => {
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height))
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(img.width * scale))
      canvas.height = Math.max(1, Math.round(img.height * scale))

      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(objectUrl)
      resolve(canvas.toDataURL('image/jpeg', 0.7))
    }

    img.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('Could not decode image for thumbnail'))
    }

    img.src = objectUrl
  })
}
