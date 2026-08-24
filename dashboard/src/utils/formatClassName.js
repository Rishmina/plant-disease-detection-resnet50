// Class names come straight from the ONNX model's CLASS_NAMES list, e.g.
// "Pepper__bell___Bacterial_spot" or "Tomato__Tomato_YellowLeaf__Curl_Virus".
// This turns them into a { crop, condition, isHealthy, label } shape for display.

function titleCaseWord(word) {
  if (!word) return word
  return word[0].toUpperCase() + word.slice(1).toLowerCase()
}

export function parseClassName(raw) {
  if (!raw) return { crop: 'Unknown', condition: 'Unknown', isHealthy: false, label: 'Unknown' }

  let tokens = raw.split(/_+/).filter(Boolean)
  let crop = tokens[0]
  let rest = tokens.slice(1)

  // "Pepper__bell___Bacterial_spot" -> crop "Pepper (Bell)"
  if (crop.toLowerCase() === 'pepper' && rest[0]?.toLowerCase() === 'bell') {
    crop = 'Pepper (Bell)'
    rest = rest.slice(1)
  }

  // "Tomato__Tomato_YellowLeaf__Curl_Virus" -> drop the duplicated crop token
  if (rest[0]?.toLowerCase() === crop.toLowerCase()) {
    rest = rest.slice(1)
  }

  const conditionRaw = rest
    .join(' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2') // split "YellowLeaf" -> "Yellow Leaf"
    .split(' ')
    .map(titleCaseWord)
    .join(' ')
    .trim()

  const isHealthy = conditionRaw.toLowerCase() === 'healthy'

  return {
    crop: titleCaseWord(crop.split(' ')[0]) + crop.slice(crop.split(' ')[0].length),
    condition: conditionRaw || 'Healthy',
    isHealthy,
    label: `${crop} — ${conditionRaw}`,
  }
}

export function formatPercent(fraction, digits = 1) {
  return `${(fraction * 100).toFixed(digits)}%`
}
