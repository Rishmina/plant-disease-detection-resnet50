import { describe, expect, it } from 'vitest'
import { parseClassName, formatPercent } from './formatClassName.js'

describe('parseClassName', () => {
  it('splits crop and disease for a simple two-token class', () => {
    const result = parseClassName('Potato___Early_blight')
    expect(result.crop).toBe('Potato')
    expect(result.condition).toBe('Early Blight')
    expect(result.isHealthy).toBe(false)
  })

  it('handles the Pepper (Bell) crop qualifier', () => {
    const result = parseClassName('Pepper__bell___Bacterial_spot')
    expect(result.crop).toBe('Pepper (Bell)')
    expect(result.condition).toBe('Bacterial Spot')
  })

  it('flags healthy classes', () => {
    const result = parseClassName('Tomato_healthy')
    expect(result.isHealthy).toBe(true)
    expect(result.condition).toBe('Healthy')
  })

  it('drops a duplicated crop token and splits camelCase', () => {
    const result = parseClassName('Tomato__Tomato_YellowLeaf__Curl_Virus')
    expect(result.crop).toBe('Tomato')
    expect(result.condition).toBe('Yellow Leaf Curl Virus')
  })

  it('handles a long multi-word disease name', () => {
    const result = parseClassName('Tomato_Spider_mites_Two_spotted_spider_mite')
    expect(result.condition).toBe('Spider Mites Two Spotted Spider Mite')
  })
})

describe('formatPercent', () => {
  it('converts a 0-1 fraction to a percentage string', () => {
    expect(formatPercent(0.9231)).toBe('92.3%')
  })

  it('respects the digits argument', () => {
    expect(formatPercent(0.7, 0)).toBe('70%')
  })
})
