/**
 * SentimentInferer — browser-side FinBERT sentiment scoring.
 *
 * Uses @xenova/transformers (Transformers.js v2) to run Xenova/finbert
 * entirely in the browser. The quantized model (~80MB) is downloaded once
 * from HuggingFace CDN and cached permanently in IndexedDB.
 *
 * FinBERT output: {label: "positive"|"negative"|"neutral", score: float}
 * Net sentiment: positive_score − negative_score ∈ [-1, +1]
 *
 * Falls back gracefully if device memory < 4GB (navigator.deviceMemory).
 *
 * References
 * ----------
 * Malo, P., Sinha, A., Korhonen, P., Wallenius, J. & Takala, P. (2014).
 *   Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.
 *   Journal of the American Society for Information Science and Technology,
 *   65(4), 782–796. https://doi.org/10.1002/asi.23062
 *
 * Yang, Y., UY, M.C.S., & Huang, A. (2020). FinBERT: A Pretrained Language
 *   Model for Financial Communications. arXiv:2006.08097.
 *   https://arxiv.org/abs/2006.08097
 */

let _pipeline = null
let _loadPromise = null

/**
 * Load (or return cached) FinBERT pipeline.
 * @param {function} onProgress  Called with {status, progress, file} during download.
 * @returns {Promise<pipeline>}
 */
export async function loadFinBERT(onProgress) {
  if (_pipeline) return _pipeline
  if (_loadPromise) return _loadPromise

  _loadPromise = (async () => {
    const { pipeline } = await import('@xenova/transformers')
    _pipeline = await pipeline(
      'sentiment-analysis',
      'Xenova/finbert',
      { progress_callback: onProgress ?? (() => {}) },
    )
    return _pipeline
  })()

  return _loadPromise
}

/**
 * Score a list of headlines. Returns an array of {label, score, net} objects.
 * net ∈ [-1, +1]: positive_score − negative_score.
 *
 * @param {string[]} headlines
 * @param {function} onProgress  Passed to loadFinBERT on first load.
 * @returns {Promise<Array<{label:string, score:number, net:number}>>}
 */
export async function scoreHeadlines(headlines, onProgress) {
  if (!headlines?.length) return []
  const clf = await loadFinBERT(onProgress)

  const results = await Promise.all(
    headlines.map(h => clf(h.slice(0, 512)))   // FinBERT max 512 tokens
  )

  return results.map(raw => {
    const arr    = Array.isArray(raw) ? raw : [raw]
    const posObj = arr.find(x => x.label?.toLowerCase() === 'positive') ?? { score: 0 }
    const negObj = arr.find(x => x.label?.toLowerCase() === 'negative') ?? { score: 0 }
    const best   = arr.reduce((a, b) => a.score > b.score ? a : b)
    return {
      label: best.label?.toLowerCase() ?? 'neutral',
      score: best.score,
      net:   posObj.score - negObj.score,
    }
  })
}

/**
 * Aggregate per-headline scores into a ticker-level summary.
 * @param {{label:string, net:number}[]} scores
 * @returns {{net: number, positive: number, negative: number, neutral: number, n: number}}
 */
export function aggregateScores(scores) {
  if (!scores?.length) return { net: 0, positive: 0, negative: 0, neutral: 0, n: 0 }
  const n        = scores.length
  const positive = scores.filter(s => s.label === 'positive').length / n
  const negative = scores.filter(s => s.label === 'negative').length / n
  const neutral  = scores.filter(s => s.label === 'neutral').length  / n
  const net      = scores.reduce((sum, s) => sum + s.net, 0) / n
  return { net: +net.toFixed(3), positive: +positive.toFixed(3), negative: +negative.toFixed(3), neutral: +neutral.toFixed(3), n }
}

/**
 * Returns true if this device is likely capable of running FinBERT in-browser.
 * Uses navigator.deviceMemory (Chrome/Edge only; undefined on Safari/Firefox → assume capable).
 */
export function deviceCanRunFinBERT() {
  const mem = navigator?.deviceMemory
  return mem == null || mem >= 4
}
