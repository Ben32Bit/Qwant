/**
 * MetaEnsemble — Client-Side OOS R²-Weighted Ensemble (Phase C.4).
 *
 * Combines base model fan bands weighted by each method's OOS R² on a
 * 30-day shadow holdout window. Disagreement across models widens the
 * ensemble confidence bands (Krogh & Vedelsby 1995).
 *
 * Two inference paths:
 *   1. Server weights (preferred): `ensemble_weights` from ForecastResponse,
 *      computed server-side by get_ensemble_weights_from_oos_r2().
 *   2. Client fallback: `weightsFromR2()` mirrors the backend algorithm
 *      using oos_r2 from each MethodResult (used when server weights absent).
 *
 * References
 * ----------
 * Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
 * Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation,
 *   and Active Learning. NeurIPS 8, 231–238.
 * Lakshminarayanan, B. et al. (2017). Simple and Scalable Predictive Uncertainty
 *   Estimation Using Deep Ensembles. NeurIPS 30.
 */

// ── ONNX model cache ──────────────────────────────────────────────────────────
const _onnxCache = {}

// Per-regime meta ONNX models aren't built yet. The HEAD probe below used to
// fire on every forecast run (one per regime), spraying red 404s in devtools
// even though the fallback rule-based path handled it silently. Flip this
// flag when an actual training pipeline ships models to /models/meta/.
const META_ONNX_AVAILABLE = false

/**
 * Try to load a per-regime ONNX model. Returns null if unavailable.
 * Models are expected at /models/meta/{regime}.onnx (Vercel static asset).
 */
async function tryLoadOnnxModel(regime) {
  if (!META_ONNX_AVAILABLE) return null
  if (_onnxCache[regime] !== undefined) return _onnxCache[regime]
  try {
    if (typeof window === 'undefined' || !window.ort) { _onnxCache[regime] = null; return null }
    const url = `/models/meta/${regime}.onnx`
    const resp = await fetch(url, { method: 'HEAD' })
    if (!resp.ok) { _onnxCache[regime] = null; return null }
    const session = await window.ort.InferenceSession.create(url)
    _onnxCache[regime] = session
    return session
  } catch {
    _onnxCache[regime] = null
    return null
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Compute disagreement features from base model p50 arrays.
 *
 * @param {Array}  results     - array of MethodResult from useForecast
 * @param {number} horizonIdx  - forecast step index to evaluate (default 20 = 21d)
 * @returns {{level:string, spread:number, variance:number, n:number, methodVals:Object}}
 */
export function computeDisagreement(results, horizonIdx = 20) {
  const methodVals = {}
  for (const r of results) {
    if (r.forecast?.p50?.length > horizonIdx) {
      methodVals[r.method] = r.forecast.p50[horizonIdx]
    }
  }
  const vals = Object.values(methodVals)
  const n    = vals.length
  if (n < 2) return { level: 'low', spread: 0, variance: 0, n, methodVals }

  const mean     = vals.reduce((a, b) => a + b, 0) / n
  const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n
  const spread   = Math.max(...vals) - Math.min(...vals)

  const level = variance > 2.5 || spread > 8.0 ? 'high'
              : variance > 0.5  || spread > 3.0 ? 'med'
              : 'low'

  return { level, spread: +spread.toFixed(3), variance: +variance.toFixed(4), n, methodVals }
}

/**
 * Compute ensemble weights from OOS R² values (mirrors backend get_ensemble_weights_from_oos_r2).
 *
 * Used as a client-side fallback when serverWeights are not yet available.
 * TimesFM gets a 30% floor for stability on short shadow windows.
 *
 * @param  {Array}  results   - MethodResult[] with .oos_r2 fields
 * @returns {Object} {method: weight}
 */
export function weightsFromR2(results) {
  const TFM_FLOOR = 0.30
  const r2s = {}
  for (const r of results) {
    if (r.forecast?.p50?.length > 0) r2s[r.method] = r.oos_r2 ?? 0
  }

  const clipped = {}
  for (const [m, r2] of Object.entries(r2s)) clipped[m] = Math.max(0, r2)
  const total = Object.values(clipped).reduce((a, b) => a + b, 0)

  const weights = {}
  if (total <= 0) {
    const n = Object.keys(r2s).length
    for (const m of Object.keys(r2s)) weights[m] = n > 0 ? +(1 / n).toFixed(4) : 0
    return weights
  }

  for (const [m, w] of Object.entries(clipped)) weights[m] = w / total

  // TimesFM 30% floor
  if ('timesfm' in weights && weights.timesfm < TFM_FLOOR) {
    const deficit = TFM_FLOOR - weights.timesfm
    weights.timesfm = TFM_FLOOR
    const others = Object.fromEntries(Object.entries(weights).filter(([m]) => m !== 'timesfm'))
    const otherTotal = Object.values(others).reduce((a, b) => a + b, 0)
    if (otherTotal > 0) {
      const scale = (1 - TFM_FLOOR) / otherTotal
      for (const m in others) weights[m] = others[m] * scale
    }
  }

  for (const m in weights) weights[m] = +weights[m].toFixed(4)
  return weights
}

/**
 * Compute the ensemble fan band.
 *
 * Weighted blend of base model p5/p25/p50/p75/p95 arrays.
 * High disagreement adds σ_disagreement to p95 and subtracts from p5
 * to widen the ensemble's uncertainty (Krogh & Vedelsby 1995).
 *
 * @param {Array}  results      - MethodResult array (with forecast bands)
 * @param {Object} weights      - {method: weight} from weightsFromR2() or serverWeights
 * @param {Object} disagreement - from computeDisagreement()
 * @returns {{dates:string[], p5:number[], p25:number[], p50:number[], p75:number[], p95:number[]}}
 */
export function blendBands(results, weights, disagreement) {
  const QUANTILES = ['p5', 'p25', 'p50', 'p75', 'p95']

  // Gather bands per method
  const bands = {}
  for (const r of results) {
    if (r.forecast?.dates?.length && weights[r.method] > 0) {
      bands[r.method] = r.forecast
    }
  }

  if (!Object.keys(bands).length) return null

  // Use dates from the method with most forecast steps
  const dates = Object.values(bands)
    .map(b => b.dates)
    .reduce((a, b) => (a.length >= b.length ? a : b))

  const T = dates.length
  const ensemble = {}

  // Per-timestep active method set — drives weight renormalization AND the
  // model-count trajectory we expose for the "ensemble degradation" indicator.
  // A method's contribution ends at its band length (horizon caps live in
  // useForecast), not at a NaN carry-forward.
  const activeMethodsPerT = Array.from({ length: T }, () => [])
  for (const [method, band] of Object.entries(bands)) {
    const len = Math.min(band.dates?.length ?? 0, T)
    for (let t = 0; t < len; t++) {
      if (Number.isFinite(band.p50?.[t])) activeMethodsPerT[t].push(method)
    }
  }

  for (const q of QUANTILES) {
    ensemble[q] = new Array(T).fill(null)
    for (let t = 0; t < T; t++) {
      let num = 0, den = 0
      for (const method of activeMethodsPerT[t]) {
        const w = weights[method] ?? 0
        const v = bands[method][q]?.[t]
        if (w > 0 && Number.isFinite(v)) {
          num += w * v
          den += w
        }
      }
      ensemble[q][t] = den > 0 ? num / den : null
    }
  }

  // Disagreement adjustment: widen outer bands
  const disagFactor = disagreement.level === 'high' ? 0.25
                    : disagreement.level === 'med'  ? 0.10
                    : 0.0
  if (disagFactor > 0) {
    const sigma = Math.sqrt(disagreement.variance)
    for (let t = 0; t < T; t++) {
      if (ensemble.p5[t]  != null) ensemble.p5[t]  -= disagFactor * sigma
      if (ensemble.p95[t] != null) ensemble.p95[t] += disagFactor * sigma
    }
  }

  const round4 = v => v == null ? null : +v.toFixed(4)
  return {
    dates,
    p5:  ensemble.p5.map(round4),
    p25: ensemble.p25.map(round4),
    p50: ensemble.p50.map(round4),
    p75: ensemble.p75.map(round4),
    p95: ensemble.p95.map(round4),
    modelCounts:  activeMethodsPerT.map(a => a.length),
    activeMethodsPerT,
  }
}

/**
 * Main entry: compute ensemble forecast from base model results.
 *
 * @param {Array}  results       - MethodResult[] (from useForecast)
 * @param {Object} regimeProbs   - from ForecastResponse.regime_probs (for display only)
 * @param {Object} serverWeights - pre-computed weights from ForecastResponse.ensemble_weights (preferred)
 * @returns {{band, weights, disagreement, source}}
 */
export async function computeEnsemble(results, regimeProbs, serverWeights = null) {
  const available = results.filter(r => r.forecast?.p50?.length > 0).map(r => r.method)
  if (!available.length) return null

  const disagreement = computeDisagreement(results)

  // Server weights are authoritative (OOS R²-based, computed by backend).
  // Client fallback mirrors the same algorithm using oos_r2 from MethodResults.
  let weights = serverWeights ?? weightsFromR2(results)
  const source = serverWeights ? 'oos_r2_server' : 'oos_r2_client'

  const dominant = regimeProbs
    ? (regimeProbs.dominant ?? Object.entries(regimeProbs)
        .filter(([k]) => k !== 'dominant')
        .reduce((a, b) => a[1] > b[1] ? a : b)[0])
    : null

  const band = blendBands(results, weights, disagreement)
  if (!band) return null

  return { band, weights, disagreement, source, dominant }
}
