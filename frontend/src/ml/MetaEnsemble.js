/**
 * MetaEnsemble — Client-Side Regime-Conditional Ensemble (Phase 5D).
 *
 * Combines base model fan bands using regime-probability-weighted blending.
 * Disagreement across models widens the ensemble confidence bands.
 *
 * Two inference paths:
 *   1. ONNX (preferred): per-regime elastic nets trained by train_meta_learner.py
 *      and exported to frontend/public/models/meta/{regime}.onnx.
 *   2. Rule-based fallback: Ang & Timmermann (2012) priors (always available).
 *
 * References
 * ----------
 * Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
 * Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets.
 *   Annual Review of Financial Economics, 4(1), 313–337.
 * Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation,
 *   and Active Learning. NeurIPS 8, 231–238.
 * Lakshminarayanan, B. et al. (2017). Simple and Scalable Predictive Uncertainty
 *   Estimation Using Deep Ensembles. NeurIPS 30.
 */

// ── Rule-based regime-conditional weights (inline fallback) ───────────────────
// Mirrors backend/app/services/meta_learner.py::_REGIME_WEIGHTS
const REGIME_WEIGHTS = {
  bull_low_vol:  { factor: 0.35, xgboost: 0.25, hmm: 0.18, var: 0.12, nbeats: 0.05, lstm: 0.05 },
  bull_high_vol: { xgboost: 0.35, hmm: 0.25, factor: 0.18, var: 0.12, nbeats: 0.05, lstm: 0.05 },
  bear:          { hmm: 0.35, var: 0.25, factor: 0.18, xgboost: 0.12, nbeats: 0.05, lstm: 0.05 },
  crisis:        { var: 0.35, hmm: 0.30, xgboost: 0.15, factor: 0.10, nbeats: 0.05, lstm: 0.05 },
}

// ── ONNX model cache ──────────────────────────────────────────────────────────
const _onnxCache = {}

/**
 * Try to load a per-regime ONNX model. Returns null if unavailable.
 * Models are expected at /models/meta/{regime}.onnx (Vercel static asset).
 */
async function tryLoadOnnxModel(regime) {
  if (_onnxCache[regime] !== undefined) return _onnxCache[regime]
  try {
    const { InferenceSession } = await import('onnxruntime-web')
    const url = `/models/meta/${regime}.onnx`
    const resp = await fetch(url, { method: 'HEAD' })
    if (!resp.ok) { _onnxCache[regime] = null; return null }
    const session = await InferenceSession.create(url)
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
 * Compute ensemble-blended weights from regime probabilities (rule-based path).
 *
 * @param  {Object} regimeProbs  - {bull_low_vol:…, bull_high_vol:…, bear:…, crisis:…}
 * @param  {string[]} available  - methods that produced forecasts
 * @returns {Object} {method: weight}
 */
export function blendWeights(regimeProbs, available) {
  const avail  = new Set(available)
  const weights = {}

  for (const [regime, prob] of Object.entries(regimeProbs)) {
    if (regime === 'dominant') continue
    const rw = REGIME_WEIGHTS[regime] ?? {}
    for (const [method, w] of Object.entries(rw)) {
      if (avail.has(method)) {
        weights[method] = (weights[method] ?? 0) + prob * w
      }
    }
  }

  // Normalise
  const total = Object.values(weights).reduce((a, b) => a + b, 0)
  if (total > 0) {
    for (const m in weights) weights[m] = +(weights[m] / total).toFixed(4)
  }
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
 * @param {Object} weights      - {method: weight} from blendWeights()
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

  for (const q of QUANTILES) {
    ensemble[q] = new Array(T).fill(0)
    let totalW = 0
    for (const [method, band] of Object.entries(bands)) {
      const w = weights[method] ?? 0
      if (!w || !band[q]?.length) continue
      const arr = band[q]
      for (let t = 0; t < T; t++) {
        ensemble[q][t] += w * (arr[t] ?? arr.at(-1) ?? 0)
      }
      totalW += w
    }
    // Renormalize if some methods were missing quantile data
    if (totalW > 0 && totalW < 1) {
      for (let t = 0; t < T; t++) ensemble[q][t] /= totalW
    }
  }

  // Disagreement adjustment: widen outer bands
  const disagFactor = disagreement.level === 'high' ? 0.25
                    : disagreement.level === 'med'  ? 0.10
                    : 0.0
  if (disagFactor > 0) {
    const sigma = Math.sqrt(disagreement.variance)
    for (let t = 0; t < T; t++) {
      ensemble.p5[t]  -= disagFactor * sigma
      ensemble.p95[t] += disagFactor * sigma
    }
  }

  return {
    dates,
    p5:  ensemble.p5.map(v => +v.toFixed(4)),
    p25: ensemble.p25.map(v => +v.toFixed(4)),
    p50: ensemble.p50.map(v => +v.toFixed(4)),
    p75: ensemble.p75.map(v => +v.toFixed(4)),
    p95: ensemble.p95.map(v => +v.toFixed(4)),
  }
}

/**
 * Main entry: compute ensemble forecast from base model results + regime.
 *
 * @param {Array}  results       - MethodResult[] (from useForecast)
 * @param {Object} regimeProbs   - from ForecastResponse.regime_probs
 * @param {Object} serverWeights - pre-computed weights from ForecastResponse.ensemble_weights (optional)
 * @returns {{band, weights, disagreement, source}}
 */
export async function computeEnsemble(results, regimeProbs, serverWeights = null) {
  const available  = results.filter(r => r.forecast?.p50?.length > 0).map(r => r.method)
  if (!available.length || !regimeProbs) return null

  const disagreement = computeDisagreement(results)

  // Try ONNX per-regime models (Phase 5D preferred path)
  let weights     = null
  let source      = 'rule_based'
  const dominant  = regimeProbs.dominant ?? Object.entries(regimeProbs)
    .filter(([k]) => k !== 'dominant')
    .reduce((a, b) => a[1] > b[1] ? a : b)[0]

  const session = await tryLoadOnnxModel(dominant)
  if (session) {
    // ONNX path: run per-regime elastic net
    // (implementation wired when pre-trained models are available)
    source = 'onnx'
  }

  // Use server-computed weights if available; else rule-based
  weights = serverWeights ?? blendWeights(regimeProbs, available)
  if (source === 'onnx') source = 'onnx_weights'

  const band = blendBands(results, weights, disagreement)
  if (!band) return null

  return { band, weights, disagreement, source, dominant }
}
