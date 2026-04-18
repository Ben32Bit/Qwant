/**
 * XGBoostInferer.js — Client-side XGBoost quantile forecast via ONNX Runtime Web.
 *
 * Loads 5 pre-trained ONNX models (one per quantile: p5/p25/p50/p75/p95).
 * Each model is a sklearn Pipeline(StandardScaler, GradientBoostingRegressor)
 * exported to ONNX — the scaler is baked in, so raw features are passed directly.
 *
 * Input:  9 raw features computed by the server (prepare_xgboost_features).
 * Output: {dates, p5, p25, p50, p75, p95} — 252-day fan chart as cumulative % returns.
 *
 * Horizon extrapolation (21d → 252d):
 *   The models predict 21-day cumulative return distributions.
 *   For the full 252-day fan chart, we compound at the median rate and scale
 *   the uncertainty spread by sqrt(t/21) — the standard i.i.d. result.
 *
 *   For day t (t=1..252):
 *     k         = t / 21                         (fractional period)
 *     median(t) = (1 + p50_21d)^k − 1            (compounding at median rate)
 *     p50(t)    = median(t)
 *     p25/p75   = median(t) ± (p75_21d − p25_21d)/2 × sqrt(k)
 *     p5/p95    = median(t) ± (p95_21d − p5_21d)/2  × sqrt(k)
 *
 * References:
 *   Chen & Guestrin (2016, KDD). XGBoost: A Scalable Tree Boosting System.
 *   Gu, Kelly & Xiu (2020, RFS). Empirical Asset Pricing via Machine Learning.
 *   Friedman (2001, AoS). Greedy function approximation: a gradient boosting machine.
 *   López de Prado (2018). AFML, Ch. 7 (purged walk-forward CV, embargo).
 */

const QUANTILE_TAGS = ['q05', 'q25', 'q50', 'q75', 'q95']
const MODEL_BASE    = '/models/xgboost'
const HORIZON       = 252  // trading days in the fan chart
const PERIOD_DAYS   = 21   // horizon the models were trained on

let _ort     = null
const _sessions = {}  // tag → InferenceSession (lazy, cached)

async function getOrt() {
  if (!_ort) {
    _ort = await import('onnxruntime-web')
    // Use WASM backend; disable verbose logging
    _ort.env.wasm.numThreads = 1
  }
  return _ort
}

async function getSession(tag) {
  if (_sessions[tag]) return _sessions[tag]
  const ort = await getOrt()
  _sessions[tag] = await ort.InferenceSession.create(
    `${MODEL_BASE}/${tag}.onnx`,
    { executionProviders: ['wasm'] }
  )
  return _sessions[tag]
}

/**
 * Run all 5 quantile models and return the 21-day predictions.
 * @param {number[]} features - 9-element raw feature array from the server
 * @returns {Promise<{q05, q25, q50, q75, q95}>} 21-day cumulative % returns
 */
async function predict21d(features) {
  const ort = await getOrt()
  const floatArr = new Float32Array(features)

  const results = {}
  for (const tag of QUANTILE_TAGS) {
    const session = await getSession(tag)
    // Input name from skl2onnx Pipeline export is 'float_input'
    const tensor = new ort.Tensor('float32', floatArr, [1, features.length])
    const output = await session.run({ float_input: tensor })
    // skl2onnx Pipeline output name is 'variable'
    const val = output.variable?.data[0] ?? output[Object.keys(output)[0]].data[0]
    results[tag] = Number(val)
  }
  return results
}

/**
 * Extrapolate 21-day quantile predictions to a 252-day fan chart.
 *
 * @param {{q05, q25, q50, q75, q95}} preds21 - 21-day cumulative % returns (as decimals)
 * @param {string[]} forecastDates - 252 trading-day date strings from the server
 * @returns {{dates, p5, p25, p50, p75, p95}} - cumulative % return arrays
 */
function extrapolate252(preds21, forecastDates) {
  const { q05, q25, q50, q75, q95 } = preds21

  // Half-spreads from the 21-day predictions
  const halfInner = (q75 - q25) / 2   // IQR half-width
  const halfOuter = (q95 - q05) / 2   // 90th-pctile half-width

  const p5  = []
  const p25 = []
  const p50 = []
  const p75 = []
  const p95 = []

  for (let t = 1; t <= forecastDates.length; t++) {
    const k      = t / PERIOD_DAYS              // fractional periods
    const med    = Math.pow(1 + q50, k) - 1     // compounded median
    const sqrtK  = Math.sqrt(k)

    p50.push(med * 100)
    p25.push((med - halfInner * sqrtK) * 100)
    p75.push((med + halfInner * sqrtK) * 100)
    p5.push( (med - halfOuter * sqrtK) * 100)
    p95.push((med + halfOuter * sqrtK) * 100)
  }

  return {
    dates: forecastDates,
    p5, p25, p50, p75, p95,
  }
}

/**
 * Main entry point — called by useForecast.js after Phase 1 server response.
 *
 * @param {object} opts
 * @param {number[]} opts.features       - 9 raw features from server
 * @param {string[]} opts.forecastDates  - 252 date strings from server
 * @returns {Promise<{dates, p5, p25, p50, p75, p95}>} fan chart bands
 */
export async function inferXGBoost({ features, forecastDates }) {
  const preds21 = await predict21d(features)
  return extrapolate252(preds21, forecastDates)
}
