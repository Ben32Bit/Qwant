/**
 * NBeatsInferer.js — Client-side N-BEATS quantile forecast (pure JS, no runtime).
 *
 * Loads raw float32 weights from weights.bin + manifest from meta.json.
 * Implements linear+ReLU forward pass — no ONNX or TF.js dependency.
 *
 * Model architecture (Oreshkin et al. 2020, ICLR):
 *   3 residual blocks. Each block: FC(256)×4 → backcast(30) + forecast(21×5).
 *   Residual composition: input = input - backcast; total_forecast += forecast.
 *   Output: [21, 5] daily return quantiles [p5, p25, p50, p75, p95].
 *
 * Fan chart (12 × 21 = 252 days):
 *   12 recursive periods. Each period runs one forward pass on the current 30-day
 *   window, then advances the window using p50 predictions.
 *   Uncertainty scales ×√(k+1) across periods (i.i.d. approximation).
 *
 * References:
 *   Oreshkin, B. et al. (2020). N-BEATS. ICLR. https://arxiv.org/abs/1905.10437
 *   Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica.
 *   López de Prado, M. (2018). AFML, Ch. 7.
 */

const WEIGHTS_PATH = '/models/nbeats/weights.bin'
const META_PATH    = '/models/nbeats/meta.json'
const LOOKBACK     = 30
const HORIZON      = 21
const PERIODS      = 12
const HIDDEN       = 256
const Q_IDX        = { p5: 0, p25: 1, p50: 2, p75: 3, p95: 4 }

let _weights = null
let _meta    = null

async function getModel() {
  if (_weights && _meta) return { weights: _weights, meta: _meta }

  const [metaRes, binRes] = await Promise.all([
    fetch(META_PATH),
    fetch(WEIGHTS_PATH),
  ])
  if (!metaRes.ok) throw new Error(`N-BEATS meta fetch failed: ${metaRes.status}`)
  if (!binRes.ok)  throw new Error(`N-BEATS weights fetch failed: ${binRes.status}`)

  _meta = await metaRes.json()
  const binBuffer = await binRes.arrayBuffer()

  _weights = {}
  for (const entry of _meta.weights_manifest) {
    // entry.offset is in bytes; Float32Array view with element count entry.size
    _weights[entry.name] = new Float32Array(binBuffer, entry.offset, entry.size)
  }

  return { weights: _weights, meta: _meta }
}

// ── Pure-JS linear algebra ────────────────────────────────────────────────────

function linear(x, W, b, inSize, outSize) {
  // W stored row-major [outSize × inSize]; y[i] = dot(W[i], x) + b[i]
  const y = new Float32Array(outSize)
  for (let i = 0; i < outSize; i++) {
    let sum = b[i]
    const off = i * inSize
    for (let j = 0; j < inSize; j++) sum += W[off + j] * x[j]
    y[i] = sum
  }
  return y
}

function relu(x) {
  const out = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0
  return out
}

// ── N-BEATS forward pass ──────────────────────────────────────────────────────

function nbeatsForward(input, weights, nBlocks) {
  // input: Float32Array[LOOKBACK] (normalised)
  // returns: number[][] shape [HORIZON][5]
  const residual      = Float32Array.from(input)
  const totalForecast = new Float32Array(HORIZON * 5)

  // FC layer indices within each block's Sequential (ReLU layers have no params)
  const fcSpec = [
    [0, LOOKBACK, HIDDEN],
    [2, HIDDEN,   HIDDEN],
    [4, HIDDEN,   HIDDEN],
    [6, HIDDEN,   HIDDEN],
  ]

  for (let b = 0; b < nBlocks; b++) {
    const p = `blocks.${b}`

    // Four-layer FC stack
    let h = residual
    for (const [idx, inSz, outSz] of fcSpec) {
      h = relu(linear(h,
        weights[`${p}.fc.${idx}.weight`],
        weights[`${p}.fc.${idx}.bias`],
        inSz, outSz))
    }

    // Backcast and forecast heads (no activation)
    const backcast = linear(h,
      weights[`${p}.backcast_head.weight`],
      weights[`${p}.backcast_head.bias`],
      HIDDEN, LOOKBACK)

    const fcast = linear(h,
      weights[`${p}.forecast_head.weight`],
      weights[`${p}.forecast_head.bias`],
      HIDDEN, HORIZON * 5)

    // Residual subtraction
    for (let i = 0; i < LOOKBACK; i++) residual[i] -= backcast[i]
    // Accumulate forecast
    for (let i = 0; i < HORIZON * 5; i++) totalForecast[i] += fcast[i]
  }

  // Reshape [HORIZON*5] → [HORIZON][5]
  const result = []
  for (let t = 0; t < HORIZON; t++) {
    result.push(Array.from(totalForecast.subarray(t * 5, t * 5 + 5)))
  }
  return result  // [21][5] normalised
}

// ── Per-period inference ──────────────────────────────────────────────────────

function runOnePeriod(window, weights, meta) {
  const { mu_r, sigma_r, n_blocks } = meta

  const normalised = new Float32Array(LOOKBACK)
  for (let i = 0; i < LOOKBACK; i++) normalised[i] = (window[i] - mu_r) / sigma_r

  const normPreds = nbeatsForward(normalised, weights, n_blocks)

  // Denormalise → raw daily returns [21][5]
  return normPreds.map(row => row.map(v => v * sigma_r + mu_r))
}

function compoundPeriod(daily) {
  return daily.reduce((acc, r) => (1 + acc) * (1 + r) - 1, 0)
}

// ── Main export ───────────────────────────────────────────────────────────────

/**
 * @param {object} opts
 * @param {number[]} opts.lastWindow     - 30 raw daily returns (from server)
 * @param {string[]} opts.forecastDates  - 252 business-day date strings
 * @returns {Promise<{dates, p5, p25, p50, p75, p95}>} 252-day fan chart (cumulative %)
 */
export async function inferNBeats({ lastWindow, forecastDates }) {
  const { weights, meta } = await getModel()

  let window = [...lastWindow]
  const periods = []

  for (let k = 0; k < PERIODS; k++) {
    const preds    = runOnePeriod(window, weights, meta)  // [21][5]
    const p50Daily = preds.map(row => row[Q_IDX.p50])

    periods.push({
      p5:  compoundPeriod(preds.map(row => row[Q_IDX.p5])),
      p25: compoundPeriod(preds.map(row => row[Q_IDX.p25])),
      p50: compoundPeriod(p50Daily),
      p75: compoundPeriod(preds.map(row => row[Q_IDX.p75])),
      p95: compoundPeriod(preds.map(row => row[Q_IDX.p95])),
    })

    // Advance window: drop oldest HORIZON values, append p50 daily predictions
    window = [...window.slice(HORIZON), ...p50Daily]
  }

  // Build 252-day fan chart from period-level predictions
  const p5 = [], p25 = [], p50 = [], p75 = [], p95 = []
  let cumP50 = 0

  for (let k = 0; k < PERIODS; k++) {
    const prev = cumP50
    cumP50 = (1 + cumP50) * (1 + periods[k].p50) - 1

    const spreadDown_inner = periods[k].p50 - periods[k].p25
    const spreadUp_inner   = periods[k].p75 - periods[k].p50
    const spreadDown_outer = periods[k].p50 - periods[k].p5
    const spreadUp_outer   = periods[k].p95 - periods[k].p50
    const sqrtK = Math.sqrt(k + 1)

    const cumP25 = (1 + prev) * (1 + periods[k].p50 - spreadDown_inner * sqrtK) - 1
    const cumP75 = (1 + prev) * (1 + periods[k].p50 + spreadUp_inner   * sqrtK) - 1
    const cumP5  = (1 + prev) * (1 + periods[k].p50 - spreadDown_outer * sqrtK) - 1
    const cumP95 = (1 + prev) * (1 + periods[k].p50 + spreadUp_outer   * sqrtK) - 1

    for (let d = 0; d < HORIZON; d++) {
      const frac = (d + 1) / HORIZON
      p50.push((prev + (cumP50 - prev) * frac) * 100)
      p25.push((prev + (cumP25 - prev) * frac) * 100)
      p75.push((prev + (cumP75 - prev) * frac) * 100)
      p5.push( (prev + (cumP5  - prev) * frac) * 100)
      p95.push((prev + (cumP95 - prev) * frac) * 100)
    }
  }

  return { dates: forecastDates, p5, p25, p50, p75, p95 }
}
