/**
 * LSTMInferer.js — Browser-side Attention-LSTM inference via TensorFlow.js
 *
 * PATCHED: correct feature rollout (Priority 5 fix).
 * The previous version only updated feature[0] with the model's *output*
 * scalar and left features[1..4] (vol_21d, mom_5d, mom_21d, rsi_14) frozen
 * at historical values for the entire 252-day horizon. That produced
 * mathematically invalid forecasts after day 1.
 *
 * The corrected rollout maintains a buffer of the most recent scaled
 * returns, rebuilds all 5 features each step by mirroring the server's
 * prepare_lstm_features formulas, and re-scales them with the same
 * MinMaxScaler the server fit.
 */

const MODEL_URL    = '/models/lstm/model.json'
const N_MC_PASSES  = 200
const TRADING_DAYS = 252

// ── Sense-check envelope for the autoregressive rollout ───────────────────────
// MC Dropout × 252-step autoregression on a small LSTM can produce pathological
// paths: the scaler saturates once the simulated return distribution drifts
// outside training range, which then feeds scaled values that push the next
// prediction further out of distribution. These envelopes bound the damage.
const DAILY_CLIP_SIGMAS   = 3      // per-step return clip, multiples of training σ
const CUM_CLIP_LO         = -0.60  // hard floor on cumulative return per path
const CUM_CLIP_HI         = 2.00   // hard ceiling on cumulative return per path
const MIN_SURVIVOR_FRAC   = 0.20   // <20% survivors → the whole run is untrustworthy
const PLAUSIBILITY_21D    = 0.25   // |median 21d| must be within 25%
const SATURATION_THRESH   = 2.0    // |scaled| > 2 counts as out-of-range
const SATURATION_STEPS    = 5      // N consecutive out-of-range steps ⇒ path diverged

// Feature order must match server's prepare_lstm_features exactly:
//   [r, vol_21d, mom_5d, mom_21d, rsi_14]
const F_RET = 0, F_VOL = 1, F_MOM5 = 2, F_MOM21 = 3, F_RSI = 4

let _model = null
let _tf    = null

async function getModel() {
  if (_model) return _model
  if (!_tf) {
    _tf = await import('@tensorflow/tfjs')
    await _tf.ready()
  }
  // Pre-flight: sniff the JSON manifest before letting TF.js swallow the cause
  const headRes = await fetch(MODEL_URL, { method: 'HEAD' })
  if (!headRes.ok) {
    throw new Error(
      `LSTM model manifest not reachable (HTTP ${headRes.status} at ${MODEL_URL}). ` +
      `Run: cd backend && python scripts/train_lstm.py`
    )
  }
  const ctype = headRes.headers.get('content-type') ?? ''
  if (ctype.includes('text/html')) {
    throw new Error(
      `LSTM manifest served as text/html — a SPA rewrite is intercepting ${MODEL_URL}. ` +
      `Check vercel.json rewrites.`
    )
  }
  try {
    _model = await _tf.loadLayersModel(MODEL_URL)
  } catch (err) {
    throw new Error(`LSTM model load failed: ${err?.message ?? err}`)
  }
  return _model
}

// ── Feature recomputation helpers (inverse then forward of server's scaler) ──

function unscale(scaledVal, minArr, maxArr, idx) {
  // feature_range=(-1, 1):  scaled = 2 * (x - min) / (max - min) - 1
  // inverse:                x = (scaled + 1) * (max - min) / 2 + min
  return (scaledVal + 1) * (maxArr[idx] - minArr[idx]) / 2 + minArr[idx]
}

function scale(rawVal, minArr, maxArr, idx) {
  const range = maxArr[idx] - minArr[idx]
  if (range === 0) return 0
  return 2 * (rawVal - minArr[idx]) / range - 1
}

/**
 * Rebuild all 5 scaled features for a new time step, given a buffer of
 * recent *raw* (unscaled) returns.
 *
 * Mirrors backend/app/services/forecast_engine.py::prepare_lstm_features.
 * The buffer must contain AT LEAST the last 21 raw returns so we can compute
 * vol_21d and mom_21d. RSI-14 needs 15+ for the diff-based rolling mean.
 */
function buildScaledFeatureRow(rawReturns, minArr, maxArr) {
  const n = rawReturns.length
  if (n < 21) {
    throw new Error(`LSTM rollout needs ≥21 raw returns in buffer, got ${n}`)
  }

  const last = rawReturns[n - 1]

  // vol_21d (annualised std of last 21 returns)
  let mean = 0
  for (let i = n - 21; i < n; i++) mean += rawReturns[i]
  mean /= 21
  let ssq = 0
  for (let i = n - 21; i < n; i++) {
    const d = rawReturns[i] - mean
    ssq += d * d
  }
  const vol21 = Math.sqrt(ssq / 21) * Math.sqrt(TRADING_DAYS)

  // mom_5d, mom_21d (cumulative sum — matches server's rolling(N).sum())
  let mom5 = 0, mom21 = 0
  for (let i = n - 5;  i < n; i++) mom5  += rawReturns[i]
  for (let i = n - 21; i < n; i++) mom21 += rawReturns[i]

  // RSI-14: rolling mean of gains vs losses over last 14 diffs
  const diffs = []
  for (let i = n - 14; i < n; i++) diffs.push(rawReturns[i] - rawReturns[i - 1])
  let gain = 0, loss = 0
  for (const d of diffs) {
    if (d > 0) gain += d
    else       loss -= d
  }
  gain /= 14; loss /= 14
  const rsi14 = gain / (gain + loss + 1e-9)

  return [
    scale(last,   minArr, maxArr, F_RET),
    scale(vol21,  minArr, maxArr, F_VOL),
    scale(mom5,   minArr, maxArr, F_MOM5),
    scale(mom21,  minArr, maxArr, F_MOM21),
    scale(rsi14,  minArr, maxArr, F_RSI),
  ]
}

/**
 * Run Attention-LSTM MC Dropout inference.
 *
 * @param {Object} params
 * @param {number[][]}  params.seedWindow    — (60 × 5) scaled feature window from server
 * @param {string[]}    params.forecastDates — business-day dates for the horizon
 * @param {number[]}    params.scalerMin     — 5 MinMaxScaler data_min_ values (one per feature)
 * @param {number[]}    params.scalerMax     — 5 MinMaxScaler data_max_ values (one per feature)
 * @param {number[]}    [params.rawReturnSeed] — last 22+ raw returns from the portfolio
 * @param {number}      [params.nPasses=200]
 */
export async function inferLSTM({
  seedWindow,
  forecastDates,
  scalerMin,
  scalerMax,
  rawReturnSeed = null,
  nPasses = N_MC_PASSES,
}) {
  // ── Input contract validation ───────────────────────────────────────────────
  if (!Array.isArray(seedWindow) || !seedWindow.length)
    throw new Error('inferLSTM: seedWindow must be a non-empty array')
  if (!Array.isArray(seedWindow[0]))
    throw new Error(
      `inferLSTM: seedWindow must be 2D (lookback × features); got 1D. ` +
      `LSTM requires the full 5-feature window from the server.`
    )
  if (!Array.isArray(scalerMin) || scalerMin.length !== seedWindow[0].length)
    throw new Error(
      `inferLSTM: scalerMin must be an array of ${seedWindow[0].length} elements (one per feature); ` +
      `got ${Array.isArray(scalerMin) ? `length ${scalerMin.length}` : typeof scalerMin}`
    )
  if (!Array.isArray(scalerMax) || scalerMax.length !== seedWindow[0].length)
    throw new Error(`inferLSTM: scalerMax array length must match feature count`)

  const tf      = _tf ?? await import('@tensorflow/tfjs')
  const model   = await getModel()
  const horizon = forecastDates.length
  const lookback  = seedWindow.length
  const nFeatures = seedWindow[0].length

  // Seed the raw return buffer from rawReturnSeed if provided, else unscale from window
  const seedRaw = rawReturnSeed && rawReturnSeed.length >= 22
    ? rawReturnSeed.slice(-Math.max(lookback, 22))
    : seedWindow.map(row => unscale(row[F_RET], scalerMin, scalerMax, F_RET))

  // Training-distribution per-day σ for the ±3σ clip — estimated from the seed
  // buffer, which IS the historical raw return series.
  const seedMean = seedRaw.reduce((a, b) => a + b, 0) / seedRaw.length
  const seedVar  = seedRaw.reduce((s, v) => s + (v - seedMean) ** 2, 0) / Math.max(1, seedRaw.length - 1)
  const seedStd  = Math.sqrt(seedVar) || 0.015  // fallback ~1.5% if degenerate
  const dailyClip = DAILY_CLIP_SIGMAS * seedStd

  // Per-pass state: each pass gets its own diverging window + return buffer
  let windows = Array.from({ length: nPasses }, () =>
    seedWindow.map(row => [...row])
  )
  const rawBufs  = Array.from({ length: nPasses }, () => [...seedRaw])
  const cumRs    = new Float64Array(nPasses)
  const allPaths = Array.from({ length: nPasses }, () => new Array(horizon))
  const saturCtr = new Int16Array(nPasses)          // consecutive saturated steps
  const diverged = new Uint8Array(nPasses)          // 1 = path frozen at envelope

  for (let t = 0; t < horizon; t++) {
    const flatData = new Float32Array(nPasses * lookback * nFeatures)
    for (let p = 0; p < nPasses; p++) {
      const base = p * lookback * nFeatures
      for (let row = 0; row < lookback; row++) {
        for (let f = 0; f < nFeatures; f++) {
          flatData[base + row * nFeatures + f] = windows[p][row][f]
        }
      }
    }
    const batchTensor = tf.tensor3d(flatData, [nPasses, lookback, nFeatures])
    const predTensor  = model.apply(batchTensor, { training: true })  // MC Dropout ON
    const preds       = await predTensor.data()
    batchTensor.dispose()
    predTensor.dispose()

    for (let p = 0; p < nPasses; p++) {
      if (diverged[p]) {
        // Freeze diverged paths at their last cumulative return — this keeps
        // the tensor shape consistent without letting the pathology propagate.
        allPaths[p][t] = cumRs[p]
        continue
      }

      const rScaled = preds[p]

      // Guard 1: per-step return clip to ±Nσ of the training daily-return std.
      let rReal = unscale(rScaled, scalerMin, scalerMax, F_RET)
      if      (rReal >  dailyClip) rReal =  dailyClip
      else if (rReal < -dailyClip) rReal = -dailyClip

      // Guard 2: hard envelope on cumulative return — nothing realistic ranges
      // outside [−60%, +200%] on a 1y horizon.
      let cum = (1 + cumRs[p]) * (1 + rReal) - 1
      if      (cum > CUM_CLIP_HI) { cum = CUM_CLIP_HI; diverged[p] = 1 }
      else if (cum < CUM_CLIP_LO) { cum = CUM_CLIP_LO; diverged[p] = 1 }
      cumRs[p] = cum
      allPaths[p][t] = cum

      rawBufs[p].push(rReal)
      if (rawBufs[p].length > 64) rawBufs[p].shift()

      const newRow = buildScaledFeatureRow(rawBufs[p], scalerMin, scalerMax)

      // Guard 3: scaler-saturation detector. If this pass's new feature row
      // sits persistently outside the training scaler range, the model is
      // operating off-distribution and its outputs are unreliable.
      let anySat = false
      for (let f = 0; f < newRow.length; f++) {
        if (Math.abs(newRow[f]) > SATURATION_THRESH) { anySat = true; break }
      }
      if (anySat) saturCtr[p] += 1
      else        saturCtr[p]  = 0
      if (saturCtr[p] >= SATURATION_STEPS) diverged[p] = 1

      windows[p] = [...windows[p].slice(1), newRow]
    }
  }

  // ── Post-rollout survivor check ────────────────────────────────────────────
  const survivorPaths = []
  for (let p = 0; p < nPasses; p++) if (!diverged[p]) survivorPaths.push(allPaths[p])
  if (survivorPaths.length / nPasses < MIN_SURVIVOR_FRAC) {
    throw new Error(
      `LSTM rollout diverged: only ${survivorPaths.length}/${nPasses} paths stayed ` +
      `within the sanity envelope. Model is operating off-distribution for this ` +
      `return series (likely unusual volatility or drift vs training data).`,
    )
  }

  // ── Percentile bands across MC passes ──────────────────────────────────────
  const p5  = new Array(horizon)
  const p25 = new Array(horizon)
  const p50 = new Array(horizon)
  const p75 = new Array(horizon)
  const p95 = new Array(horizon)

  for (let t = 0; t < horizon; t++) {
    const vals = survivorPaths.map(path => path[t]).sort((a, b) => a - b)
    const n    = vals.length
    p5[t]  = vals[Math.max(0, Math.floor(0.05 * n))] * 100
    p25[t] = vals[Math.max(0, Math.floor(0.25 * n))] * 100
    p50[t] = vals[Math.max(0, Math.floor(0.50 * n))] * 100
    p75[t] = vals[Math.max(0, Math.floor(0.75 * n))] * 100
    p95[t] = vals[Math.min(n - 1, Math.floor(0.95 * n))] * 100
  }

  // Guard 4: post-hoc plausibility gate on the 21-day median.  Even if enough
  // paths survive, a 21d median outside ±25% is almost certainly rollout
  // pathology — surface it as an error rather than drawing a fantasy number.
  const idx21 = Math.min(20, horizon - 1)
  if (Math.abs(p50[idx21] / 100) > PLAUSIBILITY_21D) {
    throw new Error(
      `LSTM rollout implausible: 21d median of ${p50[idx21].toFixed(1)}% exceeds ` +
      `±${(PLAUSIBILITY_21D * 100).toFixed(0)}% envelope. Training distribution likely ` +
      `doesn't match this portfolio's return regime.`,
    )
  }

  return { dates: forecastDates, p5, p25, p50, p75, p95 }
}

export function isModelLoaded() {
  return _model !== null
}
