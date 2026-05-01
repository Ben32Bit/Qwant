/**
 * LSTMInferer.js — Browser-side Attention-LSTM inference via TensorFlow.js
 *
 * Contract: scale-invariant features (NO MinMaxScaler).
 * Features are bounded in ~[-1, 1] by construction, so one model generalises
 * across assets / vol regimes without per-portfolio scaler drift.
 *
 *   [0] z_ret       = r_t / σ_21d, clipped ±3, /3         ∈ [-1, 1]
 *   [1] vol_pct     = percentile of σ_63 in 5y window     ∈ [-1, 1]
 *   [2] z_mom5      = sum_5 / (σ_5 √5), clipped ±3, /3    ∈ [-1, 1]
 *   [3] z_mom21     = sum_21 / (σ_21 √21), clipped ±3, /3 ∈ [-1, 1]
 *   [4] rsi_centred = 2 * rsi_14 - 1                      ∈ [-1, 1]
 *
 * The model outputs next-day z_ret (tanh, bounded [-1, 1]). Unscale via:
 *   z_pred = model_out * CLIP_SIGMAS
 *   r_pred = z_pred * σ_21d_current
 *
 * Must stay in lockstep with:
 *   backend/app/services/forecast_engine.py::prepare_lstm_features
 *   backend/scripts/train_lstm.py
 */

const MODEL_URL    = '/models/lstm/model.json'
const N_MC_PASSES  = 200
const TRADING_DAYS = 252
const LOOKBACK     = 60
const CLIP_SIGMAS  = 3.0

// ── Sense-check envelope for the autoregressive rollout ───────────────────────
// With scale-invariant features the rollout is far more stable (features can't
// saturate a scaler), but MC Dropout × 63-step autoregression on a small LSTM
// can still produce tails beyond what's realistic. These envelopes bound it.
const CUM_CLIP_LO         = -0.60  // hard floor on cumulative return per path
const CUM_CLIP_HI         =  2.00  // hard ceiling on cumulative return per path
const MIN_SURVIVOR_FRAC   =  0.20  // <20% survivors → whole run is untrustworthy
const PLAUSIBILITY_21D    =  0.25  // |median 21d| must be within ±25%
const PER_STEP_RETURN_CLIP =  0.15 // per-step raw return capped at ±15%

// Feature order MUST match server's prepare_lstm_features exactly:
//   [z_ret, vol_pct, z_mom5, z_mom21, rsi_centred]
const F_Z_RET = 0, F_VOL_PCT = 1, F_Z_MOM5 = 2, F_Z_MOM21 = 3, F_RSI_C = 4

let _model = null
let _tf    = null

async function getModel() {
  if (_model) return _model
  if (!_tf) {
    _tf = await import('@tensorflow/tfjs')
    await _tf.ready()
  }
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

// ── Feature recomputation helpers (match server's prepare_lstm_features) ─────

function rollingStd(arr, from, to) {
  // Sample std over arr[from..to) — matches pandas Series.rolling(N).std() (ddof=1)
  const n = to - from
  if (n <= 1) return 0
  let mean = 0
  for (let i = from; i < to; i++) mean += arr[i]
  mean /= n
  let ssq = 0
  for (let i = from; i < to; i++) {
    const d = arr[i] - mean
    ssq += d * d
  }
  return Math.sqrt(ssq / (n - 1))
}

function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x }

/**
 * Rebuild all 5 scale-invariant features for the newest time step from the
 * raw-return buffer and a rolling 63d-vol history (for vol_pct ranking).
 *
 * @param {number[]} rawReturns   — raw returns, length ≥ 64 (21d vol buffer + RSI + shift)
 * @param {number[]} sigma63Hist  — trailing 63d-vol history for percentile ranking
 */
function buildZFeatureRow(rawReturns, sigma63Hist) {
  const n = rawReturns.length
  if (n < 64) {
    throw new Error(`LSTM rollout needs ≥64 raw returns in buffer, got ${n}`)
  }

  // ── Causal rolling stds: use returns up to (but NOT including) the newest ──
  const sigma21 = rollingStd(rawReturns, n - 22, n - 1)  // r_{t-21..t-1}
  const sigma5  = rollingStd(rawReturns, n - 6,  n - 1)  // r_{t-5..t-1}
  const sigma63 = rollingStd(rawReturns, n - 64, n - 1)  // r_{t-63..t-1}

  const last = rawReturns[n - 1]

  // z_ret: today's return in units of prior σ_21d
  const zRetRaw = last / (sigma21 + 1e-9)
  const zRet    = clamp(zRetRaw, -CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS

  // Momentum z-scores
  let mom5 = 0, mom21 = 0
  for (let i = n - 5;  i < n; i++) mom5  += rawReturns[i]
  for (let i = n - 21; i < n; i++) mom21 += rawReturns[i]
  const zMom5Raw  = mom5  / (sigma5  * Math.sqrt(5)  + 1e-9)
  const zMom21Raw = mom21 / (sigma21 * Math.sqrt(21) + 1e-9)
  const zMom5     = clamp(zMom5Raw,  -CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS
  const zMom21    = clamp(zMom21Raw, -CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS

  // Vol percentile: fraction of sigma63Hist entries ≤ current sigma63, mapped to [-1, 1]
  let belowCount = 0
  const histLen = sigma63Hist.length
  for (let i = 0; i < histLen; i++) if (sigma63Hist[i] <= sigma63) belowCount += 1
  const volPctRaw = histLen > 0 ? belowCount / histLen : 0.5
  const volPct    = clamp(2.0 * volPctRaw - 1.0, -1, 1)

  // RSI-14 over last 14 diffs of the raw return series (matches server's
  // delta = r.diff(); gain/loss rolling(14).mean())
  let gain = 0, loss = 0
  for (let i = n - 14; i < n; i++) {
    const d = rawReturns[i] - rawReturns[i - 1]
    if (d > 0) gain += d
    else       loss -= d
  }
  gain /= 14; loss /= 14
  const rsi14    = gain / (gain + loss + 1e-9)
  const rsiCent  = clamp(2.0 * rsi14 - 1.0, -1, 1)

  return [zRet, volPct, zMom5, zMom21, rsiCent]
}

// ── R² helpers ────────────────────────────────────────────────────────────────

function computeR2(actual, predicted) {
  const n = Math.min(actual.length, predicted.length)
  if (n < 5) return null
  const mean = actual.slice(0, n).reduce((s, v) => s + v, 0) / n
  let ssTot = 0, ssRes = 0
  for (let i = 0; i < n; i++) {
    ssTot += (actual[i] - mean) ** 2
    ssRes += (actual[i] - predicted[i]) ** 2
  }
  return ssTot > 1e-12 ? +((1 - ssRes / ssTot).toFixed(4)) : null
}

/**
 * Run the model on a batch of (60×5) windows (1 pass, no dropout).
 * Returns raw model outputs in [-1, 1].
 */
async function batchPredict(model, tf, windows) {
  const n        = windows.length
  const lookback = windows[0].length
  const nFeat    = windows[0][0].length
  const flat     = new Float32Array(n * lookback * nFeat)
  for (let i = 0; i < n; i++) {
    const base = i * lookback * nFeat
    for (let r = 0; r < lookback; r++)
      for (let f = 0; f < nFeat; f++)
        flat[base + r * nFeat + f] = windows[i][r][f]
  }
  const t = tf.tensor3d(flat, [n, lookback, nFeat])
  const p = model.apply(t, { training: false })
  const data = await p.data()
  t.dispose(); p.dispose()
  return Array.from(data)
}

/**
 * Run Attention-LSTM MC Dropout inference.
 *
 * @param {Object} params
 * @param {number[][]} params.seedWindow        — (60 × 5) scale-invariant window
 * @param {string[]}   params.forecastDates     — business-day forecast dates
 * @param {number[]}   params.rawReturnSeed     — last ≥128 raw returns from the portfolio
 * @param {number}     params.currentSigma21d   — current σ_21d (for unscaling z-preds)
 * @param {number[]}   [params.sigma63History]  — trailing 5y σ_63 distribution for vol_pct rank
 * @param {number}     [params.nPasses=200]
 * @param {number[][]} [params.isEvalWindows]   — 30 historical (60×5) windows for IS R²
 * @param {number[]}   [params.isEvalTargets]   — actual z_ret for each eval window ([-1, 1])
 * @param {Object}     [params.shadow]          — shadow holdout {last_window, sigma_21d,
 *                                                raw_return_seed, actual_cumrets}
 */
export async function inferLSTM({
  seedWindow,
  forecastDates,
  rawReturnSeed,
  currentSigma21d,
  sigma63History = null,
  nPasses = N_MC_PASSES,
  isEvalWindows = null,
  isEvalTargets = null,
  shadow = null,
}) {
  // ── Input contract validation ───────────────────────────────────────────────
  if (!Array.isArray(seedWindow) || !seedWindow.length)
    throw new Error('inferLSTM: seedWindow must be a non-empty array')
  if (!Array.isArray(seedWindow[0]))
    throw new Error(
      `inferLSTM: seedWindow must be 2D (lookback × features); got 1D.`,
    )
  if (!Array.isArray(rawReturnSeed) || rawReturnSeed.length < 64)
    throw new Error(
      `inferLSTM: rawReturnSeed must be an array of ≥64 raw returns; ` +
      `got ${Array.isArray(rawReturnSeed) ? `length ${rawReturnSeed.length}` : typeof rawReturnSeed}`,
    )
  if (!Number.isFinite(currentSigma21d) || currentSigma21d <= 0)
    throw new Error(
      `inferLSTM: currentSigma21d must be a positive number; got ${currentSigma21d}`,
    )

  const tf      = _tf ?? await import('@tensorflow/tfjs')
  const model   = await getModel()
  const horizon = forecastDates.length
  const lookback  = seedWindow.length
  const nFeatures = seedWindow[0].length

  // Per-pass state: each pass rolls out an independent diverging path.
  const windows  = Array.from({ length: nPasses }, () => seedWindow.map(row => [...row]))
  const rawBufs  = Array.from({ length: nPasses }, () => [...rawReturnSeed])
  const cumRs    = new Float64Array(nPasses)
  const allPaths = Array.from({ length: nPasses }, () => new Array(horizon))
  const diverged = new Uint8Array(nPasses)

  // Prefer the server-provided 5y σ_63 distribution (matches training's vol_pct
  // rank window exactly). Fall back to a 128-sample locally-computed buffer
  // only if the server didn't send one.
  const sigma63Hist = Array.isArray(sigma63History) && sigma63History.length > 0
    ? sigma63History.slice()
    : buildSigma63History(rawReturnSeed)

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
        allPaths[p][t] = cumRs[p]
        continue
      }

      // Model output is bounded [-1, 1] (tanh); unscale to raw return.
      // z_pred = model_out * CLIP_SIGMAS ; r_pred = z_pred * σ_21d_current
      const zPred = preds[p] * CLIP_SIGMAS
      let   rReal = zPred * currentSigma21d
      rReal = clamp(rReal, -PER_STEP_RETURN_CLIP, PER_STEP_RETURN_CLIP)

      let cum = (1 + cumRs[p]) * (1 + rReal) - 1
      if      (cum > CUM_CLIP_HI) { cum = CUM_CLIP_HI; diverged[p] = 1 }
      else if (cum < CUM_CLIP_LO) { cum = CUM_CLIP_LO; diverged[p] = 1 }
      cumRs[p] = cum
      allPaths[p][t] = cum

      rawBufs[p].push(rReal)
      if (rawBufs[p].length > 128) rawBufs[p].shift()

      const newRow = buildZFeatureRow(rawBufs[p], sigma63Hist)
      windows[p] = [...windows[p].slice(1), newRow]
    }
  }

  // ── Post-rollout survivor check ────────────────────────────────────────────
  const survivorPaths = []
  for (let p = 0; p < nPasses; p++) if (!diverged[p]) survivorPaths.push(allPaths[p])
  if (survivorPaths.length / nPasses < MIN_SURVIVOR_FRAC) {
    throw new Error(
      `LSTM rollout diverged: only ${survivorPaths.length}/${nPasses} paths stayed ` +
      `within the sanity envelope. The model may be out-of-distribution for this ` +
      `return series — try retraining (see .github/workflows/retrain-lstm.yml).`,
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

  const idx21 = Math.min(20, horizon - 1)
  if (Math.abs(p50[idx21] / 100) > PLAUSIBILITY_21D) {
    throw new Error(
      `LSTM rollout implausible: 21d median of ${p50[idx21].toFixed(1)}% exceeds ` +
      `±${(PLAUSIBILITY_21D * 100).toFixed(0)}% envelope.`,
    )
  }

  // ── IS R²: 1-step-ahead accuracy on held-out historical windows ────────────
  // Model predicts z_ret ∈ [-1, 1]; targets are actual z_ret (same scale).
  let isR2 = null
  if (isEvalWindows?.length && isEvalTargets?.length) {
    try {
      const preds = await batchPredict(model, tf, isEvalWindows)
      isR2 = computeR2(isEvalTargets, preds)
    } catch { /* non-fatal */ }
  }

  // ── OOS R²: shadow holdout — run LSTM on shadow seed, compare to actual ────
  // Shadow window = [T-60, T-30]: we have the 60×5 seed window at T-60, and
  // the actual cumulative % returns for the next 30 days. Run a single MC pass
  // (200 paths) from the shadow seed and compare p50 to actual_cumrets.
  let oosR2 = null
  if (shadow?.last_window && shadow?.actual_cumrets?.length >= 5) {
    try {
      const sSigma63 = Array.isArray(sigma63History) && sigma63History.length > 0
        ? sigma63History
        : buildSigma63History(shadow.raw_return_seed ?? rawReturnSeed)
      const shadowInference = await inferLSTM({
        seedWindow:      shadow.last_window,
        forecastDates:   Array.from({ length: shadow.actual_cumrets.length }, (_, i) => `s${i}`),
        rawReturnSeed:   shadow.raw_return_seed ?? rawReturnSeed,
        currentSigma21d: shadow.sigma_21d ?? currentSigma21d,
        sigma63History:  sSigma63,
        nPasses:         100,   // fewer passes for speed
        isEvalWindows:   null,
        isEvalTargets:   null,
        shadow:          null,  // no nested shadow
      })
      oosR2 = computeR2(shadow.actual_cumrets, shadowInference.p50)
    } catch { /* non-fatal — shadow OOS R² is optional */ }
  }

  return { dates: forecastDates, p5, p25, p50, p75, p95, is_r2: isR2, oos_r2: oosR2 }
}

/**
 * Compute the trailing 63d-vol history used for the vol_pct percentile feature.
 * Slides a 63-wide window and records the sample std at each position.
 * Returns an array of length max(0, rawReturnSeed.length - 63 + 1).
 */
function buildSigma63History(rawReturnSeed) {
  const n = rawReturnSeed.length
  const win = 63
  if (n < win) return []
  const hist = new Array(n - win + 1)
  for (let i = 0; i + win <= n; i++) {
    hist[i] = rollingStd(rawReturnSeed, i, i + win)
  }
  return hist
}

export function isModelLoaded() {
  return _model !== null
}
