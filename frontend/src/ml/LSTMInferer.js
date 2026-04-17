/**
 * LSTMInferer.js — Browser-side Attention-LSTM inference via TensorFlow.js
 *
 * Loads the pre-trained model from /models/lstm/model.json (served by Vercel
 * as a static asset) and runs 200 MC Dropout passes to produce Bayesian
 * uncertainty fan bands (p5/p25/p50/p75/p95).
 *
 * The server prepares the scaled 60-day feature window and MinMaxScaler
 * parameters; this module handles inference only.
 *
 * MC Dropout inference: model.apply(x, { training: true }) keeps Dropout
 * layers active at test time — each pass samples a different dropout mask,
 * giving a distribution of predictions that represents model uncertainty.
 *
 * Vectorised: all 200 passes run as a single batch at each horizon step,
 * so the total inference cost is 252 model.apply() calls (not 200×252).
 *
 * References
 * ----------
 * Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation.
 *   Proceedings of ICML 33, 1050–1059.
 *   https://proceedings.mlr.press/v48/gal16.html
 *
 * CS230 Stanford (2020). Predicting Stock Market Returns Using Temporal
 *   Attention-Enhanced LSTM. Winter 2020 Project Reports.
 *   https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf
 *
 * Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation
 *   by jointly learning to align and translate. ICLR 2015.
 *   https://arxiv.org/abs/1409.0473
 */

const MODEL_URL      = '/models/lstm/model.json'
const N_MC_PASSES    = 200
const MODEL_NOT_FOUND = `Attention-LSTM model not found. Run: cd backend && python scripts/train_lstm.py`

let _model = null      // cached after first load
let _tf    = null      // tf module cached after dynamic import

/** Lazily import TF.js and load the model on first use. */
async function getModel() {
  if (_model) return _model

  if (!_tf) {
    _tf = await import('@tensorflow/tfjs')
    // Use WebGL backend for GPU acceleration if available; falls back to CPU
    await _tf.ready()
  }

  try {
    _model = await _tf.loadLayersModel(MODEL_URL)
  } catch (err) {
    throw new Error(MODEL_NOT_FOUND)
  }

  return _model
}

/**
 * Run Attention-LSTM MC Dropout inference.
 *
 * @param {Object} params
 * @param {number[][]}  params.seedWindow    — (60 × 5) scaled feature window from server
 * @param {string[]}    params.forecastDates — business-day dates for the horizon
 * @param {number[]}    params.scalerMin     — MinMaxScaler data_min_ per feature
 * @param {number[]}    params.scalerMax     — MinMaxScaler data_max_ per feature
 * @param {number}      [params.nPasses=200] — MC Dropout passes
 * @returns {Promise<{dates, p5, p25, p50, p75, p95}>} — cumulative % returns
 */
export async function inferLSTM({
  seedWindow,
  forecastDates,
  scalerMin,
  scalerMax,
  nPasses = N_MC_PASSES,
}) {
  const tf      = _tf ?? await import('@tensorflow/tfjs')
  const model   = await getModel()
  const horizon = forecastDates.length
  const lookback = seedWindow.length
  const nFeatures = seedWindow[0].length

  // Initialise: nPasses copies of the seed window + cumulative returns
  // windows[pass] = (lookback × nFeatures) float array (flat for tensor creation)
  let windows = Array.from({ length: nPasses }, () =>
    seedWindow.map(row => [...row])
  )
  const cumRs     = new Float64Array(nPasses)         // running compound return
  const allPaths  = Array.from({ length: nPasses }, () => new Array(horizon))

  for (let t = 0; t < horizon; t++) {
    // Stack all windows → batch tensor (nPasses, lookback, nFeatures)
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

    // MC Dropout ON: training=true keeps Dropout layers active
    // model.apply() is the low-level call that respects the training kwarg
    const predTensor  = model.apply(batchTensor, { training: true })
    const preds       = await predTensor.data()   // Float32Array(nPasses)

    batchTensor.dispose()
    predTensor.dispose()

    for (let p = 0; p < nPasses; p++) {
      const rScaled = preds[p]

      // Inverse MinMaxScaler (feature_range=(-1,1)):
      // x = (rScaled + 1) * (data_max - data_min) / 2 + data_min
      const rReal = (rScaled + 1) * (scalerMax[0] - scalerMin[0]) / 2 + scalerMin[0]

      cumRs[p] = (1 + cumRs[p]) * (1 + rReal) - 1
      allPaths[p][t] = cumRs[p]

      // Slide window: shift left by 1, append new row
      const newRow = [...windows[p][lookback - 1]]
      newRow[0] = rScaled   // update return feature with scaled prediction
      windows[p] = [...windows[p].slice(1), newRow]
    }
  }

  // Compute percentile bands across MC passes
  const p5  = new Array(horizon)
  const p25 = new Array(horizon)
  const p50 = new Array(horizon)
  const p75 = new Array(horizon)
  const p95 = new Array(horizon)

  for (let t = 0; t < horizon; t++) {
    const vals = allPaths.map(path => path[t]).sort((a, b) => a - b)
    const n    = vals.length
    p5[t]  = vals[Math.max(0, Math.floor(0.05  * n))] * 100
    p25[t] = vals[Math.max(0, Math.floor(0.25  * n))] * 100
    p50[t] = vals[Math.max(0, Math.floor(0.50  * n))] * 100
    p75[t] = vals[Math.max(0, Math.floor(0.75  * n))] * 100
    p95[t] = vals[Math.min(n - 1, Math.floor(0.95 * n))] * 100
  }

  return { dates: forecastDates, p5, p25, p50, p75, p95 }
}

/** True after first successful model load — lets UI skip the "model not found" skeleton. */
export function isModelLoaded() {
  return _model !== null
}
