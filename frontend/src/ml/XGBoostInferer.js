/**
 * XGBoostInferer.js — Client-side XGBoost quantile forecast via ONNX Runtime Web.
 *
 * PATCHED (Priority 2 + 3):
 *   - Pre-flight HEAD check on each model URL with clear "not found" errors so
 *     the UI never hangs on "waiting…" when artefacts aren't committed.
 *   - Explicit feature-count validation (meta.n_features vs features.length)
 *     so a server/model mismatch fails loudly, not silently.
 *   - Configurable expected feature count via meta.json (aligns with server's
 *     prepare_xgboost_features, currently 14 = 9 price + 5 macro).
 *
 * Input contract:
 *   - features: number[] — raw (unscaled) features. HistGradientBoostingRegressor
 *     uses histogram binning internally, so no StandardScaler is needed in the
 *     ONNX graph. The client passes raw feature values directly.
 *   - Expected length is asserted against meta.json::n_features (14 currently).
 */

const QUANTILE_TAGS = ['q05', 'q25', 'q50', 'q75', 'q95']
const MODEL_BASE    = '/models/xgboost'
const META_URL      = `${MODEL_BASE}/meta.json`
const HORIZON       = 252
const PERIOD_DAYS   = 21
const CREATE_TIMEOUT_MS = 15_000
const RUN_TIMEOUT_MS    = 10_000

let _ort      = null
const _sessions = {}
let _meta     = null

function withTimeout(promise, ms, label) {
  let timer
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${label} timed out after ${ms}ms`)),
      ms,
    )
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer))
}

async function getOrt() {
  if (_ort) return _ort
  if (typeof window === 'undefined' || !window.ort) {
    throw new Error('onnxruntime-web not loaded — check CDN <script> tag in index.html')
  }
  _ort = window.ort
  _ort.env.wasm.numThreads = 1
  return _ort
}

async function loadMeta() {
  if (_meta) return _meta
  const resp = await fetch(META_URL, { cache: 'force-cache' })
  if (!resp.ok) {
    throw new Error(
      `XGBoost meta.json not found at ${META_URL} (HTTP ${resp.status}). ` +
      `Run: cd backend && python scripts/train_xgboost.py`
    )
  }
  _meta = await resp.json()
  if (!Number.isInteger(_meta?.n_features)) {
    throw new Error(`XGBoost meta.json invalid: missing integer n_features`)
  }
  return _meta
}

async function getSession(tag) {
  if (_sessions[tag]) return _sessions[tag]
  const ort = await getOrt()
  const url = `${MODEL_BASE}/${tag}.onnx`

  // Pre-flight: confirm the file exists AND is actually ONNX binary.
  // A SPA catch-all rewrite can silently return index.html (text/html) with
  // a 200 status, which would hang InferenceSession.create as it tries to
  // parse HTML as a model. Check content-type explicitly.
  const head = await fetch(url, { method: 'HEAD' })
  if (!head.ok) {
    throw new Error(
      `XGBoost model ${tag}.onnx not found (HTTP ${head.status}). ` +
      `Run: cd backend && python scripts/train_xgboost.py`
    )
  }
  const ctype = head.headers.get('content-type') ?? ''
  if (ctype.includes('text/html')) {
    throw new Error(
      `XGBoost ${tag}.onnx served as text/html — likely a Vercel SPA rewrite ` +
      `is intercepting the model URL. Check vercel.json rewrites.`
    )
  }

  _sessions[tag] = await withTimeout(
    ort.InferenceSession.create(url, { executionProviders: ['wasm'] }),
    CREATE_TIMEOUT_MS,
    `XGBoost ${tag}.onnx session load`,
  )
  return _sessions[tag]
}

// ONNX Runtime Web sometimes throws a raw number (a WASM memory pointer)
// when the model's ONNX opset is beyond what this ort-web build supports,
// or when an internal op fails. Wrap those into a diagnostic message so
// users see something they can act on instead of "XGBoost: 8667456".
function describeOrtError(e, stage) {
  if (typeof e === 'number') {
    return new Error(
      `ORT WASM error code ${e} during ${stage}. Likely cause: ONNX model ` +
      `uses an opset beyond what the pinned onnxruntime-web build supports. ` +
      `Either bump the CDN pin in index.html or re-export train_xgboost.py ` +
      `with target_opset={'': 18, 'ai.onnx.ml': 3}.`,
    )
  }
  if (e instanceof Error) return e
  return new Error(`ORT error during ${stage}: ${String(e)}`)
}

async function predict21d(features) {
  const ort  = await getOrt()
  const meta = await loadMeta()

  if (features.length !== meta.n_features) {
    throw new Error(
      `XGBoost feature count mismatch: model expects ${meta.n_features}, got ${features.length}. ` +
      `Server's prepare_xgboost_features and the trained ONNX models must agree. ` +
      `Either retrain with the current feature list or align the server output.`
    )
  }

  const floatArr = new Float32Array(features)
  const results  = {}
  for (const tag of QUANTILE_TAGS) {
    let session
    try {
      session = await getSession(tag)
    } catch (e) {
      throw describeOrtError(e, `${tag} session load`)
    }
    const tensor  = new ort.Tensor('float32', floatArr, [1, features.length])
    let output
    try {
      output = await withTimeout(
        session.run({ float_input: tensor }),
        RUN_TIMEOUT_MS,
        `XGBoost ${tag} inference`,
      )
    } catch (e) {
      throw describeOrtError(e, `${tag} run`)
    }
    const val = output.variable?.data[0] ?? output[Object.keys(output)[0]].data[0]
    results[tag] = Number(val)
  }
  return results
}

function extrapolate252(preds21, forecastDates) {
  const { q05, q25, q50, q75, q95 } = preds21
  const halfInner = (q75 - q25) / 2
  const halfOuter = (q95 - q05) / 2

  const p5 = [], p25 = [], p50 = [], p75 = [], p95 = []
  for (let t = 1; t <= forecastDates.length; t++) {
    const k     = t / PERIOD_DAYS
    const med   = Math.pow(1 + q50, k) - 1
    const sqrtK = Math.sqrt(k)
    p50.push(med * 100)
    p25.push((med - halfInner * sqrtK) * 100)
    p75.push((med + halfInner * sqrtK) * 100)
    p5.push( (med - halfOuter * sqrtK) * 100)
    p95.push((med + halfOuter * sqrtK) * 100)
  }
  return { dates: forecastDates, p5, p25, p50, p75, p95 }
}

export async function inferXGBoost({ features, forecastDates }) {
  const preds21 = await predict21d(features)
  return extrapolate252(preds21, forecastDates)
}
