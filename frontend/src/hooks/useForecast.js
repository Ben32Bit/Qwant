import { useState, useCallback, useRef, useMemo } from 'react'

const PHASE1_METHODS = ['xgboost', 'nbeats', 'factor']
const PHASE2_METHODS = ['hmm', 'var', 'lstm']

// Per-method horizon caps (trading days). Methods trained on a short target
// shouldn't masquerade as full-year forecasters — XGBoost's 21-day quantile
// regressor extrapolates to 252d via √t scaling (heuristic, not earned), and
// N-BEATS stacks twelve 21-day recursive predictions (error compounds fast).
// Factor / HMM / GP / LSTM simulate the full path directly, so they run the
// full horizon. Truncating here also drives the ensemble's "model count"
// degradation past each cap.
export const HORIZON_CAPS_DAYS = { xgboost: 21, nbeats: 63 }

function capBand(band, maxDays) {
  if (!band?.dates?.length || !maxDays) return band
  const cap = Math.min(band.dates.length, maxDays)
  if (cap === band.dates.length) return band
  return {
    dates: band.dates.slice(0, cap),
    p5:    band.p5?.slice(0, cap),
    p25:   band.p25?.slice(0, cap),
    p50:   band.p50?.slice(0, cap),
    p75:   band.p75?.slice(0, cap),
    p95:   band.p95?.slice(0, cap),
  }
}

export function useForecast(backtest, portfolio) {
  const [phase1, setPhase1] = useState(null)
  const [phase2, setPhase2] = useState(null)
  const [newsContext, setNewsContext]   = useState(null)
  const [edgarContext, setEdgarContext] = useState(null)
  const [loading, setLoading] = useState({
    phase1: false, xgb: false, nbeats: false, phase2: false, lstm: false,
  })
  const [error, setError]   = useState(null)
  const [timing, setTiming] = useState({
    phase1Ms: null, xgbMs: null, nbeatsMs: null, phase2Ms: null, lstmMs: null,
  })
  const p1Start     = useRef(null)
  const xgbStart    = useRef(null)
  const nbeatsStart = useRef(null)
  const p2Start     = useRef(null)
  const lstmStart   = useRef(null)

  const run = useCallback(async () => {
    if (!backtest?.equity_curve) return
    setPhase1(null); setPhase2(null)
    setNewsContext(null); setEdgarContext(null); setError(null)
    setTiming({ phase1Ms: null, xgbMs: null, nbeatsMs: null, phase2Ms: null, lstmMs: null })

    const body = {
      equity_curve:      backtest.equity_curve,
      assets:            portfolio?.assets ?? [],
      start_date:        portfolio?.start_date ?? backtest.equity_curve[0]?.date,
      end_date:          portfolio?.end_date   ?? backtest.equity_curve.at(-1)?.date,
      horizon_days:      252,
      n_paths:           1000,
      ff5_decomposition: backtest.ff5_decomposition ?? null,
    }

    // ── Phase 1: server ───────────────────────────────────────────────────
    p1Start.current = Date.now()
    setLoading(l => ({ ...l, phase1: true }))
    let p1Data = null
    try {
      const res = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body, methods: PHASE1_METHODS }),
      })
      if (!res.ok) throw new Error(`Forecast phase 1 failed: ${res.status}`)
      p1Data = await res.json()
      setPhase1(p1Data)
      if (p1Data?.news_context)  setNewsContext(p1Data.news_context)
      if (p1Data?.edgar_context) setEdgarContext(p1Data.edgar_context)
      setTiming(t => ({ ...t, phase1Ms: Date.now() - (p1Start.current ?? Date.now()) }))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(l => ({ ...l, phase1: false }))
    }

    // ── Phase 1B: XGBoost + N-BEATS browser ONNX (parallel) ────────────────
    if (p1Data) {
      const xgbResult    = p1Data.results?.find(r => r.method === 'xgboost' && r.metadata?.client_side)
      const nbeatsResult = p1Data.results?.find(r => r.method === 'nbeats'  && r.metadata?.client_side)
      const tasks = []

      if (xgbResult) {
        xgbStart.current = Date.now()
        setLoading(l => ({ ...l, xgb: true }))
        const xgb_features = xgbResult.metadata?.xgb_features
        // Guard: if the server contract broke and features aren't present,
        // fail the card loudly instead of letting the browser task hang
        // and leave the UI stuck on "waiting…".
        if (!xgb_features || !Array.isArray(xgb_features.features)) {
          setPhase1(p1 => p1 ? {
            ...p1,
            results: p1.results.map(r => r.method !== 'xgboost' ? r : {
              ...r, error: 'XGBoost: server returned no feature bundle',
            }),
          } : p1)
          setLoading(l => ({ ...l, xgb: false }))
        } else {
          tasks.push(
            import('../ml/XGBoostInferer.js')
              .then(({ inferXGBoost }) => inferXGBoost({
                features:      xgb_features.features,
                forecastDates: xgb_features.forecast_dates,
              }))
              .then(band => {
                const xgbMs = Date.now() - (xgbStart.current ?? Date.now())
                setTiming(t => ({ ...t, xgbMs }))
                const capped = capBand(band, HORIZON_CAPS_DAYS.xgboost)
                setPhase1(p1 => p1 ? {
                  ...p1,
                  results: p1.results.map(r => r.method !== 'xgboost' ? r : {
                    ...r, forecast: capped, compute_ms: xgbMs,
                    metadata: {
                      oos_r2: xgb_features.oos_r2 ?? null,
                      ret_21d_ann: xgb_features.ret_21d_ann,
                      vol_21d_ann: xgb_features.vol_21d_ann,
                      rsi_14: xgb_features.rsi_14,
                      vol_regime: xgb_features.vol_regime,
                      n_obs: xgb_features.n_obs,
                      client_side: false,
                    },
                  }),
                } : p1)
              })
              .catch(e => {
                console.warn('XGBoost inference failed:', e)
                setPhase1(p1 => p1 ? {
                  ...p1,
                  results: p1.results.map(r => r.method !== 'xgboost' ? r : {
                    ...r, error: `XGBoost: ${e?.message ?? String(e)}`,
                  }),
                } : p1)
              })
              .finally(() => setLoading(l => ({ ...l, xgb: false })))
          )
        }
      } else {
        // Server returned xgboost result but without client_side — surface
        // this rather than leaving the card on "waiting…" forever.
        const rawXgb = p1Data.results?.find(r => r.method === 'xgboost')
        if (rawXgb && !rawXgb.error && !rawXgb.forecast) {
          setPhase1(p1 => p1 ? {
            ...p1,
            results: p1.results.map(r => r.method !== 'xgboost' ? r : {
              ...r, error: 'XGBoost: server did not flag client_side inference',
            }),
          } : p1)
        }
      }

      if (nbeatsResult) {
        nbeatsStart.current = Date.now()
        setLoading(l => ({ ...l, nbeats: true }))
        const { nbeats_features } = nbeatsResult.metadata
        tasks.push(
          import('../ml/NBeatsInferer.js')
            .then(({ inferNBeats }) => inferNBeats({
              lastWindow:    nbeats_features.last_window,
              forecastDates: nbeats_features.forecast_dates,
            }))
            .then(band => {
              const nbeatsMs = Date.now() - (nbeatsStart.current ?? Date.now())
              setTiming(t => ({ ...t, nbeatsMs }))
              const capped = capBand(band, HORIZON_CAPS_DAYS.nbeats)
              setPhase1(p1 => p1 ? {
                ...p1,
                results: p1.results.map(r => r.method !== 'nbeats' ? r : {
                  ...r, forecast: capped, compute_ms: nbeatsMs,
                  metadata: {
                    vol_21d_ann: nbeats_features.vol_21d_ann,
                    ret_21d:     nbeats_features.ret_21d,
                    n_obs:       nbeats_features.n_obs,
                    periods:     Math.ceil(HORIZON_CAPS_DAYS.nbeats / 21),
                    client_side: false,
                  },
                }),
              } : p1)
            })
            .catch(e => setPhase1(p1 => p1 ? {
              ...p1,
              results: p1.results.map(r => r.method !== 'nbeats' ? r : { ...r, error: e.message }),
            } : p1))
            .finally(() => setLoading(l => ({ ...l, nbeats: false })))
        )
      }

      await Promise.all(tasks)
    }

    // ── Phase 2: server ────────────────────────────────────────────────────
    p2Start.current = Date.now()
    setLoading(l => ({ ...l, phase2: true }))
    let p2Data = null
    let p2FailReason = null
    try {
      const p2Controller = new AbortController()
      const p2Timeout    = setTimeout(() => p2Controller.abort(), 90_000)
      try {
        const res = await fetch('/api/forecast', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...body, methods: PHASE2_METHODS }),
          signal: p2Controller.signal,
        })
        if (!res.ok) throw new Error(`Forecast phase 2 failed: ${res.status}`)
        p2Data = await res.json()
        setPhase2(p2Data)
        setTiming(t => ({ ...t, phase2Ms: Date.now() - (p2Start.current ?? Date.now()) }))
      } finally {
        clearTimeout(p2Timeout)
      }
    } catch (e) {
      p2FailReason = e.name === 'AbortError'
        ? 'Phase 2 timed out after 90s'
        : `Phase 2 error: ${e.message}`
      console.warn(p2FailReason)
    } finally {
      setLoading(l => ({ ...l, phase2: false }))
    }

    // ── Phase 3: browser Attention-LSTM (NO fallback — server features required) ─
    const lstmServerResult = p2Data?.results?.find(
      r => r.method === 'lstm' && r.metadata?.client_side === true
    )
    const lstmFeatures = lstmServerResult?.metadata?.lstm_features

    if (!lstmFeatures) {
      const reason = p2FailReason ?? 'Server returned no LSTM feature bundle'
      if (p2Data) {
        setPhase2(p2 => p2 ? {
          ...p2,
          results: (p2.results ?? []).map(r => r.method !== 'lstm' ? r : {
            ...r, error: reason,
          }),
        } : p2)
      } else {
        // Phase 2 entirely failed — inject error entries for ALL three methods
        // so they render as "error" instead of "waiting…"
        setPhase2({
          results: [
            { method: 'hmm',  label: 'HMM',              color: '#4a9eff', error: reason },
            { method: 'var',  label: 'Gaussian Process', color: '#9c88ff', error: reason },
            { method: 'lstm', label: 'Attention-LSTM',   color: '#ff4757', error: reason },
          ],
        })
      }
      return
    }

    lstmStart.current = Date.now()
    setLoading(l => ({ ...l, lstm: true }))
    try {
      const { inferLSTM } = await import('../ml/LSTMInferer.js')
      const band = await inferLSTM({
        seedWindow:      lstmFeatures.last_window,
        forecastDates:   lstmFeatures.forecast_dates,
        rawReturnSeed:   lstmFeatures.raw_return_seed,
        currentSigma21d: lstmFeatures.current_sigma_21d,
        sigma63History:  lstmFeatures.sigma_63_history,
      })
      const lstmMs = Date.now() - (lstmStart.current ?? Date.now())
      setTiming(t => ({ ...t, lstmMs }))
      setPhase2(p2 => p2 ? {
        ...p2,
        results: p2.results.map(r => r.method !== 'lstm' ? r : {
          ...r,
          forecast:   band,
          compute_ms: lstmMs,
          metadata: {
            architecture:   r.metadata.architecture,
            attention:      r.metadata.attention,
            dropout_passes: 200,
            client_side:    false,
          },
        }),
      } : p2)
    } catch (e) {
      setPhase2(p2 => p2 ? {
        ...p2,
        results: p2.results.map(r => r.method !== 'lstm' ? r : { ...r, error: e.message }),
      } : p2)
    } finally {
      setLoading(l => ({ ...l, lstm: false }))
    }
  }, [backtest, portfolio])

  // Memoize so `results` is a stable reference when phase1/phase2 haven't
  // changed. Otherwise consumers that depend on it (e.g. the ensemble effect
  // in ForecastPanel) fire every render and can drive a runaway loop.
  const allResults = useMemo(() => mergeResults(phase1, phase2), [phase1, phase2])

  // Phase 1 carries forecast_start / forecast_end; Phase 2 carries
  // regime_probs + ensemble_weights (computed only after HMM lands).
  // Surface both under a single `meta` so ForecastPanel doesn't need to
  // know which phase owns which field.
  const meta = (phase1 || phase2)
    ? {
        ...(phase1 ?? {}),
        regime_probs:     phase2?.regime_probs     ?? phase1?.regime_probs     ?? null,
        ensemble_weights: phase2?.ensemble_weights ?? phase1?.ensemble_weights ?? null,
      }
    : null

  return {
    results:        allResults,
    meta,
    loading,
    error,
    run,
    hasData:        allResults.length > 0,
    timing,
    newsContext,
    edgarContext,
    p1StartRef:     p1Start,
    xgbStartRef:    xgbStart,
    nbeatsStartRef: nbeatsStart,
    p2StartRef:     p2Start,
    lstmStartRef:   lstmStart,
  }
}

function mergeResults(phase1, phase2) {
  if (!phase1) return []
  const map = {}
  for (const r of (phase1.results ?? [])) map[r.method] = r
  for (const r of (phase2?.results ?? [])) {
    map[r.method] = r
  }
  const order = ['xgboost', 'nbeats', 'factor', 'hmm', 'var', 'lstm']
  return order.map(m => map[m]).filter(Boolean)
}
