import { useState, useCallback, useRef } from 'react'

const PHASE1_METHODS = ['xgboost', 'nbeats', 'factor']
const PHASE2_METHODS = ['hmm', 'var', 'lstm']   // lstm returns features, not forecast

/**
 * Two-phase + client-ML forecast fetcher.
 *
 * Phase 1  (server, ~1-3s):   xgboost (features), garch, factor
 * Phase 1B (browser, ~1-3s):  XGBoost ONNX Runtime Web · N-BEATS pure-JS weights
 * Phase 2  (server, ~5-15s):  hmm, var + lstm feature window
 * Phase 3  (browser, ~2-5s):  Attention-LSTM MC Dropout via TF.js
 *
 * XGBoost replaces Monte Carlo GBM (Phase 2A upgrade).
 * LSTM replaced server-side tensorflow-cpu (Phase 1 upgrade).
 * Both run client-side to stay within Railway 512MB free-tier RAM.
 */
export function useForecast(backtest, portfolio) {
  const [phase1, setPhase1]     = useState(null)
  const [phase2, setPhase2]     = useState(null)
  const [newsContext, setNewsContext]   = useState(null)
  const [edgarContext, setEdgarContext] = useState(null)
  const [loading, setLoading] = useState({ phase1: false, xgb: false, nbeats: false, phase2: false, lstm: false })
  const [error, setError]     = useState(null)
  const [timing, setTiming]   = useState({ phase1Ms: null, xgbMs: null, nbeatsMs: null, phase2Ms: null, lstmMs: null })
  const p1Start      = useRef(null)
  const xgbStart     = useRef(null)
  const nbeatsStart  = useRef(null)
  const p2Start      = useRef(null)
  const lstmStart    = useRef(null)

  const run = useCallback(async () => {
    if (!backtest?.equity_curve) return
    setPhase1(null)
    setPhase2(null)
    setNewsContext(null)
    setEdgarContext(null)
    setError(null)
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

    // ── Phase 1: fast server methods ──────────────────────────────────────
    p1Start.current = Date.now()
    setLoading(l => ({ ...l, phase1: true }))
    let p1Data = null
    try {
      const res = await fetch('/api/forecast', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ...body, methods: PHASE1_METHODS }),
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

    // ── Phase 1B: XGBoost + N-BEATS client-side ONNX (parallel) ─────────────
    if (p1Data) {
      const xgbResult    = p1Data.results?.find(r => r.method === 'xgboost' && r.metadata?.client_side)
      const nbeatsResult = p1Data.results?.find(r => r.method === 'nbeats'  && r.metadata?.client_side)

      const inferTasks = []

      if (xgbResult) {
        xgbStart.current = Date.now()
        setLoading(l => ({ ...l, xgb: true }))
        const { xgb_features } = xgbResult.metadata
        inferTasks.push(
          import('../ml/XGBoostInferer.js')
            .then(({ inferXGBoost }) => inferXGBoost({
              features:      xgb_features.features,
              forecastDates: xgb_features.forecast_dates,
            }))
            .then(band => {
              const xgbMs = Date.now() - (xgbStart.current ?? Date.now())
              setTiming(t => ({ ...t, xgbMs }))
              setPhase1(p1 => p1 ? {
                ...p1,
                results: p1.results.map(r => r.method !== 'xgboost' ? r : {
                  ...r, forecast: band, compute_ms: xgbMs,
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
            .catch(e => setPhase1(p1 => p1 ? {
              ...p1,
              results: p1.results.map(r => r.method !== 'xgboost' ? r : { ...r, error: e.message }),
            } : p1))
            .finally(() => setLoading(l => ({ ...l, xgb: false })))
        )
      }

      if (nbeatsResult) {
        nbeatsStart.current = Date.now()
        setLoading(l => ({ ...l, nbeats: true }))
        const { nbeats_features } = nbeatsResult.metadata
        inferTasks.push(
          import('../ml/NBeatsInferer.js')
            .then(({ inferNBeats }) => inferNBeats({
              lastWindow:    nbeats_features.last_window,
              forecastDates: nbeats_features.forecast_dates,
            }))
            .then(band => {
              const nbeatsMs = Date.now() - (nbeatsStart.current ?? Date.now())
              setTiming(t => ({ ...t, nbeatsMs }))
              setPhase1(p1 => p1 ? {
                ...p1,
                results: p1.results.map(r => r.method !== 'nbeats' ? r : {
                  ...r, forecast: band, compute_ms: nbeatsMs,
                  metadata: {
                    vol_21d_ann: nbeats_features.vol_21d_ann,
                    ret_21d:     nbeats_features.ret_21d,
                    n_obs:       nbeats_features.n_obs,
                    periods:     12,
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

      // Run XGBoost (ONNX) and N-BEATS (pure-JS) in parallel
      await Promise.all(inferTasks)
    }

    // ── Phase 2: server returns HMM, VAR + LSTM feature window ────────────
    p2Start.current = Date.now()
    setLoading(l => ({ ...l, phase2: true }))
    let p2Data = null
    try {
      const p2Controller = new AbortController()
      const p2Timeout    = setTimeout(() => p2Controller.abort(), 90_000)  // 90s hard timeout
      try {
        const res = await fetch('/api/forecast', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ ...body, methods: PHASE2_METHODS }),
          signal:  p2Controller.signal,
        })
        if (!res.ok) throw new Error(`Forecast phase 2 failed: ${res.status}`)
        p2Data = await res.json()
        setPhase2(p2Data)
        setTiming(t => ({ ...t, phase2Ms: Date.now() - (p2Start.current ?? Date.now()) }))
      } finally {
        clearTimeout(p2Timeout)
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        console.warn('Forecast phase 2 timed out after 90s — HMM/GP skipped')
      } else {
        console.warn('Forecast phase 2 error:', e.message)
      }
    } finally {
      setLoading(l => ({ ...l, phase2: false }))
    }

    // ── Phase 3: client-side Attention-LSTM (browser TF.js) ───────────────
    if (!p2Data) return
    const lstmServerResult = p2Data.results?.find(
      r => r.method === 'lstm' && r.metadata?.client_side === true
    )
    if (!lstmServerResult) return

    lstmStart.current = Date.now()
    setLoading(l => ({ ...l, lstm: true }))
    try {
      const { inferLSTM } = await import('../ml/LSTMInferer.js')
      const { lstm_features } = lstmServerResult.metadata
      const band = await inferLSTM({
        seedWindow:    lstm_features.last_window,
        forecastDates: lstm_features.forecast_dates,
        scalerMin:     lstm_features.scaler_min,
        scalerMax:     lstm_features.scaler_max,
      })
      const lstmMs = Date.now() - (lstmStart.current ?? Date.now())
      setTiming(t => ({ ...t, lstmMs }))

      setPhase2(p2 => {
        if (!p2) return p2
        return {
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
        }
      })
    } catch (e) {
      setPhase2(p2 => {
        if (!p2) return p2
        return {
          ...p2,
          results: p2.results.map(r =>
            r.method !== 'lstm' ? r : { ...r, error: e.message }
          ),
        }
      })
    } finally {
      setLoading(l => ({ ...l, lstm: false }))
    }
  }, [backtest, portfolio])

  const allResults = mergeResults(phase1, phase2)

  return {
    results:         allResults,
    meta:            phase1,
    loading,
    error,
    run,
    hasData:         allResults.length > 0,
    timing,
    newsContext,
    edgarContext,
    p1StartRef:      p1Start,
    xgbStartRef:     xgbStart,
    nbeatsStartRef:  nbeatsStart,
    p2StartRef:      p2Start,
    lstmStartRef:    lstmStart,
  }
}

function mergeResults(phase1, phase2) {
  if (!phase1) return []
  const map = {}
  for (const r of (phase1.results ?? [])) map[r.method] = r
  for (const r of (phase2?.results ?? [])) {
    if (r.method === 'lstm' && r.metadata?.client_side === true) continue
    map[r.method] = r
  }
  const order = ['xgboost', 'nbeats', 'factor', 'hmm', 'var', 'lstm']
  return order.map(m => map[m]).filter(Boolean)
}
