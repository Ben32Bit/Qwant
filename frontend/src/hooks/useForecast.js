import { useState, useCallback, useRef } from 'react'

const PHASE1_METHODS = ['monte_carlo', 'garch', 'factor']
const PHASE2_METHODS = ['hmm', 'var', 'lstm']   // lstm returns features, not forecast

/**
 * Two-phase + client-LSTM forecast fetcher.
 *
 * Phase 1 (server, ~1-3s):   monte_carlo, garch, factor
 * Phase 2 (server, ~5-15s):  hmm, var + lstm feature window
 * Phase 3 (browser, ~2-5s):  Attention-LSTM MC Dropout via TF.js
 *
 * The LSTM was moved from server (tensorflow-cpu, ~450MB) to the browser
 * (TF.js, ~400KB model) in Phase 1 of the hybrid architecture upgrade.
 * The server now returns a scaled 60-day feature window + MinMaxScaler params;
 * the browser runs 200 MC Dropout passes to produce the fan bands.
 */
export function useForecast(backtest, portfolio) {
  const [phase1, setPhase1]   = useState(null)
  const [phase2, setPhase2]   = useState(null)
  const [loading, setLoading] = useState({ phase1: false, phase2: false, lstm: false })
  const [error, setError]     = useState(null)
  const [timing, setTiming]   = useState({ phase1Ms: null, phase2Ms: null, lstmMs: null })
  const p1Start = useRef(null)
  const p2Start = useRef(null)
  const lstmStart = useRef(null)

  const run = useCallback(async () => {
    if (!backtest?.equity_curve) return
    setPhase1(null)
    setPhase2(null)
    setError(null)
    setTiming({ phase1Ms: null, phase2Ms: null, lstmMs: null })

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
    try {
      const res = await fetch('/api/forecast', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ...body, methods: PHASE1_METHODS }),
      })
      if (!res.ok) throw new Error(`Forecast phase 1 failed: ${res.status}`)
      setPhase1(await res.json())
      setTiming(t => ({ ...t, phase1Ms: Date.now() - (p1Start.current ?? Date.now()) }))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(l => ({ ...l, phase1: false }))
    }

    // ── Phase 2: server returns HMM, VAR + LSTM feature window ────────────
    p2Start.current = Date.now()
    setLoading(l => ({ ...l, phase2: true }))
    let p2Data = null
    try {
      const res = await fetch('/api/forecast', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ...body, methods: PHASE2_METHODS }),
      })
      if (!res.ok) throw new Error(`Forecast phase 2 failed: ${res.status}`)
      p2Data = await res.json()
      setPhase2(p2Data)
      setTiming(t => ({ ...t, phase2Ms: Date.now() - (p2Start.current ?? Date.now()) }))
    } catch (e) {
      console.warn('Forecast phase 2 error:', e.message)
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

      // Patch the phase2 result: replace the feature-only lstm entry with the forecast band
      setPhase2(p2 => {
        if (!p2) return p2
        return {
          ...p2,
          results: p2.results.map(r => r.method !== 'lstm' ? r : {
            ...r,
            forecast:   band,
            compute_ms: lstmMs,
            metadata:   {
              architecture:   r.metadata.architecture,
              attention:      r.metadata.attention,
              dropout_passes: 200,
              client_side:    false,
            },
          }),
        }
      })
    } catch (e) {
      // Non-fatal: patch error into the LSTM result card
      setPhase2(p2 => {
        if (!p2) return p2
        return {
          ...p2,
          results: p2.results.map(r => r.method !== 'lstm' ? r : { ...r, error: e.message }),
        }
      })
    } finally {
      setLoading(l => ({ ...l, lstm: false }))
    }
  }, [backtest, portfolio])

  const allResults = mergeResults(phase1, phase2)

  return {
    results:    allResults,
    meta:       phase1,
    loading,
    error,
    run,
    hasData:    allResults.length > 0,
    timing,
    p1StartRef: p1Start,
    p2StartRef: p2Start,
    lstmStartRef: lstmStart,
  }
}

function mergeResults(phase1, phase2) {
  if (!phase1) return []
  const map = {}
  for (const r of (phase1.results ?? [])) map[r.method] = r
  for (const r of (phase2?.results ?? [])) {
    // Only include lstm result if it has a forecast band (not just features)
    if (r.method === 'lstm' && r.metadata?.client_side === true) continue
    map[r.method] = r
  }
  const order = ['monte_carlo', 'garch', 'factor', 'hmm', 'var', 'lstm']
  return order.map(m => map[m]).filter(Boolean)
}
