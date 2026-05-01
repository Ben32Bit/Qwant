import { useState, useCallback, useRef, useMemo } from 'react'

const PHASE1_METHODS = ['nbeats']
const PHASE2_METHODS = ['hmm', 'var', 'lstm', 'timesfm']

export function useForecast(backtest, portfolio) {
  const [phase1, setPhase1] = useState(null)
  const [phase2, setPhase2] = useState(null)
  const [newsContext, setNewsContext]   = useState(null)
  const [edgarContext, setEdgarContext] = useState(null)
  const [loading, setLoading] = useState({
    phase1: false, nbeats: false, phase2: false, lstm: false,
  })
  const [error, setError]   = useState(null)
  const [timing, setTiming] = useState({
    phase1Ms: null, nbeatsMs: null, phase2Ms: null, lstmMs: null,
  })
  const p1Start     = useRef(null)
  const nbeatsStart = useRef(null)
  const p2Start     = useRef(null)
  const lstmStart   = useRef(null)

  const run = useCallback(async () => {
    if (!backtest?.equity_curve) return
    setPhase1(null); setPhase2(null)
    setNewsContext(null); setEdgarContext(null); setError(null)
    setTiming({ phase1Ms: null, nbeatsMs: null, phase2Ms: null, lstmMs: null })

    const body = {
      equity_curve:      backtest.equity_curve,
      assets:            portfolio?.assets ?? [],
      start_date:        portfolio?.start_date ?? backtest.equity_curve[0]?.date,
      end_date:          portfolio?.end_date   ?? backtest.equity_curve.at(-1)?.date,
      horizon_days:      63,
      n_paths:           1000,
      ff5_decomposition: backtest.ff5_decomposition ?? null,
    }

    // ── Phase 1: server → N-BEATS features ───────────────────────────────
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

    // ── Phase 1B: N-BEATS browser inference ─────────────────────────────
    if (p1Data) {
      const nbeatsResult = p1Data.results?.find(r => r.method === 'nbeats' && r.metadata?.client_side)

      if (nbeatsResult) {
        nbeatsStart.current = Date.now()
        setLoading(l => ({ ...l, nbeats: true }))
        const { nbeats_features } = nbeatsResult.metadata
        await import('../ml/NBeatsInferer.js')
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
                  periods:     3,
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
      }
    }

    // ── Phase 2: server ────────────────────────────────────────────────────
    // Railway free-tier + Cloudflare edge impose a ~100s upstream timeout. On
    // a cold boot HMM+GP can exceed that and the edge returns 502 before the
    // container has even finished computing. One silent retry on 502/504
    // almost always succeeds because the container is now warm from attempt 1.
    p2Start.current = Date.now()
    setLoading(l => ({ ...l, phase2: true }))
    let p2Data = null
    let p2FailReason = null

    const runPhase2 = async () => {
      const p2Controller = new AbortController()
      const p2Timeout    = setTimeout(() => p2Controller.abort(), 180_000)
      try {
        const res = await fetch('/api/forecast', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...body, methods: PHASE2_METHODS }),
          signal: p2Controller.signal,
        })
        if (!res.ok) {
          const err = new Error(`Forecast phase 2 failed: ${res.status}`)
          err.status = res.status
          throw err
        }
        return await res.json()
      } finally {
        clearTimeout(p2Timeout)
      }
    }

    try {
      try {
        p2Data = await runPhase2()
      } catch (e) {
        // Retry once on edge-proxy failure (502/504) — container is warm now.
        if (e.status === 502 || e.status === 504) {
          console.warn(`Phase 2 got ${e.status}, retrying once…`)
          await new Promise(r => setTimeout(r, 1500))
          p2Data = await runPhase2()
        } else {
          throw e
        }
      }
      setPhase2(p2Data)
      if (p2Data?.news_context)  setNewsContext(p2Data.news_context)
      if (p2Data?.edgar_context) setEdgarContext(p2Data.edgar_context)
      setTiming(t => ({ ...t, phase2Ms: Date.now() - (p2Start.current ?? Date.now()) }))
    } catch (e) {
      p2FailReason = e.name === 'AbortError'
        ? 'Phase 2 timed out after 180s'
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
            { method: 'hmm',     label: 'HMM',              color: '#a855f7', error: reason },
            { method: 'var',     label: 'Gaussian Process', color: '#ff6b35', error: reason },
            { method: 'lstm',    label: 'Attention-LSTM',   color: '#ff4757', error: reason },
            { method: 'timesfm', label: 'TimesFM 2.5',      color: '#00d4aa', error: reason },
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
        isEvalWindows:   lstmFeatures.is_eval_windows  ?? null,
        isEvalTargets:   lstmFeatures.is_eval_targets  ?? null,
        shadow:          lstmFeatures.shadow            ?? null,
      })
      const lstmMs = Date.now() - (lstmStart.current ?? Date.now())
      setTiming(t => ({ ...t, lstmMs }))
      setPhase2(p2 => p2 ? {
        ...p2,
        results: p2.results.map(r => r.method !== 'lstm' ? r : {
          ...r,
          forecast:   band,
          compute_ms: lstmMs,
          oos_r2:     band.oos_r2 ?? null,
          metadata: {
            architecture:   r.metadata.architecture,
            attention:      r.metadata.attention,
            dropout_passes: 200,
            client_side:    false,
            is_r2:          band.is_r2  ?? null,
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
        regime_probs:          phase2?.regime_probs          ?? phase1?.regime_probs          ?? null,
        ensemble_weights:      phase2?.ensemble_weights      ?? phase1?.ensemble_weights      ?? null,
        shadow_forecast_start: phase2?.shadow_forecast_start ?? null,
        shadow_forecast_end:   phase2?.shadow_forecast_end   ?? null,
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
    nbeatsStartRef: nbeatsStart,
    p2StartRef:     p2Start,
    lstmStartRef:   lstmStart,
  }
}

function mergeResults(phase1, phase2) {
  if (!phase1) return []
  const map = {}
  for (const r of (phase1.results ?? [])) map[r.method] = r
  for (const r of (phase2?.results ?? [])) map[r.method] = r
  const order = ['nbeats', 'timesfm', 'hmm', 'var', 'lstm']
  return order.map(m => map[m]).filter(Boolean)
}
