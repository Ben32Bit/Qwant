import { useState, useCallback, useRef } from 'react'

const PHASE1_METHODS = ['monte_carlo', 'garch', 'factor']
const PHASE2_METHODS = ['hmm', 'var', 'lstm']

/**
 * Two-phase forecast fetcher with per-phase timing.
 *
 * Phase 1: fast statistical methods (~1-3s) — renders immediately.
 * Phase 2: ML/regime methods (~20-90s) — fills in once complete.
 *
 * Returns timing so the UI can show an ETA progress bar.
 */
export function useForecast(backtest, portfolio) {
  const [phase1, setPhase1]   = useState(null)
  const [phase2, setPhase2]   = useState(null)
  const [loading, setLoading] = useState({ phase1: false, phase2: false })
  const [error, setError]     = useState(null)
  // Timing: ms each phase took (null while pending / not run)
  const [timing, setTiming]   = useState({ phase1Ms: null, phase2Ms: null })
  // Wall-clock start times (refs avoid stale closure issues)
  const p1Start = useRef(null)
  const p2Start = useRef(null)

  const run = useCallback(async () => {
    if (!backtest?.equity_curve) return
    setPhase1(null)
    setPhase2(null)
    setError(null)
    setTiming({ phase1Ms: null, phase2Ms: null })

    const body = {
      equity_curve:      backtest.equity_curve,
      assets:            portfolio?.assets ?? [],
      start_date:        portfolio?.start_date ?? backtest.equity_curve[0]?.date,
      end_date:          portfolio?.end_date   ?? backtest.equity_curve.at(-1)?.date,
      horizon_days:      252,
      n_paths:           1000,
      ff5_decomposition: backtest.ff5_decomposition ?? null,
    }

    // ── Phase 1: fast methods ─────────────────────────────────────────────
    p1Start.current = Date.now()
    setLoading(l => ({ ...l, phase1: true }))
    try {
      const res = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body, methods: PHASE1_METHODS }),
      })
      if (!res.ok) throw new Error(`Forecast phase 1 failed: ${res.status}`)
      setPhase1(await res.json())
      setTiming(t => ({ ...t, phase1Ms: Date.now() - (p1Start.current ?? Date.now()) }))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(l => ({ ...l, phase1: false }))
    }

    // ── Phase 2: slower methods ───────────────────────────────────────────
    p2Start.current = Date.now()
    setLoading(l => ({ ...l, phase2: true }))
    try {
      const res = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body, methods: PHASE2_METHODS }),
      })
      if (!res.ok) throw new Error(`Forecast phase 2 failed: ${res.status}`)
      setPhase2(await res.json())
      setTiming(t => ({ ...t, phase2Ms: Date.now() - (p2Start.current ?? Date.now()) }))
    } catch (e) {
      // Phase 2 failure is non-fatal — phase 1 results still show
      console.warn('Forecast phase 2 error:', e.message)
    } finally {
      setLoading(l => ({ ...l, phase2: false }))
    }
  }, [backtest, portfolio])

  const allResults = mergeResults(phase1, phase2)

  return {
    results:  allResults,
    meta:     phase1,
    loading,
    error,
    run,
    hasData:  allResults.length > 0,
    timing,
    // Expose phase start refs so ForecastPanel can compute elapsed time live
    p1StartRef: p1Start,
    p2StartRef: p2Start,
  }
}

function mergeResults(phase1, phase2) {
  if (!phase1) return []
  const map = {}
  for (const r of (phase1.results ?? [])) map[r.method] = r
  for (const r of (phase2?.results ?? []))  map[r.method] = r
  const order = ['monte_carlo', 'garch', 'factor', 'hmm', 'var', 'lstm']
  return order.map(m => map[m]).filter(Boolean)
}
