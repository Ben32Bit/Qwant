import { useState, useCallback } from 'react'

const PHASE1_METHODS = ['monte_carlo', 'garch', 'factor']
const PHASE2_METHODS = ['hmm', 'var', 'lstm']

/**
 * Two-phase forecast fetcher.
 *
 * Phase 1: fast statistical methods (~1-2s) — renders immediately.
 * Phase 2: ML/regime methods (~10-40s) — fills in once complete.
 *
 * Each phase result is a ForecastResponse object. The panel merges them
 * by replacing any method that completed in phase 2.
 */
export function useForecast(backtest, portfolio) {
  const [phase1, setPhase1]   = useState(null)
  const [phase2, setPhase2]   = useState(null)
  const [loading, setLoading] = useState({ phase1: false, phase2: false })
  const [error, setError]     = useState(null)

  const run = useCallback(async () => {
    if (!backtest?.equity_curve) return
    setPhase1(null)
    setPhase2(null)
    setError(null)

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
    setLoading(l => ({ ...l, phase1: true }))
    try {
      const res = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body, methods: PHASE1_METHODS }),
      })
      if (!res.ok) throw new Error(`Forecast phase 1 failed: ${res.status}`)
      setPhase1(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(l => ({ ...l, phase1: false }))
    }

    // ── Phase 2: slower methods ───────────────────────────────────────────
    setLoading(l => ({ ...l, phase2: true }))
    try {
      const res = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body, methods: PHASE2_METHODS }),
      })
      if (!res.ok) throw new Error(`Forecast phase 2 failed: ${res.status}`)
      setPhase2(await res.json())
    } catch (e) {
      // Phase 2 failure is non-fatal — phase 1 results still show
      console.warn('Forecast phase 2 error:', e.message)
    } finally {
      setLoading(l => ({ ...l, phase2: false }))
    }
  }, [backtest, portfolio])

  // Merged results: phase1 base + phase2 fills in
  const allResults = mergeResults(phase1, phase2)

  return {
    results:  allResults,
    meta:     phase1,          // for forecast_start / forecast_end / historical_end_value
    loading,
    error,
    run,
    hasData:  allResults.length > 0,
  }
}

function mergeResults(phase1, phase2) {
  if (!phase1) return []
  const map = {}
  for (const r of (phase1.results ?? [])) map[r.method] = r
  for (const r of (phase2?.results ?? []))  map[r.method] = r
  // Preserve display order
  const order = ['monte_carlo', 'garch', 'factor', 'hmm', 'var', 'lstm']
  return order.map(m => map[m]).filter(Boolean)
}
