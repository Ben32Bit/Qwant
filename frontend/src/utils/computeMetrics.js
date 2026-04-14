/**
 * Client-side metric computation from sampled equity curve data.
 *
 * Used by ResultsPanel to recalculate all portfolio metrics when the user
 * selects a sub-period with the equity curve range slider.
 *
 * The equity curve from the backend is sampled (~500 pts for 10yr data,
 * so each step ≈ 5 calendar days). We account for variable step length
 * by annualising using actual calendar time, not a fixed 252-day assumption.
 */

const TRADING_DAYS = 252

/**
 * Compute full portfolio metrics from a slice of the equity curve.
 *
 * @param {Array<{date:string, value:number}>} portCurve  - portfolio NAV series
 * @param {Array<{date:string, value:number}>|null} bmCurve - benchmark NAV series (aligned)
 * @param {number} rfr - annual risk-free rate (default 0.05)
 * @returns {object|null} metrics object matching the backend PortfolioMetrics shape
 */
export function computeMetricsFromCurves(portCurve, bmCurve, rfr = 0.05) {
  if (!portCurve || portCurve.length < 5) return null

  const n = portCurve.length
  const startVal = portCurve[0].value
  const endVal   = portCurve[n - 1].value
  if (!startVal || !endVal || startVal <= 0) return null

  // ── Calendar years spanned ───────────────────────────────────────────────
  const startMs = new Date(portCurve[0].date).getTime()
  const endMs   = new Date(portCurve[n - 1].date).getTime()
  const years   = Math.max((endMs - startMs) / (365.25 * 86_400_000), 0.01)

  // ── Step returns ─────────────────────────────────────────────────────────
  const steps = []
  for (let i = 1; i < n; i++) {
    const r = portCurve[i].value / portCurve[i - 1].value - 1
    if (isFinite(r)) steps.push(r)
  }
  const nSteps = steps.length
  if (nSteps < 3) return null

  const periodsPerYear = nSteps / years   // annualisation factor

  // ── Total return & CAGR ──────────────────────────────────────────────────
  const totalReturn = endVal / startVal - 1
  const cagr = Math.pow(endVal / startVal, 1 / years) - 1

  // ── Volatility ───────────────────────────────────────────────────────────
  const meanStep = steps.reduce((a, b) => a + b, 0) / nSteps
  const variance = steps.reduce((a, r) => a + (r - meanStep) ** 2, 0) / Math.max(nSteps - 1, 1)
  const stdStep  = Math.sqrt(variance)
  const volatility = stdStep * Math.sqrt(periodsPerYear)

  // ── Sharpe ───────────────────────────────────────────────────────────────
  const sharpe = volatility > 0 ? (cagr - rfr) / volatility : null

  // ── Downside deviation & Sortino ─────────────────────────────────────────
  const rfrPerStep = Math.pow(1 + rfr, 1 / periodsPerYear) - 1
  const ddSteps    = steps.filter(r => r < rfrPerStep)
  const ddVariance = ddSteps.length > 0
    ? ddSteps.reduce((a, r) => a + (r - rfrPerStep) ** 2, 0) / nSteps
    : 0
  const downsideDeviation = Math.sqrt(ddVariance) * Math.sqrt(periodsPerYear)
  const sortino = downsideDeviation > 0 ? (cagr - rfr) / downsideDeviation : null

  // ── Max drawdown ─────────────────────────────────────────────────────────
  let peak = -Infinity, maxDD = 0, peakIdx = 0, maxDDDays = 0
  for (let i = 0; i < n; i++) {
    const v = portCurve[i].value
    if (v > peak) { peak = v; peakIdx = i }
    const dd = (v - peak) / peak
    if (dd < maxDD) {
      maxDD = dd
      maxDDDays = Math.round(((i - peakIdx) / nSteps) * years * TRADING_DAYS)
    }
  }

  const calmar = maxDD !== 0 ? cagr / Math.abs(maxDD) : null

  // ── VaR & CVaR (annualised approximation) ────────────────────────────────
  const sorted  = [...steps].sort((a, b) => a - b)
  const varIdx  = Math.max(0, Math.floor(0.05 * nSteps) - 1)
  const var95   = sorted[varIdx] * Math.sqrt(periodsPerYear)
  const cvar95  = sorted.slice(0, varIdx + 1).reduce((a, b) => a + b, 0) / (varIdx + 1)
                * Math.sqrt(periodsPerYear)

  // ── Skewness & kurtosis ───────────────────────────────────────────────────
  const m3 = steps.reduce((a, r) => a + (r - meanStep) ** 3, 0) / nSteps
  const m4 = steps.reduce((a, r) => a + (r - meanStep) ** 4, 0) / nSteps
  const skewness       = stdStep > 0 ? m3 / stdStep ** 3 : 0
  const excessKurtosis = stdStep > 0 ? m4 / stdStep ** 4 - 3 : 0

  // ── Benchmark metrics ─────────────────────────────────────────────────────
  let beta = null, alpha = null, rSquared = null
  let trackingError = null, informationRatio = null, treynor = null
  let upCapture = null, downCapture = null

  if (bmCurve?.length >= 5) {
    // Build date → value map
    const bmMap = {}
    bmCurve.forEach(pt => { bmMap[pt.date] = pt.value })

    const aligned = []
    for (let i = 1; i < n; i++) {
      const d  = portCurve[i].date
      const dp = portCurve[i - 1].date
      if (bmMap[d] != null && bmMap[dp] != null && bmMap[dp] > 0) {
        aligned.push({
          p: portCurve[i].value / portCurve[i - 1].value - 1,
          b: bmMap[d] / bmMap[dp] - 1,
        })
      }
    }

    if (aligned.length >= 5) {
      const na    = aligned.length
      const pMean = aligned.reduce((a, x) => a + x.p, 0) / na
      const bMean = aligned.reduce((a, x) => a + x.b, 0) / na

      let cov = 0, bVar = 0, pVar = 0
      for (const x of aligned) {
        cov  += (x.p - pMean) * (x.b - bMean)
        bVar += (x.b - bMean) ** 2
        pVar += (x.p - pMean) ** 2
      }
      cov /= na - 1; bVar /= na - 1; pVar /= na - 1

      beta = bVar > 0 ? cov / bVar : null

      // BM CAGR over the same date range
      const bmFirst = bmCurve.find(pt => pt.date >= portCurve[0].date)
      const bmLast  = [...bmCurve].reverse().find(pt => pt.date <= portCurve[n - 1].date)
      if (bmFirst?.value && bmLast?.value) {
        const bmCagr = Math.pow(bmLast.value / bmFirst.value, 1 / years) - 1
        if (beta != null) {
          alpha  = cagr - (rfr + beta * (bmCagr - rfr))
          treynor = beta !== 0 ? (cagr - rfr) / beta : null
        }
      }

      if (pVar > 0 && bVar > 0) {
        const corr = cov / Math.sqrt(pVar * bVar)
        rSquared = Math.min(1, corr * corr)
      }

      const active = aligned.map(x => x.p - x.b)
      const aMean  = active.reduce((a, b) => a + b, 0) / na
      const aVar   = active.reduce((a, r) => a + (r - aMean) ** 2, 0) / (na - 1)
      const aStd   = Math.sqrt(aVar)
      trackingError    = aStd * Math.sqrt(periodsPerYear)
      informationRatio = trackingError > 0 ? (aMean / aStd) * Math.sqrt(periodsPerYear) : null

      const upDays   = aligned.filter(x => x.b > 0)
      const downDays = aligned.filter(x => x.b < 0)
      if (upDays.length > 0) {
        const pUp = upDays.reduce((a, x) => a * (1 + x.p), 1) - 1
        const bUp = upDays.reduce((a, x) => a * (1 + x.b), 1) - 1
        upCapture = bUp !== 0 ? pUp / bUp : null
      }
      if (downDays.length > 0) {
        const pDown = downDays.reduce((a, x) => a * (1 + x.p), 1) - 1
        const bDown = downDays.reduce((a, x) => a * (1 + x.b), 1) - 1
        downCapture = bDown !== 0 ? pDown / bDown : null
      }
    }
  }

  // ── Best / worst year & month ─────────────────────────────────────────────
  const yearBuckets = {}
  for (let i = 1; i < n; i++) {
    const yr = portCurve[i].date.slice(0, 4)
    if (!yearBuckets[yr]) yearBuckets[yr] = { start: portCurve[i - 1].value }
    yearBuckets[yr].end = portCurve[i].value
  }
  const yearReturns = Object.entries(yearBuckets)
    .map(([yr, { start, end }]) => ({ period: yr, return: end / start - 1 }))
    .filter(x => isFinite(x.return))

  const bestYear  = yearReturns.length ? yearReturns.reduce((a, b) => a.return > b.return ? a : b) : null
  const worstYear = yearReturns.length ? yearReturns.reduce((a, b) => a.return < b.return ? a : b) : null

  const monthBuckets = {}
  for (let i = 1; i < n; i++) {
    const ym = portCurve[i].date.slice(0, 7)
    if (!monthBuckets[ym]) monthBuckets[ym] = { start: portCurve[i - 1].value }
    monthBuckets[ym].end = portCurve[i].value
  }
  const monthReturns = Object.entries(monthBuckets)
    .map(([ym, { start, end }]) => ({ period: ym, return: end / start - 1 }))
    .filter(x => isFinite(x.return))

  const bestMonth  = monthReturns.length ? monthReturns.reduce((a, b) => a.return > b.return ? a : b) : null
  const worstMonth = monthReturns.length ? monthReturns.reduce((a, b) => a.return < b.return ? a : b) : null

  return {
    total_return: totalReturn,
    cagr,
    volatility,
    sharpe,
    sortino,
    calmar,
    max_drawdown: maxDD,
    max_drawdown_duration_days: maxDDDays,
    downside_deviation: downsideDeviation,
    var_95:  var95,
    cvar_95: cvar95,
    beta,
    alpha,
    r_squared: rSquared,
    tracking_error:    trackingError,
    information_ratio: informationRatio,
    treynor,
    up_capture:   upCapture,
    down_capture: downCapture,
    skewness,
    excess_kurtosis: excessKurtosis,
    best_year:   bestYear,
    worst_year:  worstYear,
    best_month:  bestMonth,
    worst_month: worstMonth,
    deflated_sharpe: null,   // requires trial count — not computable client-side
  }
}

/**
 * Recompute drawdown series from a portfolio curve (fractional values, not %).
 * DrawdownChart multiplies by 100 internally.
 */
export function computeDrawdownFromCurve(portCurve) {
  if (!portCurve?.length) return []
  let peak = -Infinity
  return portCurve.map(pt => {
    if (pt.value > peak) peak = pt.value
    return { date: pt.date, drawdown: peak > 0 ? (pt.value - peak) / peak : 0 }
  })
}
