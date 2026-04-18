/**
 * KellyCalculator — Portfolio-level Kelly position sizing from ensemble fan bands.
 *
 * Derivation:
 *   For a continuous log-normal return distribution with expected return μ and
 *   variance σ², the growth-optimal (Kelly) portfolio fraction is f* = μ / σ².
 *   We estimate μ from the ensemble p50 at the 1-year horizon and σ from the
 *   90% confidence interval: σ ≈ (p95 − p5) / (2 × 1.645).
 *
 *   Half-Kelly (f*/2) is the standard practical recommendation: it gives 75% of
 *   the long-run growth rate with far lower drawdown risk.
 *
 *   Regime adjustment: scales fraction by a confidence multiplier derived from the
 *   regime probability distribution — crisis regimes warrant smaller position sizes.
 *
 * References:
 *   Kelly, J.L. (1956). A New Interpretation of Information Rate.
 *     Bell System Technical Journal, 35(4), 917–926.
 *   MacLean, L.C., Thorp, E.O. & Ziemba, W.T. (2010). Good and Bad Properties of the
 *     Kelly Criterion. Probability in the Engineering and Informational Sciences.
 *   Thorp, E.O. (2006). The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market.
 *     Handbook of Asset and Liability Management, Vol. 1, Ch. 9.
 *   Ang & Timmermann (2012). Regime Changes and Financial Markets. Annual Review of Financial
 *     Economics, 4(1), 313–337. [regime confidence adjustments]
 */

const Z90 = 1.6449  // z-score for 90% CI (one-sided 1.6449 → two-sided 90%)

/**
 * Compute portfolio-level Kelly fraction from the ensemble forecast fan bands.
 *
 * @param {Object}  ensemble  - {dates, p5, p25, p50, p75, p95} cumulative % returns
 * @param {boolean} halfKelly - use half-Kelly (default: true)
 * @returns {{fraction, fullKelly, annualizedReturn, annualizedVol, level, clipped, halfKelly} | null}
 */
export function computeKellyFromEnsemble(ensemble, halfKelly = true) {
  if (!ensemble?.p50?.length) return null

  const T    = ensemble.p50.length
  const idx  = Math.min(T - 1, 251)   // 1-year horizon (trading day 252)

  const p50  = ensemble.p50[idx] / 100   // decimal cumulative return at 1y
  const p5   = ensemble.p5[idx]  / 100
  const p95  = ensemble.p95[idx] / 100

  // Annualized vol from the 90% CI (assumes log-normal, valid approximation for 1y)
  const sigma  = Math.max((p95 - p5) / (2 * Z90), 0.01)   // floor at 1% to avoid div/0
  const sigma2 = sigma * sigma

  const fullKelly = p50 / sigma2
  const raw       = halfKelly ? fullKelly / 2 : fullKelly
  const fraction  = Math.max(0, Math.min(raw, 2.0))  // clip to [0, 2×]

  const level =
    fraction < 0   ? 'avoid'
    : fraction < 0.5 ? 'underweight'
    : fraction < 0.8 ? 'cautious'
    : fraction < 1.1 ? 'neutral'
    : fraction < 1.5 ? 'overweight'
    : 'aggressive'

  return {
    fraction:         +fraction.toFixed(3),
    fullKelly:        +fullKelly.toFixed(3),
    annualizedReturn: +(p50 * 100).toFixed(2),
    annualizedVol:    +(sigma * 100).toFixed(2),
    level,
    clipped:          raw !== fraction,
    halfKelly,
  }
}

// Confidence multiplier per regime (Ang & Timmermann 2012 — uncertainty increases in crisis)
const REGIME_CONFIDENCE = {
  bull_low_vol:  1.00,
  bull_high_vol: 0.85,
  bear:          0.70,
  crisis:        0.50,
}

/**
 * Regime-adjust a Kelly fraction by blending regime confidence multipliers.
 *
 * @param {number} fraction     - raw Kelly fraction
 * @param {Object} regimeProbs  - {bull_low_vol, bull_high_vol, bear, crisis, dominant?}
 * @returns {number} adjusted fraction
 */
export function regimeAdjustKelly(fraction, regimeProbs) {
  if (!regimeProbs || fraction <= 0) return fraction
  let adj = 0, total = 0
  for (const [regime, p] of Object.entries(regimeProbs)) {
    if (regime === 'dominant') continue
    adj   += p * (REGIME_CONFIDENCE[regime] ?? 1.0)
    total += p
  }
  const multiplier = total > 0 ? adj / total : 1.0
  return +Math.max(0, fraction * multiplier).toFixed(3)
}

/**
 * Derive regime-adjusted Kelly from ensemble + regime probabilities.
 * Convenience wrapper combining computeKellyFromEnsemble + regimeAdjustKelly.
 *
 * @param {Object}  ensemble    - from MetaEnsemble.computeEnsemble
 * @param {Object}  regimeProbs - from ForecastResponse
 * @param {boolean} halfKelly
 * @returns {{raw, adjusted, annualizedReturn, annualizedVol, fullKelly, level, regimeMultiplier} | null}
 */
export function computeKelly(ensemble, regimeProbs, halfKelly = true) {
  const base = computeKellyFromEnsemble(ensemble, halfKelly)
  if (!base) return null

  const adjusted = regimeAdjustKelly(base.fraction, regimeProbs)

  let regimeMultiplier = 1.0
  if (regimeProbs) {
    let adj = 0, total = 0
    for (const [regime, p] of Object.entries(regimeProbs)) {
      if (regime === 'dominant') continue
      adj   += p * (REGIME_CONFIDENCE[regime] ?? 1.0)
      total += p
    }
    regimeMultiplier = total > 0 ? +(adj / total).toFixed(3) : 1.0
  }

  const adjLevel =
    adjusted < 0.5 ? 'underweight'
    : adjusted < 0.8 ? 'cautious'
    : adjusted < 1.1 ? 'neutral'
    : adjusted < 1.5 ? 'overweight'
    : 'aggressive'

  return {
    ...base,
    adjusted,
    adjLevel,
    regimeMultiplier,
  }
}
