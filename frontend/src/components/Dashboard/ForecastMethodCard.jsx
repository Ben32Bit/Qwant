import React, { useState, useRef, useMemo } from 'react'
import { createPortal } from 'react-dom'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE } from '../../utils/chartConfig.js'

// ── Paper citations — matched to FamaFrenchFactors.jsx format ─────────────────

const CITATIONS = {
  monte_carlo: [
    'Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637–654. https://doi.org/10.1086/260062',
    'Merton, R.C. (1969). Lifetime portfolio selection under uncertainty: the continuous-time case. Review of Economics and Statistics, 51(3), 247–257. https://doi.org/10.2307/1926560',
  ],
  garch: [
    'Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987–1007. https://doi.org/10.2307/1912773',
    'Bollerslev, T. (1986). Generalized autoregressive conditional heteroscedasticity. Journal of Econometrics, 31(3), 307–327. https://doi.org/10.1016/0304-4076(86)90063-1',
  ],
  hmm: [
    'Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384. https://doi.org/10.2307/1912559',
    'Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. Review of Financial Studies, 15(4), 1137–1187. https://doi.org/10.1093/rfs/15.4.1137',
  ],
  factor: [
    'Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010',
    'Cochrane, J.H. (2011). Presidential address: Discount rates. Journal of Finance, 66(4), 1047–1108. https://doi.org/10.1111/j.1540-6261.2011.01671.x',
  ],
  var: [
    'Sims, C.A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1–48. https://doi.org/10.2307/1912017',
    'Campbell, J.Y., Chan, Y.L., & Viceira, L.M. (2003). A multivariate model of strategic asset allocation. Journal of Financial Economics, 67(1), 41–80. https://doi.org/10.1016/S0304-405X(02)00231-3',
  ],
  lstm: [
    'CS230 Stanford (2020). Predicting Stock Market Returns Using Temporal Attention-Enhanced LSTM. Winter 2020 Project Reports. https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf',
    'Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. ICLR 2015. https://arxiv.org/abs/1409.0473',
    'Fischer, T. & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654–669. https://doi.org/10.1016/j.ejor.2017.11.054',
    'Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. Proceedings of ICML 33, 1050–1059. https://proceedings.mlr.press/v48/gal16.html',
  ],
}

// Out-of-sample methodology citation — shown on every card
const OOS_CITATION = 'Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7 (Purged Walk-Forward CV). https://doi.org/10.1002/9781119482161'

const COMPLEXITY = {
  monte_carlo: { label: 'LOW',  color: 'var(--accent-green)' },
  garch:       { label: 'MED',  color: 'var(--accent-yellow)' },
  hmm:         { label: 'MED',  color: 'var(--accent-yellow)' },
  factor:      { label: 'LOW',  color: 'var(--accent-green)' },
  var:         { label: 'MED',  color: 'var(--accent-yellow)' },
  lstm:        { label: 'HIGH', color: 'var(--accent-red)' },
}

const METHOD_DESC = {
  monte_carlo: 'Geometric Brownian Motion — constant drift μ and volatility σ estimated from history. Simplest parametric model; GBM is the foundation of options pricing.',
  garch:       'GARCH(1,1) captures volatility clustering — periods of high volatility beget more high volatility. Parameters fit by MLE on 80% train window; residuals validated on 20% held-out set.',
  hmm:         '2-state Hidden Markov Model (Bull / Bear). Transition probabilities and regime-conditional return distributions estimated via Baum-Welch EM with 10 random initialisations to escape local optima.',
  factor:      'Factor-anchored GBM: expected return derived from Fama-French 5-factor loadings × consensus long-run premia (Damodaran 2024), replacing naive historical mean. Reduces look-ahead bias from short backtests.',
  var:         'Vector Autoregression captures lead-lag cross-asset relationships. Lag order selected by AIC; out-of-sample residual covariance used for simulation to avoid covariance inflation.',
  lstm:        'Attention-LSTM: single LSTM(64) encoder → Bahdanau temporal attention → Dense(32) → Dense(1). Attention lets the model selectively weight past hidden states, focusing on regime-relevant windows. MC Dropout (200 passes) produces Bayesian uncertainty bands. Chronological 70/15/15 split; early stopping on val loss.',
}

// ── InfoTooltip — mirrors FamaFrenchFactors.jsx pattern ──────────────────────

const TOOLTIP_WIDTH = 320

function InfoTooltip({ content, citations }) {
  const [show, setShow] = useState(false)
  const [pos, setPos]   = useState({ top: 0, left: 0 })
  const btnRef          = useRef(null)

  const handleShow = () => {
    if (btnRef.current) {
      const rect      = btnRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - rect.bottom
      const top        = spaceBelow > 200 ? rect.bottom + 6 : rect.top - 220
      const left       = Math.min(rect.left, window.innerWidth - TOOLTIP_WIDTH - 12)
      setPos({ top, left })
    }
    setShow(true)
  }

  return (
    <span className="inline-block ml-1 align-middle">
      <button
        ref={btnRef}
        className="mono rounded-full flex items-center justify-center"
        style={{
          width: 15, height: 15,
          background: 'rgba(74,158,255,0.15)',
          color: 'var(--accent-blue)',
          border: '1px solid rgba(74,158,255,0.3)',
          cursor: 'pointer', fontSize: 9, lineHeight: 1, flexShrink: 0,
        }}
        onMouseEnter={handleShow}
        onMouseLeave={() => setShow(false)}
        aria-label="Method description and citations"
      >
        ?
      </button>
      {show && createPortal(
        <div
          className="rounded-lg border p-3 text-xs"
          style={{
            position: 'fixed', top: pos.top, left: pos.left,
            width: TOOLTIP_WIDTH, zIndex: 9999,
            background: 'var(--bg-secondary)', borderColor: 'var(--border)',
            color: 'var(--text-primary)', boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
            pointerEvents: 'none',
          }}
        >
          <p className="mb-2 leading-relaxed" style={{ color: 'var(--text-primary)' }}>
            {content}
          </p>
          {citations.map((c, i) => (
            <p key={i} className="mb-1 leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
              📄 {c}
            </p>
          ))}
          <p className="mt-1 leading-relaxed" style={{ color: 'rgba(136,136,160,0.7)', fontSize: 10, borderTop: '1px solid var(--border)', paddingTop: 4 }}>
            🔬 OOS: {OOS_CITATION}
          </p>
        </div>,
        document.body
      )}
    </span>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDollar(v) {
  if (v == null) return '—'
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`
  return `$${v.toFixed(0)}`
}

// ── Fan chart tooltip ─────────────────────────────────────────────────────────

function FanTooltip({ active, payload, label, color }) {
  if (!active || !payload?.length) return null
  const p50 = payload.find(p => p.dataKey === 'p50')?.value
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '8px 12px' }}>
      <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      {p50 != null && (
        <div className="mono text-sm" style={{ color }}>
          median: {fmtDollar(p50)}
        </div>
      )}
    </div>
  )
}

// ── Build Recharts-compatible data from forecast band ─────────────────────────
// Converts cumulative-% forecast bands to actual portfolio dollar values.
// Uses stackId trick: transparent base area + coloured range area.

function buildChartData(band, lastValue) {
  if (!band?.dates || !lastValue) return []
  const toVal = pct => lastValue * (1 + pct / 100)
  return band.dates.map((date, i) => {
    const p5  = toVal(band.p5[i])
    const p95 = toVal(band.p95[i])
    const p25 = toVal(band.p25[i])
    const p75 = toVal(band.p75[i])
    return {
      date,
      // Outer band (p5 → p95)
      outer_base:   p5,
      outer_height: p95 - p5,
      // Inner band (p25 → p75)
      inner_base:   p25,
      inner_height: p75 - p25,
      p50:          toVal(band.p50[i]),
    }
  })
}

// ── Metadata strip ────────────────────────────────────────────────────────────

function MetaStrip({ method, meta, color }) {
  const items = getMetaItems(method, meta)
  if (!items.length) return null
  return (
    <div className="flex flex-wrap gap-3 mt-2">
      {items.map(({ label, value, warn }) => (
        <div key={label}>
          <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{label} </span>
          <span className="mono text-xs font-bold"
            style={{ color: warn ? 'var(--accent-yellow)' : color }}>
            {value}
          </span>
        </div>
      ))}
    </div>
  )
}

function fmt(v, decimals = 2) {
  if (v == null || isNaN(v)) return '—'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}
function fmtPct(v) { return v != null ? `${(v * 100).toFixed(1)}%` : '—' }

function getMetaItems(method, meta) {
  if (!meta) return []
  switch (method) {
    case 'monte_carlo':
      return [
        { label: 'μ (ann)', value: fmtPct(meta.mu_ann) },
        { label: 'σ (ann)', value: fmtPct(meta.sigma_ann) },
        { label: 'n obs',   value: meta.n_obs },
      ]
    case 'garch':
      return [
        { label: 'α+β',       value: fmt(meta.persistence, 3), warn: meta.persistence > 0.98 },
        { label: 'cur vol',   value: fmtPct(meta.current_vol_ann) },
        { label: 'LR vol',    value: fmtPct(meta.longrun_vol_ann) },
        { label: 'LB ok',     value: meta.oos_ljungbox_ok ? '✓' : '✗', warn: !meta.oos_ljungbox_ok },
      ]
    case 'hmm':
      return [
        { label: 'P(bull)',   value: `${(meta.current_bull_prob * 100).toFixed(0)}%` },
        { label: 'state',     value: meta.current_state, warn: meta.current_state === 'bear' },
        { label: 'bull μ',    value: fmtPct(meta.bull_state_mu_ann) },
        { label: 'bear μ',    value: fmtPct(meta.bear_state_mu_ann) },
      ]
    case 'factor':
      return [
        { label: 'μ factor', value: fmtPct(meta.mu_factor_ann) },
        { label: 'μ hist',   value: fmtPct(meta.mu_hist_ann) },
        { label: 'R²',       value: fmt(meta.r_squared, 2) },
        { label: 'source',   value: meta.source === 'historical_fallback' ? 'fallback' : 'FF5', warn: meta.source === 'historical_fallback' },
      ]
    case 'var':
      return [
        { label: 'lag p',    value: meta.lag_order },
        { label: 'assets',   value: meta.n_assets },
        { label: 'Granger',  value: meta.granger_significant ? 'sig' : 'n.s.', warn: !meta.granger_significant },
        { label: 'OOS R²',   value: meta.oos_r2 != null ? fmt(meta.oos_r2, 3) : '—' },
      ]
    case 'lstm':
      return [
        { label: 'params',   value: meta.n_params?.toLocaleString() },
        { label: 'epochs',   value: meta.epochs_trained },
        { label: 'val loss', value: fmt(meta.val_loss, 5) },
        { label: 'OOS MSE',  value: fmt(meta.oos_mse, 5) },
      ]
    default:
      return []
  }
}

// ── Main card ─────────────────────────────────────────────────────────────────

export default function ForecastMethodCard({ result, loading, lastValue }) {
  const { method, label, color, forecast, metadata, error, compute_ms } = result ?? {}
  const complexity = COMPLEXITY[method] ?? { label: '—', color: 'var(--text-secondary)' }
  const citations  = CITATIONS[method] ?? []
  const desc       = METHOD_DESC[method] ?? ''

  const chartData = useMemo(() => buildChartData(forecast, lastValue), [forecast, lastValue])

  const tickCount = 4
  const tickDates = chartData.length
    ? Array.from({ length: tickCount }, (_, i) =>
        chartData[Math.floor((i / (tickCount - 1)) * (chartData.length - 1))]?.date
      ).filter(Boolean)
    : []

  if (loading && !forecast) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div className="skeleton h-3 w-32 mb-3" />
        <div className="skeleton h-40 w-full rounded" />
        <div className="skeleton h-3 w-48 mt-3" />
      </div>
    )
  }

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>

      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
          <span className="mono font-bold text-xs" style={{ color: 'var(--text-primary)' }}>
            {label}
          </span>
          <InfoTooltip content={desc} citations={citations} />
        </div>
        <div className="flex items-center gap-2">
          <span className="mono text-xs px-1.5 py-0.5 rounded"
            style={{ background: 'rgba(0,0,0,0.3)', color: complexity.color, border: `1px solid ${complexity.color}`, opacity: 0.85, fontSize: 9 }}>
            {complexity.label}
          </span>
          {compute_ms > 0 && (
            <span className="mono text-xs" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
              {compute_ms < 1000 ? `${compute_ms}ms` : `${(compute_ms / 1000).toFixed(1)}s`}
            </span>
          )}
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="rounded px-3 py-2 text-xs mono mt-2"
          style={{ background: 'rgba(255,71,87,0.1)', border: '1px solid rgba(255,71,87,0.3)', color: 'var(--accent-red)' }}>
          {error}
        </div>
      )}

      {/* Fan chart */}
      {forecast && chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={180}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 2 }}>
            <CartesianGrid {...GRID_STYLE} />
            <XAxis
              dataKey="date"
              ticks={tickDates}
              tick={{ ...AXIS_STYLE.tick, fontSize: 9 }}
              axisLine={AXIS_STYLE.axisLine}
              tickLine={AXIS_STYLE.tickLine}
              tickFormatter={d => d?.slice(0, 7)}
            />
            <YAxis
              tick={{ ...AXIS_STYLE.tick, fontSize: 9 }}
              axisLine={AXIS_STYLE.axisLine}
              tickLine={AXIS_STYLE.tickLine}
              tickFormatter={fmtDollar}
              width={50}
            />
            <ReferenceLine y={0} stroke="var(--border)" strokeWidth={1} />
            <Tooltip content={<FanTooltip color={color} />} />

            {/* Outer band p5→p95 */}
            <Area type="monotone" dataKey="outer_base"   stackId="outer" stroke="none" fill="transparent" isAnimationActive={false} legendType="none" />
            <Area type="monotone" dataKey="outer_height" stackId="outer" stroke="none" fill={color} fillOpacity={0.10} isAnimationActive={false} legendType="none" />

            {/* Inner band p25→p75 */}
            <Area type="monotone" dataKey="inner_base"   stackId="inner" stroke="none" fill="transparent" isAnimationActive={false} legendType="none" />
            <Area type="monotone" dataKey="inner_height" stackId="inner" stroke="none" fill={color} fillOpacity={0.22} isAnimationActive={false} legendType="none" />

            {/* Median line */}
            <Line type="monotone" dataKey="p50" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} name="median" />
          </ComposedChart>
        </ResponsiveContainer>
      )}

      {/* Metadata strip */}
      {forecast && <MetaStrip method={method} meta={metadata} color={color} />}

      {/* Citation footer — matches FamaFrenchFactors.jsx */}
      <div className="mt-3 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
        {citations.map((c, i) => (
          <p key={i} className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
            📄 {c}
          </p>
        ))}
      </div>
    </div>
  )
}
