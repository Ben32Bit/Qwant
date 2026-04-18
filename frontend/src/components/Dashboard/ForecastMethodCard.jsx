import { useState, useRef, useMemo } from 'react'
import { createPortal } from 'react-dom'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE } from '../../utils/chartConfig.js'

// ── Paper citations — matched to FamaFrenchFactors.jsx format ─────────────────

const CITATIONS = {
  xgboost: [
    'Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD \'16, 785–794. https://doi.org/10.1145/2939672.2939785',
    'Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. Review of Financial Studies, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009',
    'Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451',
  ],
  nbeats: [
    'Oreshkin, B., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. ICLR 2020. https://arxiv.org/abs/1905.10437',
    'Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1), 33–50. https://doi.org/10.2307/1913643',
  ],
  hmm: [
    'Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384. https://doi.org/10.2307/1912559',
    'Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. Review of Financial Studies, 15(4), 1137–1187. https://doi.org/10.1093/rfs/15.4.1137',
  ],
  factor: [
    'Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010',
    'Carhart, M.M. (1997). On persistence in mutual fund performance. Journal of Finance, 52(1), 57–82. https://doi.org/10.1111/j.1540-6261.1997.tb03808.x',
    'Novy-Marx, R. (2013). The other side of value: The gross profitability premium. Journal of Financial Economics, 108(1), 1–28. https://doi.org/10.1016/j.jfineco.2013.01.003',
    'Seyhun, H.N. (1986). Insiders\' profits, costs of trading, and market efficiency. Journal of Financial Economics, 16(2), 189–212. https://doi.org/10.1016/0304-405X(86)90060-7',
    'Lakonishok, J. & Lee, I. (2001). Are insider trades informative? Review of Financial Studies, 14(1), 79–111. https://doi.org/10.1093/rfs/14.1.79',
  ],
  var: [
    'Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT Press. https://gaussianprocess.org/gpml/',
    'Roberts, S. et al. (2013). Gaussian Processes for time-series modelling. Phil. Trans. R. Soc. A, 371(1984). https://doi.org/10.1098/rsta.2011.0550',
    'Matérn, B. (1960). Spatial Variation. Meddelanden från Statens Skogsforskningsinstitut, 49(5). [ν=5/2 kernel]',
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
  xgboost:     { label: 'MED',  color: 'var(--accent-yellow)' },
  nbeats:      { label: 'HIGH', color: 'var(--accent-red)' },
  hmm:         { label: 'MED',  color: 'var(--accent-yellow)' },
  factor:      { label: 'LOW',  color: 'var(--accent-green)' },
  var:         { label: 'HIGH', color: 'var(--accent-red)' },
  lstm:        { label: 'HIGH', color: 'var(--accent-red)' },
}

const METHOD_DESC = {
  xgboost:     'Histogram gradient boosting quantile regressors (sklearn HistGBR) — 5 models trained per quantile (p5/p25/p50/p75/p95) on 14 features: 9 market microstructure signals (momentum, realized vol, vol regime, RSI) + 5 live macro signals (10Y-2Y yield curve, BAA credit spread, TIPS real yield, VIX percentile rank, VIX term slope). Purged walk-forward CV with 21-day embargo + early stopping (López de Prado 2018). Runs in browser via ONNX Runtime Web; fan width scales by √(t/21) for the 252-day horizon. Rank IC ≥ 0.03 is considered useful in financial ML practice (Gu, Kelly & Xiu 2020).',
  nbeats:      'N-BEATS (Neural Basis Expansion Analysis): 3 residual stacks of fully-connected blocks. Each block explains part of the input (backcast) and contributes a partial multi-horizon forecast. Replaces GARCH which only forecasts conditional variance. N-BEATS directly predicts 21-day return sequences at 5 quantiles (pinball loss) and runs 12 recursive 21-day periods for the 252-day chart. Runs in browser via pure-JS matrix math (no runtime dependency).',
  hmm:         '2-state Hidden Markov Model (Bull / Bear). Transition probabilities and regime-conditional return distributions estimated via Baum-Welch EM with 10 random initialisations to escape local optima.',
  factor:      'Factor-anchored GBM: expected return derived from FF5 + Momentum (UMD, Carhart 1997) loadings × consensus long-run premia (Damodaran 2024), replacing naive historical mean. RMW serves as the quality/profitability factor (Novy-Marx 2013). Momentum beta is automatically included when Ken French daily data is available. Idiosyncratic vol (σ_ε = total vol × √(1−R²)) is used as the noise term — avoids double-counting systematic risk. Reduces look-ahead bias from short backtests.',
  var:         'Gaussian Process autoregression (Phase 2D): replaces VAR with a Bayesian nonparametric model. ARD Matérn 5/2 kernel learns which of the last 7 daily returns are most predictive (one length scale per lag). WhiteKernel absorbs observation noise. GP posterior gives exact Bayesian uncertainty — no Monte Carlo needed. Fan chart uses Gaussian approximation: cumulative return at T ~ N(Σμₜ, Σσₜ²). No stationarity requirement; complexity tuned by marginal likelihood. OOS NLPD measures calibration quality.',
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
    case 'xgboost': {
      const mc = meta.macro_context ?? {}
      const yieldCurve = mc.yield_curve_10y2y != null ? `${mc.yield_curve_10y2y.toFixed(2)}%` : '—'
      const yieldRegime = mc.yield_curve_10y2y != null
        ? (mc.yield_curve_10y2y < 0 ? 'inverted' : mc.yield_curve_10y2y < 0.5 ? 'flat' : 'normal')
        : null
      // rank_ic (Spearman) is more meaningful than OOS R² for financial return prediction
      const icVal = meta.rank_ic != null ? fmt(meta.rank_ic, 3) : (meta.oos_r2 != null ? fmt(meta.oos_r2, 3) : '—')
      const icLabel = meta.rank_ic != null ? 'rank IC' : 'OOS R²'
      const items = [
        { label: icLabel,    value: icVal },
        { label: 'vol rgm',  value: meta.vol_regime ?? '—', warn: meta.vol_regime === 'high' },
        { label: 'RSI-14',   value: meta.rsi_14 != null ? fmt(meta.rsi_14, 2) : '—' },
        { label: 'n obs',    value: meta.n_obs },
      ]
      if (mc.vix_spot != null)
        items.push({ label: 'VIX',    value: mc.vix_spot.toFixed(1), warn: mc.vix_pct_rank > 0.75 })
      if (mc.yield_curve_10y2y != null)
        items.push({ label: '10Y-2Y', value: yieldCurve, warn: yieldRegime === 'inverted' })
      if (!mc.macro_available)
        items.push({ label: 'macro',  value: 'neutral', warn: true })
      const ins = meta.insider_context
      if (ins?.available) {
        const netBuy = ins.portfolio_net_buying_30d
        items.push({
          label: 'insiders',
          value: `${netBuy >= 0 ? '+' : ''}$${Math.abs(netBuy).toFixed(1)}M`,
          warn: netBuy < -5,
        })
      }
      return items
    }
    case 'nbeats': {
      const mc = meta.macro_context ?? {}
      const items = [
        { label: 'periods',   value: meta.periods ?? 12 },
        { label: 'vol 21d',   value: fmtPct(meta.vol_21d_ann) },
        { label: 'ret 21d',   value: meta.ret_21d != null ? `${(meta.ret_21d * 100).toFixed(1)}%` : '—' },
        { label: 'n obs',     value: meta.n_obs },
      ]
      if (mc.vix_pct_rank != null)
        items.push({ label: 'VIX rank',  value: `${(mc.vix_pct_rank * 100).toFixed(0)}%`, warn: mc.vix_pct_rank > 0.75 })
      if (mc.yield_curve_10y2y != null)
        items.push({ label: '10Y-2Y',    value: `${mc.yield_curve_10y2y?.toFixed(2)}%`, warn: mc.yield_curve_10y2y < 0 })
      return items
    }
    case 'hmm': {
      const items = [
        { label: 'P(bull)',   value: `${(meta.current_bull_prob * 100).toFixed(0)}%` },
        { label: 'state',     value: meta.current_state, warn: meta.current_state === 'bear' },
        { label: 'bull μ',    value: fmtPct(meta.bull_state_mu_ann) },
        { label: 'bear μ',    value: fmtPct(meta.bear_state_mu_ann) },
      ]
      if (meta.macro_env)
        items.push({ label: 'macro',    value: meta.macro_env, warn: meta.macro_env === 'restrictive' })
      if (meta.yield_curve_regime)
        items.push({ label: 'YC rgm',  value: meta.yield_curve_regime, warn: meta.yield_curve_regime === 'inverted' })
      if (meta.vix_regime)
        items.push({ label: 'VIX rgm', value: meta.vix_regime, warn: meta.vix_regime === 'elevated' })
      return items
    }
    case 'factor': {
      const factorLabel = meta.source === 'historical_fallback' ? 'hist'
        : meta.source === 'ff6' ? 'FF5+Mom'
        : 'FF5'
      const items = [
        { label: 'μ factor', value: fmtPct(meta.mu_factor_ann) },
        { label: 'μ hist',   value: fmtPct(meta.mu_hist_ann) },
        { label: 'R²',       value: fmt(meta.r_squared, 2) },
        { label: 'model',    value: factorLabel, warn: meta.source === 'historical_fallback' },
      ]
      if (meta.mom_beta != null)
        items.push({ label: 'Mom β',     value: fmt(meta.mom_beta, 2) })
      if (meta.rmw_beta != null)
        items.push({ label: 'Quality β', value: fmt(meta.rmw_beta, 2) })
      if (meta.macro_adjustment) {
        const adj = meta.macro_adjustment
        items.push({ label: 'mkt scale', value: `${(adj.cycle_scale * 100).toFixed(0)}%`, warn: adj.cycle_scale < 0.85 })
        items.push({ label: '10Y-2Y',    value: `${adj.yield_curve_10y2y?.toFixed(2)}%`,  warn: adj.yield_curve_10y2y < 0 })
      }
      const ins = meta.insider_context
      if (ins?.available) {
        const netBuy = ins.portfolio_net_buying_30d
        items.push({
          label: 'insiders 30d',
          value: `${netBuy >= 0 ? '+' : ''}$${Math.abs(netBuy).toFixed(1)}M`,
          warn:  netBuy < -5,
        })
        items.push({
          label: 'buy ratio',
          value: `${(ins.portfolio_buy_ratio_30d * 100).toFixed(0)}%`,
          warn:  ins.portfolio_buy_ratio_30d < 0.3,
        })
      }
      return items
    }
    case 'var': {
      const items = [
        { label: 'OOS R²',   value: meta.oos_r2 != null ? fmt(meta.oos_r2, 3) : '—' },
        { label: 'NLPD',     value: meta.nlpd    != null ? fmt(meta.nlpd, 3)   : '—' },
        { label: 'lookback', value: `${meta.lookback ?? 7}d` },
        { label: 'kernel',   value: 'Matérn ν=5/2' },
      ]
      if (meta.vix_pct_rank != null)
        items.push({ label: 'VIX rank',  value: `${(meta.vix_pct_rank * 100).toFixed(0)}%`, warn: meta.vix_pct_rank > 0.80 })
      if (meta.vix_term_slope != null)
        items.push({ label: 'VIX slope', value: fmt(meta.vix_term_slope, 2), warn: meta.vix_term_slope < 0 })
      if (meta.yield_curve_10y2y != null)
        items.push({ label: '10Y-2Y',    value: `${meta.yield_curve_10y2y?.toFixed(2)}%`, warn: meta.yield_curve_10y2y < 0 })
      return items
    }
    case 'lstm':
      return [
        { label: 'engine',    value: meta.client_side === false ? 'TF.js browser' : 'browser' },
        { label: 'passes',    value: meta.dropout_passes ?? 200 },
        { label: 'attention', value: meta.attention ? 'Bahdanau' : '—' },
      ]
    default:
      return []
  }
}

// ── Main card ─────────────────────────────────────────────────────────────────

export default function ForecastMethodCard({ result, loading, browserCompute, lastValue }) {
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
      <div className="rounded-lg border p-4 flex flex-col" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', minHeight: 220 }}>
        {browserCompute ? (
          <>
            <div className="flex items-center gap-2 mb-3">
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: color ?? '#4a9eff', display: 'inline-block' }} />
              <span className="mono text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{label}</span>
            </div>
            <div className="flex-1 flex flex-col items-center justify-center text-center">
              <div className="mono text-xs mb-1" style={{ color: color ?? 'var(--accent-blue)' }}>⟳ Computing in browser</div>
              <div className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
                {method === 'xgboost'
                  ? '5 quantile models · ONNX Runtime Web'
                  : method === 'nbeats'
                  ? '12-period recursive · pure-JS weights'
                  : '200 MC Dropout passes · TensorFlow.js'
                }
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="skeleton h-3 w-32 mb-3" />
            <div className="skeleton h-40 w-full rounded" />
            <div className="skeleton h-3 w-48 mt-3" />
          </>
        )}
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
