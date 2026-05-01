import { useState, useRef, useMemo, memo } from 'react'
import { createPortal } from 'react-dom'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE } from '../../utils/chartConfig.js'

// Methods whose shadow OOS R² falls below this threshold are excluded from the
// composite chart and shown with a warning instead of a fan chart.
const OOS_R2_MIN = -0.5

// ── Paper citations — matched to FamaFrenchFactors.jsx format ─────────────────

const CITATIONS = {
  nbeats: [
    'Oreshkin, B., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. ICLR 2020. https://arxiv.org/abs/1905.10437',
    'Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1), 33–50. https://doi.org/10.2307/1913643',
  ],
  hmm: [
    'Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384. https://doi.org/10.2307/1912559',
    'Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. Review of Financial Studies, 15(4), 1137–1187. https://doi.org/10.1093/rfs/15.4.1137',
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
  timesfm: [
    'Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2024). A decoder-only foundation model for time-series forecasting. ICML 2024. https://arxiv.org/abs/2310.10688',
    'Google Research (2024). TimesFM 2.5 — improved zero-shot time-series forecasting. HuggingFace Hub: google/timesfm-2.5-200m-pytorch.',
  ],
}

// Out-of-sample methodology citation — shown on every card
const OOS_CITATION = 'Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7 (Purged Walk-Forward CV). https://doi.org/10.1002/9781119482161'

const METHOD_DESC = {
  nbeats: 'N-BEATS (Neural Basis Expansion Analysis): 3 residual stacks of fully-connected blocks. Each block explains part of the input (backcast) and contributes a partial multi-horizon forecast. Replaces GARCH which only forecasts conditional variance. N-BEATS directly predicts 21-day return sequences at 5 quantiles (pinball loss) and runs 3 recursive 21-day periods to cover the 63-day chart natively. Runs in browser via pure-JS matrix math (no runtime dependency).',
  hmm:    '2-state Hidden Markov Model (Bull / Bear). Transition probabilities and regime-conditional return distributions estimated via Baum-Welch EM with 10 random initialisations to escape local optima.',
  var:    'Gaussian Process autoregression: replaces VAR with a Bayesian nonparametric model. ARD Matérn 5/2 kernel learns which of the last 7 daily returns are most predictive (one length scale per lag). WhiteKernel absorbs observation noise. GP posterior gives exact Bayesian uncertainty — no Monte Carlo needed. Fan chart uses Gaussian approximation: cumulative return at T ~ N(Σμₜ, Σσₜ²). No stationarity requirement; complexity tuned by marginal likelihood. OOS NLPD measures calibration quality.',
  lstm:    'Attention-LSTM: single LSTM(64) encoder → Bahdanau temporal attention → Dense(32) → Dense(1). Attention lets the model selectively weight past hidden states, focusing on regime-relevant windows. MC Dropout (200 passes) produces Bayesian uncertainty bands. Chronological 70/15/15 split; early stopping on val loss.',
  timesfm: 'TimesFM 2.5 (Google Research, ICML 2024): a 200M-parameter decoder-only foundation model pre-trained on a large corpus of real-world time series. Runs zero-shot — no fine-tuning on your portfolio data. The equity curve is converted to a normalised level series (last = 1.0), fed to the always-warm server singleton, and the 9-quantile output is interpolated to p5/p25/p50/p75/p95. The singleton is pre-loaded at server startup so it does not block your first forecast request.',
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

// ── Fan chart tooltip ─────────────────────────────────────────────────────────

function FanTooltip({ active, payload, label, color }) {
  if (!active || !payload?.length) return null
  const p50 = payload.find(p => p.dataKey === 'p50')?.value
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '8px 12px' }}>
      <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      {p50 != null && (
        <div className="mono text-sm" style={{ color }}>
          median: {p50 >= 0 ? '+' : ''}{p50.toFixed(2)}%
        </div>
      )}
    </div>
  )
}

// ── Build Recharts-compatible data from forecast band ─────────────────────────
// Uses the raw cumulative-% values directly (p50[i] = cumulative % from forecast
// start). stackId trick: transparent base area + coloured range area.

function buildChartData(band) {
  if (!band?.dates) return []
  return band.dates.map((date, i) => ({
    date,
    outer_base:   band.p5[i],
    outer_height: band.p95[i] - band.p5[i],
    inner_base:   band.p25[i],
    inner_height: band.p75[i] - band.p25[i],
    p50:          band.p50[i],
  }))
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
      if (meta.news_article_count != null)
        items.push({ label: 'articles', value: meta.news_article_count })
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
    case 'lstm': {
      const items = [
        { label: 'engine',    value: meta.client_side === false ? 'TF.js browser' : 'browser' },
        { label: 'passes',    value: meta.dropout_passes ?? 200 },
        { label: 'attention', value: meta.attention ? 'Bahdanau' : '—' },
      ]
      if (meta.news_article_count != null)
        items.push({ label: 'articles', value: meta.news_article_count })
      return items
    }
    case 'timesfm': {
      return [
        { label: 'model',     value: '200M params' },
        { label: 'inference', value: 'zero-shot' },
        { label: 'ctx len',   value: `${meta.context_len ?? 512}d` },
        { label: 'quantiles', value: `${meta.q_levels?.length ?? 9}` },
        { label: 'source',    value: 'Google (ICML 2024)' },
      ]
    }
    default:
      return []
  }
}

function r2Color(v) {
  if (v == null) return 'var(--text-secondary)'
  if (v >= 0.5)  return 'var(--accent-green)'
  if (v >= 0.0)  return '#ffd43b'
  return '#ff4757'
}

function R2Pair({ isR2, oosR2 }) {
  const fmt = v => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}` : '—'
  return (
    <div className="flex gap-4 mt-2 mb-1">
      {[['IS R²', isR2], ['OOS R²', oosR2]].map(([lbl, val]) => (
        <div key={lbl}>
          <div className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.65 }}>{lbl}</div>
          <div className="mono font-bold" style={{ fontSize: 13, color: r2Color(val), lineHeight: 1.2 }}>
            {fmt(val)}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Main card ─────────────────────────────────────────────────────────────────

function ForecastMethodCardImpl({ result, loading, browserCompute, lastValue }) {
  const [showDetails, setShowDetails] = useState(false)
  const { method, label, color, forecast, metadata, error, compute_ms, oos_r2 } = result ?? {}
  const citations = CITATIONS[method] ?? []
  const desc      = METHOD_DESC[method] ?? ''

  const chartData = useMemo(() => buildChartData(forecast), [forecast])

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
                {method === 'nbeats'
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
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
          <span className="mono font-bold text-xs" style={{ color: 'var(--text-primary)' }}>
            {label}
          </span>
          <InfoTooltip content={desc} citations={citations} />
        </div>
        {compute_ms > 0 && (
          <span className="mono" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
            {compute_ms < 1000 ? `${compute_ms}ms` : `${(compute_ms / 1000).toFixed(1)}s`}
          </span>
        )}
      </div>

      {/* IS R² / OOS R² — primary quality signal */}
      <R2Pair isR2={metadata?.is_r2 ?? null} oosR2={oos_r2 ?? null} />

      {/* Error state */}
      {error && (
        <div className="rounded px-3 py-2 text-xs mono mt-2"
          style={{ background: 'rgba(255,71,87,0.1)', border: '1px solid rgba(255,71,87,0.3)', color: 'var(--accent-red)' }}>
          {error}
        </div>
      )}

      {/* Low OOS R² warning — chart still rendered below */}
      {forecast && oos_r2 != null && oos_r2 < OOS_R2_MIN && (
        <div className="rounded px-3 py-2 mono mt-2"
          style={{ background: 'rgba(255,212,59,0.06)', border: '1px solid rgba(255,212,59,0.25)' }}>
          <div className="text-xs font-bold mb-1" style={{ color: '#ffd43b' }}>
            Low OOS R² — interpret with caution
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            This model scored OOS R² = {oos_r2 >= 0 ? '+' : ''}{oos_r2.toFixed(2)} on the shadow holdout window,
            meaning its errors were {Math.abs(oos_r2 * 100).toFixed(0)}% larger than simply predicting
            the historical mean. Excluded from the composite ensemble — shown here for transparency.
          </div>
        </div>
      )}

      {/* Fan chart — always shown when a forecast exists */}
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
              tickFormatter={v => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
              width={46}
            />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
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

      {/* Collapsible detail strip */}
      {forecast && (
        <div>
          <button
            onClick={() => setShowDetails(d => !d)}
            className="mono"
            style={{ fontSize: 9, color: 'var(--text-secondary)', background: 'none', border: 'none', padding: '4px 0', cursor: 'pointer', opacity: 0.6 }}
          >
            {showDetails ? '▾ details' : '▸ details'}
          </button>
          {showDetails && <MetaStrip method={method} meta={metadata} color={color} />}
        </div>
      )}

    </div>
  )
}

// Memoized so a parent re-render (e.g. ensemble recompute, tooltip state
// elsewhere) doesn't force four full chart re-layouts on every click.
export default memo(ForecastMethodCardImpl, (prev, next) =>
  prev.result === next.result &&
  prev.loading === next.loading &&
  prev.browserCompute === next.browserCompute &&
  prev.lastValue === next.lastValue
)
