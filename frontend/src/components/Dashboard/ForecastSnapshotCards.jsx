import React from 'react'

const METHOD_ORDER  = ['xgboost', 'nbeats', 'factor', 'hmm', 'var', 'lstm']
const METHOD_LABELS = { xgboost: 'XGB', nbeats: 'NBT', factor: 'FAC', hmm: 'HMM', var: 'GP', lstm: 'LSTM' }
const METHOD_COLORS = { xgboost: '#4a9eff', nbeats: '#ffd43b', factor: '#00d4aa', hmm: '#a855f7', var: '#ff6b35', lstm: '#ff4757' }

const SNAPSHOTS = [
  { label: '1 week',   days: 5,  idx: 4,  basis: 'all 6 models legitimate — within every training window' },
  { label: '1 month',  days: 21, idx: 20, basis: 'all 6 models legitimate — = XGBoost training horizon' },
  { label: '3 months', days: 63, idx: 62, basis: '5 models (XGBoost capped at 21d)' },
]

function fmtPct(v) {
  if (v == null || !Number.isFinite(v)) return '—'
  const sign = v >= 0 ? '+' : ''
  return `${sign}${v.toFixed(1)}%`
}

function fmtUsd(v) {
  if (v == null || !Number.isFinite(v)) return '—'
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`
  return `$${v.toFixed(0)}`
}

function colourForReturn(pct) {
  if (pct == null || !Number.isFinite(pct)) return 'var(--text-secondary)'
  if (pct > 0.5)  return 'var(--accent-green)'
  if (pct < -0.5) return 'var(--accent-red)'
  return 'var(--text-secondary)'
}

function MethodDot({ method, value, ensembleMedian }) {
  const hasValue = value != null && Number.isFinite(value)
  const color    = METHOD_COLORS[method]

  // Deviation from ensemble median — larger deviation = lower consensus
  const devPct = hasValue && ensembleMedian != null
    ? Math.abs(value - ensembleMedian)
    : null
  const opacity = devPct == null ? 0.25
                : devPct > 10    ? 0.45
                : devPct > 5     ? 0.70
                : 1.0

  return (
    <div className="flex flex-col items-center gap-0.5" title={hasValue ? `${METHOD_LABELS[method]}: ${fmtPct(value)}` : `${METHOD_LABELS[method]}: no forecast at this horizon`}>
      <span style={{
        display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
        background: hasValue ? color : 'transparent',
        border: hasValue ? 'none' : `1px dashed ${color}55`,
        opacity,
        boxShadow: hasValue ? `0 0 4px ${color}66` : 'none',
      }} />
      <span className="mono" style={{ fontSize: 7, color, opacity: opacity * 0.9 }}>
        {METHOD_LABELS[method]}
      </span>
      <span className="mono" style={{ fontSize: 8, color: hasValue ? color : 'var(--text-secondary)', opacity: hasValue ? 1 : 0.3, minHeight: 10 }}>
        {hasValue ? fmtPct(value) : '—'}
      </span>
    </div>
  )
}

function SnapshotCard({ label, days, idx, basis, results, ensemble, lastValue }) {
  const band = ensemble?.band
  const p5   = band?.p5?.[idx]
  const p50  = band?.p50?.[idx]
  const p95  = band?.p95?.[idx]
  const hasEnsemble = Number.isFinite(p50)

  const active = (results ?? []).filter(r => Number.isFinite(r?.forecast?.p50?.[idx])).length
  const total  = METHOD_ORDER.length

  const projValue = hasEnsemble && lastValue != null ? lastValue * (1 + p50 / 100) : null
  const valueLow  = Number.isFinite(p5)  && lastValue != null ? lastValue * (1 + p5  / 100) : null
  const valueHigh = Number.isFinite(p95) && lastValue != null ? lastValue * (1 + p95 / 100) : null

  const headlineColor = colourForReturn(p50)

  return (
    <div className="rounded-lg border p-3" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      {/* Header */}
      <div className="flex items-baseline justify-between mb-2">
        <div>
          <div className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>{label}</div>
          <div className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.7 }}>
            {days} trading days
          </div>
        </div>
        <div className="mono text-right" style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
          <span style={{ color: active === total ? 'var(--accent-green)' : active >= total - 1 ? '#ffd43b' : '#ff6b35' }}>
            n={active}
          </span>
          <span style={{ opacity: 0.5 }}>/{total} active</span>
        </div>
      </div>

      {/* Headline ensemble median */}
      <div className="mb-2">
        <div className="mono font-bold" style={{ fontSize: 22, color: headlineColor, lineHeight: 1.1 }}>
          {fmtPct(p50)}
        </div>
        {projValue != null && (
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', opacity: 0.75 }}>
            projected: {fmtUsd(projValue)}
          </div>
        )}
      </div>

      {/* p5 / p95 range */}
      {Number.isFinite(p5) && Number.isFinite(p95) && (
        <div className="mono mb-2" style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
          <span style={{ opacity: 0.6 }}>90% range:</span>{' '}
          <span style={{ color: colourForReturn(p5) }}>{fmtPct(p5)}</span>
          <span style={{ opacity: 0.4 }}> … </span>
          <span style={{ color: colourForReturn(p95) }}>{fmtPct(p95)}</span>
          {valueLow != null && valueHigh != null && (
            <span style={{ opacity: 0.5, marginLeft: 6 }}>
              ({fmtUsd(valueLow)} … {fmtUsd(valueHigh)})
            </span>
          )}
        </div>
      )}

      {/* Per-method dots */}
      <div className="grid grid-cols-6 gap-1 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
        {METHOD_ORDER.map(method => {
          const r = (results ?? []).find(x => x.method === method)
          const v = r?.forecast?.p50?.[idx]
          return <MethodDot key={method} method={method} value={v} ensembleMedian={p50} />
        })}
      </div>

      {/* Methodology footnote */}
      <div className="mono mt-2" style={{ fontSize: 8, color: 'var(--text-secondary)', opacity: 0.55 }}>
        {basis}
      </div>
    </div>
  )
}

/**
 * Three-card snapshot: 1w / 1m / 3m ensemble medians + range + per-method dots.
 *
 * Each horizon is chosen so the shortest-training model (XGBoost, 21d) is still
 * within (or at) its cap, keeping all figures academically defensible. The
 * continuous 3-month chart elsewhere is exploratory only.
 */
export default function ForecastSnapshotCards({ results, ensemble, lastValue, loading }) {
  const hasAny = (results ?? []).some(r => r?.forecast?.p50?.length)
  if (!hasAny && !loading) return null

  return (
    <div>
      <div className="mb-2">
        <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
          Forecast Snapshots
          <span className="ml-2 font-normal text-xs" style={{ color: 'var(--text-secondary)' }}>
            · horizon-honest · ensemble median + 90% range + per-model consensus
          </span>
        </h3>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {SNAPSHOTS.map(s => (
          <SnapshotCard
            key={s.days}
            label={s.label}
            days={s.days}
            idx={s.idx}
            basis={s.basis}
            results={results}
            ensemble={ensemble}
            lastValue={lastValue}
          />
        ))}
      </div>
    </div>
  )
}
