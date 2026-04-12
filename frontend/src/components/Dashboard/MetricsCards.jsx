import React from 'react'
import { fmtPct, fmtRatio, colorClass } from '../../utils/formatters.js'

// ── Compact row ───────────────────────────────────────────────────────────────
function Row({ label, value, valueClass, sub }) {
  return (
    <div className="flex items-center justify-between py-1.5"
      style={{ borderBottom: '1px solid var(--border)' }}>
      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <div className="text-right">
        <span className={`mono text-sm font-semibold ${valueClass || ''}`}>{value}</span>
        {sub && <div className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{sub}</div>}
      </div>
    </div>
  )
}

function SectionTitle({ children }) {
  return (
    <div className="mono text-xs font-bold uppercase tracking-widest pt-3 pb-1.5"
      style={{ color: 'var(--accent-blue)', borderBottom: '1px solid var(--border)' }}>
      {children}
    </div>
  )
}

function SkeletonRows({ n }) {
  return Array(n).fill(0).map((_, i) => (
    <div key={i} className="flex justify-between items-center py-1.5"
      style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="skeleton h-3 w-24" />
      <div className="skeleton h-3 w-16" />
    </div>
  ))
}

// ── Main ─────────────────────────────────────────────────────────────────────
export default function MetricsCards({ metrics, loading }) {
  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div className="grid grid-cols-2 gap-x-6">
          <div><SkeletonRows n={6} /></div>
          <div><SkeletonRows n={6} /></div>
        </div>
      </div>
    )
  }

  if (!metrics) return null
  const m = metrics
  const hasBenchmark = m.beta != null

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <div className="grid grid-cols-2 gap-x-6">

        {/* Left column: Returns + Risk */}
        <div>
          <SectionTitle>Returns</SectionTitle>
          <Row label="Total Return"  value={fmtPct(m.total_return)}       valueClass={colorClass(m.total_return)} />
          <Row label="CAGR"          value={fmtPct(m.cagr)}               valueClass={colorClass(m.cagr)} />
          <Row label="Best Year"     value={m.best_year  ? fmtPct(m.best_year.return)  : '—'} sub={m.best_year?.period?.slice(0,4)}  valueClass="positive" />
          <Row label="Worst Year"    value={m.worst_year ? fmtPct(m.worst_year.return) : '—'} sub={m.worst_year?.period?.slice(0,4)} valueClass="negative" />
          <Row label="Best Month"    value={m.best_month  ? fmtPct(m.best_month.return)  : '—'} valueClass="positive" />
          <Row label="Worst Month"   value={m.worst_month ? fmtPct(m.worst_month.return) : '—'} valueClass="negative" />

          <SectionTitle>Risk</SectionTitle>
          <Row label="Volatility"      value={fmtPct(m.volatility)} />
          <Row label="Max Drawdown"    value={fmtPct(m.max_drawdown)} sub={`${m.max_drawdown_duration_days}d`} valueClass="negative" />
          <Row label="Downside Dev"    value={fmtPct(m.downside_deviation)} />
          <Row label="VaR 95%"         value={fmtPct(m.var_95)}  valueClass="negative" />
          <Row label="CVaR 95%"        value={fmtPct(m.cvar_95)} valueClass="negative" />
        </div>

        {/* Right column: Risk-Adjusted + Benchmark */}
        <div>
          <SectionTitle>Risk-Adjusted</SectionTitle>
          <Row label="Sharpe"   value={fmtRatio(m.sharpe)}  valueClass={colorClass(m.sharpe)} />
          <Row label="Sortino"  value={fmtRatio(m.sortino)} valueClass={colorClass(m.sortino)} />
          <Row label="Calmar"   value={fmtRatio(m.calmar)}  valueClass={colorClass(m.calmar)} />

          {hasBenchmark && (
            <>
              <SectionTitle>vs Benchmark</SectionTitle>
              <Row label="Beta"           value={fmtRatio(m.beta)} />
              <Row label="Alpha"          value={fmtPct(m.alpha)}             valueClass={colorClass(m.alpha)} />
              <Row label="R²"             value={fmtRatio(m.r_squared)} />
              <Row label="Info Ratio"     value={fmtRatio(m.information_ratio)} valueClass={colorClass(m.information_ratio)} />
              <Row label="Tracking Error" value={fmtPct(m.tracking_error)} />
              <Row label="Treynor"        value={fmtRatio(m.treynor)}          valueClass={colorClass(m.treynor)} />
              <Row label="Up Capture"     value={fmtPct(m.up_capture)}   valueClass="positive" />
              <Row label="Down Capture"   value={fmtPct(m.down_capture)} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
