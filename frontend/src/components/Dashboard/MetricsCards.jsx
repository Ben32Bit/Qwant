import React from 'react'
import { fmtPct, fmtRatio, fmtNum, colorClass, fmtDate } from '../../utils/formatters.js'

function MetricCard({ label, value, sub, valueClass }) {
  return (
    <div
      className="rounded-lg p-3 border"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
    >
      <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </div>
      <div className={`mono font-bold text-lg ${valueClass || ''}`}>
        {value}
      </div>
      {sub && (
        <div className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
          {sub}
        </div>
      )}
    </div>
  )
}

function SkeletonCard() {
  return (
    <div className="rounded-lg p-3 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <div className="skeleton h-3 w-20 mb-2" />
      <div className="skeleton h-6 w-24 mb-1" />
      <div className="skeleton h-2 w-16" />
    </div>
  )
}

export default function MetricsCards({ metrics, loading }) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 p-4">
        {Array(8).fill(0).map((_, i) => <SkeletonCard key={i} />)}
      </div>
    )
  }

  if (!metrics) return null

  const m = metrics

  return (
    <div className="p-4">
      {/* Return Metrics */}
      <h3 className="mono text-xs mb-3 font-bold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
        Returns
      </h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <MetricCard
          label="Total Return"
          value={fmtPct(m.total_return)}
          valueClass={colorClass(m.total_return)}
        />
        <MetricCard
          label="CAGR"
          value={fmtPct(m.cagr)}
          valueClass={colorClass(m.cagr)}
        />
        <MetricCard
          label="Best Year"
          value={m.best_year ? fmtPct(m.best_year.return) : '—'}
          sub={m.best_year?.period?.slice(0, 4)}
          valueClass="positive"
        />
        <MetricCard
          label="Worst Year"
          value={m.worst_year ? fmtPct(m.worst_year.return) : '—'}
          sub={m.worst_year?.period?.slice(0, 4)}
          valueClass="negative"
        />
      </div>

      {/* Risk Metrics */}
      <h3 className="mono text-xs mb-3 font-bold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
        Risk
      </h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <MetricCard
          label="Volatility"
          value={fmtPct(m.volatility)}
        />
        <MetricCard
          label="Max Drawdown"
          value={fmtPct(m.max_drawdown)}
          sub={`${m.max_drawdown_duration_days}d duration`}
          valueClass="negative"
        />
        <MetricCard label="VaR (95%)" value={fmtPct(m.var_95)} valueClass="negative" />
        <MetricCard label="CVaR (95%)" value={fmtPct(m.cvar_95)} valueClass="negative" />
      </div>

      {/* Risk-Adjusted */}
      <h3 className="mono text-xs mb-3 font-bold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
        Risk-Adjusted
      </h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <MetricCard label="Sharpe Ratio" value={fmtRatio(m.sharpe)} valueClass={colorClass(m.sharpe)} />
        <MetricCard label="Sortino Ratio" value={fmtRatio(m.sortino)} valueClass={colorClass(m.sortino)} />
        <MetricCard label="Calmar Ratio" value={fmtRatio(m.calmar)} valueClass={colorClass(m.calmar)} />
        <MetricCard label="Downside Dev" value={fmtPct(m.downside_deviation)} />
      </div>

      {/* Benchmark-Relative */}
      {m.beta != null && (
        <>
          <h3 className="mono text-xs mb-3 font-bold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
            vs Benchmark
          </h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard label="Beta" value={fmtRatio(m.beta)} />
            <MetricCard label="Alpha" value={fmtPct(m.alpha)} valueClass={colorClass(m.alpha)} />
            <MetricCard label="R²" value={fmtRatio(m.r_squared)} />
            <MetricCard label="Info Ratio" value={fmtRatio(m.information_ratio)} valueClass={colorClass(m.information_ratio)} />
            <MetricCard label="Tracking Error" value={fmtPct(m.tracking_error)} />
            <MetricCard label="Treynor" value={fmtRatio(m.treynor)} valueClass={colorClass(m.treynor)} />
            <MetricCard label="Up Capture" value={fmtPct(m.up_capture)} valueClass="positive" />
            <MetricCard label="Down Capture" value={fmtPct(m.down_capture)} />
          </div>
        </>
      )}
    </div>
  )
}
