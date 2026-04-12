import React from 'react'
import EquityCurve from './EquityCurve.jsx'
import DrawdownChart from './DrawdownChart.jsx'
import MetricsCards from './MetricsCards.jsx'
import MonthlyHeatmap from './MonthlyHeatmap.jsx'

export default function ResultsPanel({ backtest, portfolio, loading }) {
  const isEmpty = !backtest && !loading

  if (isEmpty) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-8">
        <div
          className="mono text-5xl mb-6 opacity-20 select-none"
          style={{ color: 'var(--accent-blue)' }}
        >
          ▷
        </div>
        <h2 className="mono font-bold text-lg mb-2" style={{ color: 'var(--text-primary)' }}>
          Results appear here
        </h2>
        <p className="text-sm max-w-sm" style={{ color: 'var(--text-secondary)' }}>
          Describe a portfolio in the chat and hit Run. The backtest runs instantly and
          results will populate this panel.
        </p>
        <div
          className="mt-8 grid grid-cols-3 gap-4 w-full max-w-lg opacity-20"
          aria-hidden="true"
        >
          {['CAGR', 'Sharpe', 'Max DD', 'Vol', 'Beta', 'Alpha'].map((label) => (
            <div
              key={label}
              className="rounded border p-3"
              style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}
            >
              <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
              <div className="skeleton h-5 w-16" />
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-4 space-y-4 fade-in">
        {/* Strategy summary */}
        {portfolio?.strategy_summary && (
          <div
            className="rounded-lg border px-4 py-3 text-sm"
            style={{ borderColor: 'rgba(74, 158, 255, 0.3)', background: 'rgba(74, 158, 255, 0.07)' }}
          >
            <span className="mono font-bold text-xs" style={{ color: 'var(--accent-blue)' }}>
              STRATEGY&nbsp;&nbsp;
            </span>
            <span style={{ color: 'var(--text-primary)' }}>{portfolio.strategy_summary}</span>
          </div>
        )}

        {/* Charts */}
        <EquityCurve
          equityCurve={backtest?.equity_curve}
          benchmarkCurve={backtest?.benchmark_curve}
          loading={loading}
        />

        <DrawdownChart
          drawdownSeries={backtest?.drawdown_series}
          loading={loading}
        />

        {/* Metrics */}
        <MetricsCards metrics={backtest?.metrics} loading={loading} />

        {/* Monthly heatmap */}
        <MonthlyHeatmap monthlyReturns={backtest?.monthly_returns} loading={loading} />
      </div>
    </div>
  )
}
