import React, { useMemo } from 'react'
import EquityCurve from './EquityCurve.jsx'
import RotationEquityChart from './RotationEquityChart.jsx'
import DrawdownChart from './DrawdownChart.jsx'
import MetricsCards from './MetricsCards.jsx'
import MonthlyHeatmap from './MonthlyHeatmap.jsx'
import CorrelationMatrix from './CorrelationMatrix.jsx'
import RollingMetrics from './RollingMetrics.jsx'
import AiNarrative from './AiNarrative.jsx'
import WeightDriftChart from './WeightDriftChart.jsx'
import FamaFrenchFactors from './FamaFrenchFactors.jsx'
import { fmtPct, fmtRatio, colorClass } from '../../utils/formatters.js'
import { exportToExcel } from '../../utils/exportExcel.js'

// ── Featured metrics strip ────────────────────────────────────────────────────
const METRIC_LABELS = {
  cagr: 'CAGR', total_return: 'Total Return', sharpe: 'Sharpe',
  sortino: 'Sortino', calmar: 'Calmar', max_drawdown: 'Max DD',
  volatility: 'Volatility', beta: 'Beta', alpha: 'Alpha',
  information_ratio: 'Info Ratio', up_capture: 'Up Capture',
  down_capture: 'Down Capture', tracking_error: 'Track. Error',
  r_squared: 'R²', treynor: 'Treynor',
}

const PCT_METRICS = new Set(['cagr', 'total_return', 'max_drawdown', 'volatility',
  'alpha', 'tracking_error', 'up_capture', 'down_capture'])

function formatMetricValue(key, val) {
  if (val == null) return '—'
  return PCT_METRICS.has(key) ? fmtPct(val) : fmtRatio(val)
}

function FeaturedMetrics({ metrics, featuredKeys }) {
  if (!metrics || !featuredKeys?.length) return null
  const items = featuredKeys.filter(k => metrics[k] != null)
  if (!items.length) return null

  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(items.length, 4)}, 1fr)` }}>
      {items.map(key => {
        const val = metrics[key]
        return (
          <div key={key} className="rounded-lg border p-3 text-center"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
              {METRIC_LABELS[key] || key}
            </div>
            <div className={`mono font-bold text-xl ${colorClass(val)}`}>
              {formatMetricValue(key, val)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8">
      <div className="mono text-5xl mb-6 opacity-20 select-none" style={{ color: 'var(--accent-blue)' }}>▷</div>
      <h2 className="mono font-bold text-lg mb-2" style={{ color: 'var(--text-primary)' }}>Results appear here</h2>
      <p className="text-sm max-w-sm" style={{ color: 'var(--text-secondary)' }}>
        Describe a portfolio in the chat. The AI will research and backtest it — results populate here instantly.
      </p>
      <div className="mt-8 grid grid-cols-3 gap-4 w-full max-w-lg opacity-20" aria-hidden="true">
        {['CAGR', 'Sharpe', 'Max DD', 'Vol', 'Beta', 'Alpha'].map(label => (
          <div key={label} className="rounded border p-3" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
            <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
            <div className="skeleton h-5 w-16" />
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────
export default function ResultsPanel({ backtest, portfolio, displayConfig, loading }) {
  const isEmpty = !backtest && !loading

  const sections = useMemo(() => {
    const raw = displayConfig?.sections?.length
      ? displayConfig.sections
      : ['equity_curve', 'drawdown', 'metrics_summary']
    // If both summary and full are present, keep only full_metrics (superset)
    if (raw.includes('full_metrics') && raw.includes('metrics_summary')) {
      return raw.filter(s => s !== 'metrics_summary')
    }
    return raw
  }, [displayConfig])

  const hasBenchmark = !!backtest?.benchmark_curve

  if (isEmpty) return <EmptyState />

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-4 space-y-4 fade-in">

        {/* Strategy header + Export button */}
        <div className="flex items-center gap-3">
          {portfolio?.strategy_summary && (
            <div className="flex-1 rounded-lg border px-4 py-3 text-sm"
              style={{ borderColor: 'rgba(74,158,255,0.3)', background: 'rgba(74,158,255,0.07)' }}>
              <span className="mono font-bold text-xs" style={{ color: 'var(--accent-blue)' }}>STRATEGY&nbsp;&nbsp;</span>
              <span style={{ color: 'var(--text-primary)' }}>{portfolio.strategy_summary}</span>
            </div>
          )}
          {backtest && (
            <button
              onClick={() => exportToExcel(backtest, portfolio)}
              className="mono text-xs px-3 py-2 rounded border transition-colors whitespace-nowrap"
              style={{
                borderColor: 'var(--accent-green)',
                color: 'var(--accent-green)',
                background: 'transparent',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background = 'rgba(0,200,83,0.1)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = 'transparent'
              }}
            >
              ↓ Export Excel
            </button>
          )}
        </div>

        {/* AI Narrative */}
        {displayConfig?.narrative && <AiNarrative narrative={displayConfig.narrative} />}

        {/* Featured metrics strip — shown whenever display_config.featured_metrics is set */}
        {displayConfig?.featured_metrics?.length > 0 && (
          <FeaturedMetrics metrics={backtest?.metrics} featuredKeys={displayConfig.featured_metrics} />
        )}

        {/* Dynamic sections in AI-chosen order */}
        {sections.map(section => {
          switch (section) {
            case 'equity_curve':
              // Use rotation chart when holding_schedule is present (rotation backtest)
              return backtest?.holding_schedule?.length ? (
                <RotationEquityChart key="equity_curve"
                  equityCurve={backtest.equity_curve}
                  benchmarkCurve={backtest.benchmark_curve}
                  holdingSchedule={backtest.holding_schedule}
                />
              ) : (
                <EquityCurve key="equity_curve"
                  equityCurve={backtest?.equity_curve}
                  benchmarkCurve={backtest?.benchmark_curve}
                  fxCurves={backtest?.fx_curves}
                  loading={loading}
                />
              )
            case 'drawdown':
              return (
                <DrawdownChart key="drawdown"
                  drawdownSeries={backtest?.drawdown_series}
                  loading={loading}
                />
              )
            case 'metrics_summary':
              return (
                <MetricsCards key="metrics_summary"
                  metrics={backtest?.metrics}
                  loading={loading}
                  mode="summary"
                />
              )
            case 'full_metrics':
              return (
                <MetricsCards key="full_metrics"
                  metrics={backtest?.metrics}
                  loading={loading}
                  mode="full"
                />
              )
            case 'monthly_heatmap':
              return (
                <MonthlyHeatmap key="monthly_heatmap"
                  monthlyReturns={backtest?.monthly_returns}
                  loading={loading}
                />
              )
            case 'correlation_matrix':
              return (
                <CorrelationMatrix key="correlation_matrix"
                  correlationMatrix={backtest?.correlation_matrix}
                  loading={loading}
                />
              )
            case 'rolling_metrics':
              return (
                <RollingMetrics key="rolling_metrics"
                  rollingMetrics={backtest?.rolling_metrics}
                  hasBenchmark={hasBenchmark}
                  loading={loading}
                />
              )
            case 'weight_drift':
              return (
                <WeightDriftChart key="weight_drift"
                  weightHistory={backtest?.weight_history}
                  rebalanceDates={backtest?.rebalance_dates}
                  loading={loading}
                />
              )
            case 'ff5_decomposition':
              return (
                <FamaFrenchFactors key="ff5_decomposition"
                  ff5={backtest?.ff5_decomposition}
                  loading={loading}
                />
              )
            default:
              return null
          }
        })}
      </div>
    </div>
  )
}
