import React, { useState, useMemo, useCallback } from 'react'
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
import { computeMetricsFromCurves, computeDrawdownFromCurve } from '../../utils/computeMetrics.js'

const RISK_FREE_RATE = 0.05

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

// ── Sub-period indicator ──────────────────────────────────────────────────────
function PeriodBadge({ startDate, endDate, fullStart, fullEnd }) {
  if (!startDate || startDate === fullStart && endDate === fullEnd) return null
  return (
    <div
      className="flex items-center gap-2 px-3 py-2 rounded-lg mono text-xs"
      style={{
        background: 'rgba(74,158,255,0.07)',
        border: '1px solid rgba(74,158,255,0.25)',
        color: 'var(--accent-blue)',
      }}
    >
      <span style={{ opacity: 0.7 }}>Metrics for selected period:</span>
      <span className="font-bold">{startDate?.slice(0, 7)} → {endDate?.slice(0, 7)}</span>
      <span style={{ opacity: 0.6 }}>· Approximate client-side recalculation</span>
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

  // ── Range slider state ────────────────────────────────────────────────────
  const [rangeStart, setRangeStart] = useState(null)
  const [rangeEnd,   setRangeEnd]   = useState(null)

  const onRangeChange = useCallback((startDate, endDate) => {
    setRangeStart(startDate)
    setRangeEnd(endDate)
  }, [])

  // Full period dates (from equity curve)
  const fullStart = backtest?.equity_curve?.[0]?.date ?? null
  const fullEnd   = backtest?.equity_curve?.at(-1)?.date ?? null

  // Are we in sub-range mode?
  const isSubRange = !!(rangeStart && rangeEnd &&
    (rangeStart !== fullStart || rangeEnd !== fullEnd))

  // ── Filtered equity curve (used for metric recalculation) ─────────────────
  const slicedEquity = useMemo(() => {
    if (!isSubRange || !backtest?.equity_curve) return backtest?.equity_curve ?? null
    return backtest.equity_curve.filter(pt => pt.date >= rangeStart && pt.date <= rangeEnd)
  }, [isSubRange, backtest?.equity_curve, rangeStart, rangeEnd])

  const slicedBenchmark = useMemo(() => {
    if (!isSubRange || !backtest?.benchmark_curve) return backtest?.benchmark_curve ?? null
    return backtest.benchmark_curve.filter(pt => pt.date >= rangeStart && pt.date <= rangeEnd)
  }, [isSubRange, backtest?.benchmark_curve, rangeStart, rangeEnd])

  // ── Recomputed metrics ────────────────────────────────────────────────────
  const activeMetrics = useMemo(() => {
    if (!isSubRange) return backtest?.metrics ?? null
    if (!slicedEquity?.length) return backtest?.metrics ?? null
    return computeMetricsFromCurves(slicedEquity, slicedBenchmark, RISK_FREE_RATE)
      ?? backtest?.metrics ?? null
  }, [isSubRange, slicedEquity, slicedBenchmark, backtest?.metrics])

  // ── Recomputed drawdown series ────────────────────────────────────────────
  const activeDrawdown = useMemo(() => {
    if (!isSubRange) return backtest?.drawdown_series ?? null
    if (!slicedEquity?.length) return backtest?.drawdown_series ?? null
    return computeDrawdownFromCurve(slicedEquity)
  }, [isSubRange, slicedEquity, backtest?.drawdown_series])

  // ── Filtered rolling metrics ──────────────────────────────────────────────
  const activeRolling = useMemo(() => {
    if (!isSubRange || !backtest?.rolling_metrics) return backtest?.rolling_metrics ?? null
    const rm = backtest.rolling_metrics
    const filter = (arr) => arr?.filter(pt => pt.date >= rangeStart && pt.date <= rangeEnd) ?? []
    return { sharpe: filter(rm.sharpe), volatility: filter(rm.volatility), beta: filter(rm.beta) }
  }, [isSubRange, backtest?.rolling_metrics, rangeStart, rangeEnd])

  // ── Filtered monthly returns ──────────────────────────────────────────────
  const activeMonthly = useMemo(() => {
    if (!isSubRange || !backtest?.monthly_returns?.data) return backtest?.monthly_returns ?? null
    const startYM = rangeStart?.slice(0, 7) ?? ''
    const endYM   = rangeEnd?.slice(0, 7)   ?? '9999-12'
    const MONTHS  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    const filtered = {}
    for (const [year, months] of Object.entries(backtest.monthly_returns.data)) {
      if (year < startYM.slice(0, 4) || year > endYM.slice(0, 4)) continue
      const filteredMonths = {}
      for (const [month, val] of Object.entries(months)) {
        const mIdx = MONTHS.indexOf(month) + 1
        const ym   = `${year}-${String(mIdx).padStart(2, '0')}`
        if (ym >= startYM && ym <= endYM) filteredMonths[month] = val
      }
      if (Object.keys(filteredMonths).length > 0) filtered[year] = filteredMonths
    }
    return { data: filtered }
  }, [isSubRange, backtest?.monthly_returns, rangeStart, rangeEnd])

  // ── Section list ──────────────────────────────────────────────────────────
  const sections = useMemo(() => {
    const raw = displayConfig?.sections?.length
      ? displayConfig.sections
      : ['equity_curve', 'drawdown', 'metrics_summary']
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
              style={{ borderColor: 'var(--accent-green)', color: 'var(--accent-green)', background: 'transparent' }}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,200,83,0.1)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
            >
              ↓ Export Excel
            </button>
          )}
        </div>

        {/* AI Narrative */}
        {displayConfig?.narrative && <AiNarrative narrative={displayConfig.narrative} />}

        {/* Featured metrics strip */}
        {displayConfig?.featured_metrics?.length > 0 && (
          <FeaturedMetrics metrics={activeMetrics} featuredKeys={displayConfig.featured_metrics} />
        )}

        {/* Sub-period badge */}
        {isSubRange && (
          <PeriodBadge startDate={rangeStart} endDate={rangeEnd} fullStart={fullStart} fullEnd={fullEnd} />
        )}

        {/* Dynamic sections in AI-chosen order */}
        {sections.map(section => {
          switch (section) {

            case 'equity_curve':
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
                  onRangeChange={onRangeChange}
                />
              )

            case 'drawdown':
              return (
                <DrawdownChart key="drawdown"
                  drawdownSeries={activeDrawdown}
                  loading={loading}
                />
              )

            case 'metrics_summary':
              return (
                <MetricsCards key="metrics_summary"
                  metrics={activeMetrics}
                  loading={loading}
                  mode="summary"
                />
              )

            case 'full_metrics':
              return (
                <MetricsCards key="full_metrics"
                  metrics={activeMetrics}
                  loading={loading}
                  mode="full"
                />
              )

            case 'monthly_heatmap':
              return (
                <MonthlyHeatmap key="monthly_heatmap"
                  monthlyReturns={activeMonthly}
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
                  rollingMetrics={activeRolling}
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
