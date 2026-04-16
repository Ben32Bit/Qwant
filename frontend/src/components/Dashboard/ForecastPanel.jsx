import React from 'react'
import { useForecast } from '../../hooks/useForecast.js'
import ForecastComposite from './ForecastComposite.jsx'
import ForecastMethodCard from './ForecastMethodCard.jsx'

const METHOD_ORDER = ['monte_carlo', 'garch', 'factor', 'hmm', 'var', 'lstm']

// Placeholder cards while phase 2 methods are loading
const PHASE2_METHODS = new Set(['hmm', 'var', 'lstm'])
const PHASE1_METHODS = new Set(['monte_carlo', 'garch', 'factor'])

function LoadingCard({ method }) {
  const LABELS = {
    monte_carlo: 'Monte Carlo (GBM)',
    garch:       'GARCH(1,1)',
    hmm:         'Hidden Markov Model',
    factor:      'Factor Model (FF5)',
    var:         'VAR Multi-Asset',
    lstm:        'LSTM Neural Net',
  }
  const COLORS = {
    monte_carlo: '#4a9eff', garch: '#ffd43b', hmm: '#a855f7',
    factor: '#00d4aa', var: '#ff6b35', lstm: '#ff4757',
  }
  return (
    <ForecastMethodCard
      result={{ method, label: LABELS[method], color: COLORS[method] }}
      loading
    />
  )
}

export default function ForecastPanel({ backtest, portfolio }) {
  const { results, meta, loading, error, run, hasData } = useForecast(backtest, portfolio)

  const isRunning = loading.phase1 || loading.phase2

  // Build result map for quick lookup
  const resultMap = {}
  for (const r of results) resultMap[r.method] = r

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-4 space-y-4 fade-in">

        {/* Header + Run button */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
              ◈ FORECAST
              <span className="ml-2 font-normal text-xs" style={{ color: 'var(--text-secondary)' }}>
                · Next 12 months · 6 research-backed methods
              </span>
            </h2>
            <p className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
              All methods use walk-forward out-of-sample validation (Lopez de Prado, 2018)
            </p>
          </div>
          <button
            onClick={run}
            disabled={isRunning || !backtest}
            className="mono text-xs px-4 py-2 rounded border transition-colors"
            style={{
              borderColor:  isRunning ? 'var(--border)' : 'var(--accent-blue)',
              color:        isRunning ? 'var(--text-secondary)' : 'var(--accent-blue)',
              background:   'transparent',
              cursor:       isRunning ? 'not-allowed' : 'pointer',
              opacity:      isRunning ? 0.6 : 1,
            }}
            onMouseEnter={e => { if (!isRunning) e.currentTarget.style.background = 'rgba(74,158,255,0.1)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
          >
            {loading.phase1 ? 'Running fast methods…'
              : loading.phase2 ? 'Running ML methods…'
              : hasData ? '↺ Re-run Forecast'
              : '▶ Run Forecast'}
          </button>
        </div>

        {/* Loading progress indicator */}
        {isRunning && (
          <div className="rounded-lg border px-4 py-3 mono text-xs"
            style={{ borderColor: 'rgba(74,158,255,0.25)', background: 'rgba(74,158,255,0.04)', color: 'var(--text-secondary)' }}>
            <div className="flex items-center gap-3">
              <span style={{ color: 'var(--accent-blue)' }}>
                {loading.phase1
                  ? '⟳ Phase 1: Monte Carlo, GARCH, Factor Model (~1–2s)…'
                  : '⟳ Phase 2: HMM, VAR, LSTM (~10–40s depending on history length)…'}
              </span>
            </div>
            {loading.phase2 && hasData && (
              <p className="mt-1" style={{ color: 'rgba(136,136,160,0.7)' }}>
                Phase 1 results shown below. Slower methods will appear when ready.
              </p>
            )}
          </div>
        )}

        {/* Error */}
        {error && !isRunning && (
          <div className="rounded-lg border px-4 py-3 mono text-xs"
            style={{ borderColor: 'rgba(255,71,87,0.3)', background: 'rgba(255,71,87,0.05)', color: 'var(--accent-red)' }}>
            {error}
          </div>
        )}

        {/* Empty state */}
        {!hasData && !isRunning && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="mono text-4xl mb-4 opacity-20 select-none" style={{ color: 'var(--accent-blue)' }}>◈</div>
            <p className="mono text-sm mb-1" style={{ color: 'var(--text-primary)' }}>6 probabilistic forecast methods</p>
            <p className="text-xs max-w-sm" style={{ color: 'var(--text-secondary)' }}>
              Monte Carlo · GARCH · Factor Model · HMM Regimes · VAR · LSTM
            </p>
            <p className="text-xs mt-2" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
              All methods cite peer-reviewed research and use strict walk-forward out-of-sample testing.
            </p>
          </div>
        )}

        {/* Composite chart — shows as soon as any result is available */}
        {(hasData || loading.phase1) && (
          <ForecastComposite
            results={results}
            equityCurve={backtest?.equity_curve}
            forecastStart={meta?.forecast_start}
            loading={loading.phase1}
          />
        )}

        {/* 2×3 method card grid */}
        {(hasData || isRunning) && (
          <div className="grid grid-cols-2 gap-4">
            {METHOD_ORDER.map(method => {
              const result  = resultMap[method]
              const p1Done  = !loading.phase1
              const p2Done  = !loading.phase2
              const inPhase1 = PHASE1_METHODS.has(method)
              const inPhase2 = PHASE2_METHODS.has(method)

              // Show skeleton if the phase for this method is still running and no result yet
              if (!result && ((inPhase1 && loading.phase1) || (inPhase2 && loading.phase2))) {
                return <LoadingCard key={method} method={method} />
              }
              if (!result) return null
              return (
                <ForecastMethodCard
                  key={method}
                  result={result}
                  loading={inPhase2 && loading.phase2 && !result.forecast}
                />
              )
            })}
          </div>
        )}

        {/* OOS methodology footer */}
        {hasData && (
          <div className="rounded-lg border px-4 py-3"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
            <p className="mono text-xs font-bold mb-1" style={{ color: 'var(--text-secondary)' }}>
              Out-of-Sample Methodology
            </p>
            <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)', opacity: 0.8 }}>
              All methods use walk-forward (expanding-window) train/test splits. Random k-fold is never
              applied — it violates temporal ordering and leaks future data. Each method reports an
              out-of-sample validation diagnostic (Ljung-Box, OOS R², regime sanity check, LSTM OOS MSE).
            </p>
            <p className="mono text-xs mt-2 leading-relaxed" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
              📄 Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.
              https://doi.org/10.1002/9781119482161 ·
              Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
              Journal of Portfolio Management, 40(5), 94–107.
              https://doi.org/10.3905/jpm.2014.40.5.094
            </p>
          </div>
        )}

      </div>
    </div>
  )
}
