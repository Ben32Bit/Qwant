import React, { useState, useEffect } from 'react'
import { useForecast } from '../../hooks/useForecast.js'
import ForecastComposite from './ForecastComposite.jsx'
import ForecastMethodCard from './ForecastMethodCard.jsx'

const METHOD_ORDER = ['monte_carlo', 'garch', 'factor', 'hmm', 'var', 'lstm']

const PHASE2_METHODS = new Set(['hmm', 'var', 'lstm'])
const PHASE1_METHODS = new Set(['monte_carlo', 'garch', 'factor'])

// Typical durations for the ETA bar (milliseconds)
const PHASE1_EST_MS = 4_000
const PHASE2_EST_MS = 75_000   // HMM + VAR + Attention-LSTM can take 30-90s

// Placeholder cards while a phase is loading
function LoadingCard({ method }) {
  const LABELS = {
    monte_carlo: 'Monte Carlo (GBM)',
    garch:       'GARCH(1,1)',
    hmm:         'Hidden Markov Model',
    factor:      'Factor Model (FF5)',
    var:         'VAR Multi-Asset',
    lstm:        'Attention-LSTM',
  }
  const COLORS = {
    monte_carlo: '#4a9eff', garch: '#ffd43b', hmm: '#a855f7',
    factor: '#00d4aa', var: '#ff6b35', lstm: '#ff4757',
  }
  return (
    <ForecastMethodCard
      result={{ method, label: LABELS[method], color: COLORS[method] }}
      loading
      lastValue={null}
    />
  )
}

// ── ETA Progress Bar ──────────────────────────────────────────────────────────

function EtaBar({ loading, p1StartRef, p2StartRef, timing }) {
  const [elapsed, setElapsed] = useState(0)

  // Tick every 250ms while any phase is running
  useEffect(() => {
    if (!loading.phase1 && !loading.phase2) {
      setElapsed(0)
      return
    }
    const startRef = loading.phase1 ? p1StartRef : p2StartRef
    const tick = () => setElapsed(Date.now() - (startRef.current ?? Date.now()))
    tick()
    const id = setInterval(tick, 250)
    return () => clearInterval(id)
  }, [loading.phase1, loading.phase2, p1StartRef, p2StartRef])

  if (!loading.phase1 && !loading.phase2) return null

  const estMs    = loading.phase1 ? PHASE1_EST_MS : PHASE2_EST_MS
  const progress = Math.min(elapsed / estMs, 0.97)   // never fill completely while pending
  const etaMs    = Math.max(0, estMs - elapsed)

  const fmtEta = (ms) => {
    if (ms >= 60_000) return `~${Math.ceil(ms / 60_000)}m`
    return `~${Math.ceil(ms / 1_000)}s`
  }

  const phase = loading.phase1 ? 1 : 2
  const methods = loading.phase1
    ? 'Monte Carlo · GARCH · Factor Model'
    : 'Hidden Markov Model · VAR · Attention-LSTM'

  return (
    <div className="rounded-lg border px-4 py-3"
      style={{ borderColor: 'rgba(74,158,255,0.2)', background: 'rgba(74,158,255,0.04)' }}>
      {/* Label row */}
      <div className="flex items-center justify-between mb-2">
        <span className="mono text-xs" style={{ color: 'var(--accent-blue)' }}>
          ⟳ Phase {phase}: {methods}
        </span>
        <span className="mono text-xs font-bold" style={{ color: 'var(--accent-blue)' }}>
          ETA {fmtEta(etaMs)}
        </span>
      </div>

      {/* Progress bar */}
      <div className="rounded-full overflow-hidden" style={{ height: 4, background: 'var(--border)' }}>
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${progress * 100}%`,
            background: 'linear-gradient(90deg, var(--accent-blue), #00d4aa)',
            transition: 'width 0.25s linear',
          }}
        />
      </div>

      {/* Phase 2 note when phase 1 results are already shown */}
      {loading.phase2 && (
        <p className="mono text-xs mt-2" style={{ color: 'rgba(136,136,160,0.7)' }}>
          Phase 1 results shown below. ML methods will appear when ready.
        </p>
      )}
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function ForecastPanel({ backtest, portfolio }) {
  const { results, meta, loading, error, run, hasData, timing, p1StartRef, p2StartRef } =
    useForecast(backtest, portfolio)

  const isRunning = loading.phase1 || loading.phase2

  // Last historical portfolio value — anchor for projecting actual dollar values
  const lastValue = backtest?.equity_curve?.at(-1)?.value ?? null

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
            {isRunning ? 'Running…'
              : hasData ? '↺ Re-run Forecast'
              : '▶ Run Forecast'}
          </button>
        </div>

        {/* ETA progress bar */}
        {isRunning && (
          <EtaBar
            loading={loading}
            p1StartRef={p1StartRef}
            p2StartRef={p2StartRef}
            timing={timing}
          />
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
              Monte Carlo · GARCH · Factor Model · HMM Regimes · VAR · Attention-LSTM
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
              const result   = resultMap[method]
              const inPhase1 = PHASE1_METHODS.has(method)
              const inPhase2 = PHASE2_METHODS.has(method)

              if (!result && ((inPhase1 && loading.phase1) || (inPhase2 && loading.phase2))) {
                return <LoadingCard key={method} method={method} />
              }
              if (!result) return null
              return (
                <ForecastMethodCard
                  key={method}
                  result={result}
                  loading={inPhase2 && loading.phase2 && !result.forecast}
                  lastValue={lastValue}
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
              out-of-sample validation diagnostic (Ljung-Box, OOS R², regime sanity check, Attention-LSTM OOS MSE).
            </p>
            <p className="mono text-xs mt-2 leading-relaxed" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
              📄 Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.
              https://doi.org/10.1002/9781119482161 ·
              Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
              Journal of Portfolio Management, 40(5), 94–107.
              https://doi.org/10.3905/jpm.2014.40.5.094 ·
              CS230 Stanford (2020). Temporal Attention-Enhanced LSTM.
              https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf
            </p>
          </div>
        )}

      </div>
    </div>
  )
}
