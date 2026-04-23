import { useState, useEffect, useRef } from 'react'
import ForecastComposite from './ForecastComposite.jsx'
import ForecastSnapshotCards from './ForecastSnapshotCards.jsx'
import ForecastMethodCard from './ForecastMethodCard.jsx'
import ForecastExport from './ForecastExport.jsx'
import SentimentPanel from './SentimentPanel.jsx'
import EnsembleCard from './EnsembleCard.jsx'
import { computeEnsemble } from '../../ml/MetaEnsemble.js'
import KellyPanel from './KellyPanel.jsx'
import ScenarioPanel from './ScenarioPanel.jsx'
import ForecastArchitecture from './ForecastArchitecture.jsx'

const METHOD_ORDER = ['xgboost', 'nbeats', 'factor', 'hmm', 'var', 'lstm']

const PHASE2_METHODS    = new Set(['hmm', 'var', 'lstm'])
const PHASE1_METHODS    = new Set(['xgboost', 'nbeats', 'factor'])

// Typical durations for the ETA bar (milliseconds)
const PHASE1_EST_MS  = 4_000
const XGB_EST_MS     = 3_000    // ONNX Runtime Web: ~1-3s (XGBoost)
const NBEATS_EST_MS  = 5_000    // ONNX Runtime Web: ~3-5s (N-BEATS 12 periods)
const PHASE2_EST_MS  = 15_000   // HMM + GP
const LSTM_EST_MS    = 5_000    // TF.js browser inference: ~2-5s

// Expandable wrapper for the heavy 12m composite chart. Uses local state so
// the chart literally unmounts when collapsed — a `<details>` element only
// toggles CSS visibility and would still run Recharts layout on every click.
function CompositeChartSection({ children }) {
  const [open, setOpen] = useState(true)
  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        className="mono text-xs cursor-pointer select-none mb-2 w-full text-left"
        style={{ color: 'var(--text-secondary)', opacity: 0.75, background: 'transparent', border: 'none', padding: '4px 0' }}
      >
        {open ? '▾' : '▸'} 12-month exploratory chart
        <span className="ml-2" style={{ opacity: 0.6 }}>
          · short-horizon models capped; interpret with caution past 3 months
        </span>
      </button>
      {open && children}
    </div>
  )
}

// Placeholder cards while a phase is loading
function LoadingCard({ method, browserCompute = false }) {
  const LABELS = {
    xgboost: 'XGBoost Quantile',
    nbeats:  'N-BEATS Neural',
    hmm:     'Hidden Markov Model',
    factor:  'Factor Model (FF5)',
    var:     'Gaussian Process (GP)',
    lstm:    'Attention-LSTM',
  }
  const COLORS = {
    xgboost: '#4a9eff', nbeats: '#ffd43b', hmm: '#a855f7',
    factor: '#00d4aa', var: '#ff6b35', lstm: '#ff4757',
  }
  return (
    <ForecastMethodCard
      result={{ method, label: LABELS[method], color: COLORS[method] }}
      loading
      browserCompute={browserCompute}
      lastValue={null}
    />
  )
}

// ── ETA Progress Bar ──────────────────────────────────────────────────────────

function EtaBar({ loading, p1StartRef, xgbStartRef, nbeatsStartRef, p2StartRef, lstmStartRef }) {
  const [elapsed, setElapsed] = useState(0)

  // Priority: p1 > xgb/nbeats (show xgb if both running) > p2 > lstm
  const activePhase = loading.phase1 ? 'p1'
    : (loading.xgb || loading.nbeats) ? (loading.xgb ? 'xgb' : 'nbeats')
    : loading.phase2 ? 'p2'
    : loading.lstm   ? 'lstm'
    : null

  useEffect(() => {
    if (!activePhase) { setElapsed(0); return }
    const startRef = activePhase === 'p1'     ? p1StartRef
      : activePhase === 'xgb'    ? xgbStartRef
      : activePhase === 'nbeats' ? nbeatsStartRef
      : activePhase === 'p2'     ? p2StartRef
      : lstmStartRef
    const tick = () => setElapsed(Date.now() - (startRef.current ?? Date.now()))
    tick()
    const id = setInterval(tick, 500)
    return () => clearInterval(id)
  }, [activePhase, p1StartRef, xgbStartRef, nbeatsStartRef, p2StartRef, lstmStartRef])

  if (!activePhase) return null

  const estMs = activePhase === 'p1'     ? PHASE1_EST_MS
    : activePhase === 'xgb'    ? XGB_EST_MS
    : activePhase === 'nbeats' ? NBEATS_EST_MS
    : activePhase === 'p2'     ? PHASE2_EST_MS
    : LSTM_EST_MS
  const overrun   = elapsed > estMs
  const progress  = overrun ? 0.97 : elapsed / estMs
  const etaMs     = Math.max(0, estMs - elapsed)
  const overrunMs = overrun ? elapsed - estMs : 0

  const fmtEta = (ms, over) =>
    over   ? `+${Math.ceil(overrunMs / 1_000)}s — still running`
    : ms >= 60_000 ? `~${Math.ceil(ms / 60_000)}m`
    : `~${Math.ceil(ms / 1_000)}s`

  const isBrowser = activePhase !== 'p1' && activePhase !== 'p2'
  const color     = overrun ? '#ffd43b' : isBrowser ? 'var(--accent-green)' : 'var(--accent-blue)'

  const LABELS = {
    p1:     'Phase 1 (server): XGBoost features · N-BEATS features · Factor Model',
    xgb:    'Phase 1B (browser): XGBoost · 5 quantile ONNX models',
    nbeats: 'Phase 1B (browser): N-BEATS · 12-period recursive · pure-JS weights',
    p2:     'Phase 2 (server): Hidden Markov Model · Gaussian Process',
    lstm:   'Phase 3 (browser): Attention-LSTM · TF.js MC Dropout',
  }
  const GRADIENTS = {
    p1:     'linear-gradient(90deg, var(--accent-blue), #00d4aa)',
    xgb:    'linear-gradient(90deg, #4a9eff, #00d4aa)',
    nbeats: 'linear-gradient(90deg, #ffd43b, #00d4aa)',
    p2:     overrun ? 'linear-gradient(90deg, #ffd43b, #ff6b35)' : 'linear-gradient(90deg, var(--accent-blue), #a855f7)',
    lstm:   'linear-gradient(90deg, #ff4757, #a855f7)',
  }

  return (
    <div className="rounded-lg border px-4 py-3"
      style={{ borderColor: `${color}33`, background: `${color}0a` }}>
      <div className="flex items-center justify-between mb-2">
        <span className="mono text-xs" style={{ color }}>
          {overrun ? '⚠' : '⟳'} {LABELS[activePhase]}
        </span>
        <span className="mono text-xs font-bold" style={{ color }}>
          {overrun ? fmtEta(0, true) : `ETA ${fmtEta(etaMs, false)}`}
        </span>
      </div>
      <div className="rounded-full overflow-hidden" style={{ height: 4, background: 'var(--border)' }}>
        <div className="h-full rounded-full"
          style={{ width: `${progress * 100}%`, background: GRADIENTS[activePhase], transition: overrun ? 'none' : 'width 0.25s linear' }}
        />
      </div>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function ForecastPanel({ backtest, portfolio, forecast }) {
  // `forecast` is hoisted to SplitView so in-flight Phase 1/2 state survives
  // a Results⇄Forecast tab switch (the panel itself unmounts on switch).
  const { results, meta, loading, error, run, hasData, newsContext, edgarContext, p1StartRef, xgbStartRef, nbeatsStartRef, p2StartRef, lstmStartRef } = forecast

  const [ensemble, setEnsemble] = useState(null)

  // Recompute ensemble whenever base model results or regime probs change
  const regimeProbs   = meta?.regime_probs ?? null
  const serverWeights = meta?.ensemble_weights ?? null

  useEffect(() => {
    const forecasted = results.filter(r => r.forecast?.p50?.length > 0)
    if (!forecasted.length || !regimeProbs) { setEnsemble(null); return }
    let cancelled = false
    computeEnsemble(forecasted, regimeProbs, serverWeights)
      .then(e => { if (!cancelled) setEnsemble(e) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [results, regimeProbs, serverWeights])

  const isRunning = loading.phase1 || loading.xgb || loading.nbeats || loading.phase2 || loading.lstm

  // Last historical portfolio value — anchor for projecting actual dollar values
  const lastValue = backtest?.equity_curve?.at(-1)?.value ?? null

  const resultMap = {}
  for (const r of results) resultMap[r.method] = r

  // Ref wraps the content area that gets captured by the PNG export. It
  // excludes the header (Run/Export buttons) so the snapshot is clean.
  const exportRef = useRef(null)

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
          <div className="flex items-start gap-2">
            <ForecastExport
              portfolio={portfolio}
              backtest={backtest}
              results={results}
              ensemble={ensemble}
              meta={meta}
              targetRef={exportRef}
              disabled={isRunning}
            />
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
        </div>

        {/* Architecture dropdown */}
        <ForecastArchitecture />

        {/* ETA progress bar */}
        {isRunning && (
          <EtaBar
            loading={loading}
            p1StartRef={p1StartRef}
            xgbStartRef={xgbStartRef}
            nbeatsStartRef={nbeatsStartRef}
            p2StartRef={p2StartRef}
            lstmStartRef={lstmStartRef}
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
              XGBoost · N-BEATS · Factor Model · HMM Regimes · VAR · Attention-LSTM
            </p>
            <p className="text-xs mt-2" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
              All methods cite peer-reviewed research and use strict walk-forward out-of-sample testing.
            </p>
          </div>
        )}

        {/* ── Primary forecast content ─────────────────────────────────────── */}
        {/* exportRef wraps everything the PNG snapshot should capture — the
            header (Run/Export buttons) is intentionally outside so the image
            isn't cluttered with UI chrome. */}
        <div ref={exportRef} className="space-y-4">

        {/* Portfolio construct header — always rendered inside the export
            scope so the PNG includes the tickers/weights that generated
            these numbers. (The HTML report reads from props directly.) */}
        {portfolio?.assets?.length > 0 && (hasData || loading.phase1) && (
          <div className="rounded-lg border p-3" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <div className="flex items-baseline justify-between mb-2">
              <h3 className="mono font-bold text-xs" style={{ color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Portfolio Construct
              </h3>
              <span className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                {portfolio.start_date ?? '—'} → {portfolio.end_date ?? '—'}
                {portfolio.rebalance_frequency && ` · ${portfolio.rebalance_frequency} rebal`}
                {portfolio.benchmark && ` · vs ${portfolio.benchmark}`}
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {portfolio.assets.map((a, i) => {
                const w = (a.weight ?? 0) * 100
                const colour = w >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'
                return (
                  <div key={i} className="mono px-2 py-1 rounded"
                    style={{ fontSize: 10, background: 'rgba(255,255,255,0.04)', border: `1px solid ${colour}44` }}>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{a.ticker}</span>
                    <span style={{ color: colour, marginLeft: 6 }}>
                      {w >= 0 ? '+' : ''}{w.toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Horizon-honest snapshot cards — 1w / 1m / 3m ensemble medians.
            These three cutoffs are all at or inside each model's training
            horizon, so every number here is academically defensible. */}
        {(hasData || loading.phase1) && (
          <ForecastSnapshotCards
            results={results}
            ensemble={ensemble}
            lastValue={lastValue}
            loading={loading.phase1}
          />
        )}

        {/* Composite chart — all 6 method medians on one chart.
            Collapsed by default AND unmounted when closed: the chart draws
            ~2500 historical points × 7 lines, which is a serious render /
            SVG-node load that was making the page unresponsive on click. */}
        {(hasData || loading.phase1) && (
          <CompositeChartSection>
            <ForecastComposite
              results={results}
              equityCurve={backtest?.equity_curve}
              forecastStart={meta?.forecast_start}
              loading={loading.phase1}
              ensemble={ensemble}
            />
          </CompositeChartSection>
        )}

        {/* 2×3 individual method cards */}
        {(hasData || isRunning) && (
          <div className="grid grid-cols-2 gap-4">
            {METHOD_ORDER.map(method => {
              const result   = resultMap[method]
              const inPhase1 = PHASE1_METHODS.has(method)
              const inPhase2 = PHASE2_METHODS.has(method)

              const isXgbBrowserLoading    = method === 'xgboost' && loading.xgb
              const isNbeatsBrowserLoading = method === 'nbeats'  && loading.nbeats
              const isLstmBrowserLoading   = method === 'lstm'    && loading.lstm

              if (!result && ((inPhase1 && loading.phase1) || (inPhase2 && loading.phase2))) {
                return <div key={method} className="cv-auto"><LoadingCard method={method} /></div>
              }
              if (isXgbBrowserLoading || isNbeatsBrowserLoading || isLstmBrowserLoading) {
                return <div key={method} className="cv-auto"><LoadingCard method={method} browserCompute /></div>
              }
              if (!result) return null
              return (
                <div key={method} className="cv-auto">
                  <ForecastMethodCard
                    result={result}
                    loading={inPhase2 && loading.phase2 && !result.forecast}
                    lastValue={lastValue}
                  />
                </div>
              )
            })}
          </div>
        )}

        {/* ── Meta-analysis (shows once ensemble is ready) ─────────────────── */}

        {/* Ensemble forecast card */}
        {(hasData || isRunning) && (
          <div className="cv-auto">
            <EnsembleCard
              ensemble={ensemble}
              regimeProbs={regimeProbs}
              loading={isRunning && !ensemble}
              lastValue={lastValue}
            />
          </div>
        )}

        {/* Kelly position sizing */}
        {(hasData || isRunning) && (
          <div className="cv-auto">
            <KellyPanel
              ensemble={ensemble}
              regimeProbs={regimeProbs}
              loading={isRunning && !ensemble}
            />
          </div>
        )}

        {/* Scenario stress tester */}
        {(hasData || isRunning) && (
          <div className="cv-auto">
            <ScenarioPanel
              ensemble={ensemble}
              regimeProbs={regimeProbs}
              results={results}
            />
          </div>
        )}

        </div> {/* end exportRef wrapper — captures through scenario testing */}

        {/* FinBERT sentiment — Phase 4B */}
        {hasData && (
          <div className="cv-auto">
            <SentimentPanel newsContext={newsContext} edgarContext={edgarContext} />
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
