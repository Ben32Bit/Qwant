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

const METHOD_ORDER = ['nbeats', 'timesfm', 'hmm', 'var', 'lstm']

const PHASE2_METHODS = new Set(['hmm', 'var', 'lstm', 'timesfm'])
const PHASE1_METHODS = new Set(['nbeats'])

const PHASE1_EST_MS  = 8_000    // macro + insider provider fan-out + N-BEATS prep
const NBEATS_EST_MS  = 5_000    // pure-JS browser inference: ~2-5s
const PHASE2_EST_MS  = 90_000   // HMM + GP + tier-2 providers, cold-boot P90
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
        {open ? '▾' : '▸'} 3-month exploratory chart
      </button>
      {open && children}
    </div>
  )
}

// Placeholder cards while a phase is loading
function LoadingCard({ method, browserCompute = false, compact = false }) {
  const LABELS = {
    nbeats:  'N-BEATS Neural',
    timesfm: 'TimesFM 2.5 (Google)',
    hmm:     'Hidden Markov Model',
    var:     'Gaussian Process (GP)',
    lstm:    'Attention-LSTM',
  }
  const COLORS = {
    nbeats: '#ffd43b', timesfm: '#00d4aa', hmm: '#a855f7', var: '#ff6b35', lstm: '#ff4757',
  }
  return (
    <ForecastMethodCard
      result={{ method, label: LABELS[method], color: COLORS[method] }}
      loading
      browserCompute={browserCompute}
      lastValue={null}
      compact={compact}
    />
  )
}

// ── ETA Progress Bar ──────────────────────────────────────────────────────────

function EtaBar({ loading, p1StartRef, nbeatsStartRef, p2StartRef, lstmStartRef }) {
  const [elapsed, setElapsed] = useState(0)

  const activePhase = loading.phase1 ? 'p1'
    : loading.nbeats ? 'nbeats'
    : loading.phase2 ? 'p2'
    : loading.lstm   ? 'lstm'
    : null

  useEffect(() => {
    if (!activePhase) { setElapsed(0); return }
    const startRef = activePhase === 'p1'     ? p1StartRef
      : activePhase === 'nbeats' ? nbeatsStartRef
      : activePhase === 'p2'     ? p2StartRef
      : lstmStartRef
    const tick = () => setElapsed(Date.now() - (startRef.current ?? Date.now()))
    tick()
    const id = setInterval(tick, 500)
    return () => clearInterval(id)
  }, [activePhase, p1StartRef, nbeatsStartRef, p2StartRef, lstmStartRef])

  if (!activePhase) return null

  const estMs = activePhase === 'p1'     ? PHASE1_EST_MS
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
    p1:     'Phase 1 (server): N-BEATS features · macro + insider provider fan-out',
    nbeats: 'Phase 1B (browser): N-BEATS · 12-period recursive · pure-JS weights',
    p2:     'Phase 2 (server): HMM · Gaussian Process · LSTM features · TimesFM 2.5',
    lstm:   'Phase 3 (browser): Attention-LSTM · TF.js MC Dropout',
  }
  const GRADIENTS = {
    p1:     'linear-gradient(90deg, var(--accent-blue), #00d4aa)',
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

// ── Consolidated references ───────────────────────────────────────────────────

const REFERENCES = [
  {
    section: 'Forecast Methods',
    items: [
      'Oreshkin, B. et al. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. ICLR 2020. arxiv.org/abs/1905.10437',
      'Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384.',
      'Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. Review of Financial Studies, 15(4), 1137–1187.',
      'Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT Press. gaussianprocess.org/gpml/',
      'Matérn, B. (1960). Spatial Variation. [ν=5/2 kernel]. Meddelanden från Statens Skogsforskningsinstitut, 49(5).',
      'Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. ICLR 2015. arxiv.org/abs/1409.0473',
      'Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. ICML 33, 1050–1059.',
      'Fischer, T. & Krauss, C. (2018). Deep learning with LSTM for financial market predictions. EJOR, 270(2), 654–669.',
      'Das, A. et al. (2024). A decoder-only foundation model for time-series forecasting. ICML 2024. arxiv.org/abs/2310.10688',
    ],
  },
  {
    section: 'Ensemble & Uncertainty',
    items: [
      'Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.',
      'Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation, and Active Learning. NeurIPS 8, 231–238.',
      'Lakshminarayanan, B. et al. (2017). Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles. NeurIPS 30.',
      'Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1), 33–50.',
    ],
  },
  {
    section: 'Validation & Position Sizing',
    items: [
      'Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7 (Purged Walk-Forward CV). doi:10.1002/9781119482161',
      'Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management, 40(5), 94–107.',
      'Kelly, J.L. (1956). A new interpretation of information rate. Bell System Technical Journal, 35(4), 917–926.',
      'Ang, A., Hodrick, R.J., Xing, Y. & Zhang, X. (2006). The cross-section of volatility and expected returns. Journal of Finance, 61(1), 259–299.',
    ],
  },
]

function ForecastReferences() {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-lg border overflow-hidden"
      style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-left"
        style={{ background: 'transparent', cursor: 'pointer', border: 'none' }}
      >
        <span className="mono font-bold text-xs" style={{ color: 'var(--text-secondary)' }}>
          📄 References &amp; Methodology
        </span>
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          {open ? '▲ collapse' : '▼ expand'}
        </span>
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-3">
          <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)', opacity: 0.8 }}>
            All methods use walk-forward (expanding-window) OOS validation — random k-fold is never applied,
            as it leaks future data in time-series settings. The shadow holdout window (T−60 → T−30)
            provides the OOS R² that drives ensemble weighting.
          </p>
          {REFERENCES.map(({ section, items }) => (
            <div key={section}>
              <p className="mono font-bold" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
                {section}
              </p>
              {items.map((c, i) => (
                <p key={i} className="mono leading-relaxed" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.65, marginBottom: 2 }}>
                  📄 {c}
                </p>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function ForecastPanel({ backtest, portfolio, forecast }) {
  // `forecast` is hoisted to SplitView so in-flight Phase 1/2 state survives
  // a Results⇄Forecast tab switch (the panel itself unmounts on switch).
  const { results, meta, loading, error, run, hasData, newsContext, edgarContext, p1StartRef, nbeatsStartRef, p2StartRef, lstmStartRef } = forecast

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

  const isRunning = loading.phase1 || loading.nbeats || loading.phase2 || loading.lstm

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
                · Next 3 months · 5 research-backed methods
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
            <p className="mono text-sm mb-1" style={{ color: 'var(--text-primary)' }}>5 probabilistic forecast methods</p>
            <p className="text-xs max-w-sm" style={{ color: 'var(--text-secondary)' }}>
              N-BEATS · TimesFM 2.5 · HMM Regimes · Gaussian Process · Attention-LSTM
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

        {/* Composite chart — all 4 method medians on one chart.
            Collapsed by default AND unmounted when closed: the chart draws
            ~2500 historical points × 7 lines, which is a serious render /
            SVG-node load that was making the page unresponsive on click. */}
        {(hasData || loading.phase1) && (
          <CompositeChartSection>
            <ForecastComposite
              results={results}
              equityCurve={backtest?.equity_curve}
              forecastStart={meta?.forecast_start}
              shadowStart={meta?.shadow_forecast_start}
              shadowEnd={meta?.shadow_forecast_end}
              loading={loading.phase1}
              ensemble={ensemble}
            />
          </CompositeChartSection>
        )}

        {/* Stacked individual method cards — single shared panel */}
        {(hasData || isRunning) && (
          <div className="rounded-lg border cv-auto"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <div className="px-4 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
              <div className="mono text-xs font-bold" style={{ color: 'var(--text-primary)' }}>
                Individual Forecasts
              </div>
              <div className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>
                Per-method 63-day fan charts. Methods with OOS R² &lt; −0.5 are excluded from the composite ensemble but rendered here for transparency.
              </div>
            </div>
            {METHOD_ORDER.map((method, idx) => {
              const result   = resultMap[method]
              const inPhase1 = PHASE1_METHODS.has(method)
              const inPhase2 = PHASE2_METHODS.has(method)

              const isNbeatsBrowserLoading = method === 'nbeats' && loading.nbeats
              const isLstmBrowserLoading   = method === 'lstm'   && loading.lstm

              const sep = idx > 0 ? { borderTop: '1px solid var(--border)' } : {}

              if (!result && ((inPhase1 && loading.phase1) || (inPhase2 && loading.phase2))) {
                return <div key={method} style={sep}><LoadingCard method={method} compact /></div>
              }
              if (isNbeatsBrowserLoading || isLstmBrowserLoading) {
                return <div key={method} style={sep}><LoadingCard method={method} browserCompute compact /></div>
              }
              if (!result) return null
              return (
                <div key={method} style={sep}>
                  <ForecastMethodCard
                    result={result}
                    loading={inPhase2 && loading.phase2 && !result.forecast}
                    lastValue={lastValue}
                    compact
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

        {/* Consolidated references — collapsible */}
        {hasData && <ForecastReferences />}

      </div>
    </div>
  )
}
