import React, { useState, useCallback, useEffect, Suspense, lazy } from 'react'
import UnifiedChatPanel from '../Chat/UnifiedChatPanel.jsx'
import ManualBuilderPanel from '../Chat/ManualBuilderPanel.jsx'
import ResultsPanel from '../Dashboard/ResultsPanel.jsx'
import ScreenerResults from '../Dashboard/ScreenerResults.jsx'
import DrawdownChart from '../Dashboard/DrawdownChart.jsx'
import MetricsCards from '../Dashboard/MetricsCards.jsx'

// Heavy panels / charts only mounted for specific views. Lazy-load so the
// initial JS bundle stays lean: ForecastPanel pulls in 6 method cards +
// ensemble/kelly/scenario panels + recharts; RotationEquityChart and
// FamaFrenchFactors are only reached via the Screener→Rotation flow.
const ForecastPanel       = lazy(() => import('../Dashboard/ForecastPanel.jsx'))
const RotationEquityChart = lazy(() => import('../Dashboard/RotationEquityChart.jsx'))
const FamaFrenchFactors   = lazy(() => import('../Dashboard/FamaFrenchFactors.jsx'))

// Shared fallback while a lazy chunk is fetching — small, low-chrome so it
// doesn't flash in-and-out noticeably on fast networks.
const LazyFallback = ({ label = 'Loading…' }) => (
  <div className="flex items-center justify-center py-8">
    <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{label}</span>
  </div>
)

// ── Tab definitions ───────────────────────────────────────────────────────────
const TABS = [
  { id: 'ai',     label: 'AI Assistant',  accent: 'var(--accent-blue)'  },
  { id: 'manual', label: 'Manual Build',  accent: 'var(--accent-green)' },
]

function Tab({ tab, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontSize: 12,
        fontFamily: 'var(--font-mono, monospace)',
        padding: '6px 16px',
        cursor: 'pointer',
        border: 'none',
        background: 'transparent',
        transition: 'color 0.15s, border-color 0.15s',
        borderBottom: `2px solid ${active ? tab.accent : 'transparent'}`,
        color: active ? tab.accent : 'var(--text-secondary)',
        whiteSpace: 'nowrap',
      }}
    >
      {tab.label}
    </button>
  )
}

// ── Inline rotation results panel ─────────────────────────────────────────────
function RotationPanel({ backtest, loading, topNHeld, onPortToManual }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="mono text-sm" style={{ color: 'var(--accent-purple)' }}>
          Running rotation backtest<span className="animate-pulse">…</span>
        </span>
      </div>
    )
  }
  if (!backtest) return null

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-4 space-y-4 fade-in">
        <div className="flex items-center justify-between">
          <div>
            <div className="mono font-bold text-sm" style={{ color: 'var(--accent-purple)' }}>◈ ROTATION BACKTEST</div>
            <div className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
              Top {topNHeld} · no-lookahead · each window uses previous winner
            </div>
          </div>
          <button
            onClick={onPortToManual}
            className="mono text-xs px-3 py-2 rounded border transition-colors"
            style={{ borderColor: 'var(--accent-green)', color: 'var(--accent-green)', background: 'transparent', cursor: 'pointer' }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,212,170,0.1)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
          >
            → Port to Manual Build
          </button>
        </div>

        {/* Featured metrics */}
        <div className="grid grid-cols-4 gap-2">
          {[
            { label: 'CAGR',   value: backtest.metrics?.cagr,        fmt: v => `${(v*100).toFixed(1)}%`, pos: true  },
            { label: 'Sharpe', value: backtest.metrics?.sharpe,       fmt: v => v.toFixed(2),             pos: true  },
            { label: 'Max DD', value: backtest.metrics?.max_drawdown, fmt: v => `${(v*100).toFixed(1)}%`, pos: false },
            { label: 'Vol',    value: backtest.metrics?.volatility,   fmt: v => `${(v*100).toFixed(1)}%`, pos: null  },
          ].map(({ label, value, fmt, pos }) => (
            <div key={label} className="rounded-lg border p-2 text-center"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
              <div className="text-xs mb-0.5" style={{ color: 'var(--text-secondary)' }}>{label}</div>
              <div className="mono font-bold text-base" style={{
                color: value == null ? 'var(--text-secondary)'
                  : pos === true  ? (value >= 0 ? 'var(--accent-green)' : 'var(--accent-red)')
                  : pos === false ? (value <= 0 ? 'var(--accent-green)' : 'var(--accent-red)')
                  : 'var(--text-primary)',
              }}>
                {value != null ? fmt(value) : '—'}
              </div>
            </div>
          ))}
        </div>

        <Suspense fallback={<LazyFallback label="Loading rotation chart…" />}>
          <RotationEquityChart
            equityCurve={backtest.equity_curve}
            benchmarkCurve={backtest.benchmark_curve}
            holdingSchedule={backtest.holding_schedule}
          />
        </Suspense>
        <DrawdownChart drawdownSeries={backtest.drawdown_series} loading={false} />
        <MetricsCards metrics={backtest.metrics} loading={false} />
        <Suspense fallback={<LazyFallback label="Loading factor analysis…" />}>
          <FamaFrenchFactors ff5={backtest.ff5_decomposition} loading={false} />
        </Suspense>
      </div>
    </div>
  )
}

// ── Right-panel tab bar (Results / Forecast) ──────────────────────────────────
const RIGHT_TABS = [
  { id: 'results',  label: 'Results',  accent: 'var(--accent-blue)'   },
  { id: 'forecast', label: 'Forecast', accent: 'var(--accent-yellow)' },
]

function RightTab({ tab, active, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        fontSize: 11,
        fontFamily: 'var(--font-mono, monospace)',
        padding: '5px 14px',
        cursor: disabled ? 'not-allowed' : 'pointer',
        border: 'none',
        background: 'transparent',
        transition: 'color 0.15s, border-color 0.15s',
        borderBottom: `2px solid ${active ? tab.accent : 'transparent'}`,
        color: active ? tab.accent : disabled ? 'var(--border)' : 'var(--text-secondary)',
        whiteSpace: 'nowrap',
        opacity: disabled ? 0.4 : 1,
      }}
    >
      {tab.label}
    </button>
  )
}

// ── Main ─────────────────────────────────────────────────────────────────────
export default function SplitView() {
  const [mode, setMode] = useState('ai')         // 'ai' | 'manual'
  const [rightTab, setRightTab] = useState('results')  // 'results' | 'forecast'

  // ── AI assistant state ────────────────────────────────────────────────────
  const [aiLoading, setAiLoading] = useState(false)
  // Last result from /api/unified/chat
  const [unifiedResult, setUnifiedResult] = useState(null)
  // Screener sub-state (rotation backtest triggered from ScreenerResults)
  const [rotationBacktest, setRotationBacktest] = useState(null)
  const [rotationLoading, setRotationLoading] = useState(false)
  const [rotationTopN, setRotationTopN] = useState(1)

  // ── Manual build state ────────────────────────────────────────────────────
  const [manualBacktest, setManualBacktest] = useState(null)
  const [manualPortfolio, setManualPortfolio] = useState(null)
  const [manualLoading, setManualLoading] = useState(false)

  // ── Import bridges ────────────────────────────────────────────────────────
  const [screenerImport, setScreenerImport] = useState(null)   // last-window tickers
  const [rotationImport, setRotationImport] = useState(null)   // full rotation strategy

  // Derived from unifiedResult
  const screenResult = unifiedResult?.type === 'screener' ? unifiedResult.screen_result : null
  const portfolio    = unifiedResult?.type === 'portfolio' ? unifiedResult.portfolio     : null
  const backtest     = unifiedResult?.type === 'portfolio' ? unifiedResult.backtest       : null
  const displayConfig = unifiedResult?.type === 'portfolio' ? unifiedResult.display_config : null

  // Clear rotation backtest whenever a new screen result arrives
  useEffect(() => {
    if (screenResult) setRotationBacktest(null)
  }, [screenResult])

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleUnifiedResult = useCallback((data) => {
    setUnifiedResult(data)
  }, [])

  const handleManualResult = useCallback((bt, pf) => {
    setManualBacktest(bt)
    setManualPortfolio(pf)
  }, [])

  const handleBacktestRotation = useCallback(async (topNHeld) => {
    if (!screenResult) return
    setRotationTopN(topNHeld)
    setRotationLoading(true)
    try {
      const res = await fetch('/api/screen/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          screen_result: screenResult,
          initial_capital: 10000,
          benchmark: 'SPY',
          top_n_held: topNHeld,
        }),
      })
      if (!res.ok) throw new Error('Rotation backtest failed')
      setRotationBacktest(await res.json())
    } catch (e) {
      console.error(e)
    } finally {
      setRotationLoading(false)
    }
  }, [screenResult])

  // Import screener's last window tickers → manual build
  const handleImportToManual = useCallback(() => {
    if (!screenResult?.windows?.length) return
    const lastWindow = screenResult.windows[screenResult.windows.length - 1]
    const topAssets = lastWindow.rankings
      .filter(r => r.rank <= screenResult.top_n)
      .map(r => r.ticker)
    const freqMap = { weekly: 'weekly', monthly: 'monthly', quarterly: 'quarterly', annually: 'annually' }
    setScreenerImport({
      tickers: topAssets,
      rebalance: freqMap[screenResult.window_freq] ?? 'quarterly',
      startDate: screenResult.windows[0]?.window_start,
      endDate: screenResult.windows[screenResult.windows.length - 1]?.window_end,
    })
    setRotationImport(null)
    setMode('manual')
  }, [screenResult])

  // Port full rotation strategy → manual build
  const handlePortRotationToManual = useCallback(() => {
    if (!screenResult || !rotationBacktest) return
    setRotationImport({
      screenResult,
      topN: rotationTopN,
      holdingSchedule: rotationBacktest.holding_schedule ?? [],
    })
    setScreenerImport(null)
    setMode('manual')
  }, [screenResult, rotationBacktest, rotationTopN])

  // ── Right panel content ───────────────────────────────────────────────────
  // Derive active backtest/portfolio based on which left-panel mode is live.
  // These are plain variable computations — no hooks — so they're safe to
  // reference anywhere in the render without dependency-array concerns.
  let activeBacktest, activePortfolio, activeDisplay, activeLoading
  if (mode === 'manual') {
    activeBacktest  = manualBacktest
    activePortfolio = manualPortfolio
    activeLoading   = manualLoading
    activeDisplay   = manualBacktest ? {
      sections: ['equity_curve', 'drawdown', 'metrics_summary', 'weight_drift', 'monthly_heatmap', 'correlation_matrix'],
      featured_metrics: ['cagr', 'sharpe', 'max_drawdown', 'volatility'],
    } : null
  } else {
    activeBacktest  = backtest
    activePortfolio = portfolio
    activeLoading   = aiLoading
    activeDisplay   = displayConfig
  }

  // Show the Results/Forecast tab bar only for portfolio backtests
  const isScreenerView  = !!(screenResult || rotationLoading)
  const isPortfolioView = !isScreenerView && !!(activeBacktest || activeLoading)

  return (
    <div className="flex h-full">

      {/* ── Left panel ───────────────────────────────────────────────────── */}
      <div
        className="flex flex-col border-r"
        style={{ width: '40%', minWidth: 320, borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
      >
        {/* Tab bar */}
        <div
          className="flex border-b px-2 overflow-x-auto"
          style={{ borderColor: 'var(--border)', flexShrink: 0, scrollbarWidth: 'none' }}
        >
          {TABS.map(tab => (
            <Tab key={tab.id} tab={tab} active={mode === tab.id} onClick={() => setMode(tab.id)} />
          ))}
        </div>

        {/* Panel */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {mode === 'ai' && (
            <UnifiedChatPanel
              onResult={handleUnifiedResult}
              loading={aiLoading}
              setLoading={setAiLoading}
            />
          )}
          {mode === 'manual' && (
            <ManualBuilderPanel
              onResult={handleManualResult}
              loading={manualLoading}
              setLoading={setManualLoading}
              aiPortfolio={portfolio}
              aiBacktest={backtest}
              screenerImport={screenerImport}
              rotationImport={rotationImport}
            />
          )}
        </div>
      </div>

      {/* ── Right panel ──────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden" style={{ background: 'var(--bg-primary)' }}>

        {/* Results / Forecast tab bar — portfolio view only */}
        {isPortfolioView && (
          <div
            className="flex border-b px-2"
            style={{ borderColor: 'var(--border)', flexShrink: 0, background: 'var(--bg-secondary)' }}
          >
            {RIGHT_TABS.map(tab => (
              <RightTab
                key={tab.id}
                tab={tab}
                active={rightTab === tab.id}
                onClick={() => setRightTab(tab.id)}
                disabled={tab.id === 'forecast' && !activeBacktest}
              />
            ))}
          </div>
        )}

        {/* ── Screener view ─────────────────────────────────────────────── */}
        {isScreenerView && (
          <div className="flex h-full overflow-hidden">
            <div className="flex-1 overflow-hidden border-r" style={{ borderColor: 'var(--border)' }}>
              <ScreenerResults
                screenResult={screenResult}
                onBacktestRotation={handleBacktestRotation}
                onImportToManual={handleImportToManual}
                backtestLoading={rotationLoading}
              />
            </div>
            {(rotationBacktest || rotationLoading) && (
              <div style={{ width: '50%', overflow: 'hidden' }}>
                <RotationPanel
                  backtest={rotationBacktest}
                  loading={rotationLoading}
                  topNHeld={rotationTopN}
                  onPortToManual={handlePortRotationToManual}
                />
              </div>
            )}
          </div>
        )}

        {/* ── Portfolio view ─────────────────────────────────────────────── */}
        {isPortfolioView && (
          <div className="flex-1 min-h-0 overflow-hidden">
            {rightTab === 'results' && (
              <ResultsPanel
                backtest={activeBacktest}
                portfolio={activePortfolio}
                displayConfig={activeDisplay}
                loading={activeLoading}
              />
            )}
            {rightTab === 'forecast' && (
              <Suspense fallback={<LazyFallback label="Loading forecast engine…" />}>
                <ForecastPanel
                  backtest={activeBacktest}
                  portfolio={activePortfolio}
                />
              </Suspense>
            )}
          </div>
        )}

        {/* ── Empty state ────────────────────────────────────────────────── */}
        {!isScreenerView && !isPortfolioView && (
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <div className="mono text-5xl mb-6 opacity-20 select-none" style={{ color: 'var(--accent-blue)' }}>▷</div>
            <h2 className="mono font-bold text-lg mb-2" style={{ color: 'var(--text-primary)' }}>Results appear here</h2>
            <p className="text-sm max-w-sm" style={{ color: 'var(--text-secondary)' }}>
              Ask the AI to build a portfolio or screen assets by performance — results populate here instantly.
            </p>
          </div>
        )}

      </div>
    </div>
  )
}
