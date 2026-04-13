import React, { useState, useCallback, useEffect } from 'react'
import ChatPanel from '../Chat/ChatPanel.jsx'
import ManualBuilderPanel from '../Chat/ManualBuilderPanel.jsx'
import StockScreenerPanel from '../Chat/StockScreenerPanel.jsx'
import ResultsPanel from '../Dashboard/ResultsPanel.jsx'
import ScreenerResults from '../Dashboard/ScreenerResults.jsx'
import RotationEquityChart from '../Dashboard/RotationEquityChart.jsx'
import DrawdownChart from '../Dashboard/DrawdownChart.jsx'
import MetricsCards from '../Dashboard/MetricsCards.jsx'

// ── Tab definitions ───────────────────────────────────────────────────────────
const TABS = [
  { id: 'portfolio', label: 'AI Portfolio Builder', accent: 'var(--accent-blue)' },
  { id: 'screener',  label: 'AI Stock Screener',    accent: 'var(--accent-purple)' },
  { id: 'manual',    label: 'Manual Build',          accent: 'var(--accent-green)' },
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

// ── Rotation results panel (right side of screener) ───────────────────────────
function RotationPanel({ backtest, loading, screenResult, topNHeld, onPortToManual }) {
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

        {/* Header + Port button */}
        <div className="flex items-center justify-between">
          <div>
            <div className="mono font-bold text-sm" style={{ color: 'var(--accent-purple)' }}>
              ◈ ROTATION BACKTEST
            </div>
            <div className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
              Top {topNHeld} · no-lookahead · each window uses previous winner
            </div>
          </div>
          <button
            onClick={onPortToManual}
            className="mono text-xs px-3 py-2 rounded border transition-colors"
            style={{
              borderColor: 'var(--accent-green)',
              color: 'var(--accent-green)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,212,170,0.1)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
          >
            → Port to Manual Build
          </button>
        </div>

        {/* Featured metrics */}
        <div className="grid grid-cols-4 gap-2">
          {[
            { label: 'CAGR',     value: backtest.metrics?.cagr,         fmt: v => `${(v*100).toFixed(1)}%`, pos: true },
            { label: 'Sharpe',   value: backtest.metrics?.sharpe,        fmt: v => v.toFixed(2),              pos: true },
            { label: 'Max DD',   value: backtest.metrics?.max_drawdown,  fmt: v => `${(v*100).toFixed(1)}%`, pos: false },
            { label: 'Vol',      value: backtest.metrics?.volatility,    fmt: v => `${(v*100).toFixed(1)}%`, pos: null },
          ].map(({ label, value, fmt, pos }) => (
            <div
              key={label}
              className="rounded-lg border p-2 text-center"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
            >
              <div className="text-xs mb-0.5" style={{ color: 'var(--text-secondary)' }}>{label}</div>
              <div
                className="mono font-bold text-base"
                style={{
                  color: value == null ? 'var(--text-secondary)'
                    : pos === true  ? (value >= 0 ? 'var(--accent-green)' : 'var(--accent-red)')
                    : pos === false ? (value <= 0 ? 'var(--accent-green)' : 'var(--accent-red)')
                    : 'var(--text-primary)',
                }}
              >
                {value != null ? fmt(value) : '—'}
              </div>
            </div>
          ))}
        </div>

        {/* Rotation equity chart with holding bands */}
        <RotationEquityChart
          equityCurve={backtest.equity_curve}
          benchmarkCurve={backtest.benchmark_curve}
          holdingSchedule={backtest.holding_schedule}
        />

        {/* Drawdown */}
        <DrawdownChart drawdownSeries={backtest.drawdown_series} loading={false} />

        {/* Metrics */}
        <MetricsCards metrics={backtest.metrics} loading={false} />
      </div>
    </div>
  )
}

// ── Main ─────────────────────────────────────────────────────────────────────
export default function SplitView({ messages, portfolio, backtest, displayConfig, loading, error, sendMessage, clearChat }) {
  const [mode, setMode] = useState('portfolio')   // 'portfolio' | 'screener' | 'manual'

  // Portfolio builder state
  const [manualBacktest, setManualBacktest]   = useState(null)
  const [manualPortfolio, setManualPortfolio] = useState(null)
  const [manualLoading, setManualLoading]     = useState(false)

  // Screener state
  const [screenResult, setScreenResult]           = useState(null)
  const [screenerLoading, setScreenerLoading]     = useState(false)
  const [rotationBacktest, setRotationBacktest]   = useState(null)
  const [rotationLoading, setRotationLoading]     = useState(false)
  const [rotationTopN, setRotationTopN]           = useState(1)

  // For pre-populating manual build from screener (last-window tickers)
  const [screenerImport, setScreenerImport] = useState(null)
  // For porting the full rotation strategy to manual build
  const [rotationImport, setRotationImport] = useState(null)

  // Apply data-mode to root so CSS variables switch (screener = purple theme)
  useEffect(() => {
    const root = document.getElementById('root') || document.body
    if (mode === 'screener') {
      root.setAttribute('data-mode', 'screener')
    } else {
      root.removeAttribute('data-mode')
    }
    return () => root.removeAttribute('data-mode')
  }, [mode])

  const handleManualResult = useCallback((backtestData, portfolioData) => {
    setManualBacktest(backtestData)
    setManualPortfolio(portfolioData)
  }, [])

  const handleScreenResult = useCallback((result) => {
    setScreenResult(result)
    setRotationBacktest(null)   // clear old rotation backtest when new screen arrives
  }, [])

  // Run rotation backtest from screener results
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
      const data = await res.json()
      setRotationBacktest(data)
    } catch (e) {
      console.error(e)
    } finally {
      setRotationLoading(false)
    }
  }, [screenResult])

  // Import screener's last window top assets → manual build (ticker grid only)
  const handleImportToManual = useCallback(() => {
    if (!screenResult?.windows?.length) return
    const lastWindow = screenResult.windows[screenResult.windows.length - 1]
    const topAssets = lastWindow.rankings
      .filter(r => r.rank <= screenResult.top_n)
      .map(r => r.ticker)

    const freqMap = {
      weekly: 'weekly', monthly: 'monthly',
      quarterly: 'quarterly', annually: 'annually',
    }
    const rebalance = freqMap[screenResult.window_freq] ?? 'quarterly'

    setScreenerImport({
      tickers: topAssets,
      rebalance,
      startDate: screenResult.windows[0]?.window_start,
      endDate: screenResult.windows[screenResult.windows.length - 1]?.window_end,
    })
    setRotationImport(null)  // clear rotation import when importing by tickers
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
    setScreenerImport(null)  // clear ticker import
    setMode('manual')
  }, [screenResult, rotationBacktest, rotationTopN])

  // ── Active right-panel data ───────────────────────────────────────────────
  const rightPanel = (() => {
    if (mode === 'screener') {
      return { type: 'screener' }
    }
    if (mode === 'portfolio') {
      return { type: 'backtest', backtest, portfolio, displayConfig, loading }
    }
    // manual
    return {
      type: 'backtest',
      backtest: manualBacktest,
      portfolio: manualPortfolio,
      loading: manualLoading,
      displayConfig: manualBacktest
        ? {
            sections: ['equity_curve', 'drawdown', 'metrics_summary', 'weight_drift', 'monthly_heatmap', 'correlation_matrix'],
            featured_metrics: ['cagr', 'sharpe', 'max_drawdown', 'volatility'],
          }
        : null,
    }
  })()

  return (
    <div className="flex h-full">

      {/* ── Left panel ───────────────────────────────────────────────────── */}
      <div
        className="flex flex-col border-r"
        style={{
          width: '40%',
          minWidth: 320,
          borderColor: 'var(--border)',
          background: 'var(--bg-secondary)',
          transition: 'background 0.25s',
        }}
      >
        {/* Tab bar */}
        <div
          className="flex border-b px-2 overflow-x-auto"
          style={{ borderColor: 'var(--border)', flexShrink: 0, scrollbarWidth: 'none' }}
        >
          {TABS.map((tab) => (
            <Tab
              key={tab.id}
              tab={tab}
              active={mode === tab.id}
              onClick={() => setMode(tab.id)}
            />
          ))}
        </div>

        {/* Panel content */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {mode === 'portfolio' && (
            <ChatPanel
              messages={messages}
              loading={loading}
              onSend={sendMessage}
              portfolio={portfolio}
            />
          )}
          {mode === 'screener' && (
            <StockScreenerPanel
              onScreenResult={handleScreenResult}
              loading={screenerLoading}
              setLoading={setScreenerLoading}
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
      <div
        className="flex-1 flex flex-col overflow-hidden"
        style={{ background: 'var(--bg-primary)', transition: 'background 0.25s' }}
      >
        {mode === 'screener' ? (
          screenResult || rotationLoading ? (
            <div className="flex h-full">
              {/* Screener results (left ~50%) */}
              <div className="flex-1 overflow-hidden border-r" style={{ borderColor: 'var(--border)' }}>
                <ScreenerResults
                  screenResult={screenResult}
                  onBacktestRotation={handleBacktestRotation}
                  onImportToManual={handleImportToManual}
                  backtestLoading={rotationLoading}
                />
              </div>
              {/* Rotation backtest results (right ~50%) */}
              {(rotationBacktest || rotationLoading) && (
                <div style={{ width: '50%', overflow: 'hidden' }}>
                  <RotationPanel
                    backtest={rotationBacktest}
                    loading={rotationLoading}
                    screenResult={screenResult}
                    topNHeld={rotationTopN}
                    onPortToManual={handlePortRotationToManual}
                  />
                </div>
              )}
            </div>
          ) : (
            // Empty screener state
            <div className="flex flex-col items-center justify-center h-full text-center px-8">
              <div className="mono text-5xl mb-6 opacity-20 select-none" style={{ color: 'var(--accent-purple)' }}>◈</div>
              <h2 className="mono font-bold text-lg mb-2" style={{ color: 'var(--text-primary)' }}>Screener results appear here</h2>
              <p className="text-sm max-w-sm" style={{ color: 'var(--text-secondary)' }}>
                Ask which assets performed best in any time window. Results show rank tables and link directly to backtesting.
              </p>
            </div>
          )
        ) : (
          <ResultsPanel
            backtest={rightPanel.backtest}
            portfolio={rightPanel.portfolio}
            displayConfig={rightPanel.displayConfig}
            loading={rightPanel.loading}
          />
        )}
      </div>
    </div>
  )
}
