import React, { useState, useCallback, useEffect } from 'react'
import ChatPanel from '../Chat/ChatPanel.jsx'
import ManualBuilderPanel from '../Chat/ManualBuilderPanel.jsx'
import StockScreenerPanel from '../Chat/StockScreenerPanel.jsx'
import ResultsPanel from '../Dashboard/ResultsPanel.jsx'
import ScreenerResults from '../Dashboard/ScreenerResults.jsx'

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

  // For pre-populating manual build from screener
  const [screenerImport, setScreenerImport] = useState(null)

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

  // Import screener's last window top assets → manual build
  const handleImportToManual = useCallback(() => {
    if (!screenResult?.windows?.length) return
    const lastWindow = screenResult.windows[screenResult.windows.length - 1]
    const topAssets = lastWindow.rankings
      .filter(r => r.rank <= screenResult.top_n)
      .map(r => r.ticker)

    // Map window_freq → rebalance_frequency
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
    setMode('manual')
  }, [screenResult])

  // ── Active right-panel data ───────────────────────────────────────────────
  const rightPanel = (() => {
    if (mode === 'screener') {
      return { type: 'screener' }
    }
    if (mode === 'ai') {
      return { type: 'backtest', backtest, portfolio, displayConfig, loading }
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
              {/* Screener results (left 55%) */}
              <div className="flex-1 overflow-hidden border-r" style={{ borderColor: 'var(--border)' }}>
                <ScreenerResults
                  screenResult={screenResult}
                  onBacktestRotation={handleBacktestRotation}
                  onImportToManual={handleImportToManual}
                  backtestLoading={rotationLoading}
                />
              </div>
              {/* Rotation backtest results (right 45%) */}
              {(rotationBacktest || rotationLoading) && (
                <div style={{ width: '45%', overflow: 'hidden' }}>
                  <ResultsPanel
                    backtest={rotationBacktest}
                    portfolio={null}
                    displayConfig={rotationBacktest ? {
                      sections: ['equity_curve', 'drawdown', 'metrics_summary', 'monthly_heatmap'],
                      featured_metrics: ['cagr', 'sharpe', 'max_drawdown', 'volatility'],
                    } : null}
                    loading={rotationLoading}
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
