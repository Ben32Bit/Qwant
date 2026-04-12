import React, { useState, useCallback } from 'react'
import ChatPanel from '../Chat/ChatPanel.jsx'
import ManualBuilderPanel from '../Chat/ManualBuilderPanel.jsx'
import ResultsPanel from '../Dashboard/ResultsPanel.jsx'

const TAB_STYLE_BASE = {
  fontSize: 12,
  fontFamily: 'var(--font-mono, monospace)',
  padding: '6px 16px',
  cursor: 'pointer',
  border: 'none',
  background: 'transparent',
  transition: 'color 0.15s, border-color 0.15s',
  borderBottom: '2px solid transparent',
  whiteSpace: 'nowrap',
}

function Tab({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        ...TAB_STYLE_BASE,
        color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
        borderBottomColor: active ? 'var(--accent-blue)' : 'transparent',
      }}
    >
      {label}
    </button>
  )
}

export default function SplitView({ messages, portfolio, backtest, displayConfig, loading, error, sendMessage, clearChat }) {
  const [mode, setMode] = useState('ai')           // 'ai' | 'manual'
  const [manualBacktest, setManualBacktest] = useState(null)
  const [manualPortfolio, setManualPortfolio] = useState(null)
  const [manualLoading, setManualLoading] = useState(false)

  const handleManualResult = useCallback((backtestData, portfolioData) => {
    setManualBacktest(backtestData)
    setManualPortfolio(portfolioData)
  }, [])

  const handleModeSwitch = (newMode) => {
    setMode(newMode)
  }

  // Decide which data to show in results panel based on active mode
  const activeBacktest  = mode === 'ai' ? backtest  : manualBacktest
  const activePortfolio = mode === 'ai' ? portfolio : manualPortfolio
  const activeLoading   = mode === 'ai' ? loading   : manualLoading
  // Manual mode uses a standard display config (all sections)
  const activeDisplayConfig = mode === 'ai'
    ? displayConfig
    : manualBacktest
      ? {
          sections: ['equity_curve', 'drawdown', 'metrics_summary', 'weight_drift', 'monthly_heatmap', 'correlation_matrix'],
          featured_metrics: ['cagr', 'sharpe', 'max_drawdown', 'volatility'],
        }
      : null

  return (
    <div className="flex h-full">
      {/* Left panel */}
      <div
        className="flex flex-col border-r"
        style={{
          width: '40%',
          minWidth: 320,
          borderColor: 'var(--border)',
          background: 'var(--bg-secondary)',
        }}
      >
        {/* Tab bar */}
        <div className="flex border-b px-2" style={{ borderColor: 'var(--border)', flexShrink: 0 }}>
          <Tab label="AI Chat"      active={mode === 'ai'}     onClick={() => handleModeSwitch('ai')} />
          <Tab label="Manual Build" active={mode === 'manual'} onClick={() => handleModeSwitch('manual')} />
        </div>

        {/* Panel content */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {mode === 'ai' ? (
            <ChatPanel
              messages={messages}
              loading={loading}
              onSend={sendMessage}
              portfolio={portfolio}
            />
          ) : (
            <ManualBuilderPanel
              onResult={handleManualResult}
              loading={manualLoading}
              setLoading={setManualLoading}
              aiPortfolio={portfolio}
              aiBacktest={backtest}
            />
          )}
        </div>
      </div>

      {/* Right — Results (60%) */}
      <div
        className="flex-1 flex flex-col overflow-hidden"
        style={{ background: 'var(--bg-primary)' }}
      >
        <ResultsPanel
          backtest={activeBacktest}
          portfolio={activePortfolio}
          displayConfig={activeDisplayConfig}
          loading={activeLoading}
        />
      </div>
    </div>
  )
}
