import React from 'react'
import SplitView from './SplitView.jsx'

export default function App() {
  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <header
        className="flex items-center justify-between px-6 py-3 border-b"
        style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
      >
        <div className="flex items-center gap-3">
          <span className="mono font-bold text-lg" style={{ color: 'var(--accent-blue)' }}>
            ▶ QWANT
          </span>
          <span style={{ color: 'var(--text-secondary)' }} className="text-sm">
            AI Investment Analyst
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs mono" style={{ color: 'var(--text-secondary)' }}>
          <span>Free · No signup · Real market data</span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 min-h-0">
        <SplitView />
      </div>
    </div>
  )
}
