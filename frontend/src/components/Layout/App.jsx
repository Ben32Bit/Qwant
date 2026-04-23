import React, { useEffect } from 'react'
import SplitView from './SplitView.jsx'
import TickerBar from './TickerBar.jsx'
import { warmupBackend } from '../../utils/api.js'

export default function App() {
  // Wake the Railway container on mount so it's warm by the time the user
  // runs their first backtest or forecast. Cold boots cost 30-60 s.
  useEffect(() => {
    warmupBackend()
  }, [])

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-primary)' }}>
      {/* Rolling market ticker — live prices + GDELT headlines + Reddit trending */}
      <TickerBar />

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
            All you ever Qwanted
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs mono" style={{ color: 'var(--text-secondary)' }}>
          <span>Free · No signup · Real market data</span>
          <a
            href={`https://github.com/Ben32Bit/Qwant/commit/${__GIT_SHA__}`}
            target="_blank"
            rel="noopener noreferrer"
            title={`Deployed commit ${__GIT_SHA__} (${__GIT_DATE__})`}
            style={{
              padding: '2px 8px',
              borderRadius: 4,
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
              textDecoration: 'none',
              background: 'var(--bg-card)',
            }}
          >
            v·{__GIT_SHA__}·{__GIT_DATE__}
          </a>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 min-h-0">
        <SplitView />
      </div>
    </div>
  )
}
