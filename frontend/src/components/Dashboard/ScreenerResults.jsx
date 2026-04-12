import React, { useState, useMemo } from 'react'
import { fmtPct, fmtRatio } from '../../utils/formatters.js'

const RANK_COLORS = ['#a855f7', '#7c3aed', '#6d28d9', '#5b21b6', '#4c1d95']
const RANK_BG = [
  'rgba(168,85,247,0.18)',
  'rgba(124,58,237,0.12)',
  'rgba(109,40,217,0.08)',
  'rgba(91,33,182,0.06)',
  'rgba(76,29,149,0.04)',
]

function isPercent(metric) {
  return ['return', 'volatility', 'max_drawdown', 'momentum_3m'].includes(metric)
}

function fmtMetric(val, metric) {
  if (val == null) return '—'
  return isPercent(metric) ? fmtPct(val) : fmtRatio(val)
}

// ── Winner timeline strip ─────────────────────────────────────────────────────
function WinnerTimeline({ windows, top_n }) {
  const shown = Math.min(top_n, 3)
  return (
    <div className="overflow-x-auto">
      <div className="flex gap-2 pb-2" style={{ minWidth: 'max-content' }}>
        {windows.map((w) => (
          <div
            key={w.label}
            className="rounded-lg border flex-shrink-0"
            style={{
              width: 108,
              background: 'var(--bg-card)',
              borderColor: 'rgba(168,85,247,0.25)',
            }}
          >
            {/* Window label */}
            <div
              className="mono text-xs font-bold text-center py-1.5 rounded-t-lg"
              style={{ background: 'rgba(168,85,247,0.12)', color: 'var(--accent-purple)' }}
            >
              {w.label}
            </div>
            {/* Top N rankings */}
            {w.rankings.slice(0, shown).map((r) => (
              <div
                key={r.ticker}
                className="flex items-center justify-between px-2 py-1"
                style={{ borderTop: '1px solid var(--border)' }}
              >
                <div className="flex items-center gap-1.5">
                  <span
                    className="mono text-xs font-bold"
                    style={{ color: RANK_COLORS[r.rank - 1] ?? 'var(--text-secondary)' }}
                  >
                    #{r.rank}
                  </span>
                  <span className="mono text-xs font-bold" style={{ color: 'var(--text-primary)' }}>
                    {r.ticker}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Detailed table ────────────────────────────────────────────────────────────
function DetailTable({ windows, tickers, metric, top_n }) {
  // Columns = windows, rows = tickers
  // Cell value = rank + metric value
  const windowMap = useMemo(() => {
    const m = {}
    for (const w of windows) {
      m[w.label] = {}
      for (const r of w.rankings) {
        m[w.label][r.ticker] = r
      }
    }
    return m
  }, [windows])

  // Sort tickers by number of #1 finishes desc
  const sortedTickers = useMemo(() => {
    return [...tickers].sort((a, b) => {
      const winsA = windows.filter(w => w.winner === a).length
      const winsB = windows.filter(w => w.winner === b).length
      return winsB - winsA
    })
  }, [tickers, windows])

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs" style={{ borderCollapse: 'collapse', minWidth: 'max-content' }}>
        <thead>
          <tr>
            <th
              className="mono text-left px-3 py-2 sticky left-0"
              style={{ background: 'var(--bg-card)', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)', minWidth: 80 }}
            >
              Asset
            </th>
            <th
              className="mono text-center px-2 py-2"
              style={{ background: 'var(--bg-card)', color: 'var(--accent-purple)', borderBottom: '1px solid var(--border)', minWidth: 48 }}
            >
              Wins
            </th>
            {windows.map((w) => (
              <th
                key={w.label}
                className="mono text-center px-2 py-2"
                style={{ background: 'var(--bg-card)', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)', minWidth: 88 }}
              >
                {w.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedTickers.map((ticker) => {
            const wins = windows.filter(w => w.winner === ticker).length
            return (
              <tr key={ticker} style={{ borderBottom: '1px solid var(--border)' }}>
                <td
                  className="mono font-bold px-3 py-1.5 sticky left-0"
                  style={{ background: 'var(--bg-card)', color: 'var(--text-primary)' }}
                >
                  {ticker}
                </td>
                <td className="mono text-center px-2 py-1.5" style={{ color: wins > 0 ? 'var(--accent-purple)' : 'var(--text-secondary)' }}>
                  {wins > 0 ? wins : '—'}
                </td>
                {windows.map((w) => {
                  const r = windowMap[w.label]?.[ticker]
                  if (!r) return (
                    <td key={w.label} className="text-center px-2 py-1.5" style={{ color: 'var(--text-secondary)' }}>—</td>
                  )
                  const rank = r.rank
                  const isTopN = rank <= top_n
                  return (
                    <td
                      key={w.label}
                      className="text-center px-2 py-1.5"
                      style={{
                        background: isTopN ? (RANK_BG[rank - 1] ?? 'transparent') : 'transparent',
                        color: isTopN ? (RANK_COLORS[rank - 1] ?? 'var(--text-primary)') : 'var(--text-secondary)',
                      }}
                    >
                      <div className="mono font-bold text-xs">#{rank}</div>
                      <div className="mono text-xs" style={{ opacity: 0.8 }}>
                        {fmtMetric(r.metric_value, metric)}
                      </div>
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function ScreenerResults({ screenResult, onBacktestRotation, onImportToManual, backtestLoading }) {
  const [view, setView] = useState('timeline') // 'timeline' | 'table'

  if (!screenResult) return null

  const { windows, metric, metric_label, tickers, top_n, screen_description } = screenResult

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-4 space-y-4 fade-in">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="mono font-bold text-sm" style={{ color: 'var(--accent-purple)' }}>
              ◈ SCREEN RESULTS
            </h2>
            {screen_description && (
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                {screen_description}
              </p>
            )}
          </div>
          {/* View toggle */}
          <div className="flex gap-1">
            {['timeline', 'table'].map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className="mono text-xs px-3 py-1.5 rounded border transition-colors"
                style={{
                  borderColor: view === v ? 'var(--accent-purple)' : 'var(--border)',
                  color: view === v ? 'var(--accent-purple)' : 'var(--text-secondary)',
                  background: view === v ? 'rgba(168,85,247,0.1)' : 'transparent',
                }}
              >
                {v === 'timeline' ? '▦ Cards' : '⊞ Table'}
              </button>
            ))}
          </div>
        </div>

        {/* Meta strip */}
        <div className="flex gap-3 flex-wrap">
          {[
            { label: 'Metric', value: metric_label },
            { label: 'Windows', value: windows.length },
            { label: 'Universe', value: `${tickers.length} assets` },
            { label: 'Top N shown', value: `${top_n} per window` },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded border px-3 py-1.5"
              style={{ borderColor: 'rgba(168,85,247,0.2)', background: 'rgba(168,85,247,0.05)' }}
            >
              <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{label}: </span>
              <span className="mono text-xs font-bold" style={{ color: 'var(--accent-purple)' }}>{value}</span>
            </div>
          ))}
        </div>

        {/* Main view */}
        <div
          className="rounded-lg border p-4"
          style={{ background: 'var(--bg-card)', borderColor: 'rgba(168,85,247,0.2)' }}
        >
          <div className="mono text-xs font-bold mb-3 uppercase tracking-widest" style={{ color: 'var(--accent-purple)' }}>
            {view === 'timeline' ? 'Window-by-window rankings' : 'Full ranking table'}
          </div>
          {view === 'timeline'
            ? <WinnerTimeline windows={windows} top_n={top_n} />
            : <DetailTable windows={windows} tickers={tickers} metric={metric} top_n={top_n} />
          }
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 flex-wrap">
          <button
            onClick={() => onBacktestRotation(1)}
            disabled={backtestLoading}
            className="mono text-xs px-4 py-2 rounded border font-bold transition-colors"
            style={{
              borderColor: 'var(--accent-purple)',
              color: 'var(--accent-purple)',
              background: backtestLoading ? 'rgba(168,85,247,0.05)' : 'rgba(168,85,247,0.1)',
              cursor: backtestLoading ? 'wait' : 'pointer',
              opacity: backtestLoading ? 0.7 : 1,
            }}
          >
            {backtestLoading ? '⟳ Running...' : '▶ Backtest Rotation (Top 1)'}
          </button>
          {top_n >= 3 && (
            <button
              onClick={() => onBacktestRotation(3)}
              disabled={backtestLoading}
              className="mono text-xs px-4 py-2 rounded border transition-colors"
              style={{
                borderColor: 'var(--accent-purple)',
                color: 'var(--accent-purple)',
                background: 'transparent',
                cursor: backtestLoading ? 'wait' : 'pointer',
                opacity: backtestLoading ? 0.7 : 1,
              }}
            >
              ▶ Backtest Rotation (Top 3)
            </button>
          )}
          <button
            onClick={() => onImportToManual()}
            className="mono text-xs px-4 py-2 rounded border transition-colors"
            style={{
              borderColor: 'var(--border)',
              color: 'var(--text-secondary)',
              background: 'transparent',
              cursor: 'pointer',
            }}
          >
            → Import to Manual Build
          </button>
        </div>

        {/* Caveat */}
        <div
          className="rounded border px-3 py-2 text-xs"
          style={{ borderColor: 'rgba(168,85,247,0.15)', color: 'var(--text-secondary)' }}
        >
          <span className="mono font-bold" style={{ color: 'var(--accent-yellow)' }}>⚠ </span>
          Rotation backtest uses previous-window winners — no lookahead bias.
          Historical screening results are retroactive and do not guarantee future performance.
        </div>

      </div>
    </div>
  )
}
