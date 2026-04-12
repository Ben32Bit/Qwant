import React from 'react'
import { fmtPct } from '../../utils/formatters.js'

export default function PortfolioCard({ portfolio }) {
  if (!portfolio) return null

  const totalWeight = portfolio.assets.reduce((sum, a) => sum + a.weight, 0)
  const isLeveraged = totalWeight > 1.01
  const hasShorts = portfolio.assets.some((a) => a.weight < 0)

  return (
    <div
      className="rounded-lg border mx-4 my-2 overflow-hidden"
      style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2 border-b"
        style={{ borderColor: 'var(--border)' }}
      >
        <span className="mono text-xs font-bold" style={{ color: 'var(--accent-blue)' }}>
          PORTFOLIO
        </span>
        <div className="flex items-center gap-2">
          {isLeveraged && (
            <span
              className="mono text-xs px-2 py-0.5 rounded"
              style={{ background: 'rgba(255, 212, 59, 0.15)', color: 'var(--accent-yellow)' }}
            >
              {(totalWeight * 100).toFixed(0)}% LEVERAGED
            </span>
          )}
          {hasShorts && (
            <span
              className="mono text-xs px-2 py-0.5 rounded"
              style={{ background: 'rgba(255, 71, 87, 0.15)', color: 'var(--accent-red)' }}
            >
              SHORT
            </span>
          )}
          <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
            {portfolio.rebalance_frequency} rebalance
          </span>
        </div>
      </div>

      {/* Assets */}
      <div className="p-3 space-y-1">
        {portfolio.assets.map((asset) => (
          <div key={asset.ticker} className="flex items-center gap-2">
            {/* Weight bar */}
            <div className="w-20 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-secondary)' }}>
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(Math.abs(asset.weight) * 100, 100)}%`,
                  background: asset.weight < 0 ? 'var(--accent-red)' : 'var(--accent-green)',
                }}
              />
            </div>
            <span
              className="mono text-xs font-bold w-12"
              style={{ color: asset.weight < 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}
            >
              {fmtPct(asset.weight, 0, true)}
            </span>
            <span className="mono text-xs font-bold" style={{ color: 'var(--text-primary)' }}>
              {asset.ticker}
            </span>
            {asset.rationale && (
              <span className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
                {asset.rationale}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div
        className="flex items-center gap-4 px-3 py-1.5 border-t"
        style={{ borderColor: 'var(--border)' }}
      >
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          vs {portfolio.benchmark}
        </span>
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          {portfolio.start_date?.slice(0, 4)} – {portfolio.end_date?.slice(0, 4)}
        </span>
        {portfolio.strategy !== 'custom' && (
          <span
            className="mono text-xs px-2 py-0.5 rounded"
            style={{ background: 'rgba(74, 158, 255, 0.15)', color: 'var(--accent-blue)' }}
          >
            {portfolio.strategy.replace('_', ' ')}
          </span>
        )}
      </div>
    </div>
  )
}
