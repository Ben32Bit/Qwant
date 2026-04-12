import React, { useState, useMemo } from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts'
import { AXIS_STYLE, TOOLTIP_STYLE, GRID_STYLE, CHART_COLORS } from '../../utils/chartConfig.js'
import { fmtPct, fmtRatio } from '../../utils/formatters.js'

const TABS = [
  { key: 'sharpe',     label: 'Rolling Sharpe',      color: CHART_COLORS.portfolio, fmt: v => fmtRatio(v), ref: 0 },
  { key: 'volatility', label: 'Rolling Volatility',   color: CHART_COLORS.overlay1,  fmt: v => fmtPct(v),   ref: null },
  { key: 'beta',       label: 'Rolling Beta',         color: CHART_COLORS.overlay2,  fmt: v => fmtRatio(v), ref: 1 },
]

function CustomTooltip({ active, payload, label, fmt }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '10px 14px' }}>
      <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      <div className="mono text-sm" style={{ color: payload[0]?.color }}>
        {fmt(payload[0]?.value)}
      </div>
    </div>
  )
}

export default function RollingMetrics({ rollingMetrics, hasBenchmark, loading }) {
  const availableTabs = useMemo(() => {
    if (!rollingMetrics) return []
    return TABS.filter(t => {
      if (t.key === 'beta' && !hasBenchmark) return false
      return rollingMetrics[t.key]?.length > 0
    })
  }, [rollingMetrics, hasBenchmark])

  const [activeTab, setActiveTab] = useState(0)

  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', height: 220 }}>
        <div className="skeleton h-4 w-32 mb-4" />
        <div className="skeleton h-full w-full rounded" style={{ height: 160 }} />
      </div>
    )
  }

  if (!rollingMetrics || availableTabs.length === 0) return null

  const tab = availableTabs[Math.min(activeTab, availableTabs.length - 1)]
  const data = rollingMetrics[tab.key] || []

  const tickCount = 6
  const tickIndices = Array.from({ length: tickCount }, (_, i) =>
    Math.floor((i / (tickCount - 1)) * (data.length - 1))
  )
  const tickDates = tickIndices.map(i => data[i]?.date)

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
          Rolling Metrics
        </h3>
        <div className="flex gap-1">
          {availableTabs.map((t, i) => (
            <button
              key={t.key}
              onClick={() => setActiveTab(i)}
              className="mono text-xs px-2 py-1 rounded border transition-colors"
              style={{
                borderColor: activeTab === i ? t.color : 'var(--border)',
                color: activeTab === i ? t.color : 'var(--text-secondary)',
                background: 'transparent',
              }}
            >
              {t.label.replace('Rolling ', '')}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis
            dataKey="date"
            ticks={tickDates}
            tick={AXIS_STYLE.tick}
            axisLine={AXIS_STYLE.axisLine}
            tickLine={AXIS_STYLE.tickLine}
            tickFormatter={d => d?.slice(0, 7)}
          />
          <YAxis
            tick={AXIS_STYLE.tick}
            axisLine={AXIS_STYLE.axisLine}
            tickLine={AXIS_STYLE.tickLine}
            tickFormatter={v => tab.key === 'volatility' ? `${(v * 100).toFixed(0)}%` : v.toFixed(1)}
          />
          <Tooltip content={<CustomTooltip fmt={tab.fmt} />} />
          {tab.ref != null && (
            <ReferenceLine y={tab.ref} stroke="var(--border)" strokeDasharray="4 2" />
          )}
          <Line
            type="monotone"
            dataKey="value"
            stroke={tab.color}
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>

      <p className="mono text-xs mt-2" style={{ color: 'var(--text-secondary)' }}>
        {tab.key === 'sharpe' && 'Window: 252 trading days (1 year)'}
        {tab.key === 'volatility' && 'Window: 60 trading days · Annualised'}
        {tab.key === 'beta' && 'Window: 126 trading days vs benchmark'}
      </p>
    </div>
  )
}
