import React, { useMemo } from 'react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'
import { AXIS_STYLE, TOOLTIP_STYLE, GRID_STYLE } from '../../utils/chartConfig.js'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '10px 14px' }}>
      <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      <div className="mono text-sm" style={{ color: 'var(--accent-red)' }}>
        Drawdown: {payload[0]?.value?.toFixed(2)}%
      </div>
    </div>
  )
}

export default function DrawdownChart({ drawdownSeries, loading }) {
  const data = useMemo(() => {
    if (!drawdownSeries) return []
    const step = Math.max(1, Math.floor(drawdownSeries.length / 500))
    return drawdownSeries
      .filter((_, i) => i % step === 0 || i === drawdownSeries.length - 1)
      .map((pt) => ({ date: pt.date, drawdown: pt.drawdown * 100 }))
  }, [drawdownSeries])

  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', height: 200 }}>
        <div className="skeleton h-4 w-24 mb-4" />
        <div className="skeleton h-full w-full rounded" style={{ height: 148 }} />
      </div>
    )
  }

  if (!drawdownSeries) return null

  const tickCount = 6
  const tickIndices = Array.from({ length: tickCount }, (_, i) =>
    Math.floor((i / (tickCount - 1)) * (data.length - 1))
  )
  const tickDates = tickIndices.map((i) => data[i]?.date)

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <h3 className="mono font-bold text-sm mb-4" style={{ color: 'var(--text-primary)' }}>
        Drawdown
      </h3>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
          <defs>
            <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ff4757" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ff4757" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis
            dataKey="date"
            ticks={tickDates}
            tick={AXIS_STYLE.tick}
            axisLine={AXIS_STYLE.axisLine}
            tickLine={AXIS_STYLE.tickLine}
            tickFormatter={(d) => d?.slice(0, 7)}
          />
          <YAxis
            tick={AXIS_STYLE.tick}
            axisLine={AXIS_STYLE.axisLine}
            tickLine={AXIS_STYLE.tickLine}
            tickFormatter={(v) => `${v.toFixed(0)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#ff4757"
            strokeWidth={1.5}
            fill="url(#ddGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
