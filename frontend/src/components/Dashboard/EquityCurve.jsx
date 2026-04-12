import React, { useState, useMemo } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import { CHART_COLORS, AXIS_STYLE, TOOLTIP_STYLE, GRID_STYLE } from '../../utils/chartConfig.js'
import { fmtDollar, fmtDate } from '../../utils/formatters.js'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '10px 14px' }}>
      <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </div>
      {payload.map((p) => (
        <div key={p.dataKey} className="mono text-sm flex items-center gap-2">
          <span style={{ color: p.color }}>■</span>
          <span style={{ color: 'var(--text-secondary)' }}>{p.name}:</span>
          <span style={{ color: p.color }}>{fmtDollar(p.value, 2)}</span>
        </div>
      ))}
    </div>
  )
}

const FX_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'SGD', 'HKD', 'CNH']

export default function EquityCurve({ equityCurve, benchmarkCurve, fxCurves, loading }) {
  const [logScale, setLogScale] = useState(false)
  const [currency, setCurrency] = useState('USD')

  // Use FX-adjusted curve when a non-USD currency is selected
  const activeCurve = useMemo(() => {
    if (currency === 'USD' || !fxCurves?.[currency]) return equityCurve
    return fxCurves[currency]
  }, [currency, equityCurve, fxCurves])

  const data = useMemo(() => {
    if (!activeCurve) return []
    // Build a date→benchmark map for fast lookup (benchmark stays in USD)
    const bmMap = {}
    if (currency === 'USD' && benchmarkCurve) {
      benchmarkCurve.forEach(pt => { bmMap[pt.date] = pt.value })
    }
    return activeCurve.map(pt => ({
      date: pt.date,
      Portfolio: pt.value,
      ...(currency === 'USD' && bmMap[pt.date] != null ? { Benchmark: bmMap[pt.date] } : {}),
    }))
  }, [activeCurve, benchmarkCurve, currency])

  const hasFx = fxCurves && Object.keys(fxCurves).length > 0

  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', height: 280 }}>
        <div className="skeleton h-4 w-32 mb-4" />
        <div className="skeleton h-full w-full rounded" style={{ height: 220 }} />
      </div>
    )
  }

  if (!equityCurve) return null

  // Thin out data to max ~500 points for performance
  const step = Math.max(1, Math.floor(data.length / 500))
  const thinned = data.filter((_, i) => i % step === 0 || i === data.length - 1)

  // Sample x-axis labels (show ~8)
  const tickCount = 8
  const tickIndices = Array.from({ length: tickCount }, (_, i) =>
    Math.floor((i / (tickCount - 1)) * (thinned.length - 1))
  )
  const tickDates = tickIndices.map((i) => thinned[i]?.date)

  return (
    <div
      className="rounded-lg border p-4"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
          Equity Curve
          {currency !== 'USD' && (
            <span className="ml-2 text-xs font-normal" style={{ color: 'var(--accent-yellow)' }}>
              · {currency}-adjusted
            </span>
          )}
        </h3>
        <div className="flex items-center gap-2">
          {hasFx && (
            <select
              value={currency}
              onChange={e => setCurrency(e.target.value)}
              className="mono text-xs px-2 py-1 rounded border"
              style={{
                borderColor: currency !== 'USD' ? 'var(--accent-yellow)' : 'var(--border)',
                color: currency !== 'USD' ? 'var(--accent-yellow)' : 'var(--text-secondary)',
                background: 'var(--bg-card)',
                cursor: 'pointer',
              }}
            >
              {FX_CURRENCIES.filter(c => c === 'USD' || fxCurves?.[c]).map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          )}
          <button
            onClick={() => setLogScale((v) => !v)}
            className="mono text-xs px-2 py-1 rounded border"
            style={{
              borderColor: logScale ? 'var(--accent-blue)' : 'var(--border)',
              color: logScale ? 'var(--accent-blue)' : 'var(--text-secondary)',
              background: 'transparent',
            }}
          >
            {logScale ? 'LOG' : 'LINEAR'}
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={thinned} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
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
            scale={logScale ? 'log' : 'auto'}
            domain={logScale ? ['auto', 'auto'] : ['auto', 'auto']}
            tick={AXIS_STYLE.tick}
            axisLine={AXIS_STYLE.axisLine}
            tickLine={AXIS_STYLE.tickLine}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: 'var(--text-secondary)' }}
          />
          <Line
            type="monotone"
            dataKey="Portfolio"
            stroke={CHART_COLORS.portfolio}
            dot={false}
            strokeWidth={2}
          />
          {benchmarkCurve && (
            <Line
              type="monotone"
              dataKey="Benchmark"
              stroke={CHART_COLORS.benchmark}
              dot={false}
              strokeWidth={1.5}
              strokeDasharray="4 4"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
