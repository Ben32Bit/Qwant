import React, { useMemo } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE } from '../../utils/chartConfig.js'

const METHOD_ORDER = ['xgboost', 'nbeats', 'factor', 'hmm', 'var', 'lstm']

function fmtVal(v) {
  if (v == null) return '—'
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`
  return `$${v.toFixed(0)}`
}

function CompositeTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '10px 14px' }}>
      <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      {payload
        .filter(p => p.value != null)
        .sort((a, b) => b.value - a.value)
        .map(p => (
          <div key={p.dataKey} className="mono text-xs flex items-center gap-2">
            <span style={{ color: p.color }}>■</span>
            <span style={{ color: 'var(--text-secondary)' }}>{p.name}:</span>
            <span style={{ color: p.color }}>{fmtVal(p.value)}</span>
          </div>
        ))}
    </div>
  )
}

/**
 * Full-width composite chart: historical portfolio value curve on the left,
 * all method median (p50) forecast lines continuing from the last historical
 * value on the right, separated by a dashed reference line at forecast start.
 *
 * Y-axis shows actual portfolio dollar values — not rebased % returns — so
 * users can read projected portfolio worth directly off the chart.
 *
 * Forecast values are computed as:
 *   projected_value = last_historical_value × (1 + p50_pct_return / 100)
 */
export default function ForecastComposite({ results, equityCurve, forecastStart, loading }) {
  const chartData = useMemo(() => {
    if (!equityCurve?.length) return []

    // Historical: use actual portfolio values
    const historical = equityCurve.map(pt => ({
      date:       pt.date,
      Historical: pt.value,
    }))

    // Last historical value is the anchor for all forecast projections
    const lastValue = equityCurve[equityCurve.length - 1].value

    // Convert each method's cumulative-% p50 forecast to actual dollar values
    const forecastByDate = {}
    for (const r of (results ?? [])) {
      if (!r.forecast) continue
      r.forecast.dates.forEach((d, i) => {
        if (!forecastByDate[d]) forecastByDate[d] = { date: d }
        forecastByDate[d][r.method] = lastValue * (1 + r.forecast.p50[i] / 100)
      })
    }

    const forecastRows = Object.values(forecastByDate).sort((a, b) => a.date.localeCompare(b.date))
    return [...historical, ...forecastRows]
  }, [equityCurve, results])

  const tickCount = 10
  const tickDates = chartData.length
    ? Array.from({ length: tickCount }, (_, i) =>
        chartData[Math.floor((i / (tickCount - 1)) * (chartData.length - 1))]?.date
      ).filter(Boolean)
    : []

  const activeResults = (results ?? []).filter(r => r.forecast)

  if (loading && !activeResults.length) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', height: 280 }}>
        <div className="skeleton h-4 w-48 mb-4" />
        <div className="skeleton w-full rounded" style={{ height: 220 }} />
      </div>
    )
  }

  if (!chartData.length) return null

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
            Composite Forecast
            <span className="ml-2 font-normal text-xs" style={{ color: 'var(--text-secondary)' }}>
              · projected portfolio value · next 12 months
            </span>
          </h3>
          <p className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            Tight clustering = higher conviction · wide spread = model disagreement
          </p>
        </div>
        {activeResults.length < 6 && (
          <span className="mono text-xs px-2 py-0.5 rounded"
            style={{ background: 'rgba(255,211,59,0.1)', border: '1px solid rgba(255,211,59,0.3)', color: 'var(--accent-yellow)' }}>
            {activeResults.length}/6 ready
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
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
            tickFormatter={fmtVal}
            width={60}
          />
          {forecastStart && (
            <ReferenceLine
              x={forecastStart}
              stroke="rgba(255,255,255,0.25)"
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{ value: 'Forecast', position: 'insideTopRight', fontSize: 9, fill: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}
            />
          )}
          <Tooltip content={<CompositeTooltip />} />
          <Legend
            wrapperStyle={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: 'var(--text-secondary)' }}
          />

          {/* Historical portfolio value */}
          <Line
            type="monotone"
            dataKey="Historical"
            stroke="var(--text-secondary)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            strokeDasharray="3 3"
          />

          {/* One median projection line per method */}
          {METHOD_ORDER.map(method => {
            const r = (results ?? []).find(x => x.method === method)
            if (!r?.forecast) return null
            return (
              <Line
                key={method}
                type="monotone"
                dataKey={method}
                name={r.label}
                stroke={r.color}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
                connectNulls={false}
              />
            )
          })}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
