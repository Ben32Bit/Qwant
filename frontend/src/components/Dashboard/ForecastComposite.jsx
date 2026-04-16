import React, { useMemo } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE } from '../../utils/chartConfig.js'

const METHOD_ORDER = ['monte_carlo', 'garch', 'factor', 'hmm', 'var', 'lstm']

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
            <span style={{ color: p.value >= 0 ? p.color : 'var(--accent-red)' }}>
              {p.value >= 0 ? '+' : ''}{p.value.toFixed(1)}%
            </span>
          </div>
        ))}
    </div>
  )
}

/**
 * Full-width composite chart: historical equity curve on the left,
 * all 6 method p50 (median) forecast lines on the right, separated
 * by a vertical dashed reference line at the forecast start date.
 *
 * The historical curve is rebased to 0% at the chart's first point,
 * so the y-axis always shows % return since backtest start.
 */
export default function ForecastComposite({ results, equityCurve, forecastStart, loading }) {
  const chartData = useMemo(() => {
    if (!equityCurve?.length) return []

    // Rebase historical to % return
    const base = equityCurve[0].value
    const historical = equityCurve.map(pt => ({
      date:       pt.date,
      Historical: ((pt.value / base) - 1) * 100,
    }))

    // Collect forecast p50 per method
    const forecastByDate = {}
    for (const r of (results ?? [])) {
      if (!r.forecast) continue
      r.forecast.dates.forEach((d, i) => {
        if (!forecastByDate[d]) forecastByDate[d] = { date: d }
        forecastByDate[d][r.method] = r.forecast.p50[i]
      })
    }

    // Merge: historical rows + forecast rows
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
              · median path per method · next 12 months
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
            tickFormatter={v => `${v >= 0 ? '+' : ''}${v.toFixed(0)}%`}
          />
          <ReferenceLine y={0} stroke="var(--border)" strokeWidth={1} />
          {forecastStart && (
            <ReferenceLine
              x={forecastStart}
              stroke="rgba(255,255,255,0.2)"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
          )}
          <Tooltip content={<CompositeTooltip />} />
          <Legend
            wrapperStyle={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: 'var(--text-secondary)' }}
          />

          {/* Historical equity curve */}
          <Line
            type="monotone"
            dataKey="Historical"
            stroke="var(--text-secondary)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            strokeDasharray="3 3"
          />

          {/* One median line per method */}
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
