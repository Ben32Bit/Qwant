import React, { useMemo, useState, memo } from 'react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts'
import { AXIS_STYLE, TOOLTIP_STYLE, GRID_STYLE } from '../../utils/chartConfig.js'

// Distinct palette for up to 10 holdings
const PALETTE = [
  '#4a9eff', '#00d4aa', '#ffd43b', '#ff4757',
  '#a29bfe', '#fd79a8', '#55efc4', '#fdcb6e',
  '#e17055', '#74b9ff',
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const sorted = [...payload].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '10px 14px', minWidth: 160 }}>
      <div className="mono text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      {sorted.map(p => (
        <div key={p.dataKey} className="mono text-xs flex items-center justify-between gap-4 mb-0.5">
          <span style={{ color: p.color }}>■ {p.dataKey}</span>
          <span style={{ color: p.value < 0 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
            {p.value.toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  )
}

// Small rebalance marker rendered at top of a ReferenceLine via customized label
function RebalanceLabel({ viewBox }) {
  if (!viewBox) return null
  const { x } = viewBox
  return (
    <g>
      <polygon
        points={`${x - 4},2 ${x + 4},2 ${x},9`}
        fill="rgba(255,215,0,0.7)"
      />
    </g>
  )
}

function WeightDriftChart({ weightHistory, rebalanceDates, loading }) {
  const [mode, setMode] = useState('area') // 'area' | 'line'

  const { data, tickers, hasShorts, tickerColors } = useMemo(() => {
    if (!weightHistory?.length) return { data: [], tickers: [], hasShorts: false, tickerColors: {} }

    // Collect all ticker columns (exclude 'date')
    const allTickers = Object.keys(weightHistory[0]).filter(k => k !== 'date')

    // Normalise: convert raw weight to percentage
    const data = weightHistory.map(row => {
      const pt = { date: row.date }
      for (const t of allTickers) {
        pt[t] = row[t] != null ? parseFloat((row[t] * 100).toFixed(2)) : 0
      }
      return pt
    })

    const hasShorts = allTickers.some(t =>
      data.some(pt => pt[t] < -0.5)
    )

    const tickerColors = Object.fromEntries(allTickers.map((t, i) => [t, PALETTE[i % PALETTE.length]]))

    return { data, tickers: allTickers, hasShorts, tickerColors }
  }, [weightHistory])

  // X-axis sample ticks
  const tickCount = 7
  const tickDates = useMemo(() => {
    if (!data.length) return []
    return Array.from({ length: tickCount }, (_, i) =>
      data[Math.floor((i / (tickCount - 1)) * (data.length - 1))]?.date
    )
  }, [data])

  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', height: 280 }}>
        <div className="skeleton h-4 w-40 mb-4" />
        <div className="skeleton h-52 w-full rounded" />
      </div>
    )
  }

  if (!weightHistory?.length || tickers.length === 0) return null

  // Force line mode when shorts exist (stacking breaks with negatives)
  const chartMode = hasShorts ? 'line' : mode

  const rebalanceDateSet = new Set(rebalanceDates || [])

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <div>
          <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
            Holdings Over Time
          </h3>
          <p className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            Weight drift between rebalances · gold lines = rebalance events
          </p>
        </div>
        {!hasShorts && (
          <div className="flex gap-1">
            {['area', 'line'].map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className="mono text-xs px-2 py-1 rounded border capitalize transition-colors"
                style={{
                  borderColor: chartMode === m ? 'var(--accent-blue)' : 'var(--border)',
                  color: chartMode === m ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  background: 'transparent',
                }}
              >
                {m}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3">
        <ResponsiveContainer width="100%" height={240}>
          {chartMode === 'area' ? (
            <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
              <defs>
                {tickers.map(t => (
                  <linearGradient key={t} id={`grad-${t}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={tickerColors[t]} stopOpacity={0.7} />
                    <stop offset="95%" stopColor={tickerColors[t]} stopOpacity={0.3} />
                  </linearGradient>
                ))}
              </defs>
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
                tickFormatter={v => `${v.toFixed(0)}%`}
                tick={AXIS_STYLE.tick}
                axisLine={AXIS_STYLE.axisLine}
                tickLine={AXIS_STYLE.tickLine}
                domain={[0, 100]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: 'var(--text-secondary)', paddingTop: 8 }}
              />
              {/* Rebalance reference lines */}
              {(rebalanceDates || []).slice(1).map(d => (
                <ReferenceLine key={d} x={d} stroke="rgba(255,215,0,0.6)" strokeWidth={1.5} strokeDasharray="3 3" label={RebalanceLabel} />
              ))}
              {tickers.map(t => (
                <Area
                  key={t}
                  type="monotone"
                  dataKey={t}
                  stackId="1"
                  stroke={tickerColors[t]}
                  fill={`url(#grad-${t})`}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </AreaChart>
          ) : (
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
                tickFormatter={v => `${v.toFixed(0)}%`}
                tick={AXIS_STYLE.tick}
                axisLine={AXIS_STYLE.axisLine}
                tickLine={AXIS_STYLE.tickLine}
              />
              {hasShorts && <ReferenceLine y={0} stroke="var(--border)" strokeWidth={1} />}
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: 'var(--text-secondary)', paddingTop: 8 }}
              />
              {/* Rebalance reference lines */}
              {(rebalanceDates || []).slice(1).map(d => (
                <ReferenceLine key={d} x={d} stroke="rgba(255,215,0,0.6)" strokeWidth={1.5} strokeDasharray="3 3" label={RebalanceLabel} />
              ))}
              {tickers.map(t => (
                <Line
                  key={t}
                  type="monotone"
                  dataKey={t}
                  stroke={tickerColors[t]}
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Legend note for shorts */}
      {hasShorts && (
        <p className="mono text-xs mt-2" style={{ color: 'var(--accent-red)' }}>
          Negative values indicate short positions
        </p>
      )}
    </div>
  )
}

export default memo(WeightDriftChart)
