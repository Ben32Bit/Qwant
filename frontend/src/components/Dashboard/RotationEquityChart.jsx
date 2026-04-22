import React, { useMemo, useState, memo } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceArea, ReferenceLine, Legend,
} from 'recharts'

// ── Colour palette — one colour per unique ticker ─────────────────────────────
const TICKER_PALETTE = [
  '#a855f7', '#4a9eff', '#00d4aa', '#ffd43b', '#ff4757',
  '#f97316', '#06b6d4', '#84cc16', '#ec4899', '#8b5cf6',
  '#14b8a6', '#f59e0b', '#3b82f6', '#ef4444', '#10b981',
]

function tickerColor(ticker, colorMap) {
  return colorMap[ticker] ?? '#8888a0'
}

// ── Custom tooltip ─────────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label, holdingSchedule, colorMap }) {
  if (!active || !payload?.length) return null

  // Find which window this date falls in
  const holding = holdingSchedule?.find(h => label >= h.window_start && label <= h.window_end)

  const portfolio = payload.find(p => p.dataKey === 'value')
  const benchmark = payload.find(p => p.dataKey === 'benchmark')

  return (
    <div
      className="rounded-lg border px-3 py-2 text-xs mono"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', minWidth: 160 }}
    >
      <div className="mb-1.5" style={{ color: 'var(--text-secondary)' }}>{label}</div>

      {portfolio && (
        <div className="flex justify-between gap-4 mb-0.5">
          <span style={{ color: '#a855f7' }}>Strategy</span>
          <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
            {portfolio.value?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
      )}
      {benchmark && (
        <div className="flex justify-between gap-4 mb-1">
          <span style={{ color: '#8888a0' }}>Benchmark</span>
          <span style={{ color: 'var(--text-secondary)' }}>
            {benchmark.value?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
      )}

      {holding && (
        <div
          className="mt-1.5 pt-1.5 flex flex-col gap-0.5"
          style={{ borderTop: '1px solid var(--border)' }}
        >
          <div style={{ color: 'var(--text-secondary)' }}>{holding.label}</div>
          <div className="flex flex-wrap gap-1 mt-0.5">
            {holding.tickers.map(t => (
              <span
                key={t}
                className="px-1.5 py-0.5 rounded font-bold"
                style={{
                  background: `${tickerColor(t, colorMap)}22`,
                  color: tickerColor(t, colorMap),
                  border: `1px solid ${tickerColor(t, colorMap)}55`,
                }}
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Holding legend ─────────────────────────────────────────────────────────────
function HoldingLegend({ colorMap }) {
  return (
    <div className="flex flex-wrap gap-2 mt-2 px-2">
      {Object.entries(colorMap).map(([ticker, color]) => (
        <div key={ticker} className="flex items-center gap-1">
          <div className="rounded-sm" style={{ width: 10, height: 10, background: color, opacity: 0.6 }} />
          <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{ticker}</span>
        </div>
      ))}
      <div className="flex items-center gap-1">
        <div className="h-px w-4" style={{ background: '#8888a0' }} />
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>Benchmark</span>
      </div>
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────
function RotationEquityChart({ equityCurve, benchmarkCurve, holdingSchedule }) {
  const [scale, setScale] = useState('linear')

  // Assign a stable color per unique ticker across all windows
  const colorMap = useMemo(() => {
    const tickers = []
    for (const h of (holdingSchedule ?? [])) {
      for (const t of h.tickers) {
        if (!tickers.includes(t)) tickers.push(t)
      }
    }
    const m = {}
    tickers.forEach((t, i) => { m[t] = TICKER_PALETTE[i % TICKER_PALETTE.length] })
    return m
  }, [holdingSchedule])

  // Merge equity + benchmark into one array keyed by date
  const chartData = useMemo(() => {
    if (!equityCurve?.length) return []
    const bmMap = {}
    for (const p of (benchmarkCurve ?? [])) bmMap[p.date] = p.value
    return equityCurve.map(p => ({
      date: p.date,
      value: p.value,
      benchmark: bmMap[p.date] ?? null,
    }))
  }, [equityCurve, benchmarkCurve])

  // X-axis ticks: one per year boundary
  // MUST be declared before any early returns to satisfy Rules of Hooks.
  const xTicks = useMemo(() => {
    const years = new Set()
    return chartData.filter(p => {
      const yr = p.date?.slice(0, 4)
      if (!yr || years.has(yr)) return false
      years.add(yr)
      return true
    }).map(p => p.date)
  }, [chartData])

  if (!chartData.length) return null

  const firstDate = chartData[0].date
  const lastDate = chartData[chartData.length - 1].date

  // Y-axis domain with slight padding
  const allValues = chartData.flatMap(p => [p.value, p.benchmark].filter(Boolean))
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  const pad = (maxVal - minVal) * 0.04
  const yDomain = [Math.max(0, minVal - pad), maxVal + pad]

  // Format x-axis: show year only to keep it sparse
  const xTick = (val) => val?.slice(0, 4) ?? ''

  return (
    <div
      className="rounded-lg border p-3"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="mono text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--accent-purple)' }}>
          Rotation Equity Curve
        </div>
        <div className="flex gap-1">
          {['linear', 'log'].map(s => (
            <button
              key={s}
              onClick={() => setScale(s)}
              className="mono text-xs px-2 py-1 rounded border transition-colors"
              style={{
                borderColor: scale === s ? 'var(--accent-purple)' : 'var(--border)',
                color: scale === s ? 'var(--accent-purple)' : 'var(--text-secondary)',
                background: scale === s ? 'rgba(168,85,247,0.1)' : 'transparent',
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Holding bands legend */}
      <HoldingLegend colorMap={colorMap} />

      {/* Chart */}
      <div style={{ height: 260, marginTop: 12 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 4, right: 8, left: 8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />

            {/* Coloured bands per holding window */}
            {(holdingSchedule ?? []).map((h, i) => {
              // Each window gets the colour of its first held ticker
              const color = h.tickers[0] ? tickerColor(h.tickers[0], colorMap) : '#888'
              return (
                <ReferenceArea
                  key={i}
                  x1={h.window_start < firstDate ? firstDate : h.window_start}
                  x2={h.window_end > lastDate ? lastDate : h.window_end}
                  fill={color}
                  fillOpacity={0.08}
                  stroke={color}
                  strokeOpacity={0.15}
                  strokeWidth={1}
                />
              )
            })}

            {/* Rotation event lines (window boundaries) */}
            {(holdingSchedule ?? []).slice(1).map((h, i) => (
              <ReferenceLine
                key={`rot-${i}`}
                x={h.window_start}
                stroke="rgba(168,85,247,0.3)"
                strokeDasharray="4 2"
                strokeWidth={1}
              />
            ))}

            <XAxis
              dataKey="date"
              ticks={xTicks}
              tickFormatter={xTick}
              tick={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'monospace' }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
            />
            <YAxis
              scale={scale}
              domain={yDomain}
              tickFormatter={v => v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(0)}`}
              tick={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'monospace' }}
              axisLine={false}
              tickLine={false}
              width={56}
            />
            <Tooltip
              content={
                <CustomTooltip
                  holdingSchedule={holdingSchedule}
                  colorMap={colorMap}
                />
              }
            />

            {/* Benchmark */}
            {benchmarkCurve && (
              <Line
                type="monotone"
                dataKey="benchmark"
                stroke="#8888a0"
                strokeWidth={1}
                dot={false}
                strokeDasharray="4 2"
                connectNulls
                isAnimationActive={false}
              />
            )}

            {/* Strategy equity curve */}
            <Line
              type="monotone"
              dataKey="value"
              stroke="#a855f7"
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Window-by-window holding label strip */}
      {holdingSchedule && holdingSchedule.length > 0 && (
        <div className="mt-3 overflow-x-auto">
          <div className="flex gap-1 pb-1" style={{ minWidth: 'max-content' }}>
            {holdingSchedule.map((h, i) => (
              <div
                key={i}
                className="rounded px-2 py-1 text-center flex-shrink-0"
                style={{
                  background: `${tickerColor(h.tickers[0], colorMap)}18`,
                  border: `1px solid ${tickerColor(h.tickers[0], colorMap)}40`,
                  minWidth: 72,
                }}
              >
                <div className="mono text-xs" style={{ color: 'var(--text-secondary)', fontSize: 9 }}>
                  {h.label}
                </div>
                <div
                  className="mono font-bold"
                  style={{ color: tickerColor(h.tickers[0], colorMap), fontSize: 11 }}
                >
                  {h.tickers.join('+')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default memo(RotationEquityChart)
