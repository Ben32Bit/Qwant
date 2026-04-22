import React, { useState, useMemo, useEffect, useRef, memo } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts'
import { CHART_COLORS, AXIS_STYLE, TOOLTIP_STYLE, GRID_STYLE } from '../../utils/chartConfig.js'

// ── Inject slider thumb styles once ──────────────────────────────────────────
// (can't set ::-webkit-slider-thumb via inline styles)
function useSliderStyles() {
  useEffect(() => {
    if (document.getElementById('qw-range-slider-styles')) return
    const el = document.createElement('style')
    el.id = 'qw-range-slider-styles'
    el.textContent = `
      .qw-rslider { position: relative; height: 20px; }
      .qw-rslider input[type=range] {
        -webkit-appearance: none; appearance: none;
        position: absolute; width: 100%; top: 0; height: 100%;
        background: transparent; pointer-events: none; margin: 0;
      }
      .qw-rslider input[type=range]:focus { outline: none; }
      .qw-rslider input[type=range]::-webkit-slider-runnable-track { background: transparent; }
      .qw-rslider input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none; width: 13px; height: 13px;
        border-radius: 50%; background: #4a9eff;
        pointer-events: auto; cursor: pointer;
        border: 2px solid #0a0a0f; box-shadow: 0 0 0 1px #4a9eff;
        margin-top: 0;
      }
      .qw-rslider input[type=range]::-moz-range-thumb {
        width: 13px; height: 13px;
        border-radius: 50%; background: #4a9eff;
        pointer-events: auto; cursor: pointer;
        border: 2px solid #0a0a0f; box-shadow: 0 0 0 1px #4a9eff;
      }
    `
    document.head.appendChild(el)
  }, [])
}

// ── Dual-thumb range slider ───────────────────────────────────────────────────
function DualRangeSlider({ max, startIdx, endIdx, onStartChange, onEndChange }) {
  useSliderStyles()

  const startPct = max > 0 ? (startIdx / max) * 100 : 0
  const endPct   = max > 0 ? (endIdx   / max) * 100 : 100

  return (
    <div className="qw-rslider">
      {/* Background track */}
      <div style={{
        position: 'absolute', top: '50%', transform: 'translateY(-50%)',
        left: 0, right: 0, height: 3,
        background: 'var(--border)', borderRadius: 2, pointerEvents: 'none',
      }} />
      {/* Selected fill */}
      <div style={{
        position: 'absolute', top: '50%', transform: 'translateY(-50%)',
        left: `${startPct}%`, width: `${endPct - startPct}%`, height: 3,
        background: 'var(--accent-blue)', borderRadius: 2, pointerEvents: 'none',
      }} />
      {/* Start thumb */}
      <input
        type="range"
        min={0} max={max} value={startIdx}
        style={{ zIndex: 2 }}
        onChange={e => onStartChange(Math.min(Number(e.target.value), endIdx - 1))}
      />
      {/* End thumb (higher z-index so it wins when both overlap) */}
      <input
        type="range"
        min={0} max={max} value={endIdx}
        style={{ zIndex: 3 }}
        onChange={e => onEndChange(Math.max(Number(e.target.value), startIdx + 1))}
      />
    </div>
  )
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '10px 14px' }}>
      <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </div>
      {payload.map((p) => {
        const sign = p.value >= 0 ? '+' : ''
        return (
          <div key={p.dataKey} className="mono text-sm flex items-center gap-2">
            <span style={{ color: p.color }}>■</span>
            <span style={{ color: 'var(--text-secondary)' }}>{p.name}:</span>
            <span style={{ color: p.value >= 0 ? p.color : 'var(--accent-red)' }}>
              {sign}{p.value?.toFixed(2)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

const FX_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'SGD', 'HKD', 'CNH']

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatDuration(startDate, endDate) {
  if (!startDate || !endDate) return ''
  const ms    = new Date(endDate) - new Date(startDate)
  const years = ms / (365.25 * 86_400_000)
  if (years >= 2) return `${years.toFixed(1)}Y`
  const months = Math.round(years * 12)
  if (months >= 2) return `${months}M`
  return `${Math.round(ms / 86_400_000)}D`
}

// ── Main component ────────────────────────────────────────────────────────────
function EquityCurve({ equityCurve, benchmarkCurve, fxCurves, loading, onRangeChange }) {
  const [currency, setCurrency]  = useState('USD')
  const [startIdx, setStartIdx]  = useState(0)
  const [endIdx, setEndIdx]      = useState(0)   // 0 = uninitialised; set in effect below

  // Use FX-adjusted curve when a non-USD currency is selected
  const activeCurve = useMemo(() => {
    if (currency === 'USD' || !fxCurves?.[currency]) return equityCurve
    return fxCurves[currency]
  }, [currency, equityCurve, fxCurves])

  // Build the full merged data array (portfolio + benchmark, indexed by date)
  const fullData = useMemo(() => {
    if (!activeCurve) return []
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

  const maxIdx = Math.max(0, fullData.length - 1)

  // Reset slider to full range whenever the data source changes
  useEffect(() => {
    setStartIdx(0)
    setEndIdx(maxIdx)
    if (onRangeChange && fullData.length > 0) {
      onRangeChange(fullData[0].date, fullData[maxIdx]?.date ?? fullData[0].date)
    }
  }, [maxIdx, fullData[0]?.date])   // eslint-disable-line react-hooks/exhaustive-deps

  // Report range to parent when slider moves
  const handleStartChange = (v) => {
    setStartIdx(v)
    if (onRangeChange && fullData[v] && fullData[endIdx]) {
      onRangeChange(fullData[v].date, fullData[endIdx].date)
    }
  }
  const handleEndChange = (v) => {
    setEndIdx(v)
    if (onRangeChange && fullData[startIdx] && fullData[v]) {
      onRangeChange(fullData[startIdx].date, fullData[v].date)
    }
  }

  const hasFx = fxCurves && Object.keys(fxCurves).length > 0

  // Rebase to % return from the start of the selected window.
  // MUST be declared before any early returns to satisfy Rules of Hooks.
  const displayData = useMemo(() => {
    const sliced = fullData.slice(startIdx, endIdx + 1)
    if (!sliced.length) return []
    const basePort = sliced[0].Portfolio ?? 1
    const baseBm   = sliced.find(pt => pt.Benchmark != null)?.Benchmark ?? 1
    return sliced.map(pt => ({
      date:      pt.date,
      Portfolio: pt.Portfolio != null ? (pt.Portfolio / basePort - 1) * 100 : undefined,
      Benchmark: pt.Benchmark != null ? (pt.Benchmark / baseBm  - 1) * 100 : undefined,
    }))
  }, [startIdx, endIdx, fullData])   // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', height: 320 }}>
        <div className="skeleton h-4 w-32 mb-4" />
        <div className="skeleton h-full w-full rounded" style={{ height: 220 }} />
      </div>
    )
  }

  if (!equityCurve) return null

  const isSubRange = startIdx > 0 || endIdx < maxIdx

  const startDate   = fullData[startIdx]?.date ?? ''
  const endDate     = fullData[endIdx]?.date ?? ''
  const duration    = formatDuration(startDate, endDate)

  // X-axis tick labels (sample ~8)
  const tickCount = 8
  const tickDates = Array.from({ length: tickCount }, (_, i) =>
    displayData[Math.floor((i / (tickCount - 1)) * Math.max(displayData.length - 1, 0))]?.date
  ).filter(Boolean)

  return (
    <div
      className="rounded-lg border p-4"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
          Returns
          <span className="ml-2 text-xs font-normal" style={{ color: 'var(--text-secondary)' }}>
            · from {startDate?.slice(0, 7)}
          </span>
          {currency !== 'USD' && (
            <span className="ml-1 text-xs font-normal" style={{ color: 'var(--accent-yellow)' }}>
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
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={displayData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
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
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: 'var(--text-secondary)' }} />
          <Line type="monotone" dataKey="Portfolio"  stroke={CHART_COLORS.portfolio}  dot={false} strokeWidth={2} isAnimationActive={false} />
          {benchmarkCurve && (
            <Line type="monotone" dataKey="Benchmark" stroke={CHART_COLORS.benchmark} dot={false} strokeWidth={1.5} strokeDasharray="4 4" isAnimationActive={false} />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Range slider */}
      <div className="mt-3 px-1">
        <DualRangeSlider
          max={maxIdx}
          startIdx={startIdx}
          endIdx={endIdx}
          onStartChange={handleStartChange}
          onEndChange={handleEndChange}
        />

        {/* Date labels row */}
        <div className="flex items-center justify-between mt-1.5">
          <span className="mono text-xs" style={{ color: isSubRange ? 'var(--accent-blue)' : 'var(--text-secondary)' }}>
            {startDate?.slice(0, 7)}
          </span>
          <div className="flex items-center gap-2">
            <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
              {duration}
            </span>
            {isSubRange && (
              <button
                onClick={() => { handleStartChange(0); handleEndChange(maxIdx) }}
                className="mono text-xs px-1.5 py-0.5 rounded"
                style={{ background: 'rgba(74,158,255,0.1)', color: 'var(--accent-blue)', border: '1px solid rgba(74,158,255,0.3)' }}
              >
                Reset
              </button>
            )}
          </div>
          <span className="mono text-xs" style={{ color: isSubRange ? 'var(--accent-blue)' : 'var(--text-secondary)' }}>
            {endDate?.slice(0, 7)}
          </span>
        </div>
      </div>
    </div>
  )
}

export default memo(EquityCurve)
