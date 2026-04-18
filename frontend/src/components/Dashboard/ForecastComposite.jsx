import React, { useMemo } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE } from '../../utils/chartConfig.js'

const METHOD_ORDER  = ['xgboost', 'nbeats', 'factor', 'hmm', 'var', 'lstm']
const METHOD_LABELS = { xgboost: 'XGBoost', nbeats: 'N-BEATS', factor: 'Factor', hmm: 'HMM', var: 'GP', lstm: 'LSTM' }
const METHOD_COLORS = { xgboost: '#4a9eff', nbeats: '#ffd43b', factor: '#00d4aa', hmm: '#a855f7', var: '#ff6b35', lstm: '#ff4757' }

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

// ── Method status dots ────────────────────────────────────────────────────────

function MethodStatusDots({ results }) {
  const resultMap = {}
  for (const r of (results ?? [])) resultMap[r.method] = r

  return (
    <div className="flex gap-2 flex-wrap">
      {METHOD_ORDER.map(method => {
        const r     = resultMap[method]
        const color = METHOD_COLORS[method]
        const done  = r?.forecast != null
        const err   = r?.error
        const wait  = !r || (!done && !err)

        return (
          <div key={method} className="flex items-center gap-1">
            <span style={{
              display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
              background: err ? '#ff4757' : done ? color : 'var(--border)',
              boxShadow:  done ? `0 0 4px ${color}88` : 'none',
              flexShrink: 0,
            }} />
            <span className="mono" style={{
              fontSize: 9,
              color: done ? color : err ? '#ff4757' : 'var(--text-secondary)',
              opacity: wait ? 0.5 : 1,
            }}>
              {METHOD_LABELS[method]}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Method effectiveness + ensemble weights ───────────────────────────────────

// Extract a quality signal (0–1) from whatever OOS metric a method exposes.
function methodQuality(result) {
  const m = result?.metadata
  if (!m) return null
  if (m.oos_r2  != null) return Math.max(0, Math.min(1, (m.oos_r2 + 0.2) / 1.2))   // XGBoost
  if (m.oos_mse != null) return Math.max(0, 1 - Math.min(1, m.oos_mse * 5))         // LSTM
  if (m.regime_sanity != null) return m.regime_sanity ? 0.80 : 0.45                  // HMM
  if (m.ljung_box_ok  != null) return m.ljung_box_ok  ? 0.75 : 0.40                  // VAR/GP
  if (m.periods != null) return 0.70                                                  // N-BEATS heuristic
  if (m.factor_r2 != null) return Math.max(0, Math.min(1, m.factor_r2))              // Factor
  return null
}

function qualityLabel(q) {
  if (q == null) return null
  if (q >= 0.75) return { label: 'HIGH', color: 'var(--accent-green)' }
  if (q >= 0.50) return { label: 'MED',  color: '#ffd43b' }
  return          { label: 'LOW',  color: '#ff4757' }
}

function MethodEffectivenessTable({ results, ensemble }) {
  const weights = ensemble?.weights ?? {}
  const rows = METHOD_ORDER
    .map(method => {
      const r = (results ?? []).find(x => x.method === method)
      const w = weights[method] ?? null
      const q = methodQuality(r)
      const ql = qualityLabel(q)
      const p50end = r?.forecast?.p50?.at(-1)
      return { method, r, w, q, ql, p50end }
    })
    .filter(row => row.r || row.w != null)

  if (!rows.length) return null

  return (
    <div className="mt-3 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      <div className="grid" style={{ gridTemplateColumns: '1fr 120px 64px 72px', background: 'rgba(255,255,255,0.02)' }}>
        <div className="mono px-3 py-1.5" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>METHOD</div>
        <div className="mono px-3 py-1.5" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>ENSEMBLE WEIGHT</div>
        <div className="mono px-3 py-1.5 text-center" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>OOS FIT</div>
        <div className="mono px-3 py-1.5 text-right" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>12M MEDIAN</div>
      </div>
      {rows.map(({ method, r, w, ql, p50end }) => {
        const color = METHOD_COLORS[method]
        const done  = r?.forecast != null
        const err   = r?.error
        return (
          <div key={method} className="grid items-center"
            style={{ gridTemplateColumns: '1fr 120px 64px 72px', borderTop: '1px solid var(--border)', opacity: (!done && !err) ? 0.45 : 1 }}>
            {/* Method name + dot */}
            <div className="flex items-center gap-2 px-3 py-2">
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: err ? '#ff4757' : done ? color : 'var(--border)', display: 'inline-block', flexShrink: 0, boxShadow: done ? `0 0 5px ${color}88` : 'none' }} />
              <span className="mono font-bold" style={{ fontSize: 10, color: done ? color : 'var(--text-secondary)' }}>
                {METHOD_LABELS[method]}
              </span>
              {err && <span className="mono" style={{ fontSize: 8, color: '#ff4757' }}>error</span>}
              {!done && !err && <span className="mono" style={{ fontSize: 8, color: 'var(--text-secondary)' }}>waiting…</span>}
            </div>
            {/* Ensemble weight bar */}
            <div className="px-3 py-2">
              {w != null ? (
                <div className="flex items-center gap-1.5">
                  <div className="flex-1 rounded-full overflow-hidden" style={{ height: 4, background: 'var(--border)' }}>
                    <div style={{ width: `${w * 100}%`, height: '100%', background: color, borderRadius: 9999, transition: 'width 0.5s ease' }} />
                  </div>
                  <span className="mono font-bold flex-shrink-0" style={{ fontSize: 9, color, minWidth: 26, textAlign: 'right' }}>
                    {(w * 100).toFixed(0)}%
                  </span>
                </div>
              ) : (
                <span className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.4 }}>—</span>
              )}
            </div>
            {/* OOS quality */}
            <div className="px-3 py-2 text-center">
              {ql ? (
                <span className="mono font-bold" style={{ fontSize: 9, color: ql.color }}>{ql.label}</span>
              ) : (
                <span className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.4 }}>—</span>
              )}
            </div>
            {/* 12-month p50 endpoint */}
            <div className="px-3 py-2 text-right">
              {p50end != null ? (
                <span className="mono font-bold" style={{ fontSize: 9, color }}>
                  {p50end >= 0 ? '+' : ''}{p50end.toFixed(1)}%
                </span>
              ) : (
                <span className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.4 }}>—</span>
              )}
            </div>
          </div>
        )
      })}
      {Object.keys(weights).length > 0 && (
        <div className="px-3 py-1.5 flex items-center gap-1" style={{ borderTop: '1px solid var(--border)', background: 'rgba(74,158,255,0.04)' }}>
          <span className="mono" style={{ fontSize: 8, color: 'var(--text-secondary)', opacity: 0.6 }}>
            Weights: regime-conditional stacked ensemble (Wolpert 1992) · current regime drives allocation
          </span>
        </div>
      )}
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
export default function ForecastComposite({ results, equityCurve, forecastStart, loading, ensemble }) {
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
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
            Composite Forecast
            <span className="ml-2 font-normal text-xs" style={{ color: 'var(--text-secondary)' }}>
              · projected portfolio value · next 12 months
            </span>
          </h3>
          <div className="flex items-center gap-2">
            <MethodStatusDots results={results} />
            {activeResults.length === 6 && (
              <span className="mono text-xs px-2 py-0.5 rounded"
                style={{ background: 'rgba(0,212,170,0.1)', border: '1px solid rgba(0,212,170,0.3)', color: 'var(--accent-green)' }}>
                all 6 ✓
              </span>
            )}
          </div>
        </div>
        <MethodEffectivenessTable results={results} ensemble={ensemble} />
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
