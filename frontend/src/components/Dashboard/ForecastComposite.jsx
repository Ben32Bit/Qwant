import React, { useMemo, useState, memo } from 'react'
import { createPortal } from 'react-dom'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ReferenceArea, ResponsiveContainer,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE, CHART_COLORS } from '../../utils/chartConfig.js'

const METHOD_ORDER  = ['nbeats', 'timesfm', 'hmm', 'var', 'lstm']

// Methods whose shadow OOS R² falls below this threshold are considered
// actively harmful (≥50% worse than naive mean) and excluded from the chart.
const OOS_R2_MIN = -0.5
const METHOD_LABELS = { nbeats: 'N-BEATS', timesfm: 'TimesFM', hmm: 'HMM', var: 'GP', lstm: 'LSTM' }
const METHOD_COLORS = { nbeats: '#ffd43b', timesfm: '#00d4aa', hmm: '#a855f7', var: '#ff6b35', lstm: '#ff4757' }

// Y-axis: cumulative % return from the start of the user's backtest.
// Historical and forecast are both expressed against the same anchor so the
// line is continuous through the forecast-start divider.
function fmtPct(v) {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
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
            <span style={{ color: p.color }}>{fmtPct(p.value)}</span>
          </div>
        ))}
    </div>
  )
}

// ── Shadow window info tooltip ────────────────────────────────────────────────

function ShadowWindowInfo({ shadowStart, shadowEnd }) {
  const [show, setShow] = useState(false)
  const [pos,  setPos]  = useState({ top: 0, left: 0 })
  const btnRef = React.useRef(null)

  if (!shadowStart || !shadowEnd) return null

  const handleShow = () => {
    if (btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect()
      setPos({ top: rect.bottom + 6, left: Math.max(8, rect.left - 120) })
    }
    setShow(true)
  }

  return (
    <span className="inline-flex items-center gap-1 ml-2">
      <span className="mono" style={{ fontSize: 9, color: '#ffd43b', opacity: 0.7 }}>
        Shadow {shadowStart?.slice(0, 7)} → {shadowEnd?.slice(0, 7)}
      </span>
      <button
        ref={btnRef}
        className="mono rounded-full flex items-center justify-center flex-shrink-0"
        style={{
          width: 14, height: 14,
          background: 'rgba(255,212,59,0.15)',
          color: '#ffd43b',
          border: '1px solid rgba(255,212,59,0.4)',
          cursor: 'pointer', fontSize: 9, lineHeight: 1,
        }}
        onMouseEnter={handleShow}
        onMouseLeave={() => setShow(false)}
        aria-label="Shadow holdout window explanation"
      >
        ?
      </button>
      {show && createPortal(
        <div
          className="rounded-lg border p-3"
          style={{
            position: 'fixed', top: pos.top, left: pos.left,
            width: 300, zIndex: 9999,
            background: 'var(--bg-secondary)', borderColor: 'rgba(255,212,59,0.4)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.6)', pointerEvents: 'none',
          }}
        >
          <p className="mono font-bold text-xs mb-2" style={{ color: '#ffd43b' }}>
            Shadow Holdout (OOS Test Window)
          </p>
          <p className="mono text-xs leading-relaxed mb-2" style={{ color: 'var(--text-secondary)' }}>
            Each server-side model (HMM, GP, TimesFM) was trained on data up to{' '}
            <span style={{ color: '#ffd43b' }}>{shadowStart}</span>, then made a
            30-day forecast through{' '}
            <span style={{ color: '#ffd43b' }}>{shadowEnd}</span>.
          </p>
          <p className="mono text-xs leading-relaxed mb-2" style={{ color: 'var(--text-secondary)' }}>
            Because this window is now in the past, we can compare the model's
            predictions (dashed lines) against what actually happened (blue Portfolio
            line). The accuracy of this test is the <span style={{ color: 'var(--accent-green)' }}>OOS R²</span> shown
            in the table above — it drives each model's ensemble weight.
          </p>
          <p className="mono" style={{ fontSize: 9, color: 'rgba(136,136,160,0.6)', borderTop: '1px solid var(--border)', paddingTop: 4 }}>
            Lopez de Prado (2018) Advances in Financial Machine Learning, Ch. 7
          </p>
        </div>,
        document.body
      )}
    </span>
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

function oosR2Color(r2) {
  if (r2 == null) return null
  if (r2 >= 0.5)  return 'var(--accent-green)'
  if (r2 >= 0.0)  return '#ffd43b'
  return '#ff4757'
}

function MethodEffectivenessTable({ results, ensemble }) {
  const weights = ensemble?.weights ?? {}
  const rows = METHOD_ORDER
    .map(method => {
      const r     = (results ?? []).find(x => x.method === method)
      const w     = weights[method] ?? null
      const p50end = r?.forecast?.p50?.at(-1)
      return { method, r, w, p50end }
    })

  if (!rows.length) return null

  return (
    <div className="mt-3 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      <div className="grid" style={{ gridTemplateColumns: '1fr 120px 64px 72px', background: 'rgba(255,255,255,0.02)' }}>
        <div className="mono px-3 py-1.5" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>METHOD</div>
        <div className="mono px-3 py-1.5" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>ENSEMBLE WEIGHT</div>
        <div className="mono px-3 py-1.5 text-center" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>OOS R²</div>
        <div className="mono px-3 py-1.5 text-right" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>3M MEDIAN</div>
      </div>
      {rows.map(({ method, r, w, p50end }) => {
        const color   = METHOD_COLORS[method]
        const done    = r?.forecast != null
        const err     = r?.error
        const r2      = r?.oos_r2 ?? null
        const r2color = oosR2Color(r2)
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
              {done && r2 != null && r2 < OOS_R2_MIN && (
                <span className="mono" style={{ fontSize: 8, color: '#ff4757', background: 'rgba(255,71,87,0.1)', border: '1px solid rgba(255,71,87,0.25)', borderRadius: 3, padding: '0 3px' }}>
                  excluded
                </span>
              )}
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
            {/* OOS R² from shadow holdout */}
            <div className="px-3 py-2 text-center">
              {r2 != null ? (
                <span className="mono font-bold" style={{ fontSize: 9, color: r2color }}>
                  {r2 >= 0 ? '+' : ''}{r2.toFixed(2)}
                </span>
              ) : (
                <span className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.4 }}>—</span>
              )}
            </div>
            {/* 3-month p50 endpoint */}
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
            OOS R² from 30-day shadow holdout · OOS R²-weighted ensemble (Wolpert 1992) · methods with OOS R² &lt; −0.5 excluded
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
// Cap historical points drawn into the chart. A 10-year backtest has ~2500
// daily points; times 7 lines that's ~17.5k SVG nodes which was making the
// page stall on click/hover. Uniform stride keeps the shape of the curve.
const MAX_HISTORICAL_POINTS = 500

function downsampleHistorical(equityCurve) {
  const n = equityCurve.length
  if (n <= MAX_HISTORICAL_POINTS) return equityCurve
  const stride = Math.ceil(n / MAX_HISTORICAL_POINTS)
  const out = []
  for (let i = 0; i < n; i += stride) out.push(equityCurve[i])
  // Always keep the last point — the forecast anchors to it.
  if (out[out.length - 1] !== equityCurve[n - 1]) out.push(equityCurve[n - 1])
  return out
}

const SHADOW_METHODS = ['hmm', 'var', 'timesfm']

function ForecastCompositeImpl({ results, equityCurve, forecastStart, shadowStart, shadowEnd, loading, ensemble }) {
  const chartData = useMemo(() => {
    if (!equityCurve?.length) return []

    // Anchor at the first historical value so the Y-axis reads as cumulative
    // % return from the user's backtest start. Forecast lines compound the
    // current portfolio cumulative return with each model's projected p50.
    const firstValue = equityCurve[0].value
    const lastValue  = equityCurve[equityCurve.length - 1].value
    const toPct      = v => (v / firstValue - 1) * 100

    // Build a date→value lookup for the full (non-downsampled) equity curve.
    // Used to (a) provide Portfolio values for shadow-window rows that were
    // skipped by downsampling, and (b) find the shadow anchor value.
    const equityByDate = {}
    for (const pt of equityCurve) equityByDate[pt.date] = pt.value

    // Shadow anchor: last equity value strictly before shadowStart.
    let shadowAnchorValue = lastValue
    if (shadowStart) {
      for (const pt of equityCurve) {
        if (pt.date < shadowStart) shadowAnchorValue = pt.value
      }
    }
    const projectFromShadow = pct50 => toPct(shadowAnchorValue * (1 + pct50 / 100))

    // Build historical rows (downsampled). We then merge in any shadow dates
    // that downsampling dropped so the shadow lines render without gaps.
    const historicalRows = downsampleHistorical(equityCurve).map(pt => ({
      date:      pt.date,
      Portfolio: toPct(pt.value),
    }))
    const historicalMap = {}
    for (const row of historicalRows) historicalMap[row.date] = row

    // Add shadow projections into historical rows.
    for (const r of (results ?? [])) {
      if (!r.shadow_band?.dates?.length) continue
      const key = `${r.method}_shadow`
      r.shadow_band.dates.forEach((d, i) => {
        if (!historicalMap[d]) {
          // Inject a row the downsampler skipped, filling Portfolio from lookup.
          const actual = equityByDate[d]
          historicalMap[d] = { date: d, Portfolio: actual != null ? toPct(actual) : null }
        }
        historicalMap[d][key] = projectFromShadow(r.shadow_band.p50[i] ?? 0)
      })
    }

    const mergedHistorical = Object.values(historicalMap).sort((a, b) => a.date.localeCompare(b.date))

    // Each method's p50[i] is cumulative % return from forecast start, so
    // projected_value = lastValue × (1 + p50/100), then re-anchor to start.
    const projectFromForecast = pct50 => toPct(lastValue * (1 + pct50 / 100))

    const forecastByDate = {}
    for (const r of (results ?? [])) {
      if (!r.forecast) continue
      r.forecast.dates.forEach((d, i) => {
        if (!forecastByDate[d]) forecastByDate[d] = { date: d }
        forecastByDate[d][r.method] = projectFromForecast(r.forecast.p50[i])
      })
    }

    // Overlay the regime-conditional ensemble median as a 7th line, only once
    // at least two base methods have landed (otherwise it's degenerate).
    if (ensemble?.band?.dates?.length && ensemble.band.p50?.length) {
      ensemble.band.dates.forEach((d, i) => {
        if (!forecastByDate[d]) forecastByDate[d] = { date: d }
        forecastByDate[d].ensemble = projectFromForecast(ensemble.band.p50[i])
      })
    }

    const forecastRows = Object.values(forecastByDate).sort((a, b) => a.date.localeCompare(b.date))
    return [...mergedHistorical, ...forecastRows]
  }, [equityCurve, results, ensemble, shadowStart])

  // ── Zoom slider state ────────────────────────────────────────────────────
  // The user can drag the start of the visible window forward; the right edge
  // is fixed at the forecast end. Slider value is an index into chartData;
  // max is the last historical row (so users can't slide past forecast start
  // and end up looking at a chart with no anchor).
  const lastHistoricalIdx = useMemo(() => {
    if (!chartData.length) return 0
    if (!forecastStart) return chartData.length - 1
    let last = 0
    for (let i = 0; i < chartData.length; i++) {
      if (chartData[i].date < forecastStart) last = i
    }
    return last
  }, [chartData, forecastStart])

  const [viewStartIdx, setViewStartIdx] = useState(0)
  const clampedStartIdx = Math.min(viewStartIdx, lastHistoricalIdx)
  const visibleData     = useMemo(() =>
    chartData.slice(clampedStartIdx),
    [chartData, clampedStartIdx])

  const tickCount = 10
  const tickDates = visibleData.length
    ? Array.from({ length: tickCount }, (_, i) =>
        visibleData[Math.floor((i / (tickCount - 1)) * (visibleData.length - 1))]?.date
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
          <div className="flex items-center flex-wrap gap-x-2">
            <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
              Composite Forecast
              <span className="ml-2 font-normal text-xs" style={{ color: 'var(--text-secondary)' }}>
                · cumulative % return · next 3 months
              </span>
            </h3>
            <ShadowWindowInfo shadowStart={shadowStart} shadowEnd={shadowEnd} />
          </div>
          <div className="flex items-center gap-2">
            <MethodStatusDots results={results} />
            {activeResults.length === 5 && (
              <span className="mono text-xs px-2 py-0.5 rounded"
                style={{ background: 'rgba(0,212,170,0.1)', border: '1px solid rgba(0,212,170,0.3)', color: 'var(--accent-green)' }}>
                all 5 ✓
              </span>
            )}
          </div>
        </div>
        <MethodEffectivenessTable results={results} ensemble={ensemble} />
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={visibleData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
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
            width={50}
          />
          {/* Shadow holdout window — amber shaded region with border lines */}
          {shadowStart && shadowEnd && (
            <ReferenceArea
              x1={shadowStart}
              x2={shadowEnd}
              fill="rgba(255,212,59,0.06)"
              stroke="rgba(255,212,59,0.25)"
              strokeWidth={1}
              label={{
                value: 'OOS Test Window',
                position: 'insideTop',
                fontSize: 8,
                fill: 'rgba(255,212,59,0.55)',
                fontFamily: 'monospace',
              }}
            />
          )}
          {shadowStart && !shadowEnd && (
            <ReferenceLine
              x={shadowStart}
              stroke="rgba(255,212,59,0.4)"
              strokeDasharray="3 4"
              strokeWidth={1}
              label={{ value: 'Shadow start', position: 'insideTopLeft', fontSize: 8, fill: 'rgba(255,212,59,0.5)', fontFamily: 'monospace' }}
            />
          )}
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

          {/* Historical portfolio cumulative return — same blue used for the
              Portfolio line in the Results tab. connectNulls keeps it solid
              through the shadow window even if downsampling skipped some dates. */}
          <Line
            type="monotone"
            dataKey="Portfolio"
            name="Your Portfolio (actual)"
            stroke={CHART_COLORS.portfolio}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />

          {/* Shadow lines — shown even for excluded methods so the user can see
              WHY they were excluded (the prediction vs actual gap is visible). */}
          {SHADOW_METHODS.map(method => {
            const r = (results ?? []).find(x => x.method === method)
            if (!r?.shadow_band) return null
            return (
              <Line
                key={`${method}_shadow`}
                type="monotone"
                dataKey={`${method}_shadow`}
                name={`${METHOD_LABELS[method]} (shadow)`}
                stroke={METHOD_COLORS[method]}
                strokeWidth={1}
                strokeOpacity={0.30}
                strokeDasharray="3 4"
                dot={false}
                isAnimationActive={false}
                connectNulls={false}
                legendType="none"
              />
            )
          })}

          {/* One median projection line per method — skipped if OOS R² < threshold */}
          {METHOD_ORDER.map(method => {
            const r = (results ?? []).find(x => x.method === method)
            if (!r?.forecast) return null
            if (r.oos_r2 != null && r.oos_r2 < OOS_R2_MIN) return null
            return (
              <Line
                key={method}
                type="monotone"
                dataKey={method}
                name={r.label}
                stroke={r.color}
                strokeWidth={1.2}
                strokeOpacity={0.55}
                dot={false}
                isAnimationActive={false}
                connectNulls={false}
              />
            )
          })}

          {/* Regime-conditional ensemble median — bold white line on top */}
          {ensemble?.band?.p50?.length > 0 && (
            <Line
              type="monotone"
              dataKey="ensemble"
              name="Ensemble (regime-weighted)"
              stroke="#ffffff"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
              connectNulls={false}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Zoom slider — moves the visible window's left edge forward; right
          edge stays pinned to the forecast end so the user can zoom into
          the recent past + projection without losing the tail. */}
      {lastHistoricalIdx > 0 && (
        <div className="flex items-center gap-3 mt-2">
          <span className="mono flex-shrink-0" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.7 }}>
            Zoom
          </span>
          <span className="mono flex-shrink-0" style={{ fontSize: 9, color: 'var(--text-secondary)', minWidth: 64 }}>
            {visibleData[0]?.date ?? '—'}
          </span>
          <input
            type="range"
            min={0}
            max={lastHistoricalIdx}
            value={clampedStartIdx}
            onChange={e => setViewStartIdx(+e.target.value)}
            className="flex-1 cursor-pointer"
            style={{ accentColor: CHART_COLORS.portfolio }}
            aria-label="Forecast chart zoom — drag right to focus on the tail"
          />
          <span className="mono flex-shrink-0" style={{ fontSize: 9, color: 'var(--text-secondary)', minWidth: 64, textAlign: 'right' }}>
            {visibleData[visibleData.length - 1]?.date ?? '—'}
          </span>
          {clampedStartIdx > 0 && (
            <button
              onClick={() => setViewStartIdx(0)}
              className="mono flex-shrink-0"
              style={{
                fontSize: 9,
                padding: '2px 6px',
                borderRadius: 3,
                border: '1px solid var(--border)',
                background: 'transparent',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              reset
            </button>
          )}
        </div>
      )}

    </div>
  )
}

export default memo(ForecastCompositeImpl, (prev, next) =>
  prev.results === next.results &&
  prev.equityCurve === next.equityCurve &&
  prev.forecastStart === next.forecastStart &&
  prev.shadowStart === next.shadowStart &&
  prev.shadowEnd === next.shadowEnd &&
  prev.loading === next.loading &&
  prev.ensemble === next.ensemble
)
