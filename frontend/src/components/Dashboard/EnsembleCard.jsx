/**
 * EnsembleCard — Phase 5F: Featured ensemble forecast card.
 *
 * Displays the regime-conditional stacked ensemble at the top of the Forecast
 * panel. Shows: ensemble fan chart, regime donut, disagreement badge, method weights.
 *
 * References
 * ----------
 * Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
 * Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets.
 *   Annual Review of Financial Economics, 4(1), 313–337.
 * Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation,
 *   and Active Learning. NeurIPS 8.
 * Lakshminarayanan, B. et al. (2017). Simple and Scalable Predictive Uncertainty
 *   Estimation. NeurIPS 30.
 */

import { useMemo, memo } from 'react'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'
import { AXIS_STYLE, GRID_STYLE, TOOLTIP_STYLE } from '../../utils/chartConfig.js'

const REGIME_COLORS = {
  bull_low_vol:  '#00d4aa',
  bull_high_vol: '#ffd43b',
  bear:          '#ff6b35',
  crisis:        '#ff4757',
}

const REGIME_LABELS = {
  bull_low_vol:  'Bull / Low Vol',
  bull_high_vol: 'Bull / High Vol',
  bear:          'Bear',
  crisis:        'Crisis',
}

const METHOD_COLORS = {
  xgboost: '#4a9eff', nbeats: '#ffd43b', hmm: '#a855f7',
  factor: '#00d4aa', var: '#ff6b35', lstm: '#ff4757',
}

const DISAGREE_COLOR = { low: '#00d4aa', med: '#ffd43b', high: '#ff4757' }

// ── Regime Donut (SVG) ────────────────────────────────────────────────────────

function RegimeDonut({ regimeProbs, size = 72 }) {
  const REGIMES = ['bull_low_vol', 'bull_high_vol', 'bear', 'crisis']
  const values  = REGIMES.map(r => regimeProbs[r] ?? 0)
  const total   = values.reduce((a, b) => a + b, 0) || 1

  const cx     = size / 2
  const cy     = size / 2
  const R      = size * 0.42
  const r      = size * 0.25

  // Build SVG arc paths
  const arcs = []
  let startAngle = -Math.PI / 2

  values.forEach((val, i) => {
    const sweep = (val / total) * 2 * Math.PI
    if (sweep < 0.01) { startAngle += sweep; return }
    const endAngle = startAngle + sweep

    const x1 = cx + R * Math.cos(startAngle)
    const y1 = cy + R * Math.sin(startAngle)
    const x2 = cx + R * Math.cos(endAngle)
    const y2 = cy + R * Math.sin(endAngle)
    const ix1 = cx + r * Math.cos(startAngle)
    const iy1 = cy + r * Math.sin(startAngle)
    const ix2 = cx + r * Math.cos(endAngle)
    const iy2 = cy + r * Math.sin(endAngle)
    const large = sweep > Math.PI ? 1 : 0

    arcs.push({
      d: `M ${x1} ${y1} A ${R} ${R} 0 ${large} 1 ${x2} ${y2}
          L ${ix2} ${iy2} A ${r} ${r} 0 ${large} 0 ${ix1} ${iy1} Z`,
      color:   REGIME_COLORS[REGIMES[i]],
      label:   REGIME_LABELS[REGIMES[i]],
      pct:     Math.round((val / total) * 100),
    })
    startAngle = endAngle
  })

  const dominant = REGIMES[values.indexOf(Math.max(...values))]

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {arcs.map((arc, i) => (
          <path key={i} d={arc.d} fill={arc.color} opacity={0.85} />
        ))}
        {/* Centre hole */}
        <circle cx={cx} cy={cy} r={r - 1} fill="var(--bg-card)" />
      </svg>
      <span className="mono text-xs font-bold text-center" style={{ color: REGIME_COLORS[dominant], fontSize: 9 }}>
        {REGIME_LABELS[dominant]}
      </span>
    </div>
  )
}

// ── Method weights bar ────────────────────────────────────────────────────────

function WeightBar({ weights }) {
  if (!weights) return null
  const entries = Object.entries(weights)
    .filter(([, w]) => w > 0.01)
    .sort(([, a], [, b]) => b - a)

  return (
    <div className="space-y-1">
      {entries.map(([method, w]) => (
        <div key={method} className="flex items-center gap-2">
          <span className="mono text-xs w-14 flex-shrink-0" style={{ color: 'var(--text-secondary)', fontSize: 9 }}>
            {method}
          </span>
          <div className="flex-1 rounded-full overflow-hidden" style={{ height: 4, background: 'var(--border)' }}>
            <div className="h-full rounded-full"
              style={{ width: `${(w * 100).toFixed(0)}%`, background: METHOD_COLORS[method] ?? '#888' }} />
          </div>
          <span className="mono text-xs flex-shrink-0" style={{ color: METHOD_COLORS[method] ?? '#888', fontSize: 9 }}>
            {(w * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Fan chart ─────────────────────────────────────────────────────────────────

function buildChartData(band, lastValue) {
  if (!band?.dates || !lastValue) return []
  const toVal = pct => lastValue * (1 + pct / 100)
  return band.dates.map((date, i) => ({
    date,
    outer_base:   toVal(band.p5[i]),
    outer_height: toVal(band.p95[i]) - toVal(band.p5[i]),
    inner_base:   toVal(band.p25[i]),
    inner_height: toVal(band.p75[i]) - toVal(band.p25[i]),
    p50:          toVal(band.p50[i]),
  }))
}

function fmtDollar(v) {
  if (v == null) return '—'
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`
  return `$${v.toFixed(0)}`
}

const ENSEMBLE_COLOR = '#c084fc'   // bright violet — distinct from all 6 base model colors

// ── Main card ─────────────────────────────────────────────────────────────────

function EnsembleCard({ ensemble, regimeProbs, loading, lastValue }) {
  if (!ensemble && !loading) return null

  const chartData = useMemo(
    () => buildChartData(ensemble?.band, lastValue),
    [ensemble?.band, lastValue],
  )

  const tickDates = chartData.length
    ? Array.from({ length: 4 }, (_, i) =>
        chartData[Math.floor((i / 3) * (chartData.length - 1))]?.date
      ).filter(Boolean)
    : []

  const dominant    = ensemble?.dominant ?? regimeProbs?.dominant ?? 'bull_low_vol'
  const domColor    = REGIME_COLORS[dominant] ?? 'var(--accent-green)'
  const disagree    = ensemble?.disagreement
  const disagColor  = DISAGREE_COLOR[disagree?.level ?? 'low']

  return (
    <div className="rounded-lg border-2 p-4"
      style={{ background: 'var(--bg-card)', borderColor: `${ENSEMBLE_COLOR}55` }}>

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="mono font-bold text-sm" style={{ color: ENSEMBLE_COLOR }}>
            ◈ ENSEMBLE FORECAST
          </span>
          <span className="mono text-xs ml-2" style={{ color: 'var(--text-secondary)' }}>
            · Regime-conditional stacking · Wolpert (1992) · Ang & Timmermann (2012)
          </span>
        </div>
        <div className="flex items-center gap-3">
          {disagree && (
            <span className="mono text-xs px-2 py-0.5 rounded border"
              style={{ color: disagColor, borderColor: `${disagColor}55`, background: `${disagColor}11`, fontSize: 10 }}>
              {disagree.level.toUpperCase()} DISAGR.
            </span>
          )}
          <span className="mono text-xs px-2 py-0.5 rounded border"
            style={{ color: domColor, borderColor: `${domColor}55`, background: `${domColor}11`, fontSize: 10 }}>
            {REGIME_LABELS[dominant]}
          </span>
        </div>
      </div>

      {/* Loading skeleton */}
      {loading && !ensemble && (
        <div className="space-y-2">
          <div className="skeleton h-3 w-40" />
          <div className="skeleton h-36 w-full rounded" />
        </div>
      )}

      {/* Main content: fan chart + sidebar */}
      {ensemble && (
        <div className="flex gap-4">

          {/* Fan chart */}
          <div className="flex-1 min-w-0">
            {chartData.length > 0 && (
              <ResponsiveContainer width="100%" height={180}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 2 }}>
                  <CartesianGrid {...GRID_STYLE} />
                  <XAxis
                    dataKey="date"
                    ticks={tickDates}
                    tick={{ ...AXIS_STYLE.tick, fontSize: 9 }}
                    axisLine={AXIS_STYLE.axisLine}
                    tickLine={AXIS_STYLE.tickLine}
                    tickFormatter={d => d?.slice(0, 7)}
                  />
                  <YAxis
                    tick={{ ...AXIS_STYLE.tick, fontSize: 9 }}
                    axisLine={AXIS_STYLE.axisLine}
                    tickLine={AXIS_STYLE.tickLine}
                    tickFormatter={fmtDollar}
                    width={50}
                  />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null
                      const p50 = payload.find(p => p.dataKey === 'p50')?.value
                      return (
                        <div style={{ ...TOOLTIP_STYLE, padding: '8px 12px' }}>
                          <div className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
                          {p50 != null && (
                            <div className="mono text-sm font-bold" style={{ color: ENSEMBLE_COLOR }}>
                              {fmtDollar(p50)}
                            </div>
                          )}
                        </div>
                      )
                    }}
                  />
                  <Area type="monotone" dataKey="outer_base"   stackId="outer" stroke="none" fill="transparent" isAnimationActive={false} legendType="none" />
                  <Area type="monotone" dataKey="outer_height" stackId="outer" stroke="none" fill={ENSEMBLE_COLOR} fillOpacity={0.10} isAnimationActive={false} legendType="none" />
                  <Area type="monotone" dataKey="inner_base"   stackId="inner" stroke="none" fill="transparent" isAnimationActive={false} legendType="none" />
                  <Area type="monotone" dataKey="inner_height" stackId="inner" stroke="none" fill={ENSEMBLE_COLOR} fillOpacity={0.22} isAnimationActive={false} legendType="none" />
                  <Line type="monotone" dataKey="p50" stroke={ENSEMBLE_COLOR} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            )}

            {/* Disagreement metadata */}
            {disagree && (
              <div className="flex flex-wrap gap-3 mt-2">
                {[
                  { label: 'n models',  value: disagree.n },
                  { label: 'spread',    value: `${disagree.spread?.toFixed(2)}%` },
                  { label: 'variance',  value: disagree.variance?.toFixed(3) },
                  { label: 'source',    value: ensemble.source ?? '—' },
                ].map(({ label, value }) => (
                  <div key={label}>
                    <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{label} </span>
                    <span className="mono text-xs font-bold" style={{ color: ENSEMBLE_COLOR }}>{value}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Sidebar: regime donut + weight bars */}
          <div className="flex flex-col gap-4 flex-shrink-0" style={{ width: 130 }}>
            {regimeProbs && (
              <RegimeDonut regimeProbs={regimeProbs} size={80} />
            )}
            <WeightBar weights={ensemble.weights} />
          </div>
        </div>
      )}

      {/* Citation footer */}
      <div className="mt-3 pt-2 space-y-1" style={{ borderTop: '1px solid var(--border)' }}>
        <p className="mono leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📄 Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
        </p>
        <p className="mono leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📄 Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets.
          Annual Review of Financial Economics, 4(1), 313–337.
        </p>
        <p className="mono leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📄 Krogh & Vedelsby (1995). Neural Network Ensembles. NeurIPS 8. ·
          Lakshminarayanan et al. (2017). Deep Ensembles. NeurIPS 30.
        </p>
      </div>
    </div>
  )
}

export default memo(EnsembleCard)
