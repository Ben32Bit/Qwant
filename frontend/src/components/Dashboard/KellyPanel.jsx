/**
 * KellyPanel — Regime-adjusted Kelly position sizing recommendation card.
 *
 * Displays the optimal capital deployment fraction derived from the ensemble
 * forecast fan bands using the continuous Kelly criterion (Kelly 1956).
 * Applies a regime confidence multiplier (Ang & Timmermann 2012) to scale
 * down position size in high-uncertainty environments.
 */
import { useState, useMemo } from 'react'
import { computeKelly } from '../../ml/KellyCalculator.js'

const LEVEL_COLORS = {
  avoid:       '#ff4757',
  underweight: '#ff6b35',
  cautious:    '#ffd43b',
  neutral:     '#00d4aa',
  overweight:  '#4a9eff',
  aggressive:  '#a855f7',
}

const LEVEL_LABELS = {
  avoid:       'AVOID — negative expected return',
  underweight: 'UNDERWEIGHT — low edge relative to risk',
  cautious:    'CAUTIOUS — modest deployment recommended',
  neutral:     'NEUTRAL — full deployment supported',
  overweight:  'OVERWEIGHT — modest leverage supported',
  aggressive:  'AGGRESSIVE — high leverage (use with caution)',
}

// ── Gauge arc (SVG) ────────────────────────────────────────────────────────────

function KellyGauge({ fraction, color }) {
  // Arc from 0 (left) to 2× (right), covering 180°
  // fraction: 0 → 2, clamped for display
  const pct    = Math.min(Math.max(fraction / 2, 0), 1)
  const angle  = pct * 180 - 90    // −90° to +90° (left to right)
  const rad    = (angle * Math.PI) / 180
  const cx = 70, cy = 70, r = 52
  const markerX = cx + r * Math.cos(rad)
  const markerY = cy + r * Math.sin(rad)

  // Arc path: always from left (−90°) to current angle
  const arcAngle = pct * 180
  const largeArc = arcAngle > 180 ? 1 : 0
  const startX = cx + r * Math.cos(-Math.PI / 2)
  const startY = cy + r * Math.sin(-Math.PI / 2)
  const endX   = cx + r * Math.cos(rad)
  const endY   = cy + r * Math.sin(rad)

  return (
    <svg width={140} height={80} viewBox="0 0 140 80">
      {/* Background track */}
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke="var(--border)" strokeWidth={8} strokeLinecap="round"
      />
      {/* Filled arc */}
      {pct > 0.01 && (
        <path
          d={`M ${startX} ${startY} A ${r} ${r} 0 ${largeArc} 1 ${endX} ${endY}`}
          fill="none" stroke={color} strokeWidth={8} strokeLinecap="round"
          style={{ transition: 'all 0.5s ease' }}
        />
      )}
      {/* Needle marker */}
      <circle cx={markerX} cy={markerY} r={5} fill={color} style={{ transition: 'all 0.5s ease' }} />
      {/* Labels */}
      <text x={cx - r - 4} y={cy + 16} fill="var(--text-secondary)" fontSize={9} textAnchor="middle" className="mono">0×</text>
      <text x={cx} y={cy + 16} fill="var(--text-secondary)" fontSize={9} textAnchor="middle" className="mono">1×</text>
      <text x={cx + r + 4} y={cy + 16} fill="var(--text-secondary)" fontSize={9} textAnchor="middle" className="mono">2×</text>
      {/* Center value */}
      <text x={cx} y={cy - 10} fill={color} fontSize={18} fontWeight="bold" textAnchor="middle" className="mono">
        {(fraction * 100).toFixed(0)}%
      </text>
      <text x={cx} y={cy + 3} fill="var(--text-secondary)" fontSize={8} textAnchor="middle" className="mono">of capital</text>
    </svg>
  )
}

// ── Stat pill ─────────────────────────────────────────────────────────────────

function Stat({ label, value, color }) {
  return (
    <div className="rounded px-3 py-2" style={{ background: 'var(--bg-secondary)' }}>
      <div className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      <div className="mono text-sm font-bold" style={{ color: color ?? 'var(--text-primary)' }}>{value}</div>
    </div>
  )
}

// ── Deployment bar ─────────────────────────────────────────────────────────────

function DeploymentBar({ fraction, color }) {
  const pct = Math.min(fraction, 2) / 2 * 100
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>Recommended capital deployment</span>
        <span className="mono text-xs font-bold" style={{ color }}>
          {(fraction * 100).toFixed(0)}%{fraction > 1 ? ' (leveraged)' : ''}
        </span>
      </div>
      <div className="rounded-full overflow-hidden" style={{ height: 6, background: 'var(--border)' }}>
        {/* 100% reference tick */}
        <div className="relative h-full">
          <div className="h-full rounded-full" style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            transition: 'width 0.5s ease',
          }} />
          {/* 1× mark */}
          <div style={{
            position: 'absolute', left: '50%', top: 0, bottom: 0,
            width: 1, background: 'var(--text-secondary)', opacity: 0.4,
          }} />
        </div>
      </div>
      <div className="flex justify-between mt-0.5">
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)', opacity: 0.5 }}>0%</span>
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)', opacity: 0.5 }}>100%</span>
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)', opacity: 0.5 }}>200%</span>
      </div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function KellyPanel({ ensemble, regimeProbs, loading }) {
  const [halfKelly, setHalfKelly] = useState(true)

  const kelly = useMemo(
    () => computeKelly(ensemble?.band ?? ensemble, regimeProbs, halfKelly),
    [ensemble, regimeProbs, halfKelly]
  )

  const color = kelly ? (LEVEL_COLORS[kelly.adjLevel] ?? '#4a9eff') : 'var(--accent-blue)'

  if (loading && !kelly) {
    return (
      <div className="rounded-lg border p-4" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <div className="mono text-xs font-bold mb-3" style={{ color: 'var(--text-secondary)' }}>
          ◎ KELLY POSITION SIZING
        </div>
        <div className="h-24 rounded animate-pulse" style={{ background: 'var(--bg-secondary)' }} />
      </div>
    )
  }

  if (!kelly) return null

  const regimePct = ((1 - kelly.regimeMultiplier) * 100).toFixed(0)

  return (
    <div className="rounded-lg border p-4 space-y-4"
      style={{ borderColor: `${color}33`, background: 'var(--bg-card)' }}>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="mono text-xs font-bold" style={{ color }}>
            ◎ KELLY POSITION SIZING
          </h3>
          <p className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            Regime-adjusted · Kelly (1956) · MacLean et al. (2010)
          </p>
        </div>
        {/* Half / Full Kelly toggle */}
        <button
          onClick={() => setHalfKelly(h => !h)}
          className="mono text-xs px-3 py-1 rounded border transition-colors"
          style={{
            borderColor: halfKelly ? color : 'var(--border)',
            color:        halfKelly ? color : 'var(--text-secondary)',
            background:   halfKelly ? `${color}18` : 'transparent',
          }}
          title="Half-Kelly halves the position size, reducing drawdown risk significantly"
        >
          {halfKelly ? '½ Kelly' : 'Full Kelly'}
        </button>
      </div>

      {/* Gauge + stats */}
      <div className="flex items-center gap-4">
        <KellyGauge fraction={kelly.adjusted} color={color} />

        <div className="flex-1 space-y-2">
          {/* Level badge */}
          <div className="rounded px-2 py-1 mono text-xs font-bold inline-block"
            style={{ background: `${color}22`, color }}>
            {kelly.adjLevel.toUpperCase()}
          </div>
          <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            {LEVEL_LABELS[kelly.adjLevel]}
          </p>
          {kelly.clipped && (
            <p className="mono text-xs" style={{ color: '#ffd43b' }}>
              ⚠ Full Kelly ({(kelly.fullKelly * (halfKelly ? 0.5 : 1) * 100).toFixed(0)}%) clipped to 200% max
            </p>
          )}
        </div>
      </div>

      {/* Deployment bar */}
      <DeploymentBar fraction={kelly.adjusted} color={color} />

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-2">
        <Stat label="E[return]"  value={`${kelly.annualizedReturn > 0 ? '+' : ''}${kelly.annualizedReturn}%`}
          color={kelly.annualizedReturn >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'} />
        <Stat label="Forecast σ" value={`${kelly.annualizedVol}%`} />
        <Stat label="Full Kelly" value={`${(kelly.fullKelly * 100).toFixed(0)}%`} />
        <Stat label="Regime adj"
          value={kelly.regimeMultiplier < 1 ? `−${regimePct}%` : 'none'}
          color={kelly.regimeMultiplier < 1 ? '#ffd43b' : 'var(--text-secondary)'} />
      </div>

      {/* Regime breakdown if adjusted */}
      {kelly.regimeMultiplier < 0.99 && regimeProbs && (
        <div className="rounded px-3 py-2 mono text-xs"
          style={{ background: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
          <span className="font-bold" style={{ color: '#ffd43b' }}>Regime penalty active: </span>
          {kelly.regimeMultiplier < 0.65
            ? 'Crisis/bear conditions detected — Kelly scaled to protect capital (Ang & Timmermann 2012)'
            : 'Elevated volatility regime — modest Kelly reduction applied'}
        </div>
      )}

      {/* Methodology note */}
      <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
        {halfKelly
          ? '½ Kelly: 75% of max long-run growth, significantly lower drawdown than full Kelly (MacLean et al. 2010).'
          : 'Full Kelly maximises long-run geometric growth but produces extreme drawdowns — not recommended for most investors.'}
        {' '}σ estimated from ensemble 90% CI: (p95−p5)/(2×1.645). Not financial advice.
      </p>
    </div>
  )
}
