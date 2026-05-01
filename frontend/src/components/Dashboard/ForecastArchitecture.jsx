/**
 * ForecastArchitecture — Collapsible pipeline diagram for the forecast system.
 * Shows every layer: inputs → providers → two-pass eval → 5 methods → OOS R² ensemble → outputs.
 */
import { useState } from 'react'

const METHOD_COLORS = {
  nbeats: '#ffd43b', timesfm: '#00d4aa', hmm: '#a855f7', var: '#ff6b35', lstm: '#ff4757',
}

// ── Small building blocks ─────────────────────────────────────────────────────

function Pill({ label, color, sub }) {
  return (
    <div className="rounded px-2 py-1 text-center"
      style={{ background: `${color}18`, border: `1px solid ${color}44` }}>
      <div className="mono font-bold" style={{ fontSize: 9, color }}>{label}</div>
      {sub && <div className="mono" style={{ fontSize: 8, color: `${color}aa`, marginTop: 1 }}>{sub}</div>}
    </div>
  )
}

function Box({ label, sub, color, tag }) {
  return (
    <div className="rounded-lg px-3 py-2 flex-1"
      style={{ background: `${color}12`, border: `1px solid ${color}40`, minWidth: 0 }}>
      {tag && (
        <div className="mono mb-0.5" style={{ fontSize: 8, color: `${color}99`, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {tag}
        </div>
      )}
      <div className="mono font-bold" style={{ fontSize: 10, color }}>{label}</div>
      {sub && <div className="mono leading-relaxed" style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Arrow() {
  return (
    <div className="flex justify-center py-1">
      <div style={{ width: 1, height: 14, background: 'var(--border)' }} />
    </div>
  )
}

function LayerLabel({ children }) {
  return (
    <div className="mono mb-1.5" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
      {children}
    </div>
  )
}

// ── Architecture diagram ──────────────────────────────────────────────────────

function Diagram() {
  return (
    <div className="space-y-0 mt-2">

      {/* Layer ①: Inputs */}
      <LayerLabel>① Inputs</LayerLabel>
      <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <Pill label="Portfolio equity curve" color="#4a9eff" />
        <Pill label="Asset weights" color="#4a9eff" />
        <Pill label="Date range" color="#4a9eff" />
      </div>

      <Arrow />

      {/* Layer ②: Provider fan-out */}
      <LayerLabel>② Provider fan-out · parallel</LayerLabel>
      <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
        <Pill label="FRED" sub="macro" color="#00d4aa" />
        <Pill label="VIX" sub="vol regime" color="#ffd43b" />
        <Pill label="GDELT" sub="news" color="#a855f7" />
        <Pill label="Reddit" sub="sentiment" color="#ff6b35" />
        <Pill label="SEC" sub="insider" color="#ff4757" />
        <Pill label="EDGAR" sub="10-K/Q" color="#4a9eff" />
      </div>

      <Arrow />

      {/* Layer ③: Two-pass evaluation */}
      <LayerLabel>③ Two-pass evaluation · per method</LayerLabel>
      <div className="rounded-lg px-3 py-2"
        style={{ background: 'rgba(74,158,255,0.06)', border: '1px solid rgba(74,158,255,0.25)' }}>
        <div className="grid gap-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
          <div>
            <div className="mono font-bold mb-1" style={{ fontSize: 9, color: '#ff6b35' }}>
              SHADOW  [start → T−60] → forecast [T−60, T−30]
            </div>
            <div className="mono" style={{ fontSize: 8, color: 'var(--text-secondary)' }}>
              Shadow p50 vs actual → OOS R² · drives ensemble weights
            </div>
          </div>
          <div>
            <div className="mono font-bold mb-1" style={{ fontSize: 9, color: '#00d4aa' }}>
              FORWARD  [start → T] → forecast [T, T+63]
            </div>
            <div className="mono" style={{ fontSize: 8, color: 'var(--text-secondary)' }}>
              Full-history forward pass · displayed in fan charts
            </div>
          </div>
        </div>
      </div>

      <Arrow />

      {/* Layer ④: Five forecast methods */}
      <LayerLabel>④ Five forecast methods</LayerLabel>
      <div className="space-y-1">
        <div className="flex gap-1">
          <Box label="N-BEATS" sub="3 residual stacks · pure-JS browser · 63d recursive" color={METHOD_COLORS.nbeats} tag="browser · phase 1" />
          <Box label="LSTM" sub="Attention-LSTM(64) · MC Dropout 200× · TF.js" color={METHOD_COLORS.lstm} tag="browser · phase 3" />
        </div>
        <div className="flex gap-1">
          <Box label="HMM" sub="2-state Baum-Welch · regime-conditional sim · 1 000 paths" color={METHOD_COLORS.hmm} tag="server · phase 2" />
          <Box label="GP" sub="Matérn ν=5/2 ARD · Bayesian uncertainty · rollout" color={METHOD_COLORS.var} tag="server · phase 2" />
          <Box label="TimesFM 2.5" sub="Google 200M · zero-shot · always-warm singleton" color={METHOD_COLORS.timesfm} tag="server · phase 2" />
        </div>
      </div>

      <Arrow />

      {/* Layer ⑤: OOS R²-weighted ensemble */}
      <LayerLabel>⑤ Ensemble blending · OOS R²-weighted · Wolpert (1992)</LayerLabel>
      <div className="rounded-lg px-3 py-2"
        style={{ background: 'rgba(0,212,170,0.07)', border: '1px solid rgba(0,212,170,0.3)' }}>
        <div className="mono font-bold mb-1" style={{ fontSize: 10, color: 'var(--accent-green)' }}>
          OOS R² → clip(0) → normalise → TimesFM ≥ 30% floor → blend p5/p25/p50/p75/p95
        </div>
        <div className="flex gap-3 flex-wrap">
          {[
            ['Disagreement', 'Krogh & Vedelsby (1995) — high disagreement widens outer bands'],
            ['Regime display', 'HMM × VIX → 4-state (bull/bear × vol) — shown in donut, not used for weighting'],
          ].map(([title, desc]) => (
            <div key={title} className="rounded px-2 py-1" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
              <div className="mono font-bold" style={{ fontSize: 8, color: 'var(--text-secondary)' }}>{title}</div>
              <div className="mono" style={{ fontSize: 8, color: 'var(--text-secondary)', opacity: 0.7 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      <Arrow />

      {/* Layer ⑥: Outputs */}
      <LayerLabel>⑥ Outputs</LayerLabel>
      <div className="flex gap-2">
        <Box label="Fan charts" sub="p5/p25/p50/p75/p95 · IS R² + OOS R² per card" color="#4a9eff" tag="per method" />
        <Box label="Kelly sizing" sub="f* = μ/σ² · half-Kelly · regime confidence adj" color="#00d4aa" tag="position size" />
        <Box label="Scenario stress" sub="6 macro regimes · weight + Kelly delta" color="#ffd43b" tag="what-if" />
        <Box label="FinBERT" sub="News + SEC filings · browser NLP" color="#a855f7" tag="sentiment" />
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-3 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
        {[['Server', '#4a9eff'], ['Browser', '#00d4aa'], ['Ensemble', 'var(--accent-green)']].map(([l, c]) => (
          <span key={l} className="mono flex items-center gap-1" style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
            <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: 2, background: c }} />
            {l}
          </span>
        ))}
        <span className="mono ml-auto" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>
          Wolpert (1992) · Krogh & Vedelsby (1995) · Kelly (1956) · López de Prado (2018)
        </span>
      </div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ForecastArchitecture() {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-lg border overflow-hidden"
      style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-left"
        style={{ background: 'transparent', cursor: 'pointer', border: 'none' }}
      >
        <div className="flex items-center gap-2">
          <span className="mono font-bold text-xs" style={{ color: 'var(--text-secondary)' }}>
            ⬡ FORECAST ARCHITECTURE
          </span>
          <span className="mono text-xs" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
            · 6-layer pipeline · 5 research models · OOS R²-weighted ensemble
          </span>
        </div>
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          {open ? '▲ collapse' : '▼ expand'}
        </span>
      </button>

      {open && (
        <div className="px-4 pb-4">
          <Diagram />
        </div>
      )}
    </div>
  )
}
