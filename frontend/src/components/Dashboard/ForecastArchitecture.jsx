/**
 * ForecastArchitecture — Collapsible pipeline diagram for the forecast system.
 * Shows every layer: data ingestion → 6 base models → regime → ensemble → outputs.
 */
import { useState } from 'react'

const METHOD_COLORS = {
  factor:  '#00d4aa', xgboost: '#4a9eff', nbeats: '#ffd43b',
  hmm:     '#a855f7', var:     '#ff6b35', lstm:   '#ff4757',
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

      {/* Layer 0: Data sources */}
      <LayerLabel>① Data ingestion</LayerLabel>
      <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
        <Pill label="yfinance" sub="prices" color="#4a9eff" />
        <Pill label="FRED" sub="macro" color="#00d4aa" />
        <Pill label="GDELT" sub="news" color="#ffd43b" />
        <Pill label="EDGAR" sub="10-K/Q" color="#a855f7" />
        <Pill label="Reddit" sub="WSB" color="#ff6b35" />
      </div>

      <Arrow />

      {/* Layer 1: Server Phase 1 */}
      <LayerLabel>② Server — Phase 1 · fast (≈4s)</LayerLabel>
      <div className="flex gap-2">
        <Box label="Factor Model" sub="FF5+Mom loadings × premia" color={METHOD_COLORS.factor} tag="returns forecast" />
        <Box label="XGBoost features" sub="14 signals → ONNX browser" color={METHOD_COLORS.xgboost} tag="feature prep" />
        <Box label="N-BEATS features" sub="30-day window → pure-JS" color={METHOD_COLORS.nbeats} tag="feature prep" />
      </div>

      <Arrow />

      {/* Layer 2: Browser Phase 1B (parallel with Phase 2) */}
      <div className="flex gap-2">
        <div className="flex-1 space-y-0">
          <LayerLabel>③ Browser — Phase 1B · ONNX / pure-JS</LayerLabel>
          <div className="flex gap-2">
            <Box label="XGBoost ONNX" sub="5 quantile models · √t scaling" color={METHOD_COLORS.xgboost} tag="fan chart" />
            <Box label="N-BEATS" sub="12×21d recursive periods" color={METHOD_COLORS.nbeats} tag="fan chart" />
          </div>
        </div>
        <div className="flex-1 space-y-0">
          <LayerLabel>④ Server — Phase 2 · slow (≈15s)</LayerLabel>
          <div className="flex gap-2">
            <Box label="HMM" sub="2-state Baum-Welch · regime" color={METHOD_COLORS.hmm} tag="fan chart" />
            <Box label="GP" sub="Matérn 5/2 · Bayesian" color={METHOD_COLORS.var} tag="fan chart" />
          </div>
        </div>
      </div>

      <Arrow />

      {/* Layer 3: Browser Phase 3 */}
      <LayerLabel>⑤ Browser — Phase 3 · TF.js</LayerLabel>
      <Box label="Attention-LSTM" sub="LSTM(64) → Bahdanau attention → MC Dropout (200 passes)" color={METHOD_COLORS.lstm} tag="fan chart" />

      <Arrow />

      {/* Layer 4: Regime detection */}
      <LayerLabel>⑥ Regime classification</LayerLabel>
      <div className="rounded-lg px-3 py-2 flex items-center gap-4"
        style={{ background: 'rgba(168,85,247,0.08)', border: '1px solid rgba(168,85,247,0.3)' }}>
        <div>
          <div className="mono font-bold" style={{ fontSize: 10, color: '#a855f7' }}>HMM × VIX → 4-state regime</div>
          <div className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
            Bull/Low-Vol · Bull/High-Vol · Bear · Crisis
          </div>
        </div>
        <div className="flex gap-1 ml-auto">
          {[['Bull', '#00d4aa'], ['Bull↑σ', '#4a9eff'], ['Bear', '#ff6b35'], ['Crisis', '#ff4757']].map(([l, c]) => (
            <span key={l} className="mono px-1.5 py-0.5 rounded" style={{ fontSize: 8, background: `${c}22`, color: c, border: `1px solid ${c}44` }}>{l}</span>
          ))}
        </div>
      </div>

      <Arrow />

      {/* Layer 5: Meta-ensemble */}
      <LayerLabel>⑦ Meta-ensemble · Wolpert (1992) stacked generalisation</LayerLabel>
      <div className="rounded-lg px-3 py-2"
        style={{ background: 'rgba(74,158,255,0.08)', border: '1px solid rgba(74,158,255,0.35)' }}>
        <div className="mono font-bold mb-1" style={{ fontSize: 10, color: 'var(--accent-blue)' }}>
          Regime-conditional ensemble
        </div>
        <div className="flex gap-2 flex-wrap">
          {[
            ['Bull/LV', 'factor 35% · xgb 25% · hmm 18%', '#00d4aa'],
            ['Bull/HV', 'xgb 35% · hmm 25% · factor 18%', '#4a9eff'],
            ['Bear',    'hmm 35% · var 25% · factor 18%',  '#ff6b35'],
            ['Crisis',  'var 35% · hmm 30% · xgb 15%',    '#ff4757'],
          ].map(([regime, weights, color]) => (
            <div key={regime} className="rounded px-2 py-1" style={{ background: `${color}12`, border: `1px solid ${color}30` }}>
              <div className="mono font-bold" style={{ fontSize: 9, color }}>{regime}</div>
              <div className="mono" style={{ fontSize: 8, color: 'var(--text-secondary)' }}>{weights}</div>
            </div>
          ))}
        </div>
      </div>

      <Arrow />

      {/* Layer 6: Outputs */}
      <LayerLabel>⑧ Outputs</LayerLabel>
      <div className="flex gap-2">
        <Box label="Fan charts" sub="p5/p25/p50/p75/p95 bands at 63 days" color="#4a9eff" tag="per method" />
        <Box label="Kelly sizing" sub="f* = μ/σ² · half-Kelly · regime adj" color="#00d4aa" tag="position size" />
        <Box label="Scenario stress" sub="6 macro regimes · weight + Kelly delta" color="#ffd43b" tag="what-if" />
        <Box label="FinBERT" sub="News + SEC filings · browser NLP" color="#a855f7" tag="sentiment" />
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-3 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
        {[['Server', '#4a9eff'], ['Browser', '#00d4aa'], ['Regime', '#a855f7']].map(([l, c]) => (
          <span key={l} className="mono flex items-center gap-1" style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
            <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: 2, background: c }} />
            {l}
          </span>
        ))}
        <span className="mono ml-auto" style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.6 }}>
          Ang & Timmermann (2012) · Wolpert (1992) · Kelly (1956) · López de Prado (2018)
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
            · 8-layer pipeline · 6 research models · regime-conditional ensemble
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
