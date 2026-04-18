/**
 * ScenarioPanel — Macro Scenario Stress Tester (Phase 8).
 *
 * Re-computes regime-conditional ensemble weights and Kelly position sizing
 * under six alternative macro scenarios, comparing each against the live
 * (model-derived) baseline.
 *
 * All computation is client-side — no additional API calls.
 * Uses the same MetaEnsemble and KellyCalculator logic as the live panel.
 *
 * Scenario regime distributions are calibrated to historical precedent:
 *   Soft Landing  — Campbell & Cochrane (1999): low surplus consumption, benign vol
 *   Rate Spike    — Ang et al. (2006) "What Does the Yield Curve Tell Us About GDP Growth"
 *   Mild Recession — NBER-defined recession: VIX 40–60 range, HMM bear prob dominant
 *   Severe Crisis  — GFC / COVID peak: VIX > 60, forced de-leveraging
 *   Stagflation    — 1970s analogue: negative real rates, bonds & equities fall together
 *
 * References:
 *   Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets.
 *     Annual Review of Financial Economics, 4(1), 313–337.
 *   Campbell, J.Y. & Cochrane, J.H. (1999). By Force of Habit.
 *     Journal of Political Economy, 107(2), 205–251.
 */

import { useMemo, useState } from 'react'
import { blendWeights } from '../../ml/MetaEnsemble.js'
import { computeKellyFromEnsemble, regimeAdjustKelly } from '../../ml/KellyCalculator.js'

// ── Scenario definitions ──────────────────────────────────────────────────────

const SCENARIOS = {
  current: null,  // populated from live regimeProbs
  soft_landing:   { bull_low_vol: 0.74, bull_high_vol: 0.14, bear: 0.09, crisis: 0.03 },
  rate_spike:     { bull_low_vol: 0.14, bull_high_vol: 0.56, bear: 0.24, crisis: 0.06 },
  mild_recession: { bull_low_vol: 0.06, bull_high_vol: 0.08, bear: 0.55, crisis: 0.31 },
  severe_crisis:  { bull_low_vol: 0.02, bull_high_vol: 0.03, bear: 0.28, crisis: 0.67 },
  stagflation:    { bull_low_vol: 0.08, bull_high_vol: 0.24, bear: 0.48, crisis: 0.20 },
}

const SCENARIO_META = {
  current:       { label: 'Current',        color: '#4a9eff', icon: '◉' },
  soft_landing:  { label: 'Soft Landing',   color: '#00d4aa', icon: '↗' },
  rate_spike:    { label: 'Rate Spike',     color: '#ffd43b', icon: '↑' },
  mild_recession:{ label: 'Recession',      color: '#ff6b35', icon: '↘' },
  severe_crisis: { label: 'Crisis',         color: '#ff4757', icon: '⚠' },
  stagflation:   { label: 'Stagflation',    color: '#a855f7', icon: '≈' },
}

const SCENARIO_DESC = {
  current:       'Live regime probabilities from the HMM + VIX model.',
  soft_landing:  'Fed achieves 2% inflation without recession. Equity multiples stable, vol subdued.',
  rate_spike:    '10Y yields surge 100+ bps. Growth equities re-rate; duration pain in bonds.',
  mild_recession:'GDP contracts 2 quarters. Earnings fall 15–20%. VIX 40–60. 12-month recovery.',
  severe_crisis: 'System-wide credit event. VIX > 60. Forced de-leveraging and liquidity freeze.',
  stagflation:   '1970s analogue: persistent inflation + stagnant growth. Bonds and equities fall together.',
}

const REGIME_LABELS = { bull_low_vol: 'Bull/Low Vol', bull_high_vol: 'Bull/High Vol', bear: 'Bear', crisis: 'Crisis' }
const REGIME_COLORS = { bull_low_vol: '#00d4aa', bull_high_vol: '#4a9eff', bear: '#ff6b35', crisis: '#ff4757' }
const METHOD_COLORS = {
  factor: '#00d4aa', xgboost: '#4a9eff', hmm: '#a855f7',
  var: '#ff6b35', nbeats: '#ffd43b', lstm: '#ff4757',
}

// ── Sub-components ────────────────────────────────────────────────────────────

function RegimePips({ probs }) {
  const regimes = ['bull_low_vol', 'bull_high_vol', 'bear', 'crisis']
  return (
    <div className="flex gap-1">
      {regimes.map(r => {
        const pct = ((probs[r] ?? 0) * 100).toFixed(0)
        const color = REGIME_COLORS[r]
        return (
          <div key={r} className="flex-1">
            <div className="rounded-sm overflow-hidden mb-0.5"
              style={{ height: 4, background: 'var(--border)' }}>
              <div style={{ height: '100%', width: `${pct}%`, background: color, transition: 'width 0.4s ease' }} />
            </div>
            <div className="mono text-center" style={{ fontSize: 9, color }}>
              {pct}%
            </div>
          </div>
        )
      })}
    </div>
  )
}

function WeightBar({ method, weight, baseWeight }) {
  const color  = METHOD_COLORS[method] ?? '#888'
  const delta  = weight - (baseWeight ?? weight)
  const pct    = Math.round(weight * 100)
  const sign   = delta > 0.005 ? '+' : delta < -0.005 ? '−' : ''
  const dColor = delta > 0.005 ? 'var(--accent-green)' : delta < -0.005 ? 'var(--accent-red)' : 'var(--text-secondary)'

  return (
    <div className="flex items-center gap-2">
      <span className="mono text-xs w-16 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{method}</span>
      <div className="flex-1 rounded-full overflow-hidden" style={{ height: 5, background: 'var(--border)' }}>
        <div className="h-full rounded-full"
          style={{ width: `${pct}%`, background: color, transition: 'width 0.4s ease' }} />
      </div>
      <span className="mono text-xs w-8 text-right flex-shrink-0" style={{ color }}>{pct}%</span>
      {sign && (
        <span className="mono flex-shrink-0" style={{ fontSize: 10, color: dColor, width: 24, textAlign: 'right' }}>
          {sign}{Math.abs(Math.round(delta * 100))}
        </span>
      )}
    </div>
  )
}

function KellyDelta({ baseline, scenario, color }) {
  if (!baseline || !scenario) return null
  const delta = scenario - baseline
  const sign  = delta > 0 ? '+' : ''
  const dColor = delta > 0.01 ? 'var(--accent-green)' : delta < -0.01 ? 'var(--accent-red)' : 'var(--text-secondary)'
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1">
        <div className="flex justify-between mb-1">
          <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>Capital deployment</span>
          <span className="mono text-xs font-bold" style={{ color }}>
            {(scenario * 100).toFixed(0)}%
            <span className="ml-1" style={{ color: dColor, fontSize: 10 }}>
              ({sign}{(delta * 100).toFixed(0)}pp)
            </span>
          </span>
        </div>
        <div className="rounded-full overflow-hidden" style={{ height: 5, background: 'var(--border)' }}>
          <div className="h-full rounded-full"
            style={{
              width: `${Math.min(scenario / 2, 1) * 100}%`,
              background: `linear-gradient(90deg, ${color}88, ${color})`,
              transition: 'width 0.4s ease',
            }} />
        </div>
      </div>
    </div>
  )
}

// ── Scenario card (expanded detail) ──────────────────────────────────────────

function ScenarioDetail({ scenarioKey, scenarioProbs, availableMethods, ensemble, baseWeights, baseKelly }) {
  const weights = useMemo(
    () => blendWeights(scenarioProbs, availableMethods),
    [scenarioProbs, availableMethods]
  )

  const kelly = useMemo(() => {
    const band = ensemble?.band ?? ensemble
    if (!band) return null
    const base    = computeKellyFromEnsemble(band, true)
    if (!base) return null
    return regimeAdjustKelly(base.fraction, scenarioProbs)
  }, [ensemble, scenarioProbs])

  const meta  = SCENARIO_META[scenarioKey]
  const color = meta?.color ?? '#4a9eff'

  const methodOrder = ['factor', 'xgboost', 'hmm', 'var', 'nbeats', 'lstm']

  return (
    <div className="space-y-3">
      {/* Description */}
      <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
        {SCENARIO_DESC[scenarioKey]}
      </p>

      {/* Regime bars */}
      <div>
        <p className="mono text-xs mb-1" style={{ color: 'var(--text-secondary)', opacity: 0.7 }}>Regime distribution</p>
        <RegimePips probs={scenarioProbs} />
        <div className="flex justify-between mt-0.5">
          {['bull_low_vol','bull_high_vol','bear','crisis'].map(r => (
            <span key={r} className="mono flex-1 text-center" style={{ fontSize: 8, color: 'var(--text-secondary)', opacity: 0.6 }}>
              {REGIME_LABELS[r].split('/').pop()}
            </span>
          ))}
        </div>
      </div>

      {/* Ensemble weights */}
      <div>
        <p className="mono text-xs mb-2" style={{ color: 'var(--text-secondary)', opacity: 0.7 }}>
          Ensemble weights
          {scenarioKey !== 'current' && <span style={{ opacity: 0.5 }}> (vs baseline)</span>}
        </p>
        <div className="space-y-1.5">
          {methodOrder
            .filter(m => availableMethods.includes(m))
            .sort((a, b) => (weights[b] ?? 0) - (weights[a] ?? 0))
            .map(m => (
              <WeightBar
                key={m}
                method={m}
                weight={weights[m] ?? 0}
                baseWeight={scenarioKey !== 'current' ? (baseWeights?.[m] ?? 0) : null}
              />
            ))}
        </div>
      </div>

      {/* Kelly fraction */}
      {kelly != null && (
        <KellyDelta
          baseline={baseKelly}
          scenario={kelly}
          color={color}
        />
      )}
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ScenarioPanel({ ensemble, regimeProbs, results }) {
  const [selected, setSelected] = useState('current')

  const availableMethods = useMemo(
    () => (results ?? []).filter(r => r.forecast?.p50?.length > 0).map(r => r.method),
    [results]
  )

  const baseWeights = useMemo(
    () => regimeProbs ? blendWeights(regimeProbs, availableMethods) : {},
    [regimeProbs, availableMethods]
  )

  const baseKelly = useMemo(() => {
    const band = ensemble?.band ?? ensemble
    if (!band || !regimeProbs) return null
    const base = computeKellyFromEnsemble(band, true)
    if (!base) return null
    return regimeAdjustKelly(base.fraction, regimeProbs)
  }, [ensemble, regimeProbs])

  if (!ensemble || !regimeProbs) return null

  // Build scenarios including current
  const allScenarios = { current: regimeProbs, ...SCENARIOS }
  delete allScenarios.current
  const scenariosWithCurrent = { current: regimeProbs, ...SCENARIOS }

  const activeProbs = scenariosWithCurrent[selected] ?? regimeProbs
  const activeMeta  = SCENARIO_META[selected]

  return (
    <div className="rounded-lg border p-4 space-y-4"
      style={{ borderColor: `${activeMeta.color}33`, background: 'var(--bg-card)' }}>

      {/* Header */}
      <div>
        <h3 className="mono text-xs font-bold" style={{ color: activeMeta.color }}>
          ⚑ SCENARIO STRESS TEST
        </h3>
        <p className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
          Re-weights ensemble + Kelly under alternative macro regimes · Ang &amp; Timmermann (2012)
        </p>
      </div>

      {/* Scenario selector */}
      <div className="grid grid-cols-3 gap-1.5">
        {Object.keys(scenariosWithCurrent).map(key => {
          const m   = SCENARIO_META[key]
          const act = selected === key
          return (
            <button
              key={key}
              onClick={() => setSelected(key)}
              className="mono text-xs px-2 py-1.5 rounded border transition-colors text-left"
              style={{
                borderColor: act ? m.color : 'var(--border)',
                color:       act ? m.color : 'var(--text-secondary)',
                background:  act ? `${m.color}18` : 'transparent',
                cursor:      'pointer',
              }}
            >
              <span style={{ marginRight: 4 }}>{m.icon}</span>{m.label}
            </button>
          )
        })}
      </div>

      {/* Detail for selected scenario */}
      <ScenarioDetail
        key={selected}
        scenarioKey={selected}
        scenarioProbs={activeProbs}
        availableMethods={availableMethods}
        ensemble={ensemble}
        baseWeights={selected !== 'current' ? baseWeights : null}
        baseKelly={selected !== 'current' ? baseKelly : null}
      />

      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {['bull_low_vol','bull_high_vol','bear','crisis'].map(r => (
          <span key={r} className="mono flex items-center gap-1" style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
            <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: REGIME_COLORS[r] }} />
            {REGIME_LABELS[r]}
          </span>
        ))}
      </div>
    </div>
  )
}
