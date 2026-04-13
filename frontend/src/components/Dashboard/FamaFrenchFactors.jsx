import React, { useState, useRef } from 'react'
import { createPortal } from 'react-dom'

const PAPER_CITATION = 'Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010'

const FACTOR_META = {
  alpha: {
    label: 'Alpha (α)',
    description: 'Annualised excess return unexplained by the five factors — Jensen\'s alpha. A positive, significant alpha suggests genuine outperformance beyond compensated risk.',
    sign: 'pct',
  },
  mkt_rf: {
    label: 'Market (Mkt-RF)',
    description: 'Loading on the market excess return factor. A beta near 1.0 means the portfolio moves roughly in line with the broad market.',
    sign: 'ratio',
  },
  smb: {
    label: 'Size (SMB)',
    description: 'Small Minus Big — exposure to the size premium. Positive loading means the portfolio tilts toward small-cap stocks; negative toward large-caps.',
    sign: 'ratio',
  },
  hml: {
    label: 'Value (HML)',
    description: 'High Minus Low book-to-market — exposure to the value premium. Positive loading indicates a value tilt; negative indicates a growth tilt.',
    sign: 'ratio',
  },
  rmw: {
    label: 'Profitability (RMW)',
    description: 'Robust Minus Weak operating profitability. Positive loading means the portfolio skews toward more profitable firms.',
    sign: 'ratio',
  },
  cma: {
    label: 'Investment (CMA)',
    description: 'Conservative Minus Aggressive investment — exposure to the investment premium. Positive loading indicates conservative (low-investment) firms; negative indicates aggressive.',
    sign: 'ratio',
  },
}

const TOOLTIP_WIDTH = 288

function InfoTooltip({ content, citation }) {
  const [show, setShow] = useState(false)
  const [pos, setPos] = useState({ top: 0, left: 0 })
  const btnRef = useRef(null)

  const handleShow = () => {
    if (btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - rect.bottom
      const top = spaceBelow > 160 ? rect.bottom + 6 : rect.top - 160
      const left = Math.min(rect.left, window.innerWidth - TOOLTIP_WIDTH - 12)
      setPos({ top, left })
    }
    setShow(true)
  }

  return (
    <span className="inline-block ml-1 align-middle" style={{ position: 'relative' }}>
      <button
        ref={btnRef}
        className="mono rounded-full flex items-center justify-center"
        style={{
          width: 16, height: 16,
          background: 'rgba(74,158,255,0.15)',
          color: 'var(--accent-blue)',
          border: '1px solid rgba(74,158,255,0.3)',
          cursor: 'pointer',
          fontSize: 10,
          lineHeight: 1,
          flexShrink: 0,
        }}
        onMouseEnter={handleShow}
        onMouseLeave={() => setShow(false)}
        onFocus={handleShow}
        onBlur={() => setShow(false)}
        aria-label="Factor description"
      >
        ?
      </button>
      {show && createPortal(
        <div
          className="rounded-lg border p-3 text-xs"
          style={{
            position: 'fixed',
            top: pos.top,
            left: pos.left,
            width: TOOLTIP_WIDTH,
            zIndex: 9999,
            background: 'var(--bg-secondary)',
            borderColor: 'var(--border)',
            color: 'var(--text-primary)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
            pointerEvents: 'none',
          }}
        >
          <p className="mb-2 leading-relaxed" style={{ color: 'var(--text-primary)' }}>{content}</p>
          <p className="leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
            📄 {citation}
          </p>
        </div>,
        document.body
      )}
    </span>
  )
}

function sigColor(t) {
  if (t == null || isNaN(t)) return 'var(--text-secondary)'
  const a = Math.abs(t)
  if (a > 3.29) return 'var(--accent-green)'
  if (a > 2.58) return '#ffd43b'
  if (a > 1.96) return '#fd79a8'
  return 'var(--text-secondary)'
}

function FactorRow({ factorKey, data }) {
  const meta = FACTOR_META[factorKey]
  const val = data[factorKey]
  const t = data[`${factorKey}_t_stat`]
  const stars = data[`${factorKey}_stars`] || ''

  const formattedVal = meta.sign === 'pct'
    ? `${(val * 100).toFixed(2)}%`
    : val?.toFixed(3)

  const isPositive = val > 0
  const valColor = factorKey === 'alpha'
    ? (isPositive ? 'var(--accent-green)' : 'var(--accent-red)')
    : 'var(--text-primary)'

  return (
    <tr style={{ borderBottom: '1px solid var(--border)' }}>
      <td className="py-2 px-3">
        <span className="mono text-xs font-bold" style={{ color: 'var(--text-primary)' }}>
          {meta.label}
        </span>
        <InfoTooltip content={meta.description} citation={PAPER_CITATION} />
      </td>
      <td className="py-2 px-3 text-right mono text-sm font-bold" style={{ color: valColor }}>
        {formattedVal ?? '—'}
      </td>
      <td className="py-2 px-3 text-right mono text-xs" style={{ color: sigColor(t) }}>
        {t != null ? t.toFixed(2) : '—'}
      </td>
    </tr>
  )
}

export default function FamaFrenchFactors({ ff5, loading }) {
  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div className="skeleton h-4 w-48 mb-4" />
        <div className="skeleton h-40 w-full rounded" />
      </div>
    )
  }

  if (!ff5) return null

  const sourceLabel = ff5.source === 'ken_french'
    ? 'Ken French Data Library'
    : 'ETF Proxies (fallback)'

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="mono font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
            Fama-French Five-Factor Decomposition
            <InfoTooltip
              content="Regresses portfolio excess returns on five systematic risk factors to decompose performance into compensated risk exposures and genuine alpha."
              citation={PAPER_CITATION}
            />
          </h3>
          <p className="mono text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            R² = <span className="font-bold" style={{ color: 'var(--text-primary)' }}>{(ff5.r_squared * 100).toFixed(1)}%</span>
            &nbsp;·&nbsp;n = {ff5.n_obs} obs
            &nbsp;·&nbsp;Source: {sourceLabel}
          </p>
        </div>
        <div className="mono text-xs px-2 py-1 rounded" style={{ background: 'rgba(74,158,255,0.1)', color: 'var(--accent-blue)', border: '1px solid rgba(74,158,255,0.2)' }}>
          FF5
        </div>
      </div>

      {/* Significance legend — t-stat colour key */}
      <div className="flex gap-4 mb-3 flex-wrap">
        {[
          { label: 'p<0.001', color: 'var(--accent-green)' },
          { label: 'p<0.01', color: '#ffd43b' },
          { label: 'p<0.05', color: '#fd79a8' },
          { label: 'n.s.', color: 'var(--text-secondary)' },
        ].map(({ label, color }) => (
          <span key={label} className="mono text-xs flex items-center gap-1" style={{ color }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: 'inline-block', flexShrink: 0 }} />
            {label}
          </span>
        ))}
      </div>

      {/* Factor table */}
      <div className="overflow-x-auto">
        <table className="w-full" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th className="py-1.5 px-3 text-left mono text-xs" style={{ color: 'var(--text-secondary)' }}>Factor</th>
              <th className="py-1.5 px-3 text-right mono text-xs" style={{ color: 'var(--text-secondary)' }}>Loading</th>
              <th className="py-1.5 px-3 text-right mono text-xs" style={{ color: 'var(--text-secondary)' }}>t-stat</th>
            </tr>
          </thead>
          <tbody>
            {['alpha', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma'].map(k => (
              <FactorRow key={k} factorKey={k} data={ff5} />
            ))}
          </tbody>
        </table>
      </div>

      {/* Citation footer */}
      <p className="mono text-xs mt-3 pt-3" style={{ color: 'var(--text-secondary)', borderTop: '1px solid var(--border)' }}>
        📄 {PAPER_CITATION}
      </p>
    </div>
  )
}
