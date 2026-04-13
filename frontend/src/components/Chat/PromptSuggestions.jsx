import React from 'react'

const SUGGESTIONS = [
  '60/40 US stocks and bonds over 10 years',
  'All-weather portfolio since 2010',
  'Max Sharpe portfolio from SPY, TLT, GLD, EEM',
  'Risk parity across stocks, bonds, gold, commodities',
]

export default function PromptSuggestions({ onSelect }) {
  return (
    <div className="p-4">
      <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
        Try one of these examples:
      </p>
      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSelect(s)}
            className="px-3 py-1.5 rounded text-xs border transition-all hover:border-opacity-80 text-left"
            style={{
              borderColor: 'var(--border)',
              color: 'var(--text-secondary)',
              background: 'var(--bg-card)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-blue)'
              e.currentTarget.style.color = 'var(--text-primary)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--border)'
              e.currentTarget.style.color = 'var(--text-secondary)'
            }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}
