import React, { useMemo } from 'react'

function corrColor(val) {
  if (val == null) return 'var(--bg-secondary)'
  // -1 → red, 0 → neutral, +1 → blue
  if (val >= 0) {
    const intensity = val
    const r = Math.round(74 + (0 - 74) * intensity)
    const g = Math.round(158 + (100 - 158) * intensity)
    const b = Math.round(255 + (200 - 255) * intensity)
    return `rgba(${r},${g},${b},${0.15 + intensity * 0.55})`
  } else {
    const intensity = Math.abs(val)
    return `rgba(255,71,87,${0.1 + intensity * 0.55})`
  }
}

function corrTextColor(val) {
  if (val == null) return 'var(--text-secondary)'
  if (Math.abs(val) > 0.5) return '#fff'
  return 'var(--text-primary)'
}

export default function CorrelationMatrix({ correlationMatrix, loading }) {
  const { tickers, matrix } = useMemo(() => {
    if (!correlationMatrix) return { tickers: [], matrix: [] }
    const tickers = Object.keys(correlationMatrix)
    const matrix = tickers.map(row =>
      tickers.map(col => correlationMatrix[row]?.[col] ?? null)
    )
    return { tickers, matrix }
  }, [correlationMatrix])

  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div className="skeleton h-4 w-40 mb-4" />
        <div className="skeleton h-32 w-full rounded" />
      </div>
    )
  }

  if (!correlationMatrix || tickers.length < 2) return null

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <h3 className="mono font-bold text-sm mb-4" style={{ color: 'var(--text-primary)' }}>
        Asset Correlations
      </h3>

      <div className="overflow-x-auto">
        <table style={{ borderCollapse: 'separate', borderSpacing: 3 }}>
          <thead>
            <tr>
              <th className="pr-3" />
              {tickers.map(t => (
                <th key={t} className="mono text-xs text-center px-1" style={{ color: 'var(--text-secondary)', minWidth: 56 }}>
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((row, ri) => (
              <tr key={row}>
                <td className="mono text-xs font-bold pr-3 text-right" style={{ color: 'var(--text-secondary)' }}>
                  {row}
                </td>
                {matrix[ri].map((val, ci) => (
                  <td key={ci} className="text-center p-0.5">
                    <div
                      className="mono text-xs rounded py-1.5 px-1"
                      style={{
                        background: corrColor(val),
                        color: corrTextColor(val),
                        minWidth: 52,
                        fontSize: 11,
                      }}
                    >
                      {val != null ? val.toFixed(2) : '—'}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mono text-xs mt-3" style={{ color: 'var(--text-secondary)' }}>
        Blue = positive correlation · Red = negative correlation · Diagonal = 1.00
      </p>
    </div>
  )
}
