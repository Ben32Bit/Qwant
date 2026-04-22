import React, { useMemo, memo } from 'react'
import { heatmapColor } from '../../utils/chartConfig.js'
import { fmtPct } from '../../utils/formatters.js'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function MonthlyHeatmap({ monthlyReturns, loading }) {
  const { years, grid } = useMemo(() => {
    if (!monthlyReturns?.data) return { years: [], grid: {} }
    const years = Object.keys(monthlyReturns.data).sort()
    return { years, grid: monthlyReturns.data }
  }, [monthlyReturns])

  if (loading) {
    return (
      <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div className="skeleton h-4 w-40 mb-4" />
        <div className="skeleton h-40 w-full rounded" />
      </div>
    )
  }

  if (!monthlyReturns || years.length === 0) return null

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <h3 className="mono font-bold text-sm mb-4" style={{ color: 'var(--text-primary)' }}>
        Monthly Returns
      </h3>

      <div className="overflow-x-auto">
        <table style={{ borderCollapse: 'separate', borderSpacing: 2 }}>
          <thead>
            <tr>
              <th className="mono text-xs pr-3 text-right" style={{ color: 'var(--text-secondary)' }}>Year</th>
              {MONTHS.map((m) => (
                <th key={m} className="mono text-xs text-center w-12" style={{ color: 'var(--text-secondary)' }}>
                  {m}
                </th>
              ))}
              <th className="mono text-xs pl-2 text-right" style={{ color: 'var(--text-secondary)' }}>Annual</th>
            </tr>
          </thead>
          <tbody>
            {years.map((year) => {
              const monthData = grid[year] || {}
              const validReturns = Object.values(monthData).filter((v) => v != null)
              const annualReturn = validReturns.length > 0
                ? validReturns.reduce((acc, r) => acc * (1 + r), 1) - 1
                : null

              return (
                <tr key={year}>
                  <td
                    className="mono text-xs pr-3 text-right font-bold"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    {year}
                  </td>
                  {MONTHS.map((month) => {
                    const val = monthData[month]
                    return (
                      <td key={month} className="text-center p-0.5">
                        <div
                          className="rounded mono text-xs py-1 px-1 text-center"
                          style={{
                            background: heatmapColor(val),
                            color: val == null ? 'var(--text-secondary)' : Math.abs(val) > 0.03 ? '#fff' : 'var(--text-primary)',
                            minWidth: 44,
                            fontSize: 10,
                          }}
                          title={val != null ? fmtPct(val) : '—'}
                        >
                          {val != null ? fmtPct(val, 1, false) : '·'}
                        </div>
                      </td>
                    )
                  })}
                  <td className="pl-2 text-right">
                    <span
                      className="mono text-xs font-bold"
                      style={{
                        color: annualReturn == null
                          ? 'var(--text-secondary)'
                          : annualReturn >= 0
                          ? 'var(--accent-green)'
                          : 'var(--accent-red)',
                      }}
                    >
                      {annualReturn != null ? fmtPct(annualReturn, 1) : '—'}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default memo(MonthlyHeatmap)
