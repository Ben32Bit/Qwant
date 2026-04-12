/**
 * Format a decimal as a percentage string.
 * e.g., 0.1234 → "+12.34%"
 */
export function fmtPct(val, decimals = 2, showSign = true) {
  if (val == null || isNaN(val)) return '—'
  const pct = val * 100
  const sign = showSign && pct > 0 ? '+' : ''
  return `${sign}${pct.toFixed(decimals)}%`
}

/**
 * Format a decimal as a plain percentage (no sign).
 */
export function fmtPctAbs(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—'
  return `${(val * 100).toFixed(decimals)}%`
}

/**
 * Format a number with comma separators and fixed decimals.
 */
export function fmtNum(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—'
  return val.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

/**
 * Format a dollar value.
 */
export function fmtDollar(val, decimals = 0) {
  if (val == null || isNaN(val)) return '—'
  return '$' + fmtNum(val, decimals)
}

/**
 * Format a ratio (e.g., Sharpe ratio).
 */
export function fmtRatio(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—'
  return val.toFixed(decimals)
}

/**
 * Return CSS class for a value (positive = green, negative = red).
 */
export function colorClass(val) {
  if (val == null || isNaN(val)) return 'neutral'
  if (val > 0) return 'positive'
  if (val < 0) return 'negative'
  return 'neutral'
}

/**
 * Format a date string YYYY-MM-DD to display format.
 */
export function fmtDate(dateStr) {
  if (!dateStr) return '—'
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

/**
 * Format year-month (YYYY-MM) for display.
 */
export function fmtYearMonth(dateStr) {
  if (!dateStr) return '—'
  const [year, month] = dateStr.split('-')
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${months[parseInt(month) - 1]} ${year}`
}
