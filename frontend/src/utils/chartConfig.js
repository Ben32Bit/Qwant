export const CHART_COLORS = {
  portfolio: '#4a9eff',
  benchmark: '#8888a0',
  overlay1: '#00d4aa',
  overlay2: '#ffd43b',
  drawdown: '#ff4757',
  positive: '#00d4aa',
  negative: '#ff4757',
}

export const CHART_STYLE = {
  backgroundColor: 'transparent',
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11,
}

export const AXIS_STYLE = {
  tick: { fill: '#8888a0', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" },
  axisLine: { stroke: '#2a2a3a' },
  tickLine: { stroke: '#2a2a3a' },
}

export const TOOLTIP_STYLE = {
  backgroundColor: '#1a1a25',
  border: '1px solid #2a2a3a',
  borderRadius: '6px',
  color: '#e0e0e8',
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 12,
}

export const GRID_STYLE = {
  stroke: '#2a2a3a',
  strokeDasharray: '3 3',
}

// Heatmap color scale for monthly returns
export function heatmapColor(val) {
  if (val == null) return '#1a1a25'
  const maxAbsVal = 0.10 // 10% cap for color intensity
  const intensity = Math.min(Math.abs(val) / maxAbsVal, 1)
  if (val > 0) {
    // Green scale
    const r = Math.round(0 + intensity * 0)
    const g = Math.round(80 + intensity * 132)
    const b = Math.round(80 + intensity * 90)
    return `rgb(${r}, ${g}, ${b})`
  } else {
    // Red scale
    const r = Math.round(100 + intensity * 155)
    const g = Math.round(30 + intensity * 41 * (1 - intensity))
    const b = Math.round(30 + intensity * 57 * (1 - intensity))
    return `rgb(${r}, ${g}, ${b})`
  }
}
