import { useState, useRef, useEffect } from 'react'

const METHOD_ORDER  = ['nbeats', 'timesfm', 'hmm', 'var', 'lstm']
const METHOD_LABELS = { nbeats: 'N-BEATS', timesfm: 'TimesFM 2.5', hmm: 'HMM Regimes', var: 'Gaussian Process', lstm: 'Attention-LSTM' }
const METHOD_COLORS = { nbeats: '#ffd43b', timesfm: '#00d4aa', hmm: '#a855f7', var: '#ff6b35', lstm: '#ff4757' }

const SNAPSHOT_HORIZONS = [
  { label: '1 week',   days: 5,  idx: 4 },
  { label: '1 month',  days: 21, idx: 20 },
  { label: '3 months', days: 63, idx: 62 },
]

function fmtPct(v, digits = 2) {
  if (v == null || !Number.isFinite(v)) return '—'
  const sign = v >= 0 ? '+' : ''
  return `${sign}${v.toFixed(digits)}%`
}

function fmtPctFraction(v, digits = 2) {
  if (v == null || !Number.isFinite(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

function fmtUsd(v) {
  if (v == null || !Number.isFinite(v)) return '—'
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`
  return `$${v.toFixed(0)}`
}

function slugForFilename(portfolio) {
  const tickers = (portfolio?.assets ?? [])
    .slice(0, 4)
    .map(a => a.ticker)
    .join('-')
    .toUpperCase()
    .replace(/[^A-Z0-9-]/g, '')
  const stamp = new Date().toISOString().slice(0, 10)
  return `forecast-${stamp}${tickers ? `-${tickers}` : ''}`
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// ── HTML report generation ────────────────────────────────────────────────────

function buildHtmlReport({ portfolio, backtest, results, ensemble, meta }) {
  const assets = portfolio?.assets ?? []
  const totalWeight = assets.reduce((s, a) => s + (a.weight ?? 0), 0)
  const metrics = backtest?.metrics ?? {}

  const assetsRows = assets.map(a => {
    const w = (a.weight ?? 0) * 100
    const wStr = `${w >= 0 ? '+' : ''}${w.toFixed(2)}%`
    const wColor = w >= 0 ? '#00875a' : '#c92a2a'
    return `<tr>
      <td class="mono">${a.ticker ?? '—'}</td>
      <td class="mono num" style="color:${wColor};font-weight:700">${wStr}</td>
      <td>${(a.rationale ?? '').replace(/</g, '&lt;')}</td>
    </tr>`
  }).join('')

  const backtestRows = [
    ['Total Return',  fmtPctFraction(metrics.total_return)],
    ['CAGR',          fmtPctFraction(metrics.cagr)],
    ['Volatility',    fmtPctFraction(metrics.volatility)],
    ['Sharpe',        metrics.sharpe != null ? metrics.sharpe.toFixed(3) : '—'],
    ['Sortino',       metrics.sortino != null ? metrics.sortino.toFixed(3) : '—'],
    ['Max Drawdown',  fmtPctFraction(metrics.max_drawdown)],
    ['Beta',          metrics.beta != null ? metrics.beta.toFixed(3) : '—'],
    ['Alpha',         fmtPctFraction(metrics.alpha)],
  ].map(([k, v]) => `<tr><td>${k}</td><td class="mono num">${v}</td></tr>`).join('')

  // Snapshot cards: ensemble headline + per-method medians per horizon
  const snapshotHtml = SNAPSHOT_HORIZONS.map(s => {
    const p5  = ensemble?.band?.p5?.[s.idx]
    const p50 = ensemble?.band?.p50?.[s.idx]
    const p95 = ensemble?.band?.p95?.[s.idx]
    const active = (results ?? []).filter(r => Number.isFinite(r?.forecast?.p50?.[s.idx])).length

    const methodRows = METHOD_ORDER.map(m => {
      const r = (results ?? []).find(x => x.method === m)
      const v = r?.forecast?.p50?.[s.idx]
      const hasVal = Number.isFinite(v)
      const colour = METHOD_COLORS[m]
      return `<tr>
        <td><span class="dot" style="background:${colour}"></span>${METHOD_LABELS[m]}</td>
        <td class="mono num" style="color:${hasVal ? (v > 0 ? '#00875a' : v < 0 ? '#c92a2a' : '#555') : '#aaa'}">${hasVal ? fmtPct(v) : '—'}</td>
      </tr>`
    }).join('')

    const headlineColor = Number.isFinite(p50)
      ? (p50 > 0 ? '#00875a' : p50 < 0 ? '#c92a2a' : '#555')
      : '#aaa'

    return `<div class="snapshot">
      <div class="snapshot-header">
        <div>
          <div class="snapshot-label">${s.label}</div>
          <div class="snapshot-sub">${s.days} trading days</div>
        </div>
        <div class="snapshot-active">n=${active}/5 active</div>
      </div>
      <div class="snapshot-headline" style="color:${headlineColor}">${fmtPct(p50, 2)}</div>
      <div class="snapshot-range">90% range: ${fmtPct(p5, 2)} … ${fmtPct(p95, 2)}</div>
      <table class="method-table">${methodRows}</table>
    </div>`
  }).join('')

  const weightNote = Math.abs(totalWeight - 1) > 0.001
    ? `<div class="leverage-note">Total weight: ${(totalWeight * 100).toFixed(1)}% ${totalWeight > 1 ? '(leveraged)' : totalWeight < 1 ? '(partially invested)' : ''}</div>`
    : ''

  const generated = new Date().toLocaleString()
  const regime = meta?.regime_probs?.dominant ?? '—'

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Forecast — ${portfolio?.start_date ?? ''} to ${portfolio?.end_date ?? ''}</title>
<style>
  * { box-sizing: border-box }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 880px; margin: 0 auto; padding: 32px 24px; color: #1a1a25; background: #fff; line-height: 1.4 }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 700 }
  h2 { margin: 28px 0 10px; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #555; border-bottom: 1px solid #e5e5eb; padding-bottom: 6px }
  .meta { color: #888; font-size: 11px; margin-bottom: 8px }
  .mono, .num { font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace }
  .num { text-align: right }
  table { width: 100%; border-collapse: collapse; font-size: 12px }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #eee; vertical-align: top }
  th { background: #f7f7fa; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: #666 }
  .kv { display: grid; grid-template-columns: 140px 1fr; gap: 2px 16px; font-size: 12px; margin-top: 8px }
  .kv dt { color: #666; font-weight: 500 }
  .kv dd { margin: 0; font-family: 'JetBrains Mono', monospace }
  .leverage-note { margin-top: 8px; padding: 6px 10px; background: #fff4e5; color: #a06b00; font-size: 11px; border-radius: 4px }
  .snapshots { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 12px }
  .snapshot { border: 1px solid #e5e5eb; border-radius: 6px; padding: 12px }
  .snapshot-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px }
  .snapshot-label { font-weight: 700; font-size: 13px }
  .snapshot-sub { font-size: 10px; color: #888 }
  .snapshot-active { font-size: 10px; color: #666 }
  .snapshot-headline { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 22px; line-height: 1.1; margin-bottom: 4px }
  .snapshot-range { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666; margin-bottom: 8px }
  .method-table { font-size: 10px }
  .method-table td { padding: 3px 4px; border-bottom: 1px dotted #eee }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: middle }
  .footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid #e5e5eb; font-size: 10px; color: #999; line-height: 1.5 }
  @media print {
    body { padding: 0 }
    .snapshots { page-break-inside: avoid }
  }
</style>
</head>
<body>
<h1>Portfolio Forecast Report</h1>
<div class="meta">Generated ${generated} · Current regime: <strong>${regime}</strong></div>

<h2>Portfolio Construct</h2>
<table>
  <thead><tr><th>Ticker</th><th class="num">Weight</th><th>Rationale</th></tr></thead>
  <tbody>${assetsRows || '<tr><td colspan="3" style="color:#aaa">No assets</td></tr>'}</tbody>
</table>
${weightNote}
<dl class="kv">
  <dt>Period</dt>            <dd>${portfolio?.start_date ?? '—'} → ${portfolio?.end_date ?? '—'}</dd>
  <dt>Rebalance</dt>         <dd>${portfolio?.rebalance_frequency ?? 'quarterly'}</dd>
  <dt>Benchmark</dt>         <dd>${portfolio?.benchmark ?? 'SPY'}</dd>
  <dt>Strategy</dt>          <dd>${portfolio?.strategy ?? 'custom'}</dd>
  ${portfolio?.strategy_summary ? `<dt>Summary</dt><dd style="font-family:inherit">${portfolio.strategy_summary.replace(/</g, '&lt;')}</dd>` : ''}
</dl>

<h2>Backtest Performance</h2>
<table style="max-width:420px">
  <tbody>${backtestRows}</tbody>
</table>

<h2>Forecast Snapshots — Ensemble + Per-Method Medians</h2>
<div class="snapshots">${snapshotHtml}</div>

<div class="footer">
  <strong>Methodology.</strong> Snapshot horizons (5d / 21d / 63d) sit within every model's training horizon.
  N-BEATS, TimesFM 2.5, HMM, GP, and Attention-LSTM each cover the full 63-day forecast horizon natively.
  Ensemble uses regime-conditional stacked weights (Wolpert 1992).
  All base models use walk-forward out-of-sample validation (Lopez de Prado 2018).
</div>
</body>
</html>`
}

// ── Component ─────────────────────────────────────────────────────────────────

/**
 * Export dropdown — produces a self-contained HTML report or a PNG snapshot
 * of the forecast area. The PNG captures whatever is currently visible in
 * `targetRef`; the HTML report is built from the prop data so it's rendered
 * identically regardless of which panels the user has expanded on screen.
 */
export default function ForecastExport({ portfolio, backtest, results, ensemble, meta, targetRef, disabled }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(null)   // 'html' | 'png' | null
  const [toast, setToast] = useState(null)
  const wrapperRef = useRef(null)

  useEffect(() => {
    if (!open) return
    const onClick = e => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  async function exportHtml() {
    setBusy('html')
    try {
      const html = buildHtmlReport({ portfolio, backtest, results, ensemble, meta })
      const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
      triggerDownload(blob, `${slugForFilename(portfolio)}.html`)
      setToast('HTML report saved')
    } catch (e) {
      setToast(`HTML export failed: ${e?.message ?? e}`)
    } finally {
      setBusy(null)
      setOpen(false)
      setTimeout(() => setToast(null), 3500)
    }
  }

  async function exportPng() {
    if (!targetRef?.current) {
      setToast('No forecast content to export')
      setTimeout(() => setToast(null), 2500)
      return
    }
    setBusy('png')
    setOpen(false)

    // html2canvas historically misaligned Recharts SVG under nested flex +
    // scrollable ancestors — axis labels drifted up into section headers and
    // research-citation <p> tags clipped into the next card. html-to-image
    // clones the DOM with computed styles into a single <foreignObject>,
    // which preserves SVG positioning and flex-gap geometry faithfully.
    const node = targetRef.current

    // Temporarily un-constrain the scrollable ancestor chain so the export
    // captures at natural height. Without this, a parent with `h-full +
    // overflow-y-auto` causes the cloned tree to be measured at viewport
    // height, which is what triggers the overlap on long forecasts.
    const mutatedAncestors = []
    for (let el = node.parentElement; el && el !== document.body; el = el.parentElement) {
      const cs = getComputedStyle(el)
      if (cs.overflow !== 'visible' || cs.overflowY !== 'visible' || cs.maxHeight !== 'none') {
        mutatedAncestors.push({
          el,
          overflow:  el.style.overflow,
          overflowY: el.style.overflowY,
          maxHeight: el.style.maxHeight,
          height:    el.style.height,
        })
        el.style.overflow  = 'visible'
        el.style.overflowY = 'visible'
        el.style.maxHeight = 'none'
        el.style.height    = 'auto'
      }
    }

    try {
      // One animation frame after the ancestor mutations so Recharts'
      // ResponsiveContainer re-measures against the natural-height tree.
      await new Promise(r => requestAnimationFrame(r))

      const { toPng } = await import('html-to-image')
      const dataUrl = await toPng(node, {
        backgroundColor: '#0a0a0f',
        pixelRatio:      2,
        cacheBust:       true,
        // Give the browser one extra frame after DOM mutations so Recharts'
        // ResponsiveContainer re-measures before the snapshot.
        skipFonts:       false,
      })
      const blob = await (await fetch(dataUrl)).blob()
      triggerDownload(blob, `${slugForFilename(portfolio)}.png`)
      setToast('PNG saved')
    } catch (e) {
      setToast(`PNG export failed: ${e?.message ?? e}`)
    } finally {
      // Restore ancestor styles even if export throws.
      for (const r of mutatedAncestors) {
        r.el.style.overflow  = r.overflow
        r.el.style.overflowY = r.overflowY
        r.el.style.maxHeight = r.maxHeight
        r.el.style.height    = r.height
      }
      setBusy(null)
      setTimeout(() => setToast(null), 3500)
    }
  }

  const hasAnyData = !!(backtest || (results && results.length))

  return (
    <div ref={wrapperRef} className="relative inline-block">
      <button
        onClick={() => setOpen(o => !o)}
        disabled={disabled || !hasAnyData || busy != null}
        className="mono text-xs px-3 py-2 rounded border transition-colors"
        style={{
          borderColor: 'var(--border)',
          color: disabled || !hasAnyData ? 'var(--text-secondary)' : 'var(--text-primary)',
          background: 'transparent',
          cursor: disabled || !hasAnyData || busy != null ? 'not-allowed' : 'pointer',
          opacity: disabled || !hasAnyData ? 0.5 : 1,
        }}
      >
        {busy === 'html' ? 'Exporting HTML…'
         : busy === 'png' ? 'Rendering PNG…'
         : 'Export ▾'}
      </button>

      {open && (
        <div className="absolute right-0 mt-1 rounded border shadow-lg z-50"
          style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', minWidth: 200 }}>
          <button
            onClick={exportHtml}
            className="mono text-xs px-3 py-2 w-full text-left hover:bg-white/5"
            style={{ color: 'var(--text-primary)', background: 'transparent', border: 'none', cursor: 'pointer' }}
          >
            <div style={{ fontWeight: 700 }}>HTML Report</div>
            <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 2 }}>
              Self-contained · printable · archival
            </div>
          </button>
          <button
            onClick={exportPng}
            className="mono text-xs px-3 py-2 w-full text-left hover:bg-white/5"
            style={{ color: 'var(--text-primary)', background: 'transparent', border: 'none', cursor: 'pointer', borderTop: '1px solid var(--border)' }}
          >
            <div style={{ fontWeight: 700 }}>PNG Snapshot</div>
            <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 2 }}>
              Screenshot of what's on screen · best for sharing
            </div>
          </button>
        </div>
      )}

      {toast && (
        <div className="absolute right-0 mt-1 px-3 py-2 rounded border mono text-xs shadow-lg z-50"
          style={{
            top: '100%',
            background: 'var(--bg-card)',
            borderColor: toast.includes('failed') ? 'var(--accent-red)' : 'var(--accent-green)',
            color: toast.includes('failed') ? 'var(--accent-red)' : 'var(--accent-green)',
            whiteSpace: 'nowrap',
          }}>
          {toast}
        </div>
      )}
    </div>
  )
}
