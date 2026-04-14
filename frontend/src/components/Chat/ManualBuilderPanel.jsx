import React, { useState, useRef, useCallback, useEffect } from 'react'
import { downloadTemplate, parsePortfolioFile } from '../../utils/portfolioTemplate.js'
import { API_BASE } from '../../utils/api.js'

const REBALANCE_OPTIONS = ['none', 'daily', 'weekly', 'monthly', 'quarterly', 'annually']
const BENCHMARK_OPTIONS = ['SPY', 'QQQ', 'IWM', 'AGG', 'EFA', 'EEM', 'GLD', 'TLT', 'VTI', 'BND']

const today = new Date().toISOString().slice(0, 10)
const tenYearsAgo = `${new Date().getFullYear() - 10}-${String(new Date().getMonth() + 1).padStart(2, '0')}-${String(new Date().getDate()).padStart(2, '0')}`

const DEFAULT_ROWS = [
  { ticker: 'VOO', weight: '0.60' },
  { ticker: 'BND', weight: '0.40' },
]

const DEFAULT_SETTINGS = {
  startDate: tenYearsAgo,
  endDate: today,
  rebalance: 'quarterly',
  benchmark: 'SPY',
}

// ── Input primitives ──────────────────────────────────────────────────────────

function inputStyle(extra = {}) {
  return {
    background: 'var(--bg-primary)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    borderRadius: 6,
    outline: 'none',
    fontSize: 13,
    ...extra,
  }
}

function Btn({ children, onClick, variant = 'ghost', disabled, style: extraStyle }) {
  const base = {
    fontSize: 12,
    fontFamily: 'var(--font-mono, monospace)',
    borderRadius: 6,
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.45 : 1,
    transition: 'background 0.15s',
    padding: '6px 12px',
    border: 'none',
  }
  const variants = {
    primary: { background: 'var(--accent-blue)', color: '#fff' },
    green:   { background: 'var(--accent-green)', color: '#0a0a0f' },
    ghost:   { background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-secondary)' },
    danger:  { background: 'transparent', border: 'none', color: 'var(--accent-red)', padding: '4px 6px' },
  }
  return (
    <button onClick={disabled ? undefined : onClick} style={{ ...base, ...variants[variant], ...extraStyle }}>
      {children}
    </button>
  )
}

// ── Weight bar ────────────────────────────────────────────────────────────────

function WeightBar({ rows }) {
  const total = rows.reduce((s, r) => s + (parseFloat(r.weight) || 0), 0)
  const pct = Math.min(Math.abs(total) * 100, 200)
  const over = Math.abs(total) > 1.005
  const leveraged = total > 1.005

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          Total weight
        </span>
        <span className="mono text-xs font-bold" style={{
          color: over ? 'var(--accent-yellow)' : 'var(--accent-green)',
        }}>
          {(total * 100).toFixed(1)}%
          {leveraged && ' · leveraged'}
          {total < 0 && ' · net short'}
        </span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
        <div className="h-full rounded-full transition-all" style={{
          width: `${pct}%`,
          background: over ? 'var(--accent-yellow)' : 'var(--accent-green)',
        }} />
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ManualBuilderPanel({ onResult, loading, setLoading, aiPortfolio, aiBacktest, screenerImport, rotationImport }) {
  const [rows, setRows] = useState(DEFAULT_ROWS)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [error, setError] = useState(null)
  const [uploadMsg, setUploadMsg] = useState(null)
  const [importedFrom, setImportedFrom] = useState(null)      // tracks last imported AI portfolio
  const [importedScreener, setImportedScreener] = useState(null) // tracks last imported screener
  const [importedRotation, setImportedRotation] = useState(null) // tracks last imported rotation
  const fileInputRef = useRef(null)

  // ── Auto-import from screener when screenerImport changes ───────────────────
  useEffect(() => {
    if (!screenerImport || screenerImport === importedScreener) return
    const weight = (1 / screenerImport.tickers.length).toFixed(4)
    setRows(screenerImport.tickers.map(t => ({ ticker: t, weight })))
    setSettings(prev => ({
      startDate:  screenerImport.startDate ?? prev.startDate,
      endDate:    screenerImport.endDate   ?? prev.endDate,
      rebalance:  screenerImport.rebalance ?? prev.rebalance,
      benchmark:  prev.benchmark,
    }))
    setError(null)
    setUploadMsg(null)
    setImportedScreener(screenerImport)
    setImportedRotation(null)
  }, [screenerImport, importedScreener])

  // ── Auto-import rotation strategy ─────────────────────────────────────────
  useEffect(() => {
    if (!rotationImport || rotationImport === importedRotation) return
    // Pre-populate ticker rows with all unique tickers from the rotation universe
    const allTickers = rotationImport.screenResult?.tickers ?? []
    if (allTickers.length > 0) {
      const weight = (1 / allTickers.length).toFixed(4)
      setRows(allTickers.map(t => ({ ticker: t, weight })))
    }
    const sr = rotationImport.screenResult
    if (sr?.windows?.length) {
      const freqMap = { weekly: 'weekly', monthly: 'monthly', quarterly: 'quarterly', annually: 'annually' }
      setSettings(prev => ({
        startDate: sr.windows[0].window_start,
        endDate:   sr.windows[sr.windows.length - 1].window_end,
        rebalance: freqMap[sr.window_freq] ?? prev.rebalance,
        benchmark: prev.benchmark,
      }))
    }
    setError(null)
    setUploadMsg(null)
    setImportedRotation(rotationImport)
    setImportedScreener(null)
  }, [rotationImport, importedRotation])

  // ── Import from AI portfolio ─────────────────────────────────────────────────
  const loadFromAi = useCallback(() => {
    if (!aiPortfolio?.assets?.length) return
    setRows(aiPortfolio.assets.map(a => ({
      ticker: a.ticker,
      weight: String(a.weight),
    })))
    setSettings(prev => ({
      startDate:  aiBacktest?.equity_curve?.[0]?.date ?? prev.startDate,
      endDate:    aiBacktest?.equity_curve?.at(-1)?.date ?? prev.endDate,
      rebalance:  aiPortfolio.rebalance_frequency ?? prev.rebalance,
      benchmark:  aiPortfolio.benchmark ?? prev.benchmark,
    }))
    setError(null)
    setUploadMsg(null)
    setImportedFrom(aiPortfolio)
  }, [aiPortfolio, aiBacktest])

  // ── Row management ──────────────────────────────────────────────────────────
  const updateRow = (i, field, value) =>
    setRows(prev => prev.map((r, idx) => idx === i ? { ...r, [field]: value } : r))

  const addRow = () => setRows(prev => [...prev, { ticker: '', weight: '' }])

  const removeRow = (i) => setRows(prev => prev.filter((_, idx) => idx !== i))

  // ── Excel upload ────────────────────────────────────────────────────────────
  const handleFileUpload = useCallback(async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    setError(null)
    setUploadMsg(null)

    try {
      const { assets, settings: s } = await parsePortfolioFile(file)
      setRows(assets.map(a => ({ ticker: a.ticker, weight: String(a.weight) })))
      setSettings(prev => ({
        startDate:  s.startDate  || prev.startDate,
        endDate:    s.endDate    || prev.endDate,
        rebalance:  s.rebalance  || prev.rebalance,
        benchmark:  s.benchmark  || prev.benchmark,
      }))
      setUploadMsg(`Loaded ${assets.length} ticker${assets.length !== 1 ? 's' : ''} from ${file.name}`)
    } catch (err) {
      setError(`Upload failed: ${err.message}`)
    }
  }, [])

  // ── Run backtest ────────────────────────────────────────────────────────────
  const run = useCallback(async () => {
    setError(null)

    // Validate rows
    const assets = []
    for (const r of rows) {
      const ticker = r.ticker.trim().toUpperCase()
      const weight = parseFloat(r.weight)
      if (!ticker) continue
      if (isNaN(weight)) { setError(`Invalid weight for ${ticker}.`); return }
      assets.push({ ticker, weight })
    }
    if (!assets.length) { setError('Add at least one ticker.'); return }
    if (!settings.startDate) { setError('Start date is required.'); return }

    const payload = {
      assets,
      start_date:          settings.startDate,
      end_date:            settings.endDate || today,
      rebalance_frequency: settings.rebalance,
      benchmark:           settings.benchmark,
      strategy:            'custom',
    }

    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Backtest failed')
      onResult(data, {
        assets,
        strategy_summary: `Manual: ${assets.map(a => `${a.ticker} ${(a.weight * 100).toFixed(0)}%`).join(' / ')}`,
        start_date: payload.start_date,
        end_date: payload.end_date,
        rebalance_frequency: payload.rebalance_frequency,
        benchmark: payload.benchmark,
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [rows, settings, onResult, setLoading])

  // ── Run rotation backtest (re-runs the rotation strategy from screener) ─────
  const runRotation = useCallback(async () => {
    if (!rotationImport?.screenResult) return
    setError(null)
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/screen/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          screen_result: rotationImport.screenResult,
          initial_capital: 10000,
          benchmark: settings.benchmark,
          top_n_held: rotationImport.topN,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Rotation backtest failed')
      onResult(data, {
        assets: rotationImport.holdingSchedule?.map(h => ({ ticker: h.tickers.join('+'), weight: 1 })) ?? [],
        strategy_summary: `Rotation: Top ${rotationImport.topN} · ${rotationImport.screenResult?.window_freq} · ${rotationImport.screenResult?.metric_label ?? rotationImport.screenResult?.metric}`,
        start_date: rotationImport.screenResult?.windows?.[0]?.window_start,
        end_date:   rotationImport.screenResult?.windows?.at(-1)?.window_end,
        rebalance_frequency: rotationImport.screenResult?.window_freq,
        benchmark: settings.benchmark,
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [rotationImport, settings.benchmark, onResult, setLoading])

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="p-4 space-y-4">

        {/* Screener import banner */}
        {screenerImport && screenerImport === importedScreener && (
          <div
            className="px-3 py-2 rounded-lg mono text-xs"
            style={{
              background: 'rgba(168,85,247,0.08)',
              border: '1px solid rgba(168,85,247,0.3)',
              color: 'var(--accent-purple)',
            }}
          >
            ◈ Loaded from Screener&nbsp;
            <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>
              ({screenerImport.tickers.join(', ')}) · {screenerImport.rebalance} rebalance
            </span>
          </div>
        )}

        {/* Rotation strategy import banner + schedule */}
        {rotationImport && rotationImport === importedRotation && (
          <div
            className="rounded-lg mono text-xs overflow-hidden"
            style={{ border: '1px solid rgba(168,85,247,0.3)', background: 'rgba(168,85,247,0.05)' }}
          >
            {/* Banner */}
            <div className="px-3 py-2 flex items-center justify-between" style={{ borderBottom: '1px solid rgba(168,85,247,0.2)' }}>
              <span style={{ color: 'var(--accent-purple)' }}>
                ◈ Rotation Strategy&nbsp;
                <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>
                  Top {rotationImport.topN} · {rotationImport.screenResult?.window_freq} · {rotationImport.screenResult?.metric_label ?? rotationImport.screenResult?.metric}
                </span>
              </span>
              <button
                onClick={runRotation}
                disabled={loading}
                className="px-3 py-1 rounded font-bold transition-all"
                style={{
                  background: loading ? 'var(--border)' : 'var(--accent-purple)',
                  color: loading ? 'var(--text-secondary)' : '#fff',
                  border: 'none',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontSize: 11,
                }}
              >
                {loading ? '⟳ Running…' : '▶ Run Rotation'}
              </button>
            </div>

            {/* Rotation schedule table */}
            <div className="overflow-x-auto" style={{ maxHeight: 180 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    {['Window', 'Holding'].map(h => (
                      <th
                        key={h}
                        className="px-3 py-1.5 text-left sticky top-0"
                        style={{
                          color: 'var(--text-secondary)',
                          background: 'var(--bg-card)',
                          borderBottom: '1px solid rgba(168,85,247,0.2)',
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(rotationImport.holdingSchedule ?? []).map((h, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(168,85,247,0.08)' }}>
                      <td className="px-3 py-1" style={{ color: 'var(--text-secondary)' }}>{h.label}</td>
                      <td className="px-3 py-1">
                        <div className="flex gap-1 flex-wrap">
                          {h.tickers.map(t => (
                            <span
                              key={t}
                              className="px-1.5 py-0.5 rounded font-bold"
                              style={{
                                background: 'rgba(168,85,247,0.15)',
                                color: 'var(--accent-purple)',
                              }}
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="px-3 py-1.5 text-xs" style={{ color: 'var(--text-secondary)', borderTop: '1px solid rgba(168,85,247,0.1)' }}>
              Universe: {rotationImport.screenResult?.tickers?.join(', ')}
            </div>
          </div>
        )}

        {/* AI portfolio import banner */}
        {aiPortfolio?.assets?.length > 0 && aiPortfolio !== importedFrom && (
          <button
            onClick={loadFromAi}
            className="w-full text-left px-3 py-2 rounded-lg mono text-xs"
            style={{
              background: 'rgba(74,158,255,0.08)',
              border: '1px solid rgba(74,158,255,0.3)',
              color: 'var(--accent-blue)',
              cursor: 'pointer',
            }}
          >
            ← Load AI portfolio&nbsp;
            <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>
              ({aiPortfolio.assets.map(a => a.ticker).join(', ')})
            </span>
          </button>
        )}

        {/* Header actions */}
        <div className="flex items-center gap-2">
          <span className="mono text-xs font-bold flex-1" style={{ color: 'var(--text-secondary)' }}>
            PORTFOLIO COMPOSITION
          </span>
          <Btn onClick={downloadTemplate} variant="ghost">↓ Template</Btn>
          <Btn onClick={() => fileInputRef.current?.click()} variant="ghost">↑ Upload</Btn>
          <input
            ref={fileInputRef}
            type="file"
            accept=".xlsx,.xls,.csv"
            className="hidden"
            onChange={handleFileUpload}
          />
        </div>

        {/* Upload feedback */}
        {uploadMsg && (
          <div className="mono text-xs px-3 py-2 rounded" style={{
            background: 'rgba(0,212,170,0.1)', color: 'var(--accent-green)', border: '1px solid rgba(0,212,170,0.3)',
          }}>
            ✓ {uploadMsg}
          </div>
        )}

        {/* Weight bar */}
        <WeightBar rows={rows} />

        {/* Ticker rows */}
        <div className="space-y-1">
          {/* Column headers */}
          <div className="grid gap-2 px-1 mono text-xs" style={{
            gridTemplateColumns: '1fr 100px 28px',
            color: 'var(--text-secondary)',
          }}>
            <span>Ticker</span>
            <span>Weight <span style={{ fontWeight: 400, opacity: 0.65 }}>(−=short)</span></span>
            <span />
          </div>

          {rows.map((row, i) => {
            const isShort = parseFloat(row.weight) < 0
            return (
              <div key={i} className="grid gap-2 items-center" style={{ gridTemplateColumns: '1fr 100px 28px' }}>
                <input
                  value={row.ticker}
                  onChange={e => updateRow(i, 'ticker', e.target.value.toUpperCase())}
                  placeholder="AAPL"
                  className="mono text-sm px-3 py-2"
                  style={inputStyle(isShort ? { borderColor: 'rgba(255,71,87,0.4)' } : {})}
                />
                <input
                  value={row.weight}
                  onChange={e => updateRow(i, 'weight', e.target.value)}
                  placeholder="0.60"
                  className="mono text-sm px-3 py-2"
                  style={inputStyle(isShort ? {
                    borderColor: 'rgba(255,71,87,0.5)',
                    color: 'var(--accent-red)',
                  } : {})}
                />
                <Btn variant="danger" onClick={() => removeRow(i)}>×</Btn>
              </div>
            )
          })}

          <Btn onClick={addRow} variant="ghost" extraStyle={{ width: '100%', marginTop: 4 }}>
            + Add row
          </Btn>
        </div>

        {/* Settings */}
        <div>
          <div className="mono text-xs font-bold mb-2" style={{ color: 'var(--text-secondary)' }}>
            SETTINGS
          </div>
          <div className="space-y-2">

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="mono text-xs block mb-1" style={{ color: 'var(--text-secondary)' }}>Start Date</label>
                <input
                  type="date"
                  value={settings.startDate}
                  onChange={e => setSettings(s => ({ ...s, startDate: e.target.value }))}
                  className="mono text-xs px-3 py-2 w-full"
                  style={inputStyle()}
                />
              </div>
              <div>
                <label className="mono text-xs block mb-1" style={{ color: 'var(--text-secondary)' }}>End Date</label>
                <input
                  type="date"
                  value={settings.endDate}
                  onChange={e => setSettings(s => ({ ...s, endDate: e.target.value }))}
                  className="mono text-xs px-3 py-2 w-full"
                  style={inputStyle()}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="mono text-xs block mb-1" style={{ color: 'var(--text-secondary)' }}>Rebalance</label>
                <select
                  value={settings.rebalance}
                  onChange={e => setSettings(s => ({ ...s, rebalance: e.target.value }))}
                  className="mono text-xs px-3 py-2 w-full"
                  style={inputStyle()}
                >
                  {REBALANCE_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </div>
              <div>
                <label className="mono text-xs block mb-1" style={{ color: 'var(--text-secondary)' }}>Benchmark</label>
                <select
                  value={settings.benchmark}
                  onChange={e => setSettings(s => ({ ...s, benchmark: e.target.value }))}
                  className="mono text-xs px-3 py-2 w-full"
                  style={inputStyle()}
                >
                  {BENCHMARK_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mono text-xs px-3 py-2 rounded" style={{
            background: 'rgba(255,71,87,0.1)', color: 'var(--accent-red)', border: '1px solid rgba(255,71,87,0.3)',
          }}>
            {error}
          </div>
        )}

        {/* Run button */}
        <button
          onClick={run}
          disabled={loading}
          className="w-full mono font-bold text-sm py-3 rounded-lg transition-all"
          style={{
            background: loading ? 'var(--border)' : 'var(--accent-green)',
            color: loading ? 'var(--text-secondary)' : '#0a0a0f',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Running backtest…' : '▶ Run Backtest'}
        </button>

        {/* Tips */}
        <div className="mono text-xs space-y-0.5" style={{ color: 'var(--text-secondary)' }}>
          <div>• Weights: decimal (0.60 = 60%). Negative = short, &gt;1.0 = leveraged.</div>
          <div>• Download the Excel template to pre-fill and upload.</div>
        </div>
      </div>
    </div>
  )
}
