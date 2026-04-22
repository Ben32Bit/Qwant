// SheetJS (~400 KB) is dynamic-imported on first use so it stays out of the
// initial bundle. Module-level cache — first call pays the network cost,
// subsequent calls are synchronous reads of the cached module.
let XLSX = null
async function loadXlsx() {
  if (!XLSX) XLSX = await import('xlsx')
  return XLSX
}

// ── Download template ─────────────────────────────────────────────────────────

export async function downloadTemplate() {
  await loadXlsx()
  const wb = XLSX.utils.book_new()

  // ── Portfolio sheet ───────────────────────────────────────────────────────
  const portfolioRows = [
    { Ticker: 'VOO',  Weight: 0.60, Rationale: 'US total stock market (Vanguard S&P 500)' },
    { Ticker: 'BND',  Weight: 0.30, Rationale: 'US investment-grade bonds' },
    { Ticker: 'GLD',  Weight: 0.10, Rationale: 'Gold hedge' },
  ]
  const portfolioSheet = XLSX.utils.json_to_sheet(portfolioRows, {
    header: ['Ticker', 'Weight', 'Rationale'],
  })
  portfolioSheet['!cols'] = [{ wch: 12 }, { wch: 10 }, { wch: 40 }]

  // ── Settings sheet ────────────────────────────────────────────────────────
  const settingsRows = [
    { Setting: 'Start Date',          Value: '2015-01-01',  Notes: 'Format: YYYY-MM-DD' },
    { Setting: 'End Date',            Value: '2025-01-01',  Notes: 'Format: YYYY-MM-DD (or leave blank for today)' },
    { Setting: 'Rebalance Frequency', Value: 'quarterly',   Notes: 'daily | weekly | monthly | quarterly | annually | none' },
    { Setting: 'Benchmark',           Value: 'SPY',         Notes: 'Ticker symbol (e.g. SPY, QQQ, AGG)' },
  ]
  const settingsSheet = XLSX.utils.json_to_sheet(settingsRows, {
    header: ['Setting', 'Value', 'Notes'],
  })
  settingsSheet['!cols'] = [{ wch: 22 }, { wch: 16 }, { wch: 50 }]

  // ── Instructions sheet ────────────────────────────────────────────────────
  const instrRows = [
    { '': 'QWANT Portfolio Template — Instructions' },
    { '': '' },
    { '': 'Portfolio sheet:' },
    { '': '  • Ticker  — US-listed ticker symbol (e.g. AAPL, SPY, TLT)' },
    { '': '  • Weight  — decimal weight (0.60 = 60%). Negative = short. Weights can exceed 1.0 for leverage.' },
    { '': '  • Rationale — optional note (ignored during import)' },
    { '': '' },
    { '': 'Settings sheet:' },
    { '': '  • Fill in dates and rebalance frequency. All fields are optional (defaults used if blank).' },
    { '': '' },
    { '': 'Upload this file in QWANT using the "Upload Excel" button on the Manual Build tab.' },
  ]
  const instrSheet = XLSX.utils.json_to_sheet(instrRows, { header: [''] })
  instrSheet['!cols'] = [{ wch: 80 }]

  XLSX.utils.book_append_sheet(wb, portfolioSheet, 'Portfolio')
  XLSX.utils.book_append_sheet(wb, settingsSheet, 'Settings')
  XLSX.utils.book_append_sheet(wb, instrSheet, 'Instructions')

  XLSX.writeFile(wb, 'qwant_portfolio_template.xlsx')
}

// ── Parse uploaded file ───────────────────────────────────────────────────────

const REBALANCE_OPTIONS = ['daily', 'weekly', 'monthly', 'quarterly', 'annually', 'none']

export async function parsePortfolioFile(file) {
  await loadXlsx()
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result)
        const wb = XLSX.read(data, { type: 'array' })

        // ── Portfolio sheet ───────────────────────────────────────────────
        const portSheet = wb.Sheets['Portfolio'] || wb.Sheets[wb.SheetNames[0]]
        if (!portSheet) throw new Error('Could not find a Portfolio sheet in the uploaded file.')

        const rows = XLSX.utils.sheet_to_json(portSheet, { defval: '' })
        if (!rows.length) throw new Error('Portfolio sheet is empty.')

        const assets = []
        for (const row of rows) {
          // Accept column names case-insensitively
          const ticker = String(row['Ticker'] || row['ticker'] || row['TICKER'] || '').trim().toUpperCase()
          const rawWeight = row['Weight'] ?? row['weight'] ?? row['WEIGHT']
          const weight = parseFloat(rawWeight)

          if (!ticker) continue
          if (isNaN(weight)) throw new Error(`Invalid weight "${rawWeight}" for ticker "${ticker}". Use a decimal like 0.60.`)
          assets.push({ ticker, weight })
        }

        if (!assets.length) throw new Error('No valid ticker rows found in the Portfolio sheet.')

        // ── Settings sheet (optional) ─────────────────────────────────────
        const settings = { startDate: '', endDate: '', rebalance: 'quarterly', benchmark: 'SPY' }
        const settSheet = wb.Sheets['Settings']
        if (settSheet) {
          const settRows = XLSX.utils.sheet_to_json(settSheet, { defval: '' })
          for (const row of settRows) {
            const key = String(row['Setting'] || '').trim().toLowerCase()
            const val = String(row['Value'] || '').trim()
            if (!val) continue
            if (key.includes('start'))      settings.startDate = val
            if (key.includes('end'))        settings.endDate   = val
            if (key.includes('rebalance'))  settings.rebalance = REBALANCE_OPTIONS.includes(val.toLowerCase()) ? val.toLowerCase() : 'quarterly'
            if (key.includes('benchmark'))  settings.benchmark = val.toUpperCase()
          }
        }

        resolve({ assets, settings })
      } catch (err) {
        reject(err)
      }
    }
    reader.onerror = () => reject(new Error('Failed to read file.'))
    reader.readAsArrayBuffer(file)
  })
}
