import * as XLSX from 'xlsx'

// ── Helpers ───────────────────────────────────────────────────────────────────

function pct(val) {
  return val != null ? parseFloat((val * 100).toFixed(4)) : null
}

function round(val, dp = 4) {
  return val != null ? parseFloat(val.toFixed(dp)) : null
}

function applyHeaderStyle(ws, headers) {
  // SheetJS community edition doesn't support cell styles, but we can set column widths
  ws['!cols'] = headers.map(h => ({ wch: Math.max(h.length + 2, 14) }))
}

function makeSheet(rows, headers) {
  const ws = XLSX.utils.json_to_sheet(rows, { header: headers })
  applyHeaderStyle(ws, headers)
  return ws
}

// ── Sheet builders ────────────────────────────────────────────────────────────

function buildPerformanceSheet(backtest) {
  const { equity_curve, benchmark_curve, drawdown_series } = backtest

  const bm = {}
  if (benchmark_curve) benchmark_curve.forEach(pt => { bm[pt.date] = pt.value })

  const dd = {}
  if (drawdown_series) drawdown_series.forEach(pt => { dd[pt.date] = pt.drawdown })

  const headers = ['Date', 'Portfolio ($)', 'Benchmark ($)', 'Drawdown (%)']
  const rows = equity_curve.map(pt => ({
    'Date': pt.date,
    'Portfolio ($)': round(pt.value, 2),
    'Benchmark ($)': bm[pt.date] != null ? round(bm[pt.date], 2) : null,
    'Drawdown (%)': dd[pt.date] != null ? pct(dd[pt.date]) : null,
  }))
  return makeSheet(rows, headers)
}

function buildHoldingsSheet(backtest) {
  const { weight_history, rebalance_dates } = backtest
  if (!weight_history?.length) return null

  const tickers = Object.keys(weight_history[0]).filter(k => k !== 'date')
  const rebalanceSet = new Set(rebalance_dates || [])

  const headers = ['Date', 'Rebalance', ...tickers.map(t => `${t} (%)`)]
  const rows = weight_history.map(row => {
    const rec = {
      'Date': row.date,
      'Rebalance': rebalanceSet.has(row.date) ? 'YES' : '',
    }
    tickers.forEach(t => {
      rec[`${t} (%)`] = row[t] != null ? pct(row[t]) : null
    })
    return rec
  })
  return makeSheet(rows, headers)
}

function buildMonthlyReturnsSheet(backtest) {
  const { monthly_returns } = backtest
  if (!monthly_returns?.data) return null

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const headers = ['Year', ...months, 'Annual (%)']
  const rows = Object.entries(monthly_returns.data)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([year, mData]) => {
      const rec = { 'Year': year }
      let annualProduct = 1
      months.forEach(m => {
        const v = mData[m]
        rec[m] = v != null ? pct(v) : null
        if (v != null) annualProduct *= (1 + v)
      })
      rec['Annual (%)'] = round((annualProduct - 1) * 100, 2)
      return rec
    })
  return makeSheet(rows, headers)
}

function buildRollingMetricsSheet(backtest) {
  const { rolling_metrics } = backtest
  if (!rolling_metrics) return null

  // Merge all rolling series by date
  const dateMap = {}

  const add = (series, key) => {
    if (!series) return
    series.forEach(pt => {
      if (!dateMap[pt.date]) dateMap[pt.date] = { 'Date': pt.date }
      dateMap[pt.date][key] = round(pt.value)
    })
  }

  add(rolling_metrics.sharpe,     'Rolling Sharpe (252d)')
  add(rolling_metrics.volatility, 'Rolling Vol % (60d)')
  add(rolling_metrics.beta,       'Rolling Beta (126d)')

  const rows = Object.values(dateMap).sort((a, b) => a.Date.localeCompare(b.Date))
  if (!rows.length) return null

  // Convert vol to pct
  rows.forEach(r => {
    if (r['Rolling Vol % (60d)'] != null) {
      r['Rolling Vol % (60d)'] = pct(r['Rolling Vol % (60d)'] ?? 0)
    }
  })

  const headers = ['Date', 'Rolling Sharpe (252d)', 'Rolling Vol % (60d)', 'Rolling Beta (126d)']
  return makeSheet(rows, headers)
}

function buildCorrelationSheet(backtest) {
  const { correlation_matrix } = backtest
  if (!correlation_matrix) return null

  const tickers = Object.keys(correlation_matrix)
  const headers = ['', ...tickers]
  const rows = tickers.map(row => {
    const rec = { '': row }
    tickers.forEach(col => {
      rec[col] = correlation_matrix[row]?.[col] ?? null
    })
    return rec
  })
  return makeSheet(rows, headers)
}

function buildMetricsSheet(metrics, portfolio) {
  const rows = [
    // Portfolio info
    { Metric: 'Strategy', Value: portfolio?.strategy_summary || '' },
    { Metric: 'Start Date', Value: portfolio?.start_date || '' },
    { Metric: 'End Date', Value: portfolio?.end_date || '' },
    { Metric: 'Benchmark', Value: portfolio?.benchmark || '' },
    { Metric: 'Rebalance', Value: portfolio?.rebalance_frequency || '' },
    { Metric: '', Value: '' },
    // Return metrics
    { Metric: '── RETURNS ──', Value: '' },
    { Metric: 'Total Return (%)', Value: pct(metrics.total_return) },
    { Metric: 'CAGR (%)', Value: pct(metrics.cagr) },
    { Metric: 'Best Year (%)', Value: metrics.best_year ? pct(metrics.best_year.return) : null },
    { Metric: 'Best Year Period', Value: metrics.best_year?.period || null },
    { Metric: 'Worst Year (%)', Value: metrics.worst_year ? pct(metrics.worst_year.return) : null },
    { Metric: 'Worst Year Period', Value: metrics.worst_year?.period || null },
    { Metric: 'Best Month (%)', Value: metrics.best_month ? pct(metrics.best_month.return) : null },
    { Metric: 'Worst Month (%)', Value: metrics.worst_month ? pct(metrics.worst_month.return) : null },
    { Metric: '', Value: '' },
    // Risk metrics
    { Metric: '── RISK ──', Value: '' },
    { Metric: 'Volatility (%)', Value: pct(metrics.volatility) },
    { Metric: 'Max Drawdown (%)', Value: pct(metrics.max_drawdown) },
    { Metric: 'Max DD Duration (days)', Value: metrics.max_drawdown_duration_days },
    { Metric: 'Downside Deviation (%)', Value: pct(metrics.downside_deviation) },
    { Metric: 'VaR 95% (%)', Value: pct(metrics.var_95) },
    { Metric: 'CVaR 95% (%)', Value: pct(metrics.cvar_95) },
    { Metric: '', Value: '' },
    // Risk-adjusted
    { Metric: '── RISK-ADJUSTED ──', Value: '' },
    { Metric: 'Sharpe Ratio', Value: round(metrics.sharpe) },
    { Metric: 'Sortino Ratio', Value: round(metrics.sortino) },
    { Metric: 'Calmar Ratio', Value: round(metrics.calmar) },
    { Metric: '', Value: '' },
    // Benchmark-relative
    { Metric: '── vs BENCHMARK ──', Value: '' },
    { Metric: 'Beta', Value: round(metrics.beta) },
    { Metric: 'Alpha (%)', Value: pct(metrics.alpha) },
    { Metric: 'R-Squared', Value: round(metrics.r_squared) },
    { Metric: 'Information Ratio', Value: round(metrics.information_ratio) },
    { Metric: 'Tracking Error (%)', Value: pct(metrics.tracking_error) },
    { Metric: 'Treynor Ratio', Value: round(metrics.treynor) },
    { Metric: 'Up Capture (%)', Value: pct(metrics.up_capture) },
    { Metric: 'Down Capture (%)', Value: pct(metrics.down_capture) },
  ].filter(r => r.Value !== undefined)

  const ws = XLSX.utils.json_to_sheet(rows, { header: ['Metric', 'Value'] })
  ws['!cols'] = [{ wch: 28 }, { wch: 16 }]
  return ws
}

// ── Main export function ──────────────────────────────────────────────────────

export function exportToExcel(backtest, portfolio) {
  if (!backtest) return

  const wb = XLSX.utils.book_new()

  // Sheet 1: Summary metrics
  const metricsSheet = buildMetricsSheet(backtest.metrics, portfolio)
  XLSX.utils.book_append_sheet(wb, metricsSheet, 'Summary')

  // Sheet 2: Performance timeseries
  const perfSheet = buildPerformanceSheet(backtest)
  XLSX.utils.book_append_sheet(wb, perfSheet, 'Performance')

  // Sheet 3: Holdings over time
  const holdingsSheet = buildHoldingsSheet(backtest)
  if (holdingsSheet) XLSX.utils.book_append_sheet(wb, holdingsSheet, 'Holdings')

  // Sheet 4: Monthly returns
  const monthlySheet = buildMonthlyReturnsSheet(backtest)
  if (monthlySheet) XLSX.utils.book_append_sheet(wb, monthlySheet, 'Monthly Returns')

  // Sheet 5: Rolling metrics
  const rollingSheet = buildRollingMetricsSheet(backtest)
  if (rollingSheet) XLSX.utils.book_append_sheet(wb, rollingSheet, 'Rolling Metrics')

  // Sheet 6: Correlation matrix
  const corrSheet = buildCorrelationSheet(backtest)
  if (corrSheet) XLSX.utils.book_append_sheet(wb, corrSheet, 'Correlations')

  // Generate filename from strategy and date range
  const stratLabel = portfolio?.strategy_summary
    ? portfolio.strategy_summary.slice(0, 30).replace(/[^a-zA-Z0-9 ]/g, '').trim()
    : 'Portfolio'
  const start = portfolio?.start_date?.slice(0, 4) || ''
  const end = portfolio?.end_date?.slice(0, 4) || ''
  const filename = `${stratLabel} ${start}-${end}.xlsx`.replace(/\s+/g, '_')

  XLSX.writeFile(wb, filename)
}
