import { useState, useCallback } from 'react'
import axios from 'axios'
import { API_BASE, prewarmForecastContext } from '../utils/api.js'

export function useBacktest() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runBacktest = useCallback(async (portfolioInput) => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_BASE}/backtest`, portfolioInput)
      setResult(response.data)

      // Speculatively warm the forecast provider caches so Phase 2 is fast
      // if the user clicks Run Forecast. No-op on the wire if they don't.
      const tickers = (portfolioInput?.assets ?? []).map(a => a.ticker).filter(Boolean)
      prewarmForecastContext(tickers, portfolioInput?.end_date)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Backtest failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  return { result, loading, error, runBacktest }
}
