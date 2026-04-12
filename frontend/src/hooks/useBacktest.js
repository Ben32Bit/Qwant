import { useState, useCallback } from 'react'
import axios from 'axios'
import { API_BASE } from '../utils/api.js'

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
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Backtest failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  return { result, loading, error, runBacktest }
}
