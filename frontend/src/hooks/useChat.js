import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [portfolio, setPortfolio] = useState(null)
  const [backtest, setBacktest] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const sendMessage = useCallback(async (text) => {
    if (!text.trim()) return

    const userMessage = { role: 'user', content: text }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_BASE}/chat`, {
        message: text,
        conversation_history: messages,
        current_portfolio: portfolio,
      })

      const data = response.data
      const assistantMessage = {
        role: 'assistant',
        content: data.ai_response,
        portfolio: data.portfolio,
      }

      setMessages([...newMessages, assistantMessage])
      setPortfolio(data.portfolio)
      setBacktest(data.backtest)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Unknown error'
      setError(msg)
      setMessages([
        ...newMessages,
        { role: 'assistant', content: `Error: ${msg}`, isError: true },
      ])
    } finally {
      setLoading(false)
    }
  }, [messages, portfolio])

  const clearChat = useCallback(() => {
    setMessages([])
    setPortfolio(null)
    setBacktest(null)
    setError(null)
  }, [])

  return { messages, portfolio, backtest, loading, error, sendMessage, clearChat }
}
