import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const SUGGESTIONS = [
  'Top return sector ETF each quarter, 2022–2025',
  'Best Sharpe among FAANG stocks by year since 2018',
  'Top momentum stock (AAPL, MSFT, NVDA, META, GOOGL) each quarter',
  'Top commodity each quarter: GLD, SLV, USO, DBC',
]

function ScreenerSuggestions({ onSelect }) {
  return (
    <div className="p-4">
      <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
        Try one of these examples:
      </p>
      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSelect(s)}
            className="px-3 py-1.5 rounded text-xs border transition-all text-left"
            style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)', background: 'var(--bg-card)' }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent-purple)'; e.currentTarget.style.color = 'var(--text-primary)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-secondary)' }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

const MD_COMPONENTS = {
  ul: ({ children }) => <ul className="space-y-1 my-1.5">{children}</ul>,
  li: ({ children }) => (
    <li className="flex gap-2 text-xs leading-relaxed" style={{ color: 'var(--text-primary)' }}>
      <span style={{ color: 'var(--accent-purple)', flexShrink: 0 }}>▸</span>
      <span>{children}</span>
    </li>
  ),
  p: ({ children }) => (
    <p className="text-xs leading-relaxed mb-1" style={{ color: 'var(--text-primary)' }}>{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="mono font-bold" style={{ color: 'var(--accent-purple)' }}>{children}</strong>
  ),
  code: ({ children }) => (
    <code className="mono px-1 rounded text-xs"
      style={{ background: 'rgba(168,85,247,0.12)', color: 'var(--accent-purple)' }}>
      {children}
    </code>
  ),
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const isError = message.isError
  return (
    <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} mb-3`}>
      <span className="mono text-xs mb-1 px-1" style={{ color: 'var(--text-secondary)' }}>
        {isUser ? 'YOU' : 'QWANT SCREENER'}
      </span>
      <div
        className="rounded-lg px-4 py-3 max-w-[88%] leading-relaxed"
        style={{
          background: isUser
            ? 'rgba(168,85,247,0.12)'
            : isError
            ? 'rgba(255,71,87,0.10)'
            : 'var(--bg-card)',
          border: `1px solid ${
            isUser
              ? 'rgba(168,85,247,0.3)'
              : isError
              ? 'rgba(255,71,87,0.3)'
              : 'var(--border)'
          }`,
          color: isError ? 'var(--accent-red)' : 'var(--text-primary)',
        }}
      >
        {isUser || isError ? (
          <span className="text-sm">{message.content}</span>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
            {message.content}
          </ReactMarkdown>
        )}
      </div>
    </div>
  )
}

export default function StockScreenerPanel({ onScreenResult, loading, setLoading }) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [history, setHistory] = useState([])
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = useCallback(async (text) => {
    const msg = (text || input).trim()
    if (!msg || loading) return
    setInput('')

    const userMsg = { role: 'user', content: msg }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    const newHistory = [...history, userMsg]

    try {
      const res = await fetch('/api/screen/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: msg,
          conversation_history: history,
          current_portfolio: null,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }))
        setMessages(prev => [...prev, { role: 'assistant', content: err.detail || 'Error running screen.', isError: true }])
        setLoading(false)
        return
      }

      const data = await res.json()
      const aiMsg = { role: 'assistant', content: data.ai_response || 'Screen complete — results are on the right.' }
      setMessages(prev => [...prev, aiMsg])
      setHistory([...newHistory, aiMsg])
      onScreenResult(data.screen_result)
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Network error: ${e.message}`, isError: true }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, history, onScreenResult, setLoading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <div className="flex flex-col h-full">
            <div className="flex-1 flex flex-col items-center justify-center px-6 text-center">
              <div className="mono font-bold text-2xl mb-2" style={{ color: 'var(--accent-purple)' }}>
                ◈ QWANT
              </div>
              <p className="text-sm mb-1" style={{ color: 'var(--text-primary)' }}>
                Ask which assets performed best — by any metric, any window.
              </p>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                Results link directly to portfolio backtesting.
              </p>
            </div>
            <ScreenerSuggestions onSelect={(s) => { setInput(s); textareaRef.current?.focus() }} />
          </div>
        ) : (
          <div className="px-4 pt-4">
            {messages.map((msg, i) => <MessageBubble key={i} message={msg} />)}
            {loading && (
              <div className="flex items-start mb-3">
                <div className="flex flex-col items-start">
                  <span className="mono text-xs mb-1 px-1" style={{ color: 'var(--text-secondary)' }}>QWANT SCREENER</span>
                  <div className="rounded-lg px-4 py-3 text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <span className="mono" style={{ color: 'var(--accent-purple)' }}>
                      Screening<span className="animate-pulse">...</span>
                    </span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t p-4" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
        <div className="flex gap-2 rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
          <textarea
            ref={textareaRef}
            rows={2}
            className="flex-1 px-4 py-3 text-sm resize-none outline-none"
            style={{ background: 'transparent', color: 'var(--text-primary)' }}
            placeholder="e.g. Top return sector ETF each quarter 2022–2025…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || loading}
            className="px-4 m-2 rounded font-bold mono text-sm transition-all"
            style={{
              background: input.trim() && !loading ? 'var(--accent-purple)' : 'var(--border)',
              color: input.trim() && !loading ? '#fff' : 'var(--text-secondary)',
              cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
            }}
          >
            ↵ SCREEN
          </button>
        </div>
        <p className="mono text-xs mt-2 text-center" style={{ color: 'var(--text-secondary)' }}>
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
