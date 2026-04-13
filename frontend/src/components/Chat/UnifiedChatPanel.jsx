import React, { useState, useRef, useEffect, useCallback } from 'react'
import MessageBubble from './MessageBubble.jsx'

// ── Suggestions — 2 portfolio + 2 screener ────────────────────────────────────
const SUGGESTIONS = [
  { text: '60/40 US stocks and bonds over 10 years',              type: 'portfolio' },
  { text: 'Max Sharpe portfolio from SPY, TLT, GLD, EEM',         type: 'portfolio' },
  { text: 'Top return sector ETF each quarter, 2022–2025',         type: 'screener'  },
  { text: 'Top momentum stock (AAPL MSFT NVDA META) each quarter', type: 'screener'  },
]

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyState({ onSelect, textareaRef }) {
  return (
    <div className="flex flex-col h-full">

      {/* Hero */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 text-center">
        <div className="mono font-bold text-2xl mb-3" style={{ color: 'var(--accent-blue)' }}>
          ▶ QWANT
        </div>

        <p className="text-sm mb-4 max-w-xs leading-relaxed" style={{ color: 'var(--text-primary)' }}>
          Ask me to <strong>build and backtest a portfolio</strong>, or{' '}
          <strong>screen which assets performed best</strong> across any time window.
        </p>

        {/* Capability pills */}
        <div className="flex gap-2 mb-6 flex-wrap justify-center">
          <span
            className="mono text-xs px-2.5 py-1 rounded-full border"
            style={{ borderColor: 'rgba(74,158,255,0.4)', color: 'var(--accent-blue)', background: 'rgba(74,158,255,0.08)' }}
          >
            ▷ Portfolio Builder
          </span>
          <span
            className="mono text-xs px-2.5 py-1 rounded-full border"
            style={{ borderColor: 'rgba(168,85,247,0.4)', color: 'var(--accent-purple)', background: 'rgba(168,85,247,0.08)' }}
          >
            ◈ Asset Screener
          </span>
        </div>

        {/* How it works — brief */}
        <div
          className="rounded-lg border px-4 py-3 text-left max-w-xs w-full space-y-1.5"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}
        >
          <div className="mono text-xs font-bold mb-2" style={{ color: 'var(--text-secondary)' }}>HOW TO USE</div>
          <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            <span style={{ color: 'var(--accent-blue)' }}>Portfolio:</span>{' '}
            "Build me a 60/40 portfolio" · "Risk parity since 2015" · "Max Sharpe from tech ETFs"
          </div>
          <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            <span style={{ color: 'var(--accent-purple)' }}>Screener:</span>{' '}
            "Which sector led each quarter?" · "Best bond ETF each year" · "Top stock by Sharpe monthly"
          </div>
        </div>
      </div>

      {/* Suggestion chips */}
      <div className="p-4">
        <p className="text-xs mb-2.5" style={{ color: 'var(--text-secondary)' }}>Try an example:</p>
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s.text}
              onClick={() => { onSelect(s.text); textareaRef.current?.focus() }}
              className="px-3 py-1.5 rounded text-xs border transition-all text-left"
              style={{
                borderColor: s.type === 'screener' ? 'rgba(168,85,247,0.3)' : 'rgba(74,158,255,0.3)',
                color: 'var(--text-secondary)',
                background: s.type === 'screener' ? 'rgba(168,85,247,0.05)' : 'rgba(74,158,255,0.05)',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.color = 'var(--text-primary)'
                e.currentTarget.style.borderColor = s.type === 'screener'
                  ? 'var(--accent-purple)' : 'var(--accent-blue)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.color = 'var(--text-secondary)'
                e.currentTarget.style.borderColor = s.type === 'screener'
                  ? 'rgba(168,85,247,0.3)' : 'rgba(74,158,255,0.3)'
              }}
            >
              {s.text}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Loading indicator — adapts to detected type ───────────────────────────────
function LoadingBubble({ detectedType }) {
  const label = detectedType === 'screener' ? 'Screening assets' : 'Building portfolio'
  const color = detectedType === 'screener' ? 'var(--accent-purple)' : 'var(--accent-blue)'
  return (
    <div className="flex items-start mb-4">
      <div className="flex flex-col items-start">
        <span className="mono text-xs mb-1 px-1" style={{ color: 'var(--text-secondary)' }}>QWANT AI</span>
        <div className="rounded-lg px-4 py-3 text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <span className="mono" style={{ color }}>
            {label}<span className="animate-pulse">…</span>
          </span>
        </div>
      </div>
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function UnifiedChatPanel({ onResult, loading, setLoading }) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [history, setHistory] = useState([])
  const [detectedType, setDetectedType] = useState(null)  // 'portfolio' | 'screener' | null
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
    setDetectedType(null)

    const newHistory = [...history, userMsg]

    try {
      const res = await fetch('/api/unified/chat', {
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
        setMessages(prev => [...prev, { role: 'assistant', content: err.detail || 'Error.', isError: true }])
        setLoading(false)
        return
      }

      const data = await res.json()
      setDetectedType(data.type)

      const aiMsg = { role: 'assistant', content: data.ai_response || '…' }
      setMessages(prev => [...prev, aiMsg])
      setHistory([...newHistory, aiMsg])

      // Pass full result to parent (SplitView handles right panel)
      onResult(data)

    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Network error: ${e.message}`, isError: true }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, history, onResult, setLoading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full">

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <EmptyState
            onSelect={(s) => setInput(s)}
            textareaRef={textareaRef}
          />
        ) : (
          <div className="px-4 pt-4">
            {messages.map((msg, i) => <MessageBubble key={i} message={msg} />)}
            {loading && <LoadingBubble detectedType={detectedType} />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t p-4" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
        <div
          className="flex gap-2 rounded-lg border overflow-hidden"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}
        >
          <textarea
            ref={textareaRef}
            rows={2}
            className="flex-1 px-4 py-3 text-sm resize-none outline-none"
            style={{ background: 'transparent', color: 'var(--text-primary)' }}
            placeholder="Build a portfolio or screen which assets performed best…"
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
              background: input.trim() && !loading ? 'var(--accent-blue)' : 'var(--border)',
              color: input.trim() && !loading ? '#fff' : 'var(--text-secondary)',
              cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
            }}
          >
            ↵ ASK
          </button>
        </div>
        <p className="mono text-xs mt-2 text-center" style={{ color: 'var(--text-secondary)' }}>
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
