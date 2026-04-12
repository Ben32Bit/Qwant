import React, { useState, useRef, useEffect, useCallback } from 'react'
import MessageBubble from './MessageBubble.jsx'
import PromptSuggestions from './PromptSuggestions.jsx'

export default function ChatPanel({ messages, loading, onSend, portfolio }) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = useCallback(() => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    onSend(text)
  }, [input, loading, onSend])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <div className="flex flex-col h-full">
            {/* Hero */}
            <div className="flex-1 flex flex-col items-center justify-center px-6 text-center">
              <div
                className="mono font-bold text-2xl mb-2"
                style={{ color: 'var(--accent-blue)' }}
              >
                ▶ QWANT
              </div>
              <p className="text-sm mb-1" style={{ color: 'var(--text-primary)' }}>
                Describe a portfolio in plain English.
              </p>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                AI builds it · backtests it · shows you the numbers.
              </p>
            </div>
            {/* Prompt suggestions */}
            <PromptSuggestions onSelect={(s) => { setInput(s); textareaRef.current?.focus() }} />
          </div>
        ) : (
          <div className="px-4 pt-4">
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}

            {/* Loading indicator */}
            {loading && (
              <div className="flex items-start mb-4">
                <div className="flex flex-col items-start">
                  <span className="mono text-xs mb-1 px-1" style={{ color: 'var(--text-secondary)' }}>
                    QWANT AI
                  </span>
                  <div
                    className="rounded-lg px-4 py-3 text-sm"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                  >
                    <span
                      className="mono"
                      style={{ color: 'var(--accent-blue)' }}
                    >
                      Analyzing portfolio
                      <span className="animate-pulse">...</span>
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div
        className="border-t p-4"
        style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
      >
        <div
          className="flex gap-2 rounded-lg border overflow-hidden"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}
        >
          <textarea
            ref={textareaRef}
            rows={2}
            className="flex-1 px-4 py-3 text-sm resize-none outline-none"
            style={{
              background: 'transparent',
              color: 'var(--text-primary)',
            }}
            placeholder="e.g. 60/40 stocks and bonds since 2015, rebalanced monthly…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="px-4 m-2 rounded font-bold mono text-sm transition-all"
            style={{
              background: input.trim() && !loading ? 'var(--accent-blue)' : 'var(--border)',
              color: input.trim() && !loading ? '#fff' : 'var(--text-secondary)',
              cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
            }}
          >
            ↵ RUN
          </button>
        </div>
        <p className="mono text-xs mt-2 text-center" style={{ color: 'var(--text-secondary)' }}>
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
