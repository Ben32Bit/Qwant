import React from 'react'
import PortfolioCard from './PortfolioCard.jsx'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const isError = message.isError

  return (
    <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} mb-4`}>
      {/* Role label */}
      <span
        className="mono text-xs mb-1 px-1"
        style={{ color: 'var(--text-secondary)' }}
      >
        {isUser ? 'YOU' : 'QWANT AI'}
      </span>

      {/* Message bubble */}
      <div
        className="rounded-lg px-4 py-3 max-w-[85%] text-sm leading-relaxed"
        style={{
          background: isUser
            ? 'rgba(74, 158, 255, 0.12)'
            : isError
            ? 'rgba(255, 71, 87, 0.10)'
            : 'var(--bg-card)',
          border: `1px solid ${
            isUser
              ? 'rgba(74, 158, 255, 0.3)'
              : isError
              ? 'rgba(255, 71, 87, 0.3)'
              : 'var(--border)'
          }`,
          color: isError ? 'var(--accent-red)' : 'var(--text-primary)',
        }}
      >
        {message.content}
      </div>

      {/* Portfolio card (if AI message includes portfolio) */}
      {!isUser && message.portfolio && (
        <div className="w-full mt-2">
          <PortfolioCard portfolio={message.portfolio} />
        </div>
      )}
    </div>
  )
}
