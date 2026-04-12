import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import PortfolioCard from './PortfolioCard.jsx'

// Markdown component overrides — terminal aesthetic
const MD_COMPONENTS = {
  h2: ({ children }) => (
    <div className="mono font-bold text-xs uppercase tracking-widest mt-3 mb-1.5 pb-0.5"
      style={{ color: 'var(--accent-blue)', borderBottom: '1px solid var(--border)' }}>
      {children}
    </div>
  ),
  h3: ({ children }) => (
    <div className="mono font-bold text-xs mt-2 mb-1" style={{ color: 'var(--text-primary)' }}>
      {children}
    </div>
  ),
  ul: ({ children }) => (
    <ul className="space-y-1 my-1.5 pl-0" style={{ listStyle: 'none' }}>
      {children}
    </ul>
  ),
  li: ({ children }) => (
    <li className="flex gap-2 text-xs leading-relaxed" style={{ color: 'var(--text-primary)' }}>
      <span style={{ color: 'var(--accent-blue)', flexShrink: 0 }}>▸</span>
      <span>{children}</span>
    </li>
  ),
  p: ({ children }) => (
    <p className="text-xs leading-relaxed mb-1.5" style={{ color: 'var(--text-primary)' }}>
      {children}
    </p>
  ),
  strong: ({ children }) => (
    <strong className="mono font-bold" style={{ color: 'var(--text-primary)' }}>
      {children}
    </strong>
  ),
  em: ({ children }) => (
    <em style={{ color: 'var(--text-secondary)' }}>{children}</em>
  ),
  code: ({ children }) => (
    <code className="mono px-1 rounded text-xs"
      style={{ background: 'rgba(74,158,255,0.1)', color: 'var(--accent-blue)' }}>
      {children}
    </code>
  ),
  hr: () => (
    <hr style={{ borderColor: 'var(--border)', margin: '8px 0' }} />
  ),
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const isError = message.isError

  return (
    <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} mb-4`}>
      {/* Role label */}
      <span className="mono text-xs mb-1 px-1" style={{ color: 'var(--text-secondary)' }}>
        {isUser ? 'YOU' : 'QWANT AI'}
      </span>

      {/* Message bubble */}
      <div
        className="rounded-lg px-4 py-3 max-w-[85%] leading-relaxed"
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
        {isUser || isError ? (
          <span className="text-sm">{message.content}</span>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
            {message.content}
          </ReactMarkdown>
        )}
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
