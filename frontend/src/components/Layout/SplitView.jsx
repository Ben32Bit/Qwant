import React from 'react'
import ChatPanel from '../Chat/ChatPanel.jsx'
import ResultsPanel from '../Dashboard/ResultsPanel.jsx'

export default function SplitView({ messages, portfolio, backtest, loading, error, sendMessage, clearChat }) {
  return (
    <div className="flex h-full">
      {/* Left — Chat (40%) */}
      <div
        className="flex flex-col border-r"
        style={{
          width: '40%',
          minWidth: 320,
          borderColor: 'var(--border)',
          background: 'var(--bg-secondary)',
        }}
      >
        <ChatPanel
          messages={messages}
          loading={loading}
          onSend={sendMessage}
          portfolio={portfolio}
        />
      </div>

      {/* Right — Results (60%) */}
      <div
        className="flex-1 flex flex-col overflow-hidden"
        style={{ background: 'var(--bg-primary)' }}
      >
        <ResultsPanel backtest={backtest} portfolio={portfolio} loading={loading} />
      </div>
    </div>
  )
}
