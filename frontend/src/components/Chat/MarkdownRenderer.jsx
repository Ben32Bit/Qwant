import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/**
 * Thin wrapper over react-markdown + remark-gfm.
 *
 * Callers import this lazily via `React.lazy` so the ~80 KB markdown stack
 * stays out of the initial bundle until the first AI message renders.
 */
export default function MarkdownRenderer({ children, components }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {children}
    </ReactMarkdown>
  )
}
