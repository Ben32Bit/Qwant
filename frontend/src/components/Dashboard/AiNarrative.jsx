import React from 'react'

/**
 * Renders the AI's markdown narrative in the results panel.
 * Supports: ## headings, **bold**, bullet lists, paragraphs.
 */
function parseMarkdown(text) {
  if (!text) return []

  const elements = []
  const lines = text.split('\n')
  let key = 0
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Skip empty lines
    if (!line.trim()) { i++; continue }

    // ## Heading
    if (line.startsWith('## ')) {
      elements.push(
        <h3 key={key++} className="mono font-bold text-sm mt-4 mb-2" style={{ color: 'var(--accent-blue)' }}>
          {line.slice(3)}
        </h3>
      )
      i++; continue
    }

    // ### Sub-heading
    if (line.startsWith('### ')) {
      elements.push(
        <h4 key={key++} className="mono font-bold text-xs mt-3 mb-1 uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>
          {line.slice(4)}
        </h4>
      )
      i++; continue
    }

    // Bullet list block
    if (line.startsWith('- ') || line.startsWith('* ')) {
      const items = []
      while (i < lines.length && (lines[i].startsWith('- ') || lines[i].startsWith('* '))) {
        items.push(
          <li key={key++} className="flex gap-2 text-sm mb-1" style={{ color: 'var(--text-primary)' }}>
            <span style={{ color: 'var(--accent-blue)' }}>▸</span>
            <span>{inlineFormat(lines[i].slice(2))}</span>
          </li>
        )
        i++
      }
      elements.push(<ul key={key++} className="mb-2 space-y-0.5">{items}</ul>)
      continue
    }

    // Regular paragraph
    elements.push(
      <p key={key++} className="text-sm mb-2 leading-relaxed" style={{ color: 'var(--text-primary)' }}>
        {inlineFormat(line)}
      </p>
    )
    i++
  }

  return elements
}

function inlineFormat(text) {
  // Split on **bold** markers
  const parts = text.split(/(\*\*[^*]+\*\*)/)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} style={{ color: 'var(--text-primary)' }}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}

export default function AiNarrative({ narrative }) {
  if (!narrative) return null

  return (
    <div
      className="rounded-lg border p-4"
      style={{ borderColor: 'rgba(74,158,255,0.25)', background: 'rgba(74,158,255,0.04)' }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className="mono text-xs font-bold" style={{ color: 'var(--accent-blue)' }}>
          ▶ AI ANALYSIS
        </span>
      </div>
      <div>{parseMarkdown(narrative)}</div>
    </div>
  )
}
