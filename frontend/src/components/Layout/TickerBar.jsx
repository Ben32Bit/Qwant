import { useEffect, useState, useMemo } from 'react'
import axios from 'axios'
import { API_BASE } from '../../utils/api.js'

const REFRESH_MS = 5 * 60 * 1000   // 5 min — matches backend cache TTL

/**
 * Rolling ticker marquee at the top of the app.
 *
 * Data: GET /api/ticker/feed → prices + GDELT headlines + Reddit trending posts.
 * Refreshes every 5 minutes so users see rotating market news without a reload.
 *
 * The marquee itself is a pure CSS `@keyframes marquee-scroll` (defined in
 * globals.css). We render the items *twice* back-to-back and scroll the strip
 * by exactly -50% so the second copy lands where the first started — that's
 * what creates the seamless infinite loop.
 */
export default function TickerBar() {
  const [items, setItems]   = useState([])
  const [error, setError]   = useState(false)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const res = await axios.get(`${API_BASE}/ticker/feed`, { timeout: 20000 })
        if (cancelled) return
        setItems(res.data?.items ?? [])
        setError(false)
        setLoaded(true)
      } catch (e) {
        if (cancelled) return
        setError(true)
        setLoaded(true)
      }
    }

    load()
    const id = setInterval(load, REFRESH_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  // Duplicate the list so the -50% translate creates a seamless loop.
  const doubled = useMemo(() => [...items, ...items], [items])

  // Scroll duration scales with item count — more items need more time to
  // traverse at a readable pace. ~4s per item feels about right.
  const durationSec = Math.max(60, items.length * 4)

  if (!loaded) {
    return (
      <div style={barStyle}>
        <div className="mono text-xs" style={{ color: 'var(--text-secondary)', padding: '6px 12px' }}>
          Loading market feed…
        </div>
      </div>
    )
  }

  if (error || items.length === 0) {
    return (
      <div style={barStyle}>
        <div className="mono text-xs" style={{ color: 'var(--text-secondary)', padding: '6px 12px', opacity: 0.7 }}>
          Market feed temporarily unavailable
        </div>
      </div>
    )
  }

  return (
    <div style={barStyle} className="group">
      <div
        className="ticker-scroll"
        style={{ animationDuration: `${durationSec}s` }}
      >
        {doubled.map((it, i) => (
          <TickerItem key={i} item={it} />
        ))}
      </div>
    </div>
  )
}

const barStyle = {
  background:    'var(--bg-secondary)',
  borderBottom:  '1px solid var(--border)',
  overflow:      'hidden',
  whiteSpace:    'nowrap',
  position:      'relative',
  height:        30,
}

// ── Individual item renderers ────────────────────────────────────────────────

function TickerItem({ item }) {
  if (item.kind === 'price')  return <PriceItem  {...item} />
  if (item.kind === 'news')   return <NewsItem   {...item} />
  if (item.kind === 'reddit') return <RedditItem {...item} />
  return null
}

function PriceItem({ ticker, price, change_pct }) {
  const up = change_pct >= 0
  const colour = up ? 'var(--accent-green)' : 'var(--accent-red)'
  return (
    <span style={itemStyle}>
      <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{ticker}</span>
      <span style={{ color: 'var(--text-secondary)', marginLeft: 6 }}>
        ${Number.isFinite(price) ? price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—'}
      </span>
      <span style={{ color: colour, marginLeft: 6 }}>
        {up ? '▲' : '▼'} {Math.abs(change_pct).toFixed(2)}%
      </span>
    </span>
  )
}

function NewsItem({ headline }) {
  return (
    <span style={itemStyle}>
      <span style={{ color: 'var(--accent-yellow)', marginRight: 6 }}>📰</span>
      <span style={{ color: 'var(--text-primary)' }}>{headline}</span>
    </span>
  )
}

function RedditItem({ title, subreddit, score, url }) {
  const text = (
    <>
      <span style={{ color: 'var(--accent-purple)', marginRight: 6 }}>r/{subreddit}</span>
      <span style={{ color: 'var(--text-primary)' }}>{title}</span>
      <span style={{ color: 'var(--text-secondary)', marginLeft: 6 }}>· ↑{score}</span>
    </>
  )
  if (!url) return <span style={itemStyle}>{text}</span>
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      style={{ ...itemStyle, textDecoration: 'none' }}
    >
      {text}
    </a>
  )
}

const itemStyle = {
  display:      'inline-block',
  padding:      '6px 18px',
  borderRight:  '1px solid var(--border)',
  fontSize:     12,
  fontFamily:   '"JetBrains Mono", "SF Mono", monospace',
  verticalAlign: 'middle',
  lineHeight:   '18px',
}
