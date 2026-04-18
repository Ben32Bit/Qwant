/**
 * SentimentPanel — browser-side FinBERT sentiment for news headlines + SEC filings.
 *
 * Two tabs:
 *   News Headlines  — GDELT 2.0 DOC API headlines (per-ticker)
 *   SEC Filings     — EDGAR 10-K/10-Q excerpts (Risk Factors, MD&A)
 *
 * Both tabs run Xenova/finbert locally; the model is loaded once and cached in
 * IndexedDB. Long SEC excerpts are sentence-split before scoring.
 *
 * References
 * ----------
 * Malo et al. (2014). Good Debt or Bad Debt. JASIST, 65(4), 782–796.
 * Yang et al. (2020). FinBERT. arXiv:2006.08097
 * Loughran & McDonald (2011). When is a Liability not a Liability. JF, 66(1), 35–65.
 */

import { useState, useCallback, useRef } from 'react'

// ── Shared sub-components ─────────────────────────────────────────────────────

function sentimentColor(net) {
  if (net >  0.10) return 'var(--accent-green)'
  if (net < -0.10) return 'var(--accent-red)'
  return 'var(--accent-yellow)'
}

function SentimentBar({ ticker, label, agg }) {
  if (!agg) return null
  const pct   = Math.round(Math.abs(agg.net) * 100)
  const color = sentimentColor(agg.net)
  const word  = agg.net > 0.10 ? 'bullish' : agg.net < -0.10 ? 'bearish' : 'neutral'

  return (
    <div className="flex items-center gap-3">
      <span className="mono text-xs font-bold w-14 flex-shrink-0 truncate"
        style={{ color: 'var(--text-primary)' }} title={label ?? ticker}>
        {ticker}
      </span>
      <div className="flex-1 rounded-full overflow-hidden" style={{ height: 6, background: 'var(--border)' }}>
        <div className="h-full rounded-full"
          style={{ width: `${pct}%`, background: color, transition: 'width 0.4s ease' }} />
      </div>
      <span className="mono text-xs w-20 flex-shrink-0 text-right" style={{ color }}>
        {word} {agg.net >= 0 ? '+' : ''}{agg.net.toFixed(2)}
      </span>
      <span className="mono flex-shrink-0" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
        n={agg.n}
      </span>
    </div>
  )
}

function DownloadProgress({ progress }) {
  const pct  = progress?.progress ?? 0
  const file = progress?.file ?? ''
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="mono text-xs" style={{ color: 'var(--accent-blue)' }}>
          ⟳ Downloading FinBERT (~80MB, cached permanently)
        </span>
        <span className="mono text-xs font-bold" style={{ color: 'var(--accent-blue)' }}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div className="rounded-full overflow-hidden mb-1" style={{ height: 4, background: 'var(--border)' }}>
        <div className="h-full rounded-full" style={{
          width: `${pct}%`,
          background: 'linear-gradient(90deg, var(--accent-blue), #00d4aa)',
          transition: 'width 0.3s linear',
        }} />
      </div>
      {file && (
        <p className="mono text-xs" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          {file.split('/').pop()}
        </p>
      )}
    </div>
  )
}

// Split long text into sentences for FinBERT (works best on short spans)
function splitSentences(text, maxLen = 300) {
  const raw = text.replace(/\s+/g, ' ').trim()
  const parts = raw.split(/(?<=[.!?])\s+/)
  const out = []
  for (const p of parts) {
    if (p.length > 20) out.push(p.slice(0, maxLen))
  }
  return out.length ? out : [raw.slice(0, maxLen)]
}

// ── Tab: News Headlines ────────────────────────────────────────────────────────

function NewsTab({ newsContext }) {
  const [phase, setPhase]       = useState('can_run')
  const [progress, setProgress] = useState(null)
  const [scores, setScores]     = useState({})
  const [errMsg, setErrMsg]     = useState('')
  const abortRef                = useRef(false)

  const tickers = newsContext
    ? Object.keys(newsContext).filter(k => k !== 'portfolio_summary' && newsContext[k]?.available)
    : []
  const totalArticles = newsContext?.portfolio_summary?.total_articles ?? 0

  const handleRun = useCallback(async () => {
    const { deviceCanRunFinBERT, scoreHeadlines, aggregateScores } =
      await import('../../ml/SentimentInferer.js')
    if (!deviceCanRunFinBERT()) {
      setPhase('error')
      setErrMsg('Device memory < 4GB — FinBERT disabled.')
      return
    }
    abortRef.current = false
    setPhase('downloading')
    setProgress(null)
    try {
      const onProgress = info => {
        if (info?.status === 'progress') setProgress(info)
        if (info?.status === 'done')     setProgress(null)
      }
      setPhase('scoring')
      const result = {}
      for (const ticker of tickers) {
        if (abortRef.current) break
        const headlines = newsContext[ticker]?.headlines ?? []
        if (!headlines.length) continue
        const raw = await scoreHeadlines(headlines, onProgress)
        result[ticker] = aggregateScores(raw)
      }
      setScores(result)
      setPhase('done')
    } catch (e) {
      setPhase('error')
      setErrMsg(e.message ?? 'FinBERT inference failed')
    }
  }, [newsContext, tickers])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          {totalArticles} headlines · {tickers.length} tickers · GDELT 2.0
        </span>
        {(phase === 'can_run' || phase === 'done') && (
          <button onClick={handleRun} className="mono text-xs px-3 py-1 rounded border"
            style={{ borderColor: 'var(--accent-blue)', color: 'var(--accent-blue)', background: 'transparent', cursor: 'pointer' }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(74,158,255,0.1)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}>
            {phase === 'done' ? '↺ Rescore' : '▶ Score sentiment'}
          </button>
        )}
      </div>

      {phase === 'downloading' && <DownloadProgress progress={progress} />}
      {phase === 'scoring' && (
        <p className="mono text-xs" style={{ color: 'var(--accent-blue)' }}>
          ⟳ Scoring {totalArticles} headlines…
        </p>
      )}
      {phase === 'error' && (
        <div className="rounded px-3 py-2 text-xs mono"
          style={{ background: 'rgba(255,71,87,0.08)', border: '1px solid rgba(255,71,87,0.25)', color: 'var(--accent-red)' }}>
          {errMsg}
        </div>
      )}
      {phase === 'can_run' && (
        <p className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          Click "Score sentiment" to run Xenova/finbert in your browser (
          <span style={{ color: 'var(--text-primary)' }}>~80MB</span>, downloaded once, cached permanently).
        </p>
      )}
      {phase === 'done' && Object.keys(scores).length > 0 && (
        <div className="space-y-2">
          {tickers.filter(t => scores[t])
            .sort((a, b) => (scores[b]?.net ?? 0) - (scores[a]?.net ?? 0))
            .map(t => (
              <SentimentBar key={t} ticker={t} agg={scores[t]} />
            ))}
        </div>
      )}
    </div>
  )
}

// ── Tab: SEC Filings ──────────────────────────────────────────────────────────

function FilingsTab({ edgarContext }) {
  const [phase, setPhase]       = useState('can_run')
  const [progress, setProgress] = useState(null)
  const [scores, setScores]     = useState({})
  const [errMsg, setErrMsg]     = useState('')
  const abortRef                = useRef(false)

  const perTicker = edgarContext?.per_ticker ?? {}
  const tickers   = Object.keys(perTicker)

  const handleRun = useCallback(async () => {
    const { deviceCanRunFinBERT, scoreHeadlines, aggregateScores } =
      await import('../../ml/SentimentInferer.js')
    if (!deviceCanRunFinBERT()) {
      setPhase('error')
      setErrMsg('Device memory < 4GB — FinBERT disabled.')
      return
    }
    abortRef.current = false
    setPhase('downloading')
    setProgress(null)
    try {
      const onProgress = info => {
        if (info?.status === 'progress') setProgress(info)
        if (info?.status === 'done')     setProgress(null)
      }
      setPhase('scoring')
      const result = {}
      for (const ticker of tickers) {
        if (abortRef.current) break
        const excerpts = perTicker[ticker]?.excerpts ?? []
        if (!excerpts.length) continue
        // Sentence-split long excerpts before scoring
        const sentences = excerpts.flatMap(e => splitSentences(e))
        if (!sentences.length) continue
        const raw = await scoreHeadlines(sentences, onProgress)
        result[ticker] = aggregateScores(raw)
      }
      setScores(result)
      setPhase('done')
    } catch (e) {
      setPhase('error')
      setErrMsg(e.message ?? 'FinBERT inference failed')
    }
  }, [edgarContext, tickers, perTicker])

  if (!tickers.length) {
    return (
      <p className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
        No 10-K/10-Q excerpts available — ETF-only portfolios or EDGAR API unavailable.
      </p>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          {tickers.length} companies · most recent 10-K or 10-Q · EDGAR EFTS
        </span>
        {(phase === 'can_run' || phase === 'done') && (
          <button onClick={handleRun} className="mono text-xs px-3 py-1 rounded border"
            style={{ borderColor: '#a855f7', color: '#a855f7', background: 'transparent', cursor: 'pointer' }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(168,85,247,0.1)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}>
            {phase === 'done' ? '↺ Rescore' : '▶ Score filings'}
          </button>
        )}
      </div>

      {/* Filing metadata cards */}
      <div className="space-y-2">
        {tickers.map(ticker => {
          const filing = perTicker[ticker]
          return (
            <div key={ticker} className="rounded px-3 py-2"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <div className="flex items-center gap-2 mb-1">
                <span className="mono text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{ticker}</span>
                <span className="mono text-xs px-1 rounded"
                  style={{ background: 'rgba(168,85,247,0.15)', color: '#a855f7', fontSize: 10 }}>
                  {filing.form}
                </span>
                <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
                  {filing.filing_date}
                </span>
                <span className="mono text-xs" style={{ color: 'var(--text-secondary)', opacity: 0.5 }}>
                  · {filing.company}
                </span>
              </div>
              {filing.excerpts?.slice(0, 1).map((ex, i) => (
                <p key={i} className="mono text-xs leading-relaxed"
                  style={{ color: 'var(--text-secondary)', opacity: 0.8 }}>
                  {ex.slice(0, 220)}{ex.length > 220 ? '…' : ''}
                </p>
              ))}
              {scores[ticker] && (
                <div className="mt-2">
                  <SentimentBar ticker={ticker} label={filing.company} agg={scores[ticker]} />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {phase === 'downloading' && <DownloadProgress progress={progress} />}
      {phase === 'scoring' && (
        <p className="mono text-xs" style={{ color: '#a855f7' }}>
          ⟳ Scoring SEC filing excerpts…
        </p>
      )}
      {phase === 'error' && (
        <div className="rounded px-3 py-2 text-xs mono"
          style={{ background: 'rgba(255,71,87,0.08)', border: '1px solid rgba(255,71,87,0.25)', color: 'var(--accent-red)' }}>
          {errMsg}
        </div>
      )}
      {phase === 'can_run' && (
        <p className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          Click "Score filings" to run FinBERT on Risk Factors &amp; MD&amp;A excerpts.
        </p>
      )}
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function SentimentPanel({ newsContext, edgarContext }) {
  const [activeTab, setActiveTab] = useState('news')

  const newsAvail  = newsContext
    && Object.keys(newsContext).some(k => k !== 'portfolio_summary' && newsContext[k]?.available)
  const edgarAvail = edgarContext?.available

  if (!newsAvail && !edgarAvail) return null

  const tabs = [
    { id: 'news',   label: 'News Headlines', avail: newsAvail },
    { id: 'filings', label: 'SEC Filings',   avail: edgarAvail },
  ]

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="mono font-bold text-xs" style={{ color: 'var(--text-primary)' }}>
          ◈ FINBERT SENTIMENT
          <span className="ml-2 font-normal" style={{ color: 'var(--text-secondary)' }}>
            · browser inference · Xenova/finbert
          </span>
        </span>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-4">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => tab.avail && setActiveTab(tab.id)}
            disabled={!tab.avail}
            className="mono text-xs px-3 py-1 rounded transition-colors"
            style={{
              background:  activeTab === tab.id ? 'rgba(74,158,255,0.15)' : 'transparent',
              color:       !tab.avail           ? 'var(--text-secondary)'
                         : activeTab === tab.id ? 'var(--accent-blue)'
                         : 'var(--text-secondary)',
              border:      `1px solid ${activeTab === tab.id ? 'var(--accent-blue)' : 'var(--border)'}`,
              cursor:      tab.avail ? 'pointer' : 'default',
              opacity:     tab.avail ? 1 : 0.4,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'news' && newsAvail && <NewsTab newsContext={newsContext} />}
      {activeTab === 'filings' && <FilingsTab edgarContext={edgarContext} />}

      {/* Citation footer */}
      <div className="mt-4 pt-2 space-y-1" style={{ borderTop: '1px solid var(--border)' }}>
        <p className="mono leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📄 Malo et al. (2014). Good Debt or Bad Debt. JASIST 65(4). · Yang et al. (2020). FinBERT. arXiv:2006.08097
        </p>
        <p className="mono leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📰 Headlines: GDELT 2.0 DOC API. · Filings: SEC EDGAR EFTS (Loughran &amp; McDonald 2011).
        </p>
      </div>
    </div>
  )
}
