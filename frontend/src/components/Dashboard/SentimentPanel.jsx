/**
 * SentimentPanel — Phase 4B: browser-side FinBERT sentiment scoring.
 *
 * Receives GDELT headlines from the backend (news_context) and runs
 * Xenova/finbert in the browser to score each headline.
 *
 * States
 * ------
 * idle        : No news_context yet — panel hidden.
 * can_run     : Headlines available, user hasn't started yet — shows "Score sentiment" button.
 * downloading : FinBERT model downloading (~80MB, one-time, cached in IndexedDB).
 * scoring     : Model loaded, running inference on headlines.
 * done        : Scores available — shows per-ticker sentiment bars.
 * error       : Model load or inference failed — shows fallback message.
 */

import { useState, useCallback, useRef } from 'react'
import { scoreHeadlines, aggregateScores, deviceCanRunFinBERT } from '../../ml/SentimentInferer.js'

// Sentiment bar colour
function sentimentColor(net) {
  if (net >  0.10) return 'var(--accent-green)'
  if (net < -0.10) return 'var(--accent-red)'
  return 'var(--accent-yellow)'
}

function SentimentBar({ ticker, data, agg }) {
  if (!agg) return null
  const pct  = Math.round(Math.abs(agg.net) * 100)
  const color = sentimentColor(agg.net)
  const label = agg.net > 0.10 ? 'bullish' : agg.net < -0.10 ? 'bearish' : 'neutral'

  return (
    <div className="flex items-center gap-3">
      <span className="mono text-xs font-bold w-12 flex-shrink-0" style={{ color: 'var(--text-primary)' }}>
        {ticker}
      </span>
      <div className="flex-1 rounded-full overflow-hidden" style={{ height: 6, background: 'var(--border)' }}>
        <div className="h-full rounded-full"
          style={{ width: `${pct}%`, background: color, transition: 'width 0.4s ease' }} />
      </div>
      <span className="mono text-xs w-16 flex-shrink-0 text-right" style={{ color }}>
        {label} {agg.net >= 0 ? '+' : ''}{agg.net.toFixed(2)}
      </span>
      <span className="mono text-xs flex-shrink-0" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
        n={agg.n}
      </span>
    </div>
  )
}

function DownloadProgress({ progress }) {
  const pct = progress?.progress ?? 0
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
        <div className="h-full rounded-full"
          style={{
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

export default function SentimentPanel({ newsContext }) {
  const [phase, setPhase]       = useState('idle')  // idle | can_run | downloading | scoring | done | error
  const [progress, setProgress] = useState(null)    // {progress, file, status}
  const [scores, setScores]     = useState({})       // {ticker: aggregate}
  const [errMsg, setErrMsg]     = useState('')
  const abortRef                = useRef(false)

  const tickers = newsContext
    ? Object.keys(newsContext).filter(k => k !== 'portfolio_summary' && newsContext[k]?.available)
    : []

  const totalArticles = newsContext?.portfolio_summary?.total_articles ?? 0

  // Transition idle → can_run when news arrives
  if (newsContext && phase === 'idle' && tickers.length > 0) {
    setPhase('can_run')
  }

  const handleRun = useCallback(async () => {
    if (!deviceCanRunFinBERT()) {
      setPhase('error')
      setErrMsg('Device memory < 4GB — FinBERT disabled. GDELT headlines available in backend logs.')
      return
    }

    abortRef.current = false
    setPhase('downloading')
    setProgress(null)

    try {
      const onProgress = info => {
        if (info?.status === 'progress') setProgress(info)
        if (info?.status === 'done') setProgress(null)
      }

      // SentimentInferer caches the pipeline after first load
      const { scoreHeadlines: score, aggregateScores: agg } = await import('../../ml/SentimentInferer.js')

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

  // Hide panel entirely if no news arrived
  if (!newsContext || tickers.length === 0) return null

  return (
    <div className="rounded-lg border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="mono font-bold text-xs" style={{ color: 'var(--text-primary)' }}>
            ◈ FINBERT SENTIMENT
          </span>
          <span className="mono text-xs ml-2" style={{ color: 'var(--text-secondary)' }}>
            · {totalArticles} headlines · {tickers.length} tickers · browser inference
          </span>
        </div>
        {(phase === 'can_run' || phase === 'done') && (
          <button
            onClick={handleRun}
            className="mono text-xs px-3 py-1 rounded border"
            style={{
              borderColor: 'var(--accent-blue)',
              color:       'var(--accent-blue)',
              background:  'transparent',
              cursor:      'pointer',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(74,158,255,0.1)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
          >
            {phase === 'done' ? '↺ Rescore' : '▶ Score sentiment'}
          </button>
        )}
      </div>

      {/* Download progress */}
      {phase === 'downloading' && <DownloadProgress progress={progress} />}

      {/* Scoring progress */}
      {phase === 'scoring' && (
        <p className="mono text-xs" style={{ color: 'var(--accent-blue)' }}>
          ⟳ Running FinBERT on {totalArticles} headlines…
        </p>
      )}

      {/* Error */}
      {phase === 'error' && (
        <div className="rounded px-3 py-2 text-xs mono"
          style={{ background: 'rgba(255,71,87,0.08)', border: '1px solid rgba(255,71,87,0.25)', color: 'var(--accent-red)' }}>
          {errMsg}
        </div>
      )}

      {/* Can-run prompt */}
      {phase === 'can_run' && (
        <p className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          Click "Score sentiment" to run Xenova/finbert in your browser (
          <span style={{ color: 'var(--text-primary)' }}>~80MB</span>, downloaded once, cached permanently).
        </p>
      )}

      {/* Results */}
      {phase === 'done' && Object.keys(scores).length > 0 && (
        <div className="space-y-2 mt-2">
          {tickers
            .filter(t => scores[t])
            .sort((a, b) => (scores[b]?.net ?? 0) - (scores[a]?.net ?? 0))
            .map(ticker => (
              <SentimentBar
                key={ticker}
                ticker={ticker}
                data={newsContext[ticker]}
                agg={scores[ticker]}
              />
            ))}
        </div>
      )}

      {/* Citation footer */}
      <div className="mt-3 pt-2 space-y-1" style={{ borderTop: '1px solid var(--border)' }}>
        <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📄 Malo, P. et al. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.
          JASIST, 65(4), 782–796. https://doi.org/10.1002/asi.23062
        </p>
        <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📄 Yang, Y., UY, M.C.S. & Huang, A. (2020). FinBERT: A Pretrained Language Model for Financial
          Communications. arXiv:2006.08097
        </p>
        <p className="mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
          📰 Headlines via GDELT 2.0 DOC API (Leetaru & Schrodt, ISA 2013). Model: Xenova/finbert (quantized, ~80MB).
        </p>
      </div>
    </div>
  )
}
