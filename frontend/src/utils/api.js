/**
 * Base URL for all API calls.
 *
 * Dev:  VITE_API_URL is not set → falls back to '/api'
 *       Vite proxy rewrites '/api' → 'http://localhost:8000/api'
 *
 * Prod: Set VITE_API_URL=https://your-backend.up.railway.app/api
 *       in Vercel project → Settings → Environment Variables
 */
export const API_BASE = import.meta.env.VITE_API_URL || '/api'

// Root URL for non-prefixed endpoints like /health. Strips the trailing
// '/api' so a warm-up ping wakes the backend without triggering any
// provider I/O (the ticker feed would — /health is a 50-byte no-op).
export const ROOT_URL = API_BASE.replace(/\/api\/?$/, '') || ''

/**
 * Wake the backend container without doing any real work.
 *
 * Railway's free tier sleeps idle containers after ~15 min. Cold boots cost
 * 30-60 s (Python + pandas + hmmlearn + statsmodels imports). Firing this
 * ping as soon as the user opens the app means the container is already
 * warm by the time they click "Run Forecast".
 *
 * Fire-and-forget; failures are ignored.
 */
export function warmupBackend() {
  try {
    fetch(`${ROOT_URL}/health`, { keepalive: true }).catch(() => {})
  } catch {
    /* noop */
  }
}

/**
 * Speculatively warm the forecast provider caches for a ticker set.
 *
 * Called when a backtest completes. The backend returns immediately and
 * fans out macro/insider/news/reddit/edgar fetches in the background —
 * their in-process caches are keyed on (tickers, date), so by the time
 * the user clicks "Run Forecast" Phase 2 hits warm caches instead of
 * 10-25 s of sequential I/O.
 */
export function prewarmForecastContext(tickers, endDate) {
  if (!tickers?.length) return
  try {
    fetch(`${API_BASE}/forecast/prewarm`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ tickers, end_date: endDate ?? null }),
      keepalive: true,
    }).catch(() => {})
  } catch {
    /* noop */
  }
}
