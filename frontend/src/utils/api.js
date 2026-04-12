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
