import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { execSync } from 'node:child_process'

// Capture git SHA + commit date at build time so the deployed bundle can
// display which commit is live. Wrapped in try/catch because Vercel's build
// env includes git, but a `npm pack` / docker-less environment may not.
function gitInfo() {
  try {
    const sha   = (process.env.VERCEL_GIT_COMMIT_SHA || execSync('git rev-parse --short HEAD').toString()).trim().slice(0, 7)
    const date  = execSync('git log -1 --format=%cd --date=short').toString().trim()
    return { sha, date }
  } catch {
    return { sha: 'dev', date: new Date().toISOString().slice(0, 10) }
  }
}

const { sha: GIT_SHA, date: GIT_DATE } = gitInfo()

export default defineConfig({
  plugins: [react()],
  define: {
    __GIT_SHA__:  JSON.stringify(GIT_SHA),
    __GIT_DATE__: JSON.stringify(GIT_DATE),
  },
  optimizeDeps: {
    exclude: ['@xenova/transformers', 'onnxruntime-web'],
  },
  build: {
    rollupOptions: {
      // onnxruntime-web is loaded from CDN via <script> in index.html;
      // externalizing prevents Rollup from trying to bundle WASM binaries.
      external: ['onnxruntime-web'],
      output: {
        // Split large third-party libraries into their own chunks. Benefits:
        //  1. User navigating to a non-chart page doesn't download recharts.
        //  2. Editing a component does not invalidate the library chunk hash,
        //     so returning users replay libraries from the browser cache.
        manualChunks: {
          recharts:    ['recharts'],
          markdown:    ['react-markdown', 'remark-gfm'],
          tfjs:        ['@tensorflow/tfjs'],
          transformers: ['@xenova/transformers'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
