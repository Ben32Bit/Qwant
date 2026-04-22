import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
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
