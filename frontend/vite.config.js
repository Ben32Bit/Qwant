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
