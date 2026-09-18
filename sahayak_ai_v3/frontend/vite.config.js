import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The v2 backend runs on :8000 (uvicorn backend.api.main:app). Every /api
// call from the browser is proxied there during `npm run dev` so the frontend
// never needs CORS or a hardcoded origin — zero config in prod (FastAPI
// serves the built SPA from the same origin).
const API_TARGET = process.env.SAHAYAK_API_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
})
