import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// HuggingFace Spaces URL — update this after deploying the backend
const HF_BACKEND_URL = 'https://YOUR-USERNAME-mediai-backend.hf.space'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  },
  build: {
    outDir: 'dist',
  },
  define: {
    // Injected at build time — swap to HF URL when building for production
    __BACKEND_URL__: JSON.stringify(
      process.env.VITE_BACKEND_URL || ''
    )
  }
})
