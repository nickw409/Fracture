import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: process.env.VITE_BASE ?? '/',
  server: {
    port: 3000,
    proxy: {
      '/v1/metrics/stream': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        // SSE requires no buffering — pass headers through to disable proxy buffering.
        configure: (proxy) => {
          proxy.on('proxyReq', (_proxyReq, _req, res) => {
            // Prevent http-proxy from buffering the SSE stream.
            (res as any).flushHeaders?.();
          });
        },
      },
      '/v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
