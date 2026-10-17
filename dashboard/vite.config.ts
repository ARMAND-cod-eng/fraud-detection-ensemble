import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the dashboard runs on :5173 and proxies API calls to the FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: false },
      "/health": "http://127.0.0.1:8000",
      "/ready": "http://127.0.0.1:8000",
      "/metrics": "http://127.0.0.1:8000",
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 600 },
});
