import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // Base path
  // - If BASE is provided, use it (e.g., '/' for custom domain)
  // - Else if VITE_GH_PAGES=1, use '/uhop/' for project pages
  // - Else default '/'
  base:
    process.env.BASE || (process.env.VITE_GH_PAGES === "1" ? "/uhop/" : "/"),
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(
    Boolean,
  ),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
