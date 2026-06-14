import { readFileSync } from 'node:fs';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Single source of truth for the app version is the repo-root VERSION file —
// the backend (/api/version), the launcher and the Windows installer all read
// the same file. Baked in at build time so the UI needs no extra fetch.
//
// Resolution order: FRAGMENTA_VERSION env (Docker builds set this, since the
// flattened build stage copies only app/frontend/ and the repo-root file isn't
// on the relative path) → the repo-root VERSION file → 'dev'. The fallback
// mirrors the backend's guarded read in app.py so a missing file degrades the
// version string instead of hard-failing the whole build.
function resolveAppVersion() {
    const fromEnv = (process.env.FRAGMENTA_VERSION || '').trim();
    if (fromEnv) return fromEnv;
    try {
        return readFileSync(new URL('../../VERSION', import.meta.url), 'utf8').trim() || 'dev';
    } catch {
        return 'dev';
    }
}
const APP_VERSION = resolveAppVersion();

export default defineConfig({
    define: {
        __APP_VERSION__: JSON.stringify(APP_VERSION),
    },
    plugins: [react({ include: /\.(js|jsx)$/ })],
    esbuild: {
        loader: 'jsx',
        include: /src\/.*\.jsx?$/,
        exclude: [],
    },
    optimizeDeps: {
        esbuildOptions: {
            loader: { '.js': 'jsx' },
        },
    },
    build: {
        outDir: 'build',
        sourcemap: false,
    },
    server: {
        port: 3000,
        proxy: {
            '/api': 'http://localhost:5001',
        },
    },
});
