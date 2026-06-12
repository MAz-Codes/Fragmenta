import { readFileSync } from 'node:fs';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Single source of truth for the app version is the repo-root VERSION file —
// the backend (/api/version), the launcher and the Windows installer all read
// the same file. Baked in at build time so the UI needs no extra fetch.
const APP_VERSION = readFileSync(new URL('../../VERSION', import.meta.url), 'utf8').trim();

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
