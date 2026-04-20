import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
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
