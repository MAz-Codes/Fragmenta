// Single source of truth for the Fragmenta app version is the repo-root
// VERSION file, injected at build time by vite.config.js (define:
// __APP_VERSION__). The backend reads the same file for /api/version and
// /api/health, and packaging/windows/build_exe.ps1 passes it to the
// installer — so bumping VERSION updates every surface in one place.
//
// To bump: edit VERSION, rebuild the frontend, and tag with `git tag
// v<version>` so the GitHub release tag matches the in-app version.

/* global __APP_VERSION__ */
export const APP_VERSION = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev';
