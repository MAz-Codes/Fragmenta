// Single source of truth for the Fragmenta app version.
//
// To bump: edit the `version` field in `app/frontend/package.json` and
// rebuild. All UI surfaces (Welcome page, About dialog, …) re-read it
// automatically. For a release, tag with `git tag v<version>` after the
// bump so the GitHub release tag matches the in-app version.
import pkg from '../package.json';

export const APP_VERSION = pkg.version;
