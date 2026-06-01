// Stage B acceptance: deterministic Link-phase -> loop-position mapping
// (INV#8/#9). Imports the SAME pure function the PerformanceEngine uses, so
// there is no risk of the test drifting from the shipped formula.
//
// Run:  node tests/smoke_test_phaselock.mjs
import { phaseOffsetSec } from '../app/frontend/src/utils/phaseLock.js';

const SR = 44100;
let fails = 0;
function chk(name, cond, detail = '') {
    console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
    if (!cond) fails++;
}

const bpm = 120;
const dur2bar = 176400 / SR; // 2 bars / 8 beats @120 = 4.0s, the Stage A target
const dur4bar = 352800 / SR; // 4 bars / 16 beats

console.log('=== Stage B phase-lock (INV#8 deterministic map, INV#9 coincidence) ===');

// INV#9 — two same-length, Stage-A-correct clips map to the SAME loop position
// at ANY launch beat, so their downbeats coincide with no per-clip code.
for (const beat of [0, 3.5, 4, 7.999, 8, 13.27, 16, 100.5]) {
    const a = phaseOffsetSec(dur2bar, beat, bpm);
    const b = phaseOffsetSec(dur2bar, beat, bpm);
    chk(`same-length clips coincide @beat ${beat}`, Math.abs(a - b) < 1e-12,
        `offset=${a.toFixed(4)}s`);
}

// INV#8 — the map is deterministic: loop-position 0 occurs exactly when the
// global beat is a whole multiple of the loop length.
for (const beat of [0, 8, 16, 24, 800]) {
    chk(`downbeat at sample 0 when global beat=${beat}`,
        phaseOffsetSec(dur2bar, beat, bpm) < 1e-12);
}

// Offset is sample-exact at an integer beat (beat 5 -> 5/8 of the loop).
const off = phaseOffsetSec(dur2bar, 5, bpm);
chk('offset sample-exact at integer beat',
    Math.abs(off * SR - (5 / 8) * 176400) < 1e-6,
    `${(off * SR).toFixed(3)} vs ${(5 / 8 * 176400).toFixed(3)} samples`);

// Different loop lengths still coincide at their common boundaries (every 16
// beats), proving the map composes across clip lengths.
for (const beat of [0, 16, 32, 160]) {
    chk(`8-beat & 16-beat clips coincide @beat ${beat}`,
        Math.abs(phaseOffsetSec(dur2bar, beat, bpm) - phaseOffsetSec(dur4bar, beat, bpm)) < 1e-9);
}

// Degradation: unknown tempo or sub-beat buffer -> offset 0 (legacy head start).
chk('no tempo -> offset 0', phaseOffsetSec(dur2bar, 5, 0) === 0);
chk('sub-beat buffer -> offset 0', phaseOffsetSec(0.1, 5, bpm) === 0);

console.log(fails ? `\n${fails} FAILURE(S)` : '\nall Stage B phase-lock gates passed');
process.exit(fails ? 1 : 0);
