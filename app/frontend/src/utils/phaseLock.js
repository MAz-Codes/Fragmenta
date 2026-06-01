// Deterministic Link-phase -> loop-position mapping for Stage B (INV#8/#9).
//
// A clip's playback entry offset is the fraction of its loop that matches the
// current global beat phase. Because the map depends ONLY on (loop length,
// global beat, tempo) — never on which channel it is or when it launched —
// two bar-quantized clips at the same tempo share a downbeat automatically,
// with zero per-clip alignment code. Pure function so it is unit-testable
// without a browser AudioContext.
//
// Returns the entry offset in seconds, in [0, durationSec). Degrades to 0
// (start from the head) when tempo is unknown or the buffer is sub-beat.
export function phaseOffsetSec(durationSec, beat, bpm) {
    if (!(bpm > 0) || !(durationSec > 0)) return 0;
    const loopBeats = Math.round(durationSec * bpm / 60);
    if (loopBeats < 1) return 0;
    const phase = ((beat % loopBeats) + loopBeats) % loopBeats; // [0, loopBeats)
    return (phase / loopBeats) * durationSec;
}
