// Console-style fader taper for the performance master + channel gains.
//
// The faders store/report/MIDI-map dB, but the slider TRAVEL used to be
// linear-in-dB over -60..0 with unity pinned at the very top. That crammed
// the whole usable range into the top sliver (a quarter-throw down was
// already -15 dB / ~18% amplitude) and wasted the bottom half on the
// inaudible -30..-60 dB region.
//
// This maps slider POSITION (0 = bottom, 1 = top) to dB on a console-style
// curve: unity (0 dB) sits at ~80% of throw with +6 dB of boost above it,
// the 0..-12 dB region around unity gets fine resolution, and the lower
// region compresses toward silence at the bottom. Only the thumb position is
// tapered — the dB value the rest of the app sees is unchanged.

export const FADER_MAX_DB = 6;     // boost ceiling at the very top of throw
export const FADER_MIN_DB = -60;   // bottom of throw — dbToGain returns 0 here

// (position, dB) breakpoints, strictly increasing in both axes so both the
// forward map and its inverse are single-valued. Piecewise-linear between
// them — predictable and trivially invertible, unlike an analytic curve.
const CURVE = [
    [0.00, -60],
    [0.18, -40],
    [0.35, -24],
    [0.55, -10],
    [0.80, 0],     // unity
    [1.00, 6],
];

/** Slider position (0..1) → dB. */
export function faderPosToDb(pos) {
    const p = Math.min(1, Math.max(0, Number(pos) || 0));
    for (let i = 1; i < CURVE.length; i++) {
        const [p0, d0] = CURVE[i - 1];
        const [p1, d1] = CURVE[i];
        if (p <= p1) {
            const t = p1 === p0 ? 0 : (p - p0) / (p1 - p0);
            return d0 + t * (d1 - d0);
        }
    }
    return FADER_MAX_DB;
}

/** dB → slider position (0..1). Inverse of faderPosToDb. */
export function faderDbToPos(db) {
    const d = Math.min(FADER_MAX_DB, Math.max(FADER_MIN_DB, Number(db) || 0));
    for (let i = 1; i < CURVE.length; i++) {
        const [p0, d0] = CURVE[i - 1];
        const [p1, d1] = CURVE[i];
        if (d <= d1) {
            const t = d1 === d0 ? 0 : (d - d0) / (d1 - d0);
            return p0 + t * (p1 - p0);
        }
    }
    return 1;
}

/** Position of unity (0 dB) on the throw — used to place the unity tick. */
export const FADER_UNITY_POS = faderDbToPos(0);
