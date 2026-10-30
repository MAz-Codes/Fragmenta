// Link drift correction (PerformanceEngine._driftTick): a loop launched on
// Link's beat must stay on Link's grid while the sound card's clock drifts,
// without audible speed changes. Simulates the clocks: Link runs on the
// system clock, the AudioContext on a sound card that is `ppm` fast, with
// output latency and 200 ms Link polling. Runs the SHIPPED engine classes
// against a fake AudioContext.
//
// Run:  node tests/test_link_drift.mjs
const pending = [];
class Param {
    constructor(v = 0) { this.value = v; }
    setTargetAtTime() {} cancelScheduledValues() {} linearRampToValueAtTime() {}
    setValueAtTime(v) { this.value = v; }
}
class Node_ {
    constructor() { this.gain = new Param(1); this.frequency = new Param(); this.Q = new Param(); this.pan = new Param(); }
    connect() {} disconnect() {}
}
class Source extends Node_ {
    constructor() { super(); this.loop = false; this.loopStart = 0; this.loopEnd = 0; this.started = null; this.stoppedAt = null; this.playbackRate = new Param(1); }
    start(when, offset, duration) { this.started = { when, offset, duration }; }
    stop(t) { this.stoppedAt = t; }
}
// Simulated world: true time tau (s). Link = system clock = tau. The sound
// card runs (1 + drift) fast; audio rendered at context time T is heard
// LATENCY later in context time.
const world = { tau: 0, drift: 0, latency: 0.02 };
class FakeCtx {
    constructor() { this.sources = []; this.sampleRate = 44100; }
    get currentTime() { return world.tau * (1 + world.drift); }
    getOutputTimestamp() { return { contextTime: this.currentTime - world.latency, performanceTime: world.tau * 1000 }; }
    createBiquadFilter() { return new Node_(); }
    createGain() { return new Node_(); }
    createStereoPanner() { return new Node_(); }
    createAnalyser() { const n = new Node_(); n.frequencyBinCount = 128; return n; }
    createBufferSource() { const s = new Source(); this.sources.push(s); return s; }
}
globalThis.window = { AudioContext: FakeCtx, setTimeout: (fn) => pending.push(fn), setInterval: () => 1, clearInterval: () => {} };
globalThis.performance = { now: () => world.tau * 1000 };

const { ChannelStrip, PerformanceEngine } = await import('../app/frontend/src/utils/performanceAudio.js');

let fails = 0;
const chk = (name, cond, detail = '') => {
    console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
    if (!cond) fails++;
};
const BPM = 120;

function setup({ drift = 200e-6, clipSeconds = 4.0, jitterMs = 0 } = {}) {
    world.tau = 1; world.drift = drift;
    const strip = new ChannelStrip(new Node_(), null, null);
    strip.buffer = { duration: clipSeconds, numberOfChannels: 2, length: clipSeconds * 44100, getChannelData: () => new Float32Array(4) };
    const e = Object.create(PerformanceEngine.prototype);
    Object.assign(e, { ctx: strip.ctx, channels: [strip], linkOffsetMs: 0, launchQuantum: 4,
        internalTransport: { bpm: BPM, anchorBeat: 0, originAudioTime: 0 } });
    const snap = () => {
        const noise = jitterMs ? (Math.random() * 2 - 1) * jitterMs : 0;
        e.setLinkSnapshot({ beat: (world.tau * BPM) / 60, bpm: BPM, isPlaying: true, capturedAt: world.tau * 1000 + noise });
    };
    snap();
    return { e, strip, snap };
}

// Ground truth: how far (s) the audio heard right now sits from where Link's
// grid puts it, from the launch anchor. Positive = audio is behind the grid.
function gridError(e, strip) {
    const a = strip.linkAnchor;
    const [rs, re] = strip._regionSeconds();
    const len = re - rs;
    const T = strip.ctx.currentTime;
    const heardTau = (T + world.latency) / (1 + world.drift);     // when audio rendered at T is heard
    const beat = (heardTau * BPM) / 60;
    const spb = len / Math.round((len * BPM) / 60);
    let expected = a.offset + (beat - a.beat) * spb;
    if (expected >= re) expected = rs + ((expected - re) % len);
    let err = expected - strip.getPlayPosition() * strip.buffer.duration;
    return ((((err + len / 2) % len) + len) % len) - len / 2;
}

function run(e, strip, snap, seconds, { correct = true } = {}) {
    const stats = { maxLateErr: 0, maxRateDev: 0, resyncs: 0, finalErr: 0 };
    const start = world.tau;
    let nextSnap = world.tau + 0.2, nextTick = world.tau + 0.25;
    const sources0 = strip.ctx.sources.length;
    while (world.tau < start + seconds) {
        world.tau += 0.01;
        if (world.tau >= nextSnap) { snap(); nextSnap += 0.2; }
        if (correct && world.tau >= nextTick) { e._driftTick(); nextTick += 0.25; }
        const err = gridError(e, strip);
        if (world.tau > start + seconds / 2) stats.maxLateErr = Math.max(stats.maxLateErr, Math.abs(err));
        stats.maxRateDev = Math.max(stats.maxRateDev, Math.abs(strip.playbackRate - 1));
        stats.finalErr = err;
    }
    stats.resyncs = strip.ctx.sources.length - sources0;
    return stats;
}
const ms = (s) => `${(s * 1000).toFixed(2)} ms`;

console.log('=== clock drift (sound card 200 ppm fast) ===');
{
    const { e, strip, snap } = setup();
    e.playChannel(0, true);
    world.tau += 0.001;
    const st = run(e, strip, snap, 120, { correct: false });
    chk('without correction the loop drifts off the grid', Math.abs(st.finalErr) > 0.02, `after 2 min: ${ms(st.finalErr)}`);
}
{
    const { e, strip, snap } = setup();
    e.playChannel(0, true);
    const st = run(e, strip, snap, 120);
    chk('with correction it stays on the grid', st.maxLateErr < 0.002, `worst in the 2nd minute: ${ms(st.maxLateErr)}`);
    chk('…using inaudible rate changes only', st.maxRateDev <= 0.003, `max rate deviation ${(st.maxRateDev * 100).toFixed(3)}%`);
    chk('…and no jumps', st.resyncs === 0, `${st.resyncs} resync(s)`);
}

console.log('=== loop length not an exact number of beats ===');
{
    const { e, strip, snap } = setup({ drift: 0, clipSeconds: 4.004 });   // 0.1% long at 120 bpm
    e.playChannel(0, true);
    const st = run(e, strip, snap, 120);
    chk('a 0.1%-long loop is held on the grid', st.maxLateErr < 0.002, `worst ${ms(st.maxLateErr)}, rate ${strip.playbackRate.toFixed(5)}`);
}
{
    const { e, strip, snap } = setup({ drift: 0, clipSeconds: 4.1 });     // 2.5% long: a different tempo
    e.playChannel(0, true);
    const st = run(e, strip, snap, 20);
    chk('a clearly different tempo is left alone (no pitch bend)', st.maxRateDev === 0 && st.resyncs === 0,
        `rate ${strip.playbackRate}, ${st.resyncs} resync(s)`);
}

console.log('=== big jump → crossfaded resync ===');
{
    const { e, strip, snap } = setup({ drift: 0 });
    e.playChannel(0, true);
    run(e, strip, snap, 2);
    strip.linkAnchor.offset += 0.05;                 // the grid now says 50 ms further on
    const before = strip.ctx.sources.length;
    world.tau += 0.25; snap(); e._driftTick();
    chk('one large reading alone doesn\'t jump', strip.ctx.sources.length === before);
    world.tau += 0.25; snap(); e._driftTick();
    chk('a 50 ms error confirmed on the next tick is fixed by one resync', strip.ctx.sources.length === before + 1);
    world.tau += 0.01;                               // the handover lands 5 ms after the tick
    chk('…landing on the grid', Math.abs(gridError(e, strip)) < 0.002, ms(gridError(e, strip)));
    const st = run(e, strip, snap, 10);
    chk('…and stays there without further jumps', st.resyncs === 0 && st.maxLateErr < 0.002, ms(st.maxLateErr));
}

console.log('=== snapshot jitter (±2 ms timing noise) ===');
{
    const { e, strip, snap } = setup({ jitterMs: 2 });
    e.playChannel(0, true);
    const st = run(e, strip, snap, 120);
    chk('jitter doesn\'t throw it off the grid', st.maxLateErr < 0.003, `worst ${ms(st.maxLateErr)}`);
    chk('…or cause jumps', st.resyncs === 0);
}

console.log('=== an occasional mistimed snapshot (slow round trip) ===');
{
    const { e, strip, snap } = setup();
    e.playChannel(0, true);
    let k = 0;
    const flaky = () => { snap(); if (++k % 5 === 0) e.linkSnapshot = { ...e.linkSnapshot, capturedAt: e.linkSnapshot.capturedAt + 45 }; };
    const st = run(e, strip, flaky, 60);
    chk('a single bad snapshot never triggers a jump', st.resyncs === 0, `${st.resyncs} resync(s)`);
}

console.log('=== state changes ===');
{
    const { e, strip, snap } = setup();
    e.playChannel(0, true);
    run(e, strip, snap, 10);
    const n = strip.ctx.sources.length;
    strip.setRegion(0.25, 0.75);                     // user moves the loop brackets
    world.tau += 0.25; snap(); e._driftTick();
    chk('moving the loop brackets re-anchors (no jump)', strip.ctx.sources.length === n && strip.linkAnchor.rs === 1);
    e.playChannel(0, true);
    chk('a new launch starts at rate 1 with a fresh anchor', strip.playbackRate === 1 && strip.linkAnchor !== null);
    run(e, strip, snap, 10);
    e.setLinkSnapshot(null);
    chk('switching Link off returns to rate 1', strip.playbackRate === 1 && strip.linkAnchor === null);
}

console.log(fails ? `\n${fails} FAILED` : '\nAll Link drift checks passed.');
process.exit(fails ? 1 : 0);
