// DJ cue point (ChannelStrip.launchStart): clamping, eviction, launch offsets
// and playhead projection. Runs the SHIPPED engine class against a fake
// AudioContext that records what each source was asked to do — the real-audio
// side (what Web Audio actually plays) is checked in a browser by
// test_cue_point_render.html.
//
// Run:  node tests/test_cue_point.mjs
const pending = [];
class Param {
    constructor(v = 0) { this.value = v; }
    setTargetAtTime() {} cancelScheduledValues() {} setValueAtTime() {} linearRampToValueAtTime() {}
}
class Node_ {
    constructor() { this.gain = new Param(1); this.frequency = new Param(); this.Q = new Param(); this.pan = new Param(); }
    connect() {} disconnect() {}
}
class Source extends Node_ {
    constructor() { super(); this.loop = false; this.loopStart = 0; this.loopEnd = 0; this.started = null; this.stoppedAt = null; }
    start(when, offset, duration) { this.started = { when, offset, duration }; }
    stop(t) { this.stoppedAt = t; }
}
class FakeCtx {
    constructor() { this.currentTime = 0; this.sources = []; this.sampleRate = 44100; }
    createBiquadFilter() { return new Node_(); }
    createGain() { return new Node_(); }
    createStereoPanner() { return new Node_(); }
    createAnalyser() { const n = new Node_(); n.frequencyBinCount = 128; return n; }
    createBufferSource() { const s = new Source(); this.sources.push(s); return s; }
    async decodeAudioData() { return fakeBuffer(4); }
}
globalThis.window = { AudioContext: FakeCtx, setTimeout: (fn) => pending.push(fn) };

const { ChannelStrip } = await import('../app/frontend/src/utils/performanceAudio.js');

function fakeBuffer(duration) {
    return { duration, numberOfChannels: 2, length: duration * 44100, getChannelData: () => new Float32Array(16) };
}
let fails = 0;
function chk(name, cond, detail = '') {
    console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
    if (!cond) fails++;
}
const near = (a, b, eps = 1e-9) => a !== null && b !== null && Math.abs(a - b) < eps;
function freshStrip(duration = 10) {
    const s = new ChannelStrip(new Node_(), null, null);
    s.buffer = fakeBuffer(duration);
    s.ctx.currentTime = 0;
    return s;
}
const last = (s) => s.ctx.sources[s.ctx.sources.length - 1];

console.log('=== Cue point: setLaunchStart clamping ===');
{
    const s = freshStrip();          // 10 s clip → min gap = max(0.005, 0.05/10) = 0.005
    s.setRegion(0.2, 0.8);
    s.setLaunchStart(0.5);
    chk('cue inside region is kept', near(s.launchStart, 0.5));
    s.setLaunchStart(0.1);
    chk('cue before the in-point re-couples (null)', s.launchStart === null);
    s.setLaunchStart(0.2);
    chk('cue exactly on the in-point stores null', s.launchStart === null);
    s.setLaunchStart(0.95);
    chk('cue past the out-point clamps to out-point minus gap', near(s.launchStart, 0.795), `got ${s.launchStart}`);
    for (const bad of [null, undefined, NaN, 'abc', Infinity]) {
        s.setLaunchStart(0.5);
        s.setLaunchStart(bad);
        chk(`invalid value ${String(bad)} → null`, s.launchStart === null);
    }
}

console.log('=== Cue point: moving the loop brackets ===');
{
    const s = freshStrip();
    s.setRegion(0.2, 0.8); s.setLaunchStart(0.5);
    s.setRegion(0.6, 0.9);
    chk('in-point dragged past the cue evicts it (null)', s.launchStart === null);
    s.setRegion(0.2, 0.8); s.setLaunchStart(0.5);
    s.setRegion(0.2, 0.4);
    chk('out-point dragged below the cue clamps it inside', near(s.launchStart, 0.395), `got ${s.launchStart}`);
    s.setRegion(0.2, 0.8); s.setLaunchStart(0.5);
    s.setRegion(0.5, 0.5);
    chk('degenerate region is ignored, cue untouched', near(s.launchStart, 0.5) && near(s.regionStart, 0.2));
    s.setRegion(0.1, 0.9);
    chk('widening the loop keeps the cue where it is', near(s.launchStart, 0.5));
}

console.log('=== Cue point: what each launch path asks Web Audio to do ===');
{
    const s = freshStrip();
    s.play(true, 0);
    let src = last(s);
    chk('no cue: loop starts at the in-point (pre-cue behaviour)', near(src.started.offset, 0) && near(src.loopStart, 0) && near(src.loopEnd, 10));

    s.setRegion(0.2, 0.8); s.setLaunchStart(0.5);
    s.play(true, 0);
    src = last(s);
    chk('loop: enters at the cue, loops the region', near(src.started.offset, 5) && near(src.loopStart, 2) && near(src.loopEnd, 8) && src.loop === true,
        `offset=${src.started.offset} loop=[${src.loopStart}, ${src.loopEnd}]`);

    s.play(false, 0);
    src = last(s);
    chk('one-shot: cue → out-point, then stops', near(src.started.offset, 5) && near(src.started.duration, 3) && src.loop === false,
        `offset=${src.started.offset} dur=${src.started.duration}`);

    s.ctx.currentTime = 1;
    s.play(true, 4);               // quantized launch in the future
    src = last(s);
    chk('quantized launch keeps the scheduled time', near(src.started.when, 4));

    s.playAt(true, 6);             // gapless retrigger path
    src = last(s);
    chk('retrigger (playAt) also enters at the cue', near(src.started.offset, 5) && near(src.started.when, 6) && near(src.loopStart, 2));

    s.launchStart = 0.9;           // stale restored value outside the loop
    s.play(true, 0);
    chk('stale cue outside the loop falls back to the in-point', near(last(s).started.offset, 2));
}

console.log('=== Cue point: playhead projection ===');
{
    const s = freshStrip();
    chk('nothing playing → no playhead', s.getPlayPosition() === null);
    s.setRegion(0.2, 0.8); s.setLaunchStart(0.5);
    s.play(true, 0);
    const at = (t) => { s.ctx.currentTime = t; return s.getPlayPosition(); };
    chk('t=1 s: cue + 1 s', near(at(1), 0.6));
    chk('t=3 s: hits the out-point, wraps to the in-point', near(at(3), 0.2));
    chk('t=4 s: second pass, 1 s after the in-point', near(at(4), 0.3));
    chk('t=9 s: wrapped again (6 s loop)', near(at(9), 0.2));
    s.stop();
    chk('after stop → no playhead', s.getPlayPosition() === null);

    s.ctx.currentTime = 0;
    s.play(false, 0);
    chk('one-shot t=2.5 s', near(at(2.5), 0.75));
    chk('one-shot past the out-point → hidden', at(3.5) === null);

    s.ctx.currentTime = 1;
    s.play(true, 5);
    chk('scheduled launch parks the playhead on the cue', near(s.getPlayPosition(), 0.5));

    // Retarget a live loop: freeze against the old bounds, then project with
    // the new ones — matches Web Audio, which keeps playing to the new loopEnd.
    s.ctx.currentTime = 0;
    s.play(true, 0);
    s.ctx.currentTime = 1;           // playhead at 6 s
    s.setRegion(0.2, 0.7);           // out-point 8 s → 7 s
    chk('live loop bounds updated on the source', near(last(s).loopEnd, 7));
    chk('rebase: t=1.5 s → 6.5 s', near(at(1.5), 0.65));
    chk('rebase: t=2 s → reaches the new out-point, wraps to 2 s', near(at(2), 0.2));
}

console.log('=== Cue point: new clip resets it ===');
{
    const s = freshStrip();
    s.setRegion(0.2, 0.8); s.setLaunchStart(0.5);
    await s.loadBufferFromBlob({ arrayBuffer: async () => new ArrayBuffer(8) });
    chk('loading a different clip resets region and cue', s.launchStart === null && s.regionStart === 0 && s.regionEnd === 1);
}

console.log(fails ? `\n${fails} FAILED` : '\nAll cue-point engine checks passed.');
process.exit(fails ? 1 : 0);
