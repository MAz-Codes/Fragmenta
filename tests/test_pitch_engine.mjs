// Pitch knob in the channel engine (ChannelStrip.setPitch): what sources play,
// how a live or scheduled source is swapped without moving the playhead, and
// that stale renders never win. Runs the SHIPPED engine class against a fake
// AudioContext with the renderer stubbed — the DSP itself is checked in a
// browser by test_pitch_shift.html.
//
// Run:  node tests/test_pitch_engine.mjs
const pending = [];
class Param {
    constructor(v = 0) { this.value = v; this.events = []; }
    setTargetAtTime() {} cancelScheduledValues() {}
    setValueAtTime(v, t) { this.events.push(['set', v, t]); }
    linearRampToValueAtTime(v, t) { this.events.push(['ramp', v, t]); }
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
    async decodeAudioData() { return fakeBuffer(4, 'decoded'); }
}
globalThis.window = { AudioContext: FakeCtx, setTimeout: (fn) => pending.push(fn) };

const { ChannelStrip } = await import('../app/frontend/src/utils/performanceAudio.js');

function fakeBuffer(duration, tag) {
    return { tag, duration, numberOfChannels: 2, length: duration * 44100, getChannelData: () => new Float32Array(16) };
}
let fails = 0;
function chk(name, cond, detail = '') {
    console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
    if (!cond) fails++;
}
const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
const blob = { arrayBuffer: async () => new ArrayBuffer(8) };

// Renderer stub: each call returns a tagged copy; `gate` lets a test hold a
// render open to race it against a newer request.
function stripWithStub(duration = 10) {
    const s = new ChannelStrip(new Node_(), null, null);
    s.buffer = fakeBuffer(duration, 'orig');
    s.ctx.currentTime = 0;        // every strip shares one context: reset its clock
    s.renders = [];
    s.gate = null;
    s._renderPitch = async (buffer, n) => {
        s.renders.push(n);
        if (s.gate) await s.gate;
        return fakeBuffer(buffer.duration, `${buffer.tag}@${n}`);
    };
    return s;
}
const last = (s) => s.ctx.sources[s.ctx.sources.length - 1];

console.log('=== setPitch: range and what plays ===');
{
    const s = stripWithStub();
    await s.setPitch(30);
    chk('clamped to +12', s.pitchSemitones === 12);
    await s.setPitch(-4.6);
    chk('rounded to whole semitones', s.pitchSemitones === -5);
    s.play(true, 0);
    chk('a launch plays the pitched copy', last(s).buffer.tag === 'orig@-5', last(s).buffer.tag);
    await s.setPitch(0);
    chk('back to 0 plays the original', last(s).buffer.tag === 'orig');
    chk('the waveform buffer stays the original', s.buffer.tag === 'orig');
}

console.log('=== cache ===');
{
    const s = stripWithStub();
    await s.setPitch(3); await s.setPitch(0); await s.setPitch(3);
    chk('a pitch already rendered for this clip is reused', s.renders.filter((n) => n === 3).length === 1, `renders ${s.renders}`);
    for (const n of [1, 2, 4, 5, 6]) await s.setPitch(n);
    await s.setPitch(3);
    chk('only the last few pitches are kept', s.renders.filter((n) => n === 3).length === 2);
}

console.log('=== live swap keeps the playhead ===');
{
    const s = stripWithStub(8);
    s.play(true, 0);                              // loop 0..8 s from 0
    s.ctx.currentTime = 3;
    const old = last(s);
    await s.setPitch(7);
    const neu = last(s);
    chk('a new source takes over', neu !== old && neu.buffer.tag === 'orig@7');
    chk('…entering where the old one is', near(neu.started.offset, 3.005, 1e-6), `offset ${neu.started.offset}`);
    chk('…a few ms from now', near(neu.started.when, 3.005, 1e-6));
    chk('…looping the same region', neu.loop && near(neu.loopStart, 0) && near(neu.loopEnd, 8));
    chk('old source fades out and stops', old.stoppedAt !== null && old.stoppedAt > 3);
    const fadeIn = s.sourceFade.gain.events.find((e) => e[0] === 'ramp' && e[1] === 1);
    chk('new source fades in (no click)', !!fadeIn && fadeIn[2] > 3.005, JSON.stringify(s.sourceFade.gain.events));
    // Until the new source starts (5 ms) the playhead parks on its entry point.
    chk('playhead projection continues from the same spot', near(s.getPlayPosition() * 8, 3.0, 0.006));
    s.ctx.currentTime = 9;                        // 6 s on: 3 + 6 = 9 → wraps to 1
    chk('…and wraps the loop as before', near(s.getPlayPosition() * 8, 1.0, 1e-6), `${s.getPlayPosition() * 8}`);
}
{
    const s = stripWithStub(8);
    s.setRegion(0.25, 0.5);                       // loop 2..4 s
    s.play(true, 0);
    s.ctx.currentTime = 3.998;                    // 2 → 4, wrap, 3.998: right before the out-point
    await s.setPitch(-2);
    chk('swap right at the out-point wraps into the region', near(last(s).started.offset, 2.003, 1e-6), `${last(s).started.offset}`);
}

console.log('=== scheduled (not yet started) source ===');
{
    const s = stripWithStub(8);
    s.ctx.currentTime = 1;
    s.playAt(true, 2);                            // quantized launch at t=2
    const old = last(s);
    await s.setPitch(5);
    const neu = last(s);
    chk('rebuilt for the same launch time', neu !== old && near(neu.started.when, 2), `${neu.started.when}`);
    chk('…same entry point', near(neu.started.offset, 0));
    chk('…with the pitched copy', neu.buffer.tag === 'orig@5');
    chk('…and the old one never sounds', old.stoppedAt !== null && old.stoppedAt <= 1.01);
}

console.log('=== one-shot past its end is left alone ===');
{
    const s = stripWithStub(4);
    s.play(false, 0);
    s.ctx.currentTime = 5;                        // already past the 4 s clip
    const n = s.ctx.sources.length;
    await s.setPitch(3);
    chk('no new source for a finished one-shot', s.ctx.sources.length === n);
}

console.log('=== stale renders never win ===');
{
    const s = stripWithStub();
    let release;
    s.gate = new Promise((r) => { release = r; });
    const slow = s.setPitch(9);                   // render held open
    s.gate = null;
    await s.setPitch(2);                          // newer request finishes first
    release();
    const ok = await slow;
    chk('the overtaken request reports false', ok === false);
    chk('the newer pitch is what plays next', (s.play(true, 0), last(s).buffer.tag === 'orig@2'), last(s).buffer.tag);
}
{
    const s = stripWithStub();
    await s.setPitch(4);
    let release;
    s.gate = new Promise((r) => { release = r; });
    const slow = s.setPitch(6);                   // render of the OLD clip held open
    s.gate = null;
    await s.loadBufferFromBlob(blob);             // new clip arrives
    release();
    await slow;
    s.play(true, 0);
    chk('a new clip keeps the pitch, rendered before it plays', last(s).buffer.tag === 'decoded@6', last(s).buffer.tag);
    chk('a render of the old clip never lands on the new one', !String(last(s).buffer.tag).startsWith('orig'));
}
{
    const s = stripWithStub();
    s._renderPitch = async () => { throw new Error('boom'); };
    s.pitchSemitones = 3;
    await s.loadBufferFromBlob(blob);
    s.play(true, 0);
    chk('a failed render falls back to the original clip', last(s).buffer.tag === 'decoded');
}

console.log(fails ? `\n${fails} FAILED` : '\nAll pitch engine checks passed.');
process.exit(fails ? 1 : 0);
