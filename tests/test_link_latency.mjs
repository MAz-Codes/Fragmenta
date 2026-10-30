// Link latency compensation (PerformanceEngine.getNextQuantizedAudioTime and
// the calibration clicks): Link launches must be *heard* on Link's beat, i.e.
// scheduled earlier by the output latency plus the user's offset.
//
// Run:  node tests/test_link_latency.mjs
let intervalFn = null;
globalThis.window = { setInterval: (fn) => { intervalFn = fn; return 1; }, clearInterval: () => {}, AudioContext: function () {} };
let perfNow = 0;
globalThis.performance = { now: () => perfNow };

const { PerformanceEngine } = await import('../app/frontend/src/utils/performanceAudio.js');

let fails = 0;
const chk = (name, cond, detail = '') => {
    console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
    if (!cond) fails++;
};
const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

// Bare engine: only the fields the scheduling code reads.
function engine({ latency = 0.05, timestamp = true, base = 0, out = 0 } = {}) {
    const e = Object.create(PerformanceEngine.prototype);
    e.ctx = {
        currentTime: 10,
        baseLatency: base, outputLatency: out,
        // contextTime = the frame being heard now = rendered `latency` ago.
        getOutputTimestamp: timestamp ? () => ({ contextTime: e.ctx.currentTime - latency, performanceTime: perfNow }) : undefined,
    };
    e.channels = [{ isPlaying: true }];
    e.internalTransport = { bpm: 120, anchorBeat: 0, originAudioTime: 0 };
    e.launchQuantum = 4;
    e.linkOffsetMs = 0;
    return e;
}
// Link at 120 bpm: beat 0 at performance 0 ms → beat b at b*500 ms.
const snap = { beat: 0, bpm: 120, capturedAt: 0, isPlaying: true };
// Audio time ↔ performance time: currentTime 10 s ↔ perfNow.
const heardAt = (e, when, latency) => perfNow + (when - e.ctx.currentTime + latency) * 1000;

console.log('=== quantized Link launch is heard on the beat ===');
perfNow = 1000;                       // Link beat 2.0 now
for (const L of [0, 0.02, 0.05, 0.12]) {
    const e = engine({ latency: L }); e.setLinkSnapshot(snap);
    const { when, beat } = e.getNextQuantizedAudioTime();
    chk(`latency ${L * 1000} ms → heard exactly at beat ${beat}`, near(heardAt(e, when, L), beat * 500, 1e-6),
        `heard ${heardAt(e, when, L).toFixed(3)} ms, beat at ${beat * 500} ms`);
}
{
    const e = engine({ latency: 0.05 }); e.setLinkSnapshot(snap);
    const before = e.getNextQuantizedAudioTime().when;
    e.setLinkOffsetMs(8);
    const after = e.getNextQuantizedAudioTime().when;
    chk('user offset +8 ms → scheduled 8 ms earlier', near(before - after, 0.008));
}
{
    const e = engine({ latency: 0.05 }); e.setLinkSnapshot(snap);
    perfNow = 3990;                   // 10 ms before bar beat 8 — too late with 50 ms latency
    const { beat, when } = e.getNextQuantizedAudioTime();
    chk('boundary already missed → next bar, never scheduled in the past', beat === 12 && when > e.ctx.currentTime, `beat ${beat}`);
    perfNow = 1000;
}

console.log('=== fallback without getOutputTimestamp ===');
{
    const e = engine({ timestamp: false, base: 0.01, out: 0.03 }); e.setLinkSnapshot(snap);
    chk('reported latency = baseLatency + outputLatency', Math.round(e.getOutputLatencyMs()) === 40);
    const { when, beat } = e.getNextQuantizedAudioTime();
    chk('launch still heard on the beat', near(heardAt(e, when, 0.04), beat * 500, 1e-6));
}
{
    const e = engine({ latency: 0.05 });
    e.ctx.getOutputTimestamp = () => ({ contextTime: 0, performanceTime: 0 });  // device not started yet
    e.ctx.baseLatency = 0.02; e.setLinkSnapshot(snap);
    chk('zeroed timestamp (not started) → falls back to reported latency', Math.round(e.getOutputLatencyMs()) === 20);
}
{
    const e = engine({ latency: 0.05 });
    chk('measured latency from the timestamp', Math.round(e.getOutputLatencyMs()) === 50);
}

console.log('=== phase lock uses the heard beat ===');
{
    const e = engine({ latency: 0.05 }); e.setLinkSnapshot(snap); e.launchQuantum = 0;
    const { when, beat } = e.getNextQuantizedAudioTime();
    chk('unquantized Link launch: starts now, phase of the beat heard 50 ms later', when === 0 && near(beat, 2.1), `beat ${beat}`);
}
{
    const e = engine({ latency: 0.05 }); e.linkSnapshot = null; e.launchQuantum = 4;
    e.internalTransport = { bpm: 120, anchorBeat: 0, originAudioTime: 9 };   // 2 beats in
    const { when, beat } = e.getNextQuantizedAudioTime();
    chk('internal transport (no Link) is unchanged', beat === 4 && near(when, 11), `when ${when}`);
}

console.log('=== calibration clicks ===');
function clickEngine(latency) {
    const e = engine({ latency }); e.setLinkSnapshot(snap);
    const starts = [];
    const node = () => ({ connect: (n) => n, disconnect() {}, frequency: {}, gain: { setValueAtTime() {}, exponentialRampToValueAtTime() {} } });
    e.masterBus = node();
    e.ctx.createOscillator = () => ({ ...node(), start: (t) => starts.push(t), stop() {} });
    e.ctx.createGain = node;
    return { e, starts };
}
{
    const { e, starts } = clickEngine(0.05);
    perfNow = 1300;                   // beat 3 is heard in 200 ms
    e.startCalibrationClicks();
    chk('normal latency: the next beat gets a click', starts.length === 1, `${starts.length} click(s)`);
    chk('…heard exactly on its Link beat', starts.length && near(heardAt(e, starts[0], 0.05), 1500, 1e-6));
    e.stopCalibrationClicks();
}
{
    const { e, starts } = clickEngine(0.3);   // e.g. Bluetooth
    perfNow = 1300; e.startCalibrationClicks();
    perfNow = 1550; intervalFn();             // a later tick of the scheduler
    chk('300 ms latency: beat 4 still gets a click (look-ahead is in audio time)', starts.length === 1, `${starts.length} click(s)`);
    chk('…heard exactly on beat 4', starts.length && near(heardAt(e, starts[0], 0.3), 2000, 1e-6));
    perfNow = 1560; intervalFn();
    chk('no duplicate click on the next tick', starts.length === 1);
    e.stopCalibrationClicks();
}
perfNow = 1000;

console.log(fails ? `\n${fails} FAILED` : '\nAll Link latency checks passed.');
process.exit(fails ? 1 : 0);
