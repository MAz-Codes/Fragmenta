export const DEFAULT_CHANNEL_GAIN = 1.0; // unity (0 dB)

// Stage A renders loops at exactly 44.1 kHz and the sample-exact loop length
// (INV#2/#6) is defined at that rate. A default AudioContext adopts the system
// rate (often 48 kHz), which makes decodeAudioData resample our 44.1 kHz WAVs —
// silently changing the loop's sample count and smearing the seam. Pinning the
// context to 44.1 kHz keeps our buffers un-resampled inside Web Audio (the OS
// may still resample at the device, but that's outside the loop math).
export const ENGINE_SAMPLE_RATE = 44100;

let sharedCtx = null;

// Whether to pin the shared AudioContext to ENGINE_SAMPLE_RATE. Forcing a
// non-native rate makes Chromium route output through an internal stereo
// resampler, which collapses destination.maxChannelCount to 2 and hides every
// multi-channel main/cue output pair. The pin only benefits beatsync v2's
// sample-exact loop seams, so it's opt-in: App.js calls setSampleRatePin(true)
// when /api/environment reports beatsync_v2. Default false → native device
// rate → full channel count, so multi-channel output works out of the box.
let pinSampleRate = false;

// Set the sample-rate pin before the context is first created (i.e. before
// entering Performance mode). Once getAudioContext() has built the shared
// context this has no effect on it.
export function setSampleRatePin(on) {
    pinSampleRate = Boolean(on);
}

export function getAudioContext() {
    if (!sharedCtx) {
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (pinSampleRate) {
            try {
                sharedCtx = new Ctor({ sampleRate: ENGINE_SAMPLE_RATE });
            } catch (_) {
                // Some browsers/hosts reject a forced rate — fall back to default
                // and accept the resample rather than failing to produce audio.
                sharedCtx = new Ctor();
            }
            if (sharedCtx.sampleRate !== ENGINE_SAMPLE_RATE) {
                console.warn(
                    `[PerformanceEngine] AudioContext is ${sharedCtx.sampleRate} Hz, ` +
                    `not ${ENGINE_SAMPLE_RATE} Hz — 44.1 kHz loops will be resampled; ` +
                    `sample-exact loop length is not guaranteed on this host.`
                );
            }
        } else {
            // Native device rate: keeps destination.maxChannelCount equal to the
            // device's real channel count so multi-channel output is available.
            sharedCtx = new Ctor();
        }
    }
    if (sharedCtx.state === 'suspended') {
        sharedCtx.resume();
    }
    return sharedCtx;
}

// Original, synthesised IRs (see tools/generate_impulse_responses.py) — no
// third-party licensing. Early reflections + a multiband exponential tail per
// voice; the ConvolverNode normalizes the buffer at load.
export const IMPULSE_RESPONSES = [
    { id: 'hall',   name: 'Concert Hall',  file: 'hall.wav' },
    { id: 'room',   name: 'Drum Room',     file: 'room.wav' },
    { id: 'narrow', name: 'Narrow Space',  file: 'narrow.wav' },
];
const DEFAULT_IR_ID = 'hall';

// Master delay divisions, in fractions of a beat (quarter note = 1 beat).
// Tempo-synced via PerformanceEngine.setBpm — the actual delay-time seconds
// are recomputed whenever BPM changes so the echo always lands on the grid.
// Dotted = 1.5×, Triplet = 2/3×.
export const MASTER_DELAY_DIVISIONS = [
    { id: '1/4',   label: '1/4',   beats: 1 },
    { id: '1/8',   label: '1/8',   beats: 0.5 },
    { id: '1/16',  label: '1/16',  beats: 0.25 },
    { id: '1/4D',  label: '1/4 ·', beats: 1.5 },
    { id: '1/8D',  label: '1/8 ·', beats: 0.75 },
    { id: '1/16D', label: '1/16 ·', beats: 0.375 },
    { id: '1/4T',  label: '1/4 T', beats: 2 / 3 },
    { id: '1/8T',  label: '1/8 T', beats: 1 / 3 },
    { id: '1/16T', label: '1/16 T', beats: 1 / 6 },
];
const DEFAULT_DELAY_DIVISION_ID = '1/4';

const irBufferCache = new Map();
let irLoadPromise = null;

async function fetchAndDecodeIR(ctx, file) {
    const res = await fetch(`/ir/${encodeURIComponent(file)}`);
    if (!res.ok) throw new Error(`IR fetch failed (${res.status}): ${file}`);
    const arr = await res.arrayBuffer();
    return await ctx.decodeAudioData(arr);
}

export function loadImpulseResponses(ctx) {
    if (irLoadPromise) return irLoadPromise;
    irLoadPromise = Promise.all(
        IMPULSE_RESPONSES.map(async (ir) => {
            try {
                const buf = await fetchAndDecodeIR(ctx, ir.file);
                irBufferCache.set(ir.id, buf);
            } catch (e) {
                console.warn(`[performanceAudio] IR load failed for ${ir.id}:`, e);
            }
        })
    ).then(() => irBufferCache);
    return irLoadPromise;
}

export function getImpulseResponseBuffer(id) {
    return irBufferCache.get(id);
}

const EARLY_REFLECTIONS_MS = [
    [7, 0.55], [13, -0.42], [19, 0.36], [28, -0.30],
    [41, 0.26], [56, 0.22], [73, -0.18], [91, 0.15],
];

let sharedImpulse = null;
function getImpulse(ctx, duration = 2.8, decaySeconds = 1.6, damping = 0.55) {
    if (sharedImpulse && sharedImpulse.sampleRate === ctx.sampleRate) {
        return sharedImpulse;
    }
    const sr = ctx.sampleRate;
    const length = Math.floor(sr * duration);
    const buf = ctx.createBuffer(2, length, sr);

    for (let ch = 0; ch < 2; ch++) {
        const data = buf.getChannelData(ch);
        const stereoJitter = ch === 0 ? 1.0 : 1.037;

        for (const [timeMs, amp] of EARLY_REFLECTIONS_MS) {
            const idx = Math.floor(sr * timeMs * 0.001 * stereoJitter);
            if (idx < length) data[idx] += amp * 0.8;
        }

        let lpState = 0;
        const predelaySec = 0.012;
        for (let i = 0; i < length; i++) {
            const t = i / sr;
            if (t < predelaySec) continue;
            const env = Math.exp(-(t - predelaySec) / decaySeconds);
            const progression = Math.min(1, (t - predelaySec) / decaySeconds);
            const alpha = 1 - damping * (0.4 + 0.55 * progression);
            const noise = (Math.random() * 2 - 1) * env;
            lpState += alpha * (noise - lpState);
            data[i] += lpState * 0.72;
        }

        let peak = 0;
        for (let i = 0; i < length; i++) {
            const v = Math.abs(data[i]);
            if (v > peak) peak = v;
        }
        if (peak > 0) {
            const norm = 0.92 / peak;
            for (let i = 0; i < length; i++) data[i] *= norm;
        }
    }
    sharedImpulse = buf;
    return buf;
}

export class ChannelStrip {
    constructor(masterBus, masterDelayInput, masterReverbInput) {
        const ctx = getAudioContext();
        this.ctx = ctx;
        this.buffer = null;
        this.source = null;
        this.sourceFade = null;  // per-source fade gain — see stop() for why
        this.isPlaying = false;
        this.isLooping = false;
        this.isMuted = false;
        this.isSoloed = false;

        // High-pass and low-pass in series — together they act as a bipolar
        // "DJ filter" knob. At rest (bypass) the HPF sits at 20 Hz and the
        // LPF at 20 kHz so neither shapes audible content. The channel knob
        // drives only one side at a time depending on its sign.
        this.hpf = ctx.createBiquadFilter();
        this.hpf.type = 'highpass';
        this.hpf.frequency.value = 20;
        this.hpf.Q.value = 0.7;

        this.filter = ctx.createBiquadFilter();
        this.filter.type = 'lowpass';
        this.filter.frequency.value = 18000;
        this.filter.Q.value = 0.7;

        // Source → hpf → lpf → rest. Source connection happens in play()
        // so we only wire the static portion of the chain here.
        this.hpf.connect(this.filter);

        // Channel DLY/REV knobs control send amounts (0..1) into the shared
        // master delay and reverb — no more local FX nodes per channel. The
        // master FX run always-on; each channel's contribution to the wet
        // bus is determined by its send level.
        this.delaySend = ctx.createGain();
        this.delaySend.gain.value = 0;
        this.reverbSend = ctx.createGain();
        this.reverbSend.gain.value = 0;

        this.channelGain = ctx.createGain();
        this.channelGain.gain.value = DEFAULT_CHANNEL_GAIN;
        this._lastUserGain = DEFAULT_CHANNEL_GAIN;

        this.pan = ctx.createStereoPanner();
        this.pan.pan.value = 0;

        this.analyser = ctx.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyserData = new Uint8Array(this.analyser.frequencyBinCount);

        // Dry path — filter chain straight through to the master bus. Channels
        // run clean: the only dynamics stage is the master limiter, which
        // catches the summed peaks. (A per-channel compressor used to sit here
        // — thr -16, 2.5:1, no makeup gain — quietly costing several dB on
        // transient material and making the whole app feel quiet.)
        this.filter.connect(this.channelGain);
        this.channelGain.connect(this.pan);
        this.pan.connect(this.analyser);
        this.analyser.connect(masterBus);

        // Post-fader sends — tap from `pan` so the channel fader, mute, and
        // pan position all affect what reaches the master FX. Mute the
        // channel and the reverb tail / echoes die too.
        if (masterDelayInput) {
            this.pan.connect(this.delaySend);
            this.delaySend.connect(masterDelayInput);
        }
        if (masterReverbInput) {
            this.pan.connect(this.reverbSend);
            this.reverbSend.connect(masterReverbInput);
        }
    }

    async loadBlob(blob) {
        this.stop();
        await this.loadBufferFromBlob(blob);
    }

    // Decode a blob into the channel buffer WITHOUT touching current playback.
    // Used for gapless, scheduled clip swaps where the live source must keep
    // sounding until the next launch-quantization boundary. (loadBlob() stops
    // first; this one deliberately does not.)
    async loadBufferFromBlob(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        this.buffer = await this.ctx.decodeAudioData(arrayBuffer);
    }

    play(loop = this.isLooping, startTime = 0) {
        if (!this.buffer) return;
        this.stop();
        this.isLooping = loop;
        const src = this.ctx.createBufferSource();
        src.buffer = this.buffer;
        src.loop = loop;
        // Per-source gain so we can fade-out without touching the user's
        // channelGain. Cancels the click that would otherwise happen when
        // stop() truncates a non-zero sample.
        const fadeGain = this.ctx.createGain();
        fadeGain.gain.value = 1;
        src.connect(fadeGain);
        // Source enters at the head of the filter chain (hpf → lpf → …).
        fadeGain.connect(this.hpf);
        src.onended = () => {
            if (this.source === src) {
                this.source = null;
                this.sourceFade = null;
                this.isPlaying = false;
            }
        };

        // Start from the head of the clip at the scheduled (quantized) time.
        src.start(Math.max(0, startTime));
        this.source = src;
        this.sourceFade = fadeGain;
        this.isPlaying = true;
    }

    // Launch the current buffer from its head at `startTime` (0 = ASAP) WITHOUT
    // cutting the currently-playing source until that moment. This gives
    // gapless, grid-aligned clip switching: the live clip keeps sounding right
    // up to the boundary, then the new one takes over. When `startTime` is ~now
    // (immediate / no quantization) it degrades to a short 12 ms crossfade,
    // identical in feel to play() but without the up-front silence.
    playAt(loop = this.isLooping, startTime = 0) {
        if (!this.buffer) return;
        const now = this.ctx.currentTime;
        const when = startTime > now ? startTime : now;  // 0 / past => now
        const FADE = 0.012;

        // Hand off any current source: hold its level until just before the
        // boundary, fade over 12 ms, and stop it right at `when`. No gap.
        if (this.source) {
            const oldSrc = this.source;
            const oldFade = this.sourceFade;
            const stopAt = Math.max(when, now + FADE);
            const fadeStart = Math.max(now, stopAt - FADE);
            try {
                if (oldFade) {
                    oldFade.gain.cancelScheduledValues(now);
                    oldFade.gain.setValueAtTime(oldFade.gain.value, fadeStart);
                    oldFade.gain.linearRampToValueAtTime(0, stopAt);
                }
                oldSrc.stop(stopAt + 0.005);
            } catch (_) { /* already stopped */ }
            window.setTimeout(() => {
                try { oldSrc.disconnect(); } catch (_) { /* ok */ }
                try { oldFade && oldFade.disconnect(); } catch (_) { /* ok */ }
            }, Math.ceil((stopAt - now + 0.03) * 1000));
        }

        // Schedule the new source from the clip head at the boundary.
        this.isLooping = loop;
        const src = this.ctx.createBufferSource();
        src.buffer = this.buffer;
        src.loop = loop;
        const fadeGain = this.ctx.createGain();
        fadeGain.gain.value = 1;
        src.connect(fadeGain);
        fadeGain.connect(this.hpf);
        src.onended = () => {
            if (this.source === src) {
                this.source = null;
                this.sourceFade = null;
                this.isPlaying = false;
            }
        };
        src.start(when);
        this.source = src;
        this.sourceFade = fadeGain;
        this.isPlaying = true;
    }

    stop() {
        if (this.source) {
            const src = this.source;
            const fade = this.sourceFade;
            const now = this.ctx.currentTime;
            // 12ms gain ramp masks the click from cutting a buffer
            // mid-cycle. The source is scheduled to actually stop just
            // after the ramp finishes; disconnect happens on the next
            // tick after that so we don't truncate the tail ourselves.
            const FADE = 0.012;
            try {
                if (fade) {
                    fade.gain.cancelScheduledValues(now);
                    fade.gain.setValueAtTime(fade.gain.value, now);
                    fade.gain.linearRampToValueAtTime(0, now + FADE);
                }
                src.stop(now + FADE + 0.005);
            } catch (_) { /* already stopped */ }
            // Detach the nodes after the tail finishes — otherwise we'd
            // cut the fade short and reintroduce the click.
            window.setTimeout(() => {
                try { src.disconnect(); } catch (_) { /* ok */ }
                try { fade && fade.disconnect(); } catch (_) { /* ok */ }
            }, Math.ceil((FADE + 0.02) * 1000));
            this.source = null;
            this.sourceFade = null;
        }
        this.isPlaying = false;
    }

    setGain(value) { this.channelGain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.01); }
    setFilter(hz) { this.filter.frequency.setTargetAtTime(hz, this.ctx.currentTime, 0.01); }
    setHighpass(hz) { this.hpf.frequency.setTargetAtTime(hz, this.ctx.currentTime, 0.01); }
    // setDelayMix / setReverbMix drive the post-fader send levels into the
    // shared master delay and reverb. Knob range stays 0..1 — same numeric
    // contract as the old local-FX path, just routed differently.
    setDelayMix(value) { this.delaySend.gain.setTargetAtTime(value, this.ctx.currentTime, 0.02); }
    setReverbMix(value) { this.reverbSend.gain.setTargetAtTime(value, this.ctx.currentTime, 0.05); }
    setPan(value) { this.pan.pan.setTargetAtTime(value, this.ctx.currentTime, 0.01); }
    setLoop(value) {
        this.isLooping = value;
        if (this.source) this.source.loop = value;
    }

    applyMuteSolo(anySoloed) {
        const audible = !this.isMuted && (!anySoloed || this.isSoloed);
        this.channelGain.gain.setTargetAtTime(audible ? this._lastUserGain ?? 0 : 0, this.ctx.currentTime, 0.01);
    }

    setUserGain(value) {
        this._lastUserGain = value;
        this.channelGain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.01);
    }

    getLevel() {
        if (!this.isPlaying) return 0;
        this.analyser.getByteTimeDomainData(this.analyserData);
        let peak = 0;
        for (let i = 0; i < this.analyserData.length; i++) {
            const v = Math.abs(this.analyserData[i] - 128) / 128;
            if (v > peak) peak = v;
        }
        return peak;
    }

    drawWaveform(canvas, color) {
        if (!canvas || !this.buffer) return;
        const ctx2d = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx2d.clearRect(0, 0, w, h);
        const data = this.buffer.getChannelData(0);
        // Fixed low resolution (80 buckets) rendered as bars — matches the
        // dataset-page and Generated-Fragments waveforms.
        const PEAK_COUNT = 80;
        const bucket = Math.max(1, Math.floor(data.length / PEAK_COUNT));
        const xStep = w / PEAK_COUNT;
        const barW = Math.max(1, xStep - 1);
        ctx2d.fillStyle = color || '#279FBB';
        for (let i = 0; i < PEAK_COUNT; i++) {
            let min = 1.0, max = -1.0;
            const start = i * bucket;
            const end = Math.min(data.length, start + bucket);
            for (let j = start; j < end; j++) {
                const v = data[j];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            const yA = (1 + min) * 0.5 * h;
            const yB = (1 + max) * 0.5 * h;
            ctx2d.fillRect(i * xStep, yA, barW, Math.max(1, yB - yA));
        }
    }

    dispose() {
        this.stop();
        try {
            this.hpf.disconnect();
            this.filter.disconnect();
            this.delaySend.disconnect();
            this.reverbSend.disconnect();
            this.channelGain.disconnect();
            this.pan.disconnect();
            this.analyser.disconnect();
        } catch (_) { /* already disconnected */ }
    }
}

// AudioWorklet processor source for master capture. Loaded as an inline
// blob module (no separate served file). It accumulates the input's L/R
// Float32 blocks and flushes them to the main thread on a 'stop' message.
// Chosen over the deprecated ScriptProcessorNode, whose onaudioprocess does
// not fire reliably in the WebKitGTK / Chromium-app webviews.
const REC_WORKLET_SRC = `
class FragmentaRecorder extends AudioWorkletProcessor {
    constructor() {
        super();
        this._l = [];
        this._r = [];
        this._on = true;
        this.port.onmessage = (e) => {
            if (e.data === 'stop') {
                this._on = false;
                this.port.postMessage({ type: 'data', left: this._l, right: this._r });
                this._l = [];
                this._r = [];
            }
        };
    }
    process(inputs) {
        const input = inputs[0];
        if (this._on && input && input.length > 0) {
            const l = input[0];
            const r = input.length > 1 ? input[1] : input[0];
            if (l && l.length) this._l.push(new Float32Array(l));
            if (r && r.length) this._r.push(new Float32Array(r));
        }
        return true;
    }
}
registerProcessor('fragmenta-recorder', FragmentaRecorder);
`;

// Concatenate an array of Float32 chunks into one contiguous buffer.
function flattenFloat32(chunks) {
    let total = 0;
    for (const c of chunks) total += c.length;
    const out = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) { out.set(c, offset); offset += c.length; }
    return out;
}

// Encode interleaved 16-bit PCM stereo into a RIFF/WAVE Blob. Mirrors the
// backend's PCM_16 WAV convention so performance captures match generated
// fragments.
function encodeWavStereo(left, right, sampleRate) {
    const numFrames = left.length;
    const numChannels = 2;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const dataSize = numFrames * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    const writeStr = (off, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);                 // PCM fmt chunk size
    view.setUint16(20, 1, true);                  // audio format = PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 8 * bytesPerSample, true);
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);
    let off = 44;
    const clamp = (x) => (x < -1 ? -1 : x > 1 ? 1 : x);
    for (let i = 0; i < numFrames; i++) {
        view.setInt16(off, clamp(left[i]) * 0x7fff, true); off += 2;
        view.setInt16(off, clamp(right[i]) * 0x7fff, true); off += 2;
    }
    return new Blob([view], { type: 'audio/wav' });
}

export class PerformanceEngine {
    constructor(channelCount = 8) {
        const ctx = getAudioContext();
        this.ctx = ctx;
        this.masterBus = ctx.createGain();
        this.masterBus.gain.value = 0.9;
        this.masterLimiter = ctx.createDynamicsCompressor();
        this.masterLimiter.threshold.value = -1.0;
        this.masterLimiter.knee.value = 0;
        this.masterLimiter.ratio.value = 20;
        this.masterLimiter.attack.value = 0.002;
        this.masterLimiter.release.value = 0.1;

        this.masterAnalyser = ctx.createAnalyser();
        this.masterAnalyser.fftSize = 1024;
        this.masterAnalyserData = new Uint8Array(this.masterAnalyser.frequencyBinCount);

        // Master FX — always-on shared delay and reverb. The wet level on
        // the bus is governed entirely by how much each ChannelStrip sends
        // into them via its post-fader DLY/REV send (no master wet gain).
        // masterBus carries the dry sum; the FX outputs join it at the
        // limiter.
        this.masterDelay = ctx.createDelay(4.0);
        this.masterDelay.delayTime.value = 0.5;          // 1/4 at 120 BPM
        this.masterDelayFeedback = ctx.createGain();
        this.masterDelayFeedback.gain.value = 0.4;

        this.masterReverb = ctx.createConvolver();
        this.masterReverb.buffer = getImpulse(ctx);

        this.masterDelayDivisionId = DEFAULT_DELAY_DIVISION_ID;
        this.currentMasterImpulseId = DEFAULT_IR_ID;

        // Dry: masterBus → limiter. Wet: channel sends → master FX → limiter.
        this.masterBus.connect(this.masterLimiter);

        this.masterDelay.connect(this.masterDelayFeedback);
        this.masterDelayFeedback.connect(this.masterDelay);
        this.masterDelay.connect(this.masterLimiter);

        this.masterReverb.connect(this.masterLimiter);

        this.masterLimiter.connect(this.masterAnalyser);

        // Stage 2 multichannel routing. The stereo master bus is split into
        // two mono lines and merged into a destination sized to the device's
        // maxChannelCount. The pair selector decides which two merger inputs
        // the splitter outputs connect to — everything else stays silent.
        // On stereo-only devices the merger has 2 inputs and only pair 0 is
        // legal, which is the existing behavior.
        this.outputSplitter = null;
        this.outputMerger = null;
        this.currentMainPair = 0;
        this._buildOutputGraph();

        this.channels = Array.from({ length: channelCount }, () =>
            new ChannelStrip(this.masterBus, this.masterDelay, this.masterReverb)
        );


        this.linkSnapshot = null;
        this.launchQuantum = 0;

        // Internal transport: an always-running beat clock anchored in audio
        // time. Used for launch quantization when Ableton Link isn't active,
        // so 'Q' still lines launches up to the bar even with no peer.
        // BPM changes rebase the anchor (see setBpm) so phase is preserved.
        this.internalTransport = {
            originAudioTime: ctx.currentTime,
            anchorBeat: 0,
            bpm: 120,
        };

        loadImpulseResponses(ctx).then(() => {
            const masterBuf = getImpulseResponseBuffer(this.currentMasterImpulseId);
            if (masterBuf) this.masterReverb.buffer = masterBuf;
        });

        // Apply the default master delay division once the transport has its
        // bpm — keeps the initial 0.5s default in sync with the real BPM
        // (could be != 120 if the session restored a different tempo).
        this._applyMasterDelayTime();
    }

    _applyMasterDelayTime() {
        const div = MASTER_DELAY_DIVISIONS.find(d => d.id === this.masterDelayDivisionId);
        if (!div) return;
        const bpm = this.internalTransport?.bpm || 120;
        const seconds = (60 / Math.max(bpm, 1)) * div.beats;
        // The Delay node was created with maxDelayTime=4.0, so cap at that.
        const clamped = Math.min(seconds, 4.0);
        this.masterDelay.delayTime.setTargetAtTime(clamped, this.ctx.currentTime, 0.02);
    }

    setMasterDelayDivision(id) {
        if (!MASTER_DELAY_DIVISIONS.some(d => d.id === id)) return false;
        this.masterDelayDivisionId = id;
        this._applyMasterDelayTime();
        return true;
    }

    setMasterReverbIR(id) {
        if (!IMPULSE_RESPONSES.some(ir => ir.id === id)) return false;
        this.currentMasterImpulseId = id;
        const buf = getImpulseResponseBuffer(id);
        if (buf) this.masterReverb.buffer = buf;
        // If buf is null the IR catalog hasn't finished loading yet — the
        // constructor's loadImpulseResponses().then() will apply the right
        // buffer once it's cached, reading the updated id from this.
        return true;
    }

    /**
     * Route the engine's master output to a specific audio device. Pass `''`
     * for the system default. Stage 1: only does setSinkId — channel-pair
     * routing within the device is a follow-up.
     *
     * Returns the device's max channel count so the UI can populate
     * pair selectors (1-2, 3-4, ...).
     */
    async setOutputDevice(deviceId) {
        if (typeof this.ctx.setSinkId !== 'function') {
            console.warn('[PerformanceEngine] AudioContext.setSinkId not supported on this build');
            return this.ctx.destination.maxChannelCount ?? 2;
        }
        try {
            // Some Chromium versions reject setSinkId on a suspended context.
            if (this.ctx.state === 'suspended') {
                await this.ctx.resume();
            }
            await this.ctx.setSinkId(deviceId || '');

            // Try to coerce the destination to expose all available channels.
            // On Chromium/Linux/PipeWire, maxChannelCount is computed at
            // AudioContext-construction time bound to the original sink, and
            // setSinkId doesn't re-query it. Setting channelCount to the
            // current maxChannelCount forces a re-evaluation on some builds
            // and ensures we claim everything the destination exposes —
            // important for interfaces with more than 8 channels.
            try {
                this.ctx.destination.channelCount = this.ctx.destination.maxChannelCount;
            } catch { /* destination capped; no-op */ }
            try {
                this.ctx.destination.channelInterpretation = 'discrete';
            } catch { /* older builds may not allow this — fine */ }

            const applied = this.ctx.sinkId;
            const dest = this.ctx.destination;
            console.log(
                `[PerformanceEngine] setSinkId requested='${deviceId || '(default)'}' applied='${applied}' ` +
                `channelCount=${dest.channelCount} maxChannelCount=${dest.maxChannelCount} ` +
                `interpretation=${dest.channelInterpretation}`
            );
            if (deviceId && applied !== deviceId) {
                console.warn(
                    `[PerformanceEngine] sinkId did not stick (asked='${deviceId}', got='${applied}'). ` +
                    `Likely a placeholder/un-permissioned device id — grant mic access once to unlock real ids.`
                );
            }

            // ChannelMergerNode input count is fixed at construction. Since
            // maxChannelCount may have changed with the new device, rebuild
            // the splitter→merger→destination tail to size it correctly.
            this._buildOutputGraph();
        } catch (err) {
            console.error('[PerformanceEngine] setSinkId failed:', err);
        }
        return this.ctx.destination.maxChannelCount ?? 2;
    }

    /** Current max channel count of the bound output destination. */
    getMaxChannelCount() {
        return this.ctx.destination.maxChannelCount ?? 2;
    }

    /**
     * (Re)build the splitter → merger → destination tail of the master path.
     * Called from the constructor and again whenever the device changes
     * (since maxChannelCount may change and ChannelMergerNode's input count
     * is fixed at construction). Restores the current pair after rebuild.
     */
    _buildOutputGraph() {
        // Tear down any prior wiring. Disconnect ONLY the analyser→splitter
        // edge: an argless masterAnalyser.disconnect() would also sever the
        // recorder worklet tap (startRecording), silently truncating an
        // active master recording whenever the graph rebuilds.
        if (this.outputSplitter) {
            try { this.masterAnalyser.disconnect(this.outputSplitter); } catch { /* ok */ }
        }
        try { this.outputSplitter?.disconnect(); } catch { /* ok */ }
        try { this.outputMerger?.disconnect(); } catch { /* ok */ }

        // Claim the device's full channel count with discrete interpretation
        // here — not only in setOutputDevice — so multichannel works on the
        // system-default device and on browsers without AudioContext.setSinkId.
        try {
            this.ctx.destination.channelCount = this.ctx.destination.maxChannelCount;
        } catch { /* destination capped; no-op */ }
        try {
            this.ctx.destination.channelInterpretation = 'discrete';
        } catch { /* older builds may not allow this — fine */ }

        const channels = Math.max(2, this.ctx.destination.maxChannelCount || 2);
        this.outputSplitter = this.ctx.createChannelSplitter(2);
        this.outputMerger = this.ctx.createChannelMerger(channels);

        this.masterAnalyser.connect(this.outputSplitter);
        this.outputMerger.connect(this.ctx.destination);
        this._wireMainPair(this.currentMainPair);
    }

    /**
     * Connect the splitter's L/R outputs to a specific pair of merger inputs.
     * Pair 0 = channels 1-2, pair 1 = 3-4, etc. Clamps to the available
     * range so callers can pass stale indices safely.
     */
    _wireMainPair(pairIdx) {
        if (!this.outputSplitter || !this.outputMerger) return;
        const N = this.ctx.destination.maxChannelCount || 2;
        const maxPair = Math.max(0, Math.floor(N / 2) - 1);
        const pair = Math.min(Math.max(0, pairIdx | 0), maxPair);

        try { this.outputSplitter.disconnect(); } catch { /* ok */ }
        this.outputSplitter.connect(this.outputMerger, 0, pair * 2);
        this.outputSplitter.connect(this.outputMerger, 1, pair * 2 + 1);
    }

    /** Public: pick which channel pair the master mix routes to. The unclamped
     *  intent is stored so later graph rebuilds (device swap, or the device
     *  exposing more channels once the context is running) restore the user's
     *  selection instead of an early stereo clamp. */
    setMainOutputPair(pairIdx) {
        this.currentMainPair = Math.max(0, pairIdx | 0);
        this._wireMainPair(this.currentMainPair);
    }

    /** Re-coerce the destination and rebuild the output tail, restoring the
     *  selected pair. Chromium reports maxChannelCount=2 until the context is
     *  actually running, so callers invoke this on the 'running' statechange
     *  (and on user gestures like opening the device menu) to pick up the
     *  device's real channel count. Returns the current maxChannelCount. */
    refreshOutputGraph() {
        // Rebuild only when the device's channel count actually changed —
        // this runs on every device-menu open and 'running' statechange, and
        // a needless rebuild briefly interrupts live playback.
        const want = Math.max(2, this.ctx.destination.maxChannelCount || 2);
        if (!this.outputMerger || this.outputMerger.numberOfInputs !== want) {
            this._buildOutputGraph();
        }
        return this.getMaxChannelCount();
    }

    setBpm(bpm) {
        const safe = Number(bpm);
        if (Number.isFinite(safe) && safe > 0) {
            // Rebase the internal transport so its current beat position is
            // preserved across the BPM change. Without rebasing, switching
            // 120→140 would jump the next-quantized beat by minutes' worth
            // of time in the wrong direction.
            const tr = this.internalTransport;
            const now = this.ctx.currentTime;
            const elapsed = now - tr.originAudioTime;
            tr.anchorBeat += elapsed * tr.bpm / 60;
            tr.originAudioTime = now;
            tr.bpm = safe;
        }
        // Master delay follows the grid — recompute its seconds based on
        // the current division whenever BPM changes.
        this._applyMasterDelayTime();
    }

    setLinkSnapshot(snapshot) {
        this.linkSnapshot = snapshot;
    }

    setLaunchQuantum(beats) {
        const v = Number(beats);
        this.launchQuantum = Number.isFinite(v) && v > 0 ? v : 0;
    }

    // Returns { when, beat, bpm }: the AudioContext time to start (0 = ASAP),
    // the GLOBAL beat at that instant, and the tempo. `beat`+`bpm` let the
    // caller phase-lock each clip to the global grid (INV#8). Back-compat:
    // callers that only need `when` read the `.when` field.
    getNextQuantizedAudioTime() {
        const quantum = this.launchQuantum;
        const snap = this.linkSnapshot;

        // Current global beat + tempo from Link if running, else the internal
        // transport. Both extrapolate to "now".
        let currentBeat = 0;
        let bpm = 0;
        if (snap && snap.bpm) {
            bpm = snap.bpm;
            currentBeat = snap.beat + ((performance.now() - snap.capturedAt) / 1000) * (bpm / 60);
        } else {
            const tr = this.internalTransport;
            if (tr.bpm) {
                bpm = tr.bpm;
                currentBeat = tr.anchorBeat + (this.ctx.currentTime - tr.originAudioTime) * (bpm / 60);
            }
        }

        // First-launch handling. When slaved to Link with a quantum set, even
        // the FIRST clip waits for the shared downbeat (true slave behaviour —
        // Fragmenta drops in on Ableton's bar like any Link peer): fall through
        // to the quantize block. Standalone or quantum=None keeps the Live
        // Session-View instant-fire feel and anchors the internal transport.
        const anythingPlaying = this.channels.some(c => c.isPlaying);
        if (!anythingPlaying && !(snap && snap.bpm && quantum)) {
            if (!snap) {
                this.internalTransport.originAudioTime = this.ctx.currentTime;
                this.internalTransport.anchorBeat = 0;
                currentBeat = 0;
                bpm = this.internalTransport.bpm || bpm;
            }
            return { when: 0, beat: currentBeat, bpm };
        }

        // No quantum (None) -> start ASAP, but still report the live beat so
        // the clip phase-locks to the grid.
        if (!quantum || !bpm) {
            return { when: 0, beat: currentBeat, bpm };
        }

        // Quantize to the next quantum boundary; the global beat there is
        // exactly nextBeat.
        let nextBeat = Math.ceil(currentBeat / quantum) * quantum;
        if (nextBeat - currentBeat < 1e-6) nextBeat += quantum;
        const secondsUntil = (nextBeat - currentBeat) * 60 / bpm;
        return { when: this.ctx.currentTime + secondsUntil, beat: nextBeat, bpm };
    }

    playChannel(index, loop) {
        const ch = this.channels[index];
        if (!ch || !ch.buffer) return;
        // Clip-launch semantics (like Ableton Session View): start from the
        // clip's head on the quantized boundary. Phase coherence comes from the
        // clips being downbeat-anchored + exact-bar-length (Stage A) and
        // launched on bar boundaries — NOT from entering the buffer mid-way.
        const { when } = this.getNextQuantizedAudioTime();
        ch.play(loop, when);
    }

    // Switch a channel to its (freshly loaded) buffer and launch it from the
    // top, keeping the current clip playing until the launch point — gapless
    // clip switching. `immediate` forces an ASAP start (seconds mode); else the
    // launch aligns to the next launch-quantization boundary (which itself is
    // ASAP when the quantum is None or no tempo is running).
    relaunchChannel(index, loop, immediate = false) {
        const ch = this.channels[index];
        if (!ch || !ch.buffer) return;
        const when = immediate ? 0 : this.getNextQuantizedAudioTime().when;
        ch.playAt(loop, when);
    }

    setMasterGain(value) {
        this.masterBus.gain.setTargetAtTime(value, this.ctx.currentTime, 0.01);
    }

    getMasterPeak() {
        this.masterAnalyser.getByteTimeDomainData(this.masterAnalyserData);
        let peak = 0;
        for (let i = 0; i < this.masterAnalyserData.length; i++) {
            const v = Math.abs(this.masterAnalyserData[i] - 128) / 128;
            if (v > peak) peak = v;
        }
        return peak;
    }

    isRecording() {
        return Boolean(this._recNode);
    }

    // Lazily register the inline recorder worklet module (once per context).
    async _ensureRecorderWorklet() {
        if (this._recWorkletLoaded) return;
        if (!this.ctx.audioWorklet) {
            throw new Error('AudioWorklet is not supported in this environment.');
        }
        const blob = new Blob([REC_WORKLET_SRC], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        try {
            await this.ctx.audioWorklet.addModule(url);
        } finally {
            URL.revokeObjectURL(url);
        }
        this._recWorkletLoaded = true;
    }

    /**
     * Begin capturing the post-limiter master output to memory. Taps
     * masterAnalyser (the final master sum, after FX + limiter) into an
     * AudioWorkletNode routed through a muted sink, so the capture path
     * never re-monitors audio. Async because the worklet module loads on
     * first use. Returns false if a recording is already in progress.
     */
    async startRecording() {
        if (this._recNode) return false;
        const ctx = this.ctx;
        if (ctx.state === 'suspended') await ctx.resume().catch(() => { /* ok */ });
        await this._ensureRecorderWorklet();

        const node = new AudioWorkletNode(ctx, 'fragmenta-recorder', {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            outputChannelCount: [2],
        });
        const sink = ctx.createGain();
        sink.gain.value = 0;
        this._recSampleRate = ctx.sampleRate;

        this.masterAnalyser.connect(node);
        node.connect(sink);
        sink.connect(ctx.destination);

        this._recNode = node;
        this._recSink = sink;
        return true;
    }

    /**
     * Stop capturing and encode the buffered PCM to a stereo WAV Blob.
     * Resolves to { blob, durationSec } or null if nothing was captured /
     * no recording was active.
     */
    stopRecording() {
        if (!this._recNode) return Promise.resolve(null);
        const node = this._recNode;
        const sink = this._recSink;
        const sampleRate = this._recSampleRate;
        this._recNode = null;
        this._recSink = null;

        return new Promise((resolve) => {
            let settled = false;
            const finish = (chunksL, chunksR) => {
                if (settled) return;
                settled = true;
                try { this.masterAnalyser.disconnect(node); } catch (_) { /* ok */ }
                try { node.disconnect(); } catch (_) { /* ok */ }
                try { sink.disconnect(); } catch (_) { /* ok */ }
                const left = flattenFloat32(chunksL || []);
                const right = flattenFloat32(chunksR || []);
                if (left.length === 0) { resolve(null); return; }
                resolve({
                    blob: encodeWavStereo(left, right, sampleRate),
                    durationSec: left.length / sampleRate,
                });
            };
            node.port.onmessage = (e) => {
                if (e.data && e.data.type === 'data') finish(e.data.left, e.data.right);
            };
            // Safety net: if the worklet never replies, resolve empty rather
            // than hang the stop action.
            setTimeout(() => finish([], []), 1500);
            node.port.postMessage('stop');
        });
    }

    refreshMuteSolo() {
        const anySoloed = this.channels.some(ch => ch.isSoloed);
        this.channels.forEach(ch => ch.applyMuteSolo(anySoloed));
    }

    setMute(index, value) {
        this.channels[index].isMuted = value;
        this.refreshMuteSolo();
    }

    setSolo(index, value) {
        this.channels[index].isSoloed = value;
        this.refreshMuteSolo();
    }

    playAll(loop = true) {
        // All channels launch from their head on the same quantized boundary,
        // so equal-length downbeat-anchored clips line up (INV#9) without
        // entering mid-buffer.
        const { when } = this.getNextQuantizedAudioTime();
        this.channels.forEach(ch => {
            if (ch.buffer) ch.play(loop, when);
        });
    }

    stopAll() {
        this.channels.forEach(ch => ch.stop());
    }

    dispose() {
        this.channels.forEach(ch => ch.dispose());
        const tryDisconnect = (node) => {
            try { node?.disconnect(); } catch (_) { /* already disconnected */ }
        };
        // Drop any in-flight capture without trying to encode it.
        tryDisconnect(this._recNode);
        tryDisconnect(this._recSink);
        this._recNode = null;
        this._recSink = null;
        tryDisconnect(this.masterBus);
        tryDisconnect(this.masterDelay);
        tryDisconnect(this.masterDelayFeedback);
        tryDisconnect(this.masterReverb);
        tryDisconnect(this.masterLimiter);
        tryDisconnect(this.masterAnalyser);
    }
}
