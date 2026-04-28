export const DEFAULT_CHANNEL_GAIN = Math.pow(10, -6 / 20); // ≈ 0.5012

let sharedCtx = null;

export function getAudioContext() {
    if (!sharedCtx) {
        sharedCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (sharedCtx.state === 'suspended') {
        sharedCtx.resume();
    }
    return sharedCtx;
}

export const IMPULSE_RESPONSES = [
    { id: 'hall',   name: 'Opera Hall',    file: 'Scala Milan Opera Hall.wav' },
    { id: 'room',   name: 'Drum Room',     file: 'Nice Drum Room.wav' },
    { id: 'narrow', name: 'Narrow Space',  file: 'Narrow Bumpy Space.wav' },
];
const DEFAULT_IR_ID = 'hall';

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
    constructor(masterBus) {
        const ctx = getAudioContext();
        this.ctx = ctx;
        this.buffer = null;
        this.source = null;
        this.isPlaying = false;
        this.isLooping = false;
        this.isMuted = false;
        this.isSoloed = false;

        this.filter = ctx.createBiquadFilter();
        this.filter.type = 'lowpass';
        this.filter.frequency.value = 18000;
        this.filter.Q.value = 0.7;

        this.dryGain = ctx.createGain();
        this.dryGain.gain.value = 1.0;
        this.delayNode = ctx.createDelay(2.0);
        this.delayNode.delayTime.value = 0.25;
        this.delayFeedback = ctx.createGain();
        this.delayFeedback.gain.value = 0.42;
        this.delayWet = ctx.createGain();
        this.delayWet.gain.value = 0.0;

        this.reverbNode = ctx.createConvolver();
        this.reverbNode.buffer = getImpulse(ctx);
        this.reverbWet = ctx.createGain();
        this.reverbWet.gain.value = 0.0;

        this.channelGain = ctx.createGain();
        this.channelGain.gain.value = DEFAULT_CHANNEL_GAIN;
        this._lastUserGain = DEFAULT_CHANNEL_GAIN;

        this.compressor = ctx.createDynamicsCompressor();
        this.compressor.threshold.value = -16;
        this.compressor.knee.value = 8;
        this.compressor.ratio.value = 2.5;
        this.compressor.attack.value = 0.006;
        this.compressor.release.value = 0.14;

        this.pan = ctx.createStereoPanner();
        this.pan.pan.value = 0;

        this.analyser = ctx.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyserData = new Uint8Array(this.analyser.frequencyBinCount);

        this.filter.connect(this.dryGain);
        this.dryGain.connect(this.channelGain);

        this.filter.connect(this.delayNode);
        this.delayNode.connect(this.delayFeedback);
        this.delayFeedback.connect(this.delayNode);
        this.delayNode.connect(this.delayWet);
        this.delayWet.connect(this.channelGain);

        this.filter.connect(this.reverbNode);
        this.reverbNode.connect(this.reverbWet);
        this.reverbWet.connect(this.channelGain);

        this.channelGain.connect(this.compressor);
        this.compressor.connect(this.pan);
        this.pan.connect(this.analyser);
        this.analyser.connect(masterBus);
    }

    async loadBlob(blob) {
        this.stop();
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
        src.connect(this.filter);
        src.onended = () => {
            if (this.source === src) {
                this.source = null;
                this.isPlaying = false;
            }
        };

        src.start(Math.max(0, startTime));
        this.source = src;
        this.isPlaying = true;
    }

    stop() {
        if (this.source) {
            try { this.source.stop(0); } catch (_) { /* already stopped */ }
            this.source.disconnect();
            this.source = null;
        }
        this.isPlaying = false;
    }

    setGain(value) { this.channelGain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.01); }
    setFilter(hz) { this.filter.frequency.setTargetAtTime(hz, this.ctx.currentTime, 0.01); }
    setDelayMix(value) { this.delayWet.gain.setTargetAtTime(value, this.ctx.currentTime, 0.02); }
    setReverbMix(value) { this.reverbWet.gain.setTargetAtTime(value, this.ctx.currentTime, 0.05); }
    setPan(value) { this.pan.pan.setTargetAtTime(value, this.ctx.currentTime, 0.01); }
    setImpulseResponse(buffer) {
        if (buffer) this.reverbNode.buffer = buffer;
    }
    setDelayTimeForBpm(bpm) {
        const safeBpm = Math.max(1, bpm);
        const eighthSec = Math.min(30 / safeBpm, 2.0);
        this.delayNode.delayTime.setTargetAtTime(
            eighthSec, this.ctx.currentTime, 0.04
        );
    }
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
        const step = Math.max(1, Math.floor(data.length / w));
        ctx2d.strokeStyle = color || '#35C2D4';
        ctx2d.lineWidth = 1;
        ctx2d.beginPath();
        for (let i = 0; i < w; i++) {
            let min = 1.0, max = -1.0;
            const start = i * step;
            const end = Math.min(data.length, start + step);
            for (let j = start; j < end; j++) {
                const v = data[j];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            const yMin = (1 + min) * 0.5 * h;
            const yMax = (1 + max) * 0.5 * h;
            ctx2d.moveTo(i + 0.5, yMin);
            ctx2d.lineTo(i + 0.5, yMax);
        }
        ctx2d.stroke();
    }

    dispose() {
        this.stop();
        try {
            this.filter.disconnect();
            this.dryGain.disconnect();
            this.delayNode.disconnect();
            this.delayFeedback.disconnect();
            this.delayWet.disconnect();
            this.reverbNode.disconnect();
            this.reverbWet.disconnect();
            this.channelGain.disconnect();
            this.compressor.disconnect();
            this.pan.disconnect();
            this.analyser.disconnect();
        } catch (_) { /* already disconnected */ }
    }
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
        this.masterBus.connect(this.masterLimiter);
        this.masterLimiter.connect(this.masterAnalyser);
        this.masterAnalyser.connect(ctx.destination);
        this.channels = Array.from({ length: channelCount }, () => new ChannelStrip(this.masterBus));


        this.linkSnapshot = null;
        this.launchQuantum = 0;


        this.currentImpulseId = DEFAULT_IR_ID;
        loadImpulseResponses(ctx).then(() => {
            const buf = getImpulseResponseBuffer(this.currentImpulseId);
            if (buf) this.channels.forEach(ch => ch.setImpulseResponse(buf));
        });
    }

    setImpulseResponse(id) {
        const buf = getImpulseResponseBuffer(id);
        if (!buf) return false;
        this.currentImpulseId = id;
        this.channels.forEach(ch => ch.setImpulseResponse(buf));
        return true;
    }

    setChannelImpulseResponse(channelIndex, id) {
        const buf = getImpulseResponseBuffer(id);
        if (!buf || !this.channels[channelIndex]) return false;
        this.channels[channelIndex].setImpulseResponse(buf);
        return true;
    }

    setBpm(bpm) {
        this.channels.forEach(ch => ch.setDelayTimeForBpm(bpm));
    }

    setLinkSnapshot(snapshot) {
        this.linkSnapshot = snapshot;
    }

    setLaunchQuantum(beats) {
        const v = Number(beats);
        this.launchQuantum = Number.isFinite(v) && v > 0 ? v : 0;
    }

    getNextQuantizedAudioTime() {
        const quantum = this.launchQuantum;
        const snap = this.linkSnapshot;
        if (!quantum || !snap || !snap.bpm) return 0;
        const elapsedSec = (performance.now() - snap.capturedAt) / 1000;
        const currentBeat = snap.beat + elapsedSec * (snap.bpm / 60);
        let nextBeat = Math.ceil(currentBeat / quantum) * quantum;
        if (nextBeat - currentBeat < 1e-6) nextBeat += quantum;
        const secondsUntil = (nextBeat - currentBeat) * 60 / snap.bpm;
        return this.ctx.currentTime + secondsUntil;
    }

    playChannel(index, loop) {
        const ch = this.channels[index];
        if (!ch || !ch.buffer) return;
        ch.play(loop, this.getNextQuantizedAudioTime());
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
        const startTime = this.getNextQuantizedAudioTime();
        this.channels.forEach(ch => { if (ch.buffer) ch.play(loop, startTime); });
    }

    stopAll() {
        this.channels.forEach(ch => ch.stop());
    }

    dispose() {
        this.channels.forEach(ch => ch.dispose());
        try { this.masterBus.disconnect(); } catch (_) { /* already disconnected */ }
        try { this.masterLimiter.disconnect(); } catch (_) { /* already disconnected */ }
        try { this.masterAnalyser.disconnect(); } catch (_) { /* already disconnected */ }
    }
}
