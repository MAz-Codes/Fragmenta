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

let sharedImpulse = null;
function getImpulse(ctx, duration = 2.6, decay = 3.0) {
    if (sharedImpulse && sharedImpulse.sampleRate === ctx.sampleRate) {
        return sharedImpulse;
    }
    const length = Math.floor(ctx.sampleRate * duration);
    const buf = ctx.createBuffer(2, length, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
        const data = buf.getChannelData(ch);
        for (let i = 0; i < length; i++) {
            data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
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
        this.delayNode.delayTime.value = 0.32;
        this.delayFeedback = ctx.createGain();
        this.delayFeedback.gain.value = 0.42;
        this.delayWet = ctx.createGain();
        this.delayWet.gain.value = 0.0;

        this.reverbNode = ctx.createConvolver();
        this.reverbNode.buffer = getImpulse(ctx);
        this.reverbWet = ctx.createGain();
        this.reverbWet.gain.value = 0.0;

        this.channelGain = ctx.createGain();
        this.channelGain.gain.value = 0;

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

        this.channelGain.connect(this.pan);
        this.pan.connect(this.analyser);
        this.analyser.connect(masterBus);
    }

    async loadBlob(blob) {
        this.stop();
        const arrayBuffer = await blob.arrayBuffer();
        this.buffer = await this.ctx.decodeAudioData(arrayBuffer);
    }

    play(loop = this.isLooping) {
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
        src.start(0);
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
        this.masterAnalyser = ctx.createAnalyser();
        this.masterAnalyser.fftSize = 1024;
        this.masterAnalyserData = new Uint8Array(this.masterAnalyser.frequencyBinCount);
        this.masterBus.connect(this.masterAnalyser);
        this.masterAnalyser.connect(ctx.destination);
        this.channels = Array.from({ length: channelCount }, () => new ChannelStrip(this.masterBus));
        this.channels.forEach(ch => { ch._lastUserGain = 0; });
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
        this.channels.forEach(ch => { if (ch.buffer) ch.play(loop); });
    }

    stopAll() {
        this.channels.forEach(ch => ch.stop());
    }

    dispose() {
        this.channels.forEach(ch => ch.dispose());
        try { this.masterBus.disconnect(); } catch (_) { /* already disconnected */ }
        try { this.masterAnalyser.disconnect(); } catch (_) { /* already disconnected */ }
    }
}
