import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Button, ToggleButton, Slider, Switch, FormControlLabel, useTheme } from '@mui/material';
import { Repeat as LoopIcon, SkipForward as NextIcon } from 'lucide-react';
import Tooltip from '../Tooltip';
import api from '../../api';
import { TIPS } from '../../tooltips';
import RotaryKnob from '../RotaryKnob';
import { MidiMappable } from '../MidiContext';
import { PerformanceEngine } from '../../utils/performanceAudio';
import { faderPosToDb, faderDbToPos, FADER_MAX_DB } from '../../utils/faderLaw';
import { performanceChannelStyles as chStyles } from '../../theme';

/**
 * Live — hear a bend change while it is being changed. A mode, off by
 * default; the board and the rack play their takes exactly as before
 * until it is switched on.
 *
 * The material is a chain of chunks the model continues one from the
 * other (app/core/bending/live.py). With Loop on, the current chunk
 * repeats and every change re-renders it — same material, bent
 * differently — and the player crossfades to the new render at the same
 * position. Next moves on to the following chunk, which then loops. With
 * Loop off, every chunk is followed by the next: the sound evolves.
 *
 * Playback goes through one Performance channel strip (a one-channel
 * PerformanceEngine: the same filter, gain, delay and reverb sends, master
 * limiter and output routing as the Performance page), so the live sound
 * can be played like a channel there. One buffer source at a time on a
 * shared timeline: a re-render of the chunk being played swaps in with an
 * equal-power crossfade; a new chunk is scheduled at the loop boundary
 * (or straight after, when flowing) so the chain stays continuous.
 *
 * params: {model_id, prompt, steps, seed, loras, bend_patch} — sent on
 *   start and, debounced, whenever it changes.
 */
const XFADE = 0.08;          // s, swap crossfade
const EDGE = 0.003;          // s, fade at a chunk's edges so a loop wrap never clicks
const LEAD_MIN = 1.0;        // s of audio kept ahead at least when flowing
const STORE_KEY = 'fragmenta.bend.live.v1';

// Same knobs, ranges and tapers as a Performance channel (PerformanceChannel.js).
const GAIN_DB_MIN = -60;
const gainDbToLinear = (db) => (db <= GAIN_DB_MIN ? 0 : Math.pow(10, db / 20));
const bipolarToFilterFreqs = (pos) => ({
    lpf: pos <= 0 ? 20 * Math.pow(1000, 1 + pos) : 20000,
    hpf: pos >= 0 ? 20 * Math.pow(1000, pos) : 20,
});
const fmtHz = (hz) => (hz < 1000 ? `${Math.round(hz)}` : `${(hz / 1000).toFixed(hz < 10000 ? 1 : 0)}k`);
const fmtFilter = (pos) => {
    if (Math.abs(pos) < 0.005) return 'Off';
    const { lpf, hpf } = bipolarToFilterFreqs(pos);
    return pos < 0 ? `LP ${fmtHz(lpf)}` : `HP ${fmtHz(hpf)}`;
};
const fmtPercent = (v) => `${Math.round(v * 100)}%`;
const KNOB_DEFS = [
    { key: 'gain', label: 'Gain', min: GAIN_DB_MIN, max: FADER_MAX_DB, default: 0, scale: 'fader' },
    { key: 'filter', label: 'Filter', min: -1, max: 1, default: 0, scale: 'bipolar' },
    { key: 'delay', label: 'Delay', min: 0, max: 1, default: 0 },
    { key: 'reverb', label: 'Reverb', min: 0, max: 1, default: 0 },
];
const DEFAULT_KNOBS = Object.fromEntries(KNOB_DEFS.map(k => [k.key, k.default]));

const readStore = () => {
    try { return JSON.parse(window.localStorage.getItem(STORE_KEY) || '{}') || {}; } catch { return {}; }
};
const writeStore = (patch) => {
    try { window.localStorage.setItem(STORE_KEY, JSON.stringify({ ...readStore(), ...patch })); } catch { /* fine */ }
};

const fadeEdges = (buffer) => {
    const n = Math.floor(buffer.sampleRate * EDGE);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
        const d = buffer.getChannelData(c);
        for (let i = 0; i < n && i < d.length; i++) {
            const g = i / n;
            d[i] *= g;
            d[d.length - 1 - i] *= g;
        }
    }
    return buffer;
};

export default function BendLive({ on, onToggle, params, disabled = false, onNotice, accent }) {
    const theme = useTheme();
    const stored = useRef(readStore()).current;
    const [loop, setLoop] = useState(stored.loop !== false);
    const [chunkSec, setChunkSec] = useState(stored.chunkSec || 4);
    const [knobs, setKnobs] = useState({ ...DEFAULT_KNOBS, ...(stored.knobs || {}) });
    const [status, setStatus] = useState({ renderMs: null, count: 0, chunk: null, ahead: 0, gaps: 0, error: null, waiting: false });
    const engineRef = useRef(null);      // PerformanceEngine(1); channels[0] is ours
    const curRef = useRef(null);         // { src, gain, startAt, duration, chunk }
    const gapsRef = useRef(0);
    const runRef = useRef(0);            // bumps on every start/stop; stale loops exit
    const seqRef = useRef(0);
    const meterRef = useRef(null);
    const meterRafRef = useRef(null);
    const loopRef = useRef(loop);
    loopRef.current = loop;
    const paramsKey = JSON.stringify(params || {});
    const paramsRef = useRef(params);
    paramsRef.current = params;
    const onRef = useRef(on);
    onRef.current = on;
    const knobsRef = useRef(knobs);
    knobsRef.current = knobs;

    const applyKnob = (strip, key, value) => {
        if (key === 'gain') strip.setUserGain(gainDbToLinear(value));
        else if (key === 'filter') {
            const { lpf, hpf } = bipolarToFilterFreqs(value);
            strip.setFilter(lpf);
            strip.setHighpass(hpf);
        } else if (key === 'delay') strip.setDelayMix(value);
        else if (key === 'reverb') strip.setReverbMix(value);
    };

    // The engine (and with it the AudioContext) exists only once Live has
    // been switched on; the strip gets the remembered knobs right away.
    const engine = () => {
        if (!engineRef.current) {
            engineRef.current = new PerformanceEngine(1);
            const strip = engineRef.current.channels[0];
            Object.entries(knobsRef.current).forEach(([k, v]) => applyKnob(strip, k, v));
        }
        const c = engineRef.current.ctx;
        if (c.state === 'suspended') c.resume().catch(() => {});
        return engineRef.current;
    };
    const ctx = () => engine().ctx;
    const strip = () => engine().channels[0];

    const stopAudio = () => {
        const cur = curRef.current;
        if (cur) {
            try { cur.gain.gain.cancelScheduledValues(0); cur.gain.gain.value = 0; cur.src.stop(); } catch { /* done */ }
            curRef.current = null;
        }
        gapsRef.current = 0;
    };

    // When the chunk being played ends its current pass: the end of the
    // first pass for one not started yet, otherwise the next multiple of its
    // length. The same whether it loops or not, so switching Loop off lets
    // the pass finish instead of cutting it.
    const boundaryOf = (cur, now) => {
        if (!cur) return now + 0.05;
        if (now < cur.startAt) return cur.startAt + cur.duration;
        return cur.startAt + Math.ceil((now - cur.startAt + 1e-4) / cur.duration) * cur.duration;
    };

    const makeSource = (buffer, at, gainAt) => {
        const c = ctx();
        const gain = c.createGain();
        gain.gain.setValueAtTime(gainAt, at);
        gain.connect(strip().hpf);     // into the channel strip's filter → gain → sends → master
        const src = c.createBufferSource();
        src.buffer = buffer;
        src.loop = loopRef.current;
        src.connect(gain);
        return { src, gain };
    };

    // A re-render of the chunk being played: swap at the same position.
    const swap = (buffer, chunk) => {
        const c = ctx();
        const now = c.currentTime;
        const prev = curRef.current;
        let offset = 0;
        if (prev && now >= prev.startAt) offset = (now - prev.startAt) % prev.duration;
        const { src, gain } = makeSource(buffer, now, prev ? 0 : 1);
        if (prev) gain.gain.linearRampToValueAtTime(1, now + XFADE);
        src.start(now, offset);
        if (prev) {
            prev.gain.gain.setValueAtTime(1, now);
            prev.gain.gain.linearRampToValueAtTime(0, now + XFADE);
            try { prev.src.stop(now + XFADE + 0.01); } catch { /* done */ }
        }
        curRef.current = { src, gain, startAt: now - offset, duration: buffer.duration, chunk };
    };

    // The next chunk of the chain: after the current one's pass (or end).
    const append = (buffer, chunk) => {
        const c = ctx();
        const now = c.currentTime;
        const prev = curRef.current;
        let at = boundaryOf(prev, now);
        if (at < now + 0.02) {
            if (prev) gapsRef.current += 1;   // ran dry: a gap, then carry on
            at = now + 0.05;
        }
        const { src, gain } = makeSource(buffer, at, 1);
        src.start(at);
        if (prev) {
            try { prev.src.stop(at); } catch { /* done */ }
        }
        curRef.current = { src, gain, startAt: at, duration: buffer.duration, chunk };
    };

    const aheadNow = () => {
        const e = engineRef.current;
        const cur = curRef.current;
        if (!e || !cur) return 0;
        return Math.max(0, boundaryOf(cur, e.ctx.currentTime) - e.ctx.currentTime);
    };

    // The pull loop: one long-poll after another while Live is on.
    const pull = async (run) => {
        let lastRenderMs = null;
        while (runRef.current === run && onRef.current) {
            // Flowing: ask only when the chunk playing is about to end, so
            // what is heard stays as close to the knobs as the render allows.
            // Looping: ask straight away; the answer comes with a change or Next.
            if (!loopRef.current && curRef.current) {
                const lead = Math.max(LEAD_MIN, (lastRenderMs || 800) / 1000 * 1.3 + 0.3);
                const ahead = aheadNow();
                if (ahead > lead) {
                    setStatus(st => ({ ...st, ahead, gaps: gapsRef.current }));
                    await new Promise(r => setTimeout(r, Math.min(250, (ahead - lead) * 1000)));
                    continue;
                }
            }
            setStatus(st => ({ ...st, waiting: true }));
            let resp;
            try {
                resp = await fetch(`/api/bend/live/next?after=${seqRef.current}&timeout=25`, { cache: 'no-store' });
            } catch {
                await new Promise(r => setTimeout(r, 500));
                continue;
            }
            if (runRef.current !== run) return;
            if (resp.status === 204) continue;
            if (!resp.ok) {
                let msg = `Live render failed (HTTP ${resp.status}).`;
                try { msg = (await resp.json())?.error || msg; } catch { /* keep */ }
                setStatus(st => ({ ...st, error: msg, waiting: false }));
                onNotice?.('error', msg);
                onToggle?.(false);
                return;
            }
            const seq = parseInt(resp.headers.get('x-live-seq') || '0', 10);
            const renderMs = parseInt(resp.headers.get('x-live-render-ms') || '0', 10);
            const chunk = resp.headers.get('x-live-chunk');
            try {
                const w = JSON.parse(resp.headers.get('x-live-warnings') || '[]');
                if (Array.isArray(w) && w.length) onNotice?.('warning', w);
            } catch { /* none */ }
            let buffer;
            try {
                buffer = fadeEdges(await ctx().decodeAudioData(await resp.arrayBuffer()));
            } catch {
                continue;
            }
            if (runRef.current !== run) return;
            seqRef.current = Math.max(seqRef.current, seq);
            lastRenderMs = renderMs;
            if (curRef.current && curRef.current.chunk === chunk) swap(buffer, chunk);
            else append(buffer, chunk);
            setStatus(st => ({
                ...st, renderMs, count: st.count + 1, chunk, waiting: false,
                ahead: aheadNow(), gaps: gapsRef.current,
            }));
        }
    };

    // Switch on / off.
    useEffect(() => {
        if (!on) {
            runRef.current += 1;
            stopAudio();
            api.post('/api/bend/live/stop', {}).catch(() => {});
            setStatus(st => ({ ...st, waiting: false, ahead: 0 }));
            return undefined;
        }
        const run = ++runRef.current;
        seqRef.current = 0;
        stopAudio();
        setStatus({ renderMs: null, count: 0, chunk: null, ahead: 0, gaps: 0, error: null, waiting: true });
        ctx();
        const body = { ...paramsRef.current, loop: loopRef.current, chunk_sec: chunkSec };
        api.post('/api/bend/live/start', body)
            .then(() => { if (runRef.current === run) pull(run); })
            .catch((err) => {
                const msg = err?.response?.data?.error || 'Live could not start.';
                onNotice?.('error', typeof msg === 'string' ? msg : 'Live could not start.');
                onToggle?.(false);
            });
        return () => { if (runRef.current === run) runRef.current += 1; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [on]);

    // Level meter, as on a Performance channel, while Live is on.
    useEffect(() => {
        if (!on) {
            if (meterRef.current) meterRef.current.style.clipPath = 'inset(0 100% 0 0)';
            return undefined;
        }
        const data = new Uint8Array(128);
        const tick = () => {
            const el = meterRef.current;
            const e = engineRef.current;
            if (el && e) {
                const an = e.channels[0].analyser;
                an.getByteTimeDomainData(data);
                let peak = 0;
                for (let i = 0; i < data.length; i++) {
                    const v = Math.abs(data[i] - 128) / 128;
                    if (v > peak) peak = v;
                }
                el.style.clipPath = `inset(0 ${Math.max(0, 100 - peak * 100).toFixed(1)}% 0 0)`;
            }
            meterRafRef.current = requestAnimationFrame(tick);
        };
        meterRafRef.current = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(meterRafRef.current);
    }, [on]);

    // Params / patch changed while live: debounced, latest wins.
    const firstKeyRef = useRef(paramsKey);
    useEffect(() => {
        if (!on) { firstKeyRef.current = paramsKey; return undefined; }
        if (firstKeyRef.current === paramsKey) return undefined;
        const t = setTimeout(() => {
            firstKeyRef.current = paramsKey;
            api.post('/api/bend/live/set', paramsRef.current).catch(() => {});
        }, 150);
        return () => clearTimeout(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [paramsKey, on]);

    const advance = () => { if (on) api.post('/api/bend/live/advance', {}).catch(() => {}); };

    // Loop on/off applies to the chunk playing now, and tells the backend.
    const toggleLoop = () => {
        const next = !loop;
        setLoop(next);
        loopRef.current = next;
        writeStore({ loop: next });
        const cur = curRef.current;
        if (cur) {
            try { cur.src.loop = next; } catch { /* done */ }
        }
        if (on) {
            api.post('/api/bend/live/set', { loop: next }).catch(() => {});
            // Loop on after the chunk already ended (a gap): ask for the next
            // one so there is something to loop.
            const c = engineRef.current?.ctx;
            if (next && cur && c && c.currentTime > cur.startAt + cur.duration) advance();
        }
    };

    const chunkCommitted = (v) => {
        writeStore({ chunkSec: v });
        if (on) api.post('/api/bend/live/set', { chunk_sec: v }).catch(() => {});
    };

    const handleKnob = (key, value) => {
        setKnobs(prev => {
            const next = { ...prev, [key]: value };
            writeStore({ knobs: next });
            return next;
        });
        if (engineRef.current) applyKnob(engineRef.current.channels[0], key, value);
    };

    useEffect(() => () => {
        runRef.current += 1;
        stopAudio();
        if (onRef.current) api.post('/api/bend/live/stop', {}).catch(() => {});
        try { engineRef.current?.dispose(); } catch { /* fine */ }
        engineRef.current = null;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const line = !on ? ''
        : status.error ? status.error
            : status.count === 0 ? (status.waiting ? 'Rendering…' : '')
                : loop
                    ? `Looping chunk ${(status.chunk || '').split('.')[1] ?? ''} · last render ${status.renderMs} ms · ${status.count} renders`
                    : `Flowing · ${status.renderMs} ms per ${chunkSec} s · ${status.ahead.toFixed(1)} s ahead${
                        status.gaps ? ` · ${status.gaps} gap${status.gaps > 1 ? 's' : ''}` : ''}`;

    const deep = theme.palette.bend?.dark || accent;

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Tooltip title={TIPS.bend.live}>
                    <FormControlLabel
                        sx={{ ml: -0.5, mr: 0.5 }}
                        control={<Switch size="small" checked={on} disabled={disabled}
                                         onChange={(e) => onToggle?.(e.target.checked)} />}
                        label={<Typography variant="caption">Live</Typography>}
                    />
                </Tooltip>
                <Tooltip title={TIPS.bend.liveLoop}>
                    <ToggleButton size="small" value="loop" selected={loop} onChange={toggleLoop}
                                  color="bend" sx={{ px: 1 }} aria-label="Loop">
                        <LoopIcon size={14} />
                        <Typography variant="caption" sx={{ ml: 0.5 }}>Loop</Typography>
                    </ToggleButton>
                </Tooltip>
                <Tooltip title={TIPS.bend.liveNext}>
                    <span>
                        <Button size="small" variant="outlined" disabled={!on}
                                startIcon={<NextIcon size={14} />} onClick={advance}>
                            Next
                        </Button>
                    </span>
                </Tooltip>
            </Box>
            <Tooltip title={TIPS.bend.liveLength}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="caption" color="textSecondary">length</Typography>
                    <Slider size="small" value={chunkSec} min={1} max={12} step={0.5}
                            onChange={(_, v) => setChunkSec(v)}
                            onChangeCommitted={(_, v) => chunkCommitted(v)}
                            sx={{ width: 110 }} />
                    <Typography variant="caption">{chunkSec}s</Typography>
                </Box>
            </Tooltip>

            {/* The channel: the Performance strip's knobs and meter, so the
                live sound is played, not just heard. */}
            <Box sx={chStyles.knobsGrid}>
                {KNOB_DEFS.map((k) => {
                    const isFader = k.scale === 'fader';
                    const isBipolar = k.scale === 'bipolar';
                    const knobProps = isFader
                        ? {
                            value: faderDbToPos(knobs[k.key]),
                            min: 0, max: 1, step: 0.005, detent: 0,
                            center: faderDbToPos(k.default), origin: 0,
                            format: (pos) => {
                                const db = faderPosToDb(pos);
                                return db <= GAIN_DB_MIN + 0.05 ? '−∞' : `${db > 0 ? '+' : ''}${db.toFixed(1)} dB`;
                            },
                            onChange: (pos) => handleKnob(k.key, faderPosToDb(pos)),
                        }
                        : {
                            value: knobs[k.key],
                            min: k.min, max: k.max, step: 0.01,
                            center: k.default, detent: isBipolar ? 0.06 : 0,
                            format: isBipolar ? fmtFilter : fmtPercent,
                            onChange: (v) => handleKnob(k.key, v),
                        };
                    return (
                        <MidiMappable
                            key={k.key}
                            id={`bend.live.${k.key}`}
                            label={`Live · ${k.label}`}
                            kind="continuous"
                            curve={isFader ? 'fader' : 'linear'}
                            min={k.min}
                            max={k.max}
                            value={knobs[k.key]}
                            onChange={(v) => handleKnob(k.key, v)}
                            sx={{ alignItems: 'center' }}
                        >
                            <RotaryKnob
                                {...knobProps}
                                size={46}
                                color={deep}
                                label={k.label}
                                ariaLabel={`${k.label} · Live`}
                            />
                        </MidiMappable>
                    );
                })}
            </Box>
            <Box sx={chStyles.meterTrack}>
                <Box ref={meterRef} sx={chStyles.meterFill(accent, deep)} />
            </Box>

            {line && (
                <Typography variant="caption"
                            sx={{ color: status.error ? 'error.main' : (status.waiting ? 'text.disabled' : 'text.secondary') }}>
                    {line}
                </Typography>
            )}
        </Box>
    );
}
