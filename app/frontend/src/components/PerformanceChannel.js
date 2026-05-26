import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import {
    Box,
    Typography,
    TextField,
    IconButton,
    Slider,
    Tooltip,
    Select,
    MenuItem,
    ButtonBase,
} from '@mui/material';
import {
    Play as PlayIcon,
    Square as StopIcon,
    ArrowRight as GenerateArrowIcon,
    Volume2 as VolumeIcon,
    VolumeX as MuteIcon,
} from 'lucide-react';
import { performanceChannelStyles as styles, performancePanelStyles as panelStyles, perfTokens } from '../theme';
import { MidiMappable } from './MidiContext';
import { playBlob as playCueBlob, stopCue, isCueSupported } from '../utils/cueAudio';
import {
    channelScope,
    putTakeBlob,
    getTakeBlob,
    deleteTakeBlob,
    clearScope as clearTakeScope,
} from '../utils/takeStorage';
import api from '../api';
import ChannelTakeHistory from './ChannelTakeHistory';

const CHANNEL_COLORS = [
    '#35C2D4', '#9F8AE6', '#53C18A', '#E3A34B',
    '#E36C61', '#F08AD2', '#5BA0F0', '#A8D86B',
];

// Channel gain runs on the same dBFS scale as the master fader so the two
// scales line up: -60 dB floor, 0 dB ceiling, default at -6 dB. The knob's
// dB value is converted to linear before reaching the audio graph.
const GAIN_DB_MIN = -60;
const GAIN_DB_MAX = 0;
const GAIN_DB_DEFAULT = -6;
const gainDbToLinear = (db) => (db <= GAIN_DB_MIN ? 0 : Math.pow(10, db / 20));

const KNOB_DEFS = [
    { key: 'gain', label: 'GAIN', min: GAIN_DB_MIN, max: GAIN_DB_MAX, step: 0.5, default: GAIN_DB_DEFAULT },
    // Bipolar "DJ-filter" knob. -1..+1 with 0 = bypass. Negative side drives
    // the LPF cutoff down from 20 kHz → 20 Hz (kills highs). Positive side
    // drives the HPF cutoff up from 20 Hz → 20 kHz (kills lows). The two
    // biquads sit in series in the engine; only one side ever cuts at a time.
    { key: 'filter', label: 'FLT', min: -1, max: 1, step: 0.001, default: 0, scale: 'bipolar' },
    { key: 'delay', label: 'DLY', min: 0, max: 1.0, step: 0.01, default: 0.0 },
    { key: 'reverb', label: 'REV', min: 0, max: 1.0, step: 0.01, default: 0.0 },
];

// Map a bipolar filter position (-1..+1) to the (LPF, HPF) frequencies that
// the engine's two biquads need. 20 Hz / 20 kHz are the bypass anchors on
// each side; log-scaled so each octave gets equal slider travel.
function bipolarToFilterFreqs(pos) {
    const lpf = pos <= 0 ? 20 * Math.pow(1000, 1 + pos) : 20000;
    const hpf = pos >= 0 ? 20 * Math.pow(1000, pos) : 20;
    return { lpf, hpf };
}

const PAN_CENTER_SNAP = 0.06;

const BARS_OPTIONS = [1, 2, 4, 8, 16];
const BEATS_PER_BAR = 4;
const BATCH_OPTIONS = [1, 2, 3, 4];
// Per-channel rolling take history cap. Starred takes survive eviction.
const TAKE_CAP = 200;

export default function PerformanceChannel({
    index,
    strip,
    engine,
    playing = false,
    onGenerate,
    canGenerate,
    onMuteSoloChange,
    onStateChange,
    onFormStateChange,
    initialFormState,
    maxDuration = 380,
    bpm = 120,
}) {
    const color = CHANNEL_COLORS[index % CHANNEL_COLORS.length];
    const canvasRef = useRef(null);
    const meterRef = useRef(null);
    const meterRafRef = useRef(null);
    // IDB scope key for this channel's take blobs. Stable across the
    // component's lifetime since the channel index doesn't change.
    const scope = channelScope(index);

    const init = initialFormState || {};
    const initKnobs = init.knobs || {};
    const defaultKnobs = (() => {
        const d = Object.fromEntries(KNOB_DEFS.map(k => [k.key, k.default]));
        d.pan = 0;
        return d;
    })();

    const [prompt, setPrompt] = useState(init.prompt ?? '');
    const [duration, setDuration] = useState(init.duration ?? 8);
    const [durationMode, setDurationMode] = useState(init.durationMode ?? 'seconds');
    const [bars, setBars] = useState(init.bars ?? 4);
    const [generating, setGenerating] = useState(false);
    const [loaded, setLoaded] = useState(false);
    const [looping, setLooping] = useState(init.looping ?? true);
    const [muted, setMuted] = useState(init.muted ?? false);
    const [soloed, setSoloed] = useState(init.soloed ?? false);
    const [batchSize, setBatchSize] = useState(init.batchSize ?? 1);
    // Live progress for the Generate pill while a generation is in flight.
    // 0–100; polled from /api/generation-progress. Resets on each new run.
    const [progress, setProgress] = useState(0);
    const [knobs, setKnobs] = useState(() => {
        const merged = { ...defaultKnobs, ...initKnobs };
        // Migration: pre-bipolar `filter` was a raw Hz value (20..20000).
        // Anything outside the new -1..+1 range is a legacy save — reset
        // to bypass (0) rather than feeding nonsense into the engine.
        if (merged.filter < -1 || merged.filter > 1) merged.filter = 0;
        return merged;
    });

    // Per-channel rolling take history. Each take:
    //   { id, blob, audioUrl, prompt, duration, createdAt, starred, number }
    // Newest first. Capped at TAKE_CAP via FIFO eviction with star priority
    // (starred takes survive until everything is starred, then oldest go
    // first regardless). `nextTakeNumberRef` provides a stable T# even
    // after deletes — so T1 stays T1.
    const [takes, setTakes] = useState([]);
    const [auditioningTakeId, setAuditioningTakeId] = useState(null);
    const [committedTakeId, setCommittedTakeId] = useState(null);
    const nextTakeNumberRef = useRef(1);
    const cueSupported = useMemo(() => isCueSupported(), []);

    // Stop any active cue audition when the channel unmounts.
    useEffect(() => () => stopCue(), []);

    // Poll /api/generation-progress while a generation is in flight so the
    // Generate pill renders a real fill bar instead of a vague spinner. The
    // backend exposes a single in-flight state; performance generations are
    // sequential (the backend serves one at a time), so this naturally
    // reflects whichever channel is currently busy.
    useEffect(() => {
        if (!generating) {
            setProgress(0);
            return;
        }
        let cancelled = false;
        const tick = async () => {
            if (cancelled) return;
            try {
                const r = await api.get('/api/generation-progress');
                const pct = Number(r.data?.progress) || 0;
                if (!cancelled) {
                    // Cap at 95 until handleGenerate resolves so the bar
                    // doesn't sit at 100 while waiting for the WAV blob.
                    setProgress((prev) => Math.max(prev, Math.min(95, pct)));
                }
            } catch { /* non-fatal — bar just freezes briefly */ }
        };
        tick();
        const id = window.setInterval(tick, 250);
        return () => { cancelled = true; window.clearInterval(id); };
    }, [generating]);

    // Mirror form state up to the panel so it can persist the session. Skip the
    // first render so we don't re-write what we just loaded from localStorage.
    // Takes mirror as metadata only — the Blob bodies live in IndexedDB and
    // get rehydrated on mount by the effect below.
    const initialReportSkippedRef = useRef(false);
    useEffect(() => {
        if (!initialReportSkippedRef.current) {
            initialReportSkippedRef.current = true;
            return;
        }
        const takesMeta = takes.map(({ blob, audioUrl, ...rest }) => rest);
        onFormStateChange?.(index, {
            prompt, duration, durationMode, bars, looping, muted, soloed, batchSize, knobs,
            takes: takesMeta,
            committedTakeId,
        });
    }, [prompt, duration, durationMode, bars, looping, muted, soloed, batchSize, knobs,
        takes, committedTakeId, index, onFormStateChange]);

    // Hydrate takes on mount from the session metadata + IDB blobs. Runs once,
    // tolerates missing blobs (skips the entry), and rewinds the take numbering
    // counter so newly generated takes don't collide with the restored ones.
    const hydrationRef = useRef(false);
    useEffect(() => {
        if (hydrationRef.current) return;
        hydrationRef.current = true;
        const meta = initialFormState?.takes || [];
        const persistedCommittedId = initialFormState?.committedTakeId || null;
        if (meta.length === 0) {
            if (persistedCommittedId) setCommittedTakeId(null);
            return;
        }

        let cancelled = false;
        (async () => {
            const hydrated = [];
            for (const m of meta) {
                try {
                    const blob = await getTakeBlob(scope, m.id);
                    if (cancelled) {
                        hydrated.forEach(t => URL.revokeObjectURL(t.audioUrl));
                        return;
                    }
                    if (!blob) continue;
                    hydrated.push({
                        ...m,
                        blob,
                        audioUrl: URL.createObjectURL(blob),
                    });
                } catch {
                    /* one bad fetch — keep going */
                }
            }
            if (cancelled) {
                hydrated.forEach(t => URL.revokeObjectURL(t.audioUrl));
                return;
            }
            const maxNumber = hydrated.reduce((a, t) => Math.max(a, t.number || 0), 0);
            nextTakeNumberRef.current = maxNumber + 1;
            setTakes(hydrated);
            if (persistedCommittedId && hydrated.some(t => t.id === persistedCommittedId)) {
                setCommittedTakeId(persistedCommittedId);
                setLoaded(true);
                onStateChange?.(index, { loaded: true });
            }
        })();

        return () => { cancelled = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // When the audio strip becomes available AND we have a hydrated committed
    // take, load that take's blob into the strip so the channel comes back
    // ready to play after reload. Declared here as a ref so the effect that
    // actually does the work (after drawWave is defined below) can guard
    // against multiple loads.
    const autoLoadDoneRef = useRef(false);

    const secondsFromBars = useMemo(
        () => bars * (60 / Math.max(bpm, 1)) * BEATS_PER_BAR,
        [bars, bpm]
    );

    const availableBars = useMemo(() => {
        const maxBars = (maxDuration * bpm) / (60 * BEATS_PER_BAR);
        const opts = BARS_OPTIONS.filter(b => b <= maxBars);
        return opts.length > 0 ? opts : [BARS_OPTIONS[0]];
    }, [maxDuration, bpm]);

    useEffect(() => {
        const tick = () => {
            const el = meterRef.current;
            if (el && strip) {
                const level = strip.getLevel();
                el.style.width = `${Math.min(100, level * 140)}%`;
            }
            meterRafRef.current = requestAnimationFrame(tick);
        };
        meterRafRef.current = requestAnimationFrame(tick);
        return () => {
            if (meterRafRef.current) cancelAnimationFrame(meterRafRef.current);
        };
    }, [strip]);

    const drawWave = useCallback(() => {
        if (strip && canvasRef.current) {
            strip.drawWaveform(canvasRef.current, color);
        }
    }, [strip, color]);

    useEffect(() => { drawWave(); }, [drawWave, loaded]);

    // Auto-load the persisted committed take into the strip once Tone.js is
    // ready. Runs at most once per mount; the ref guards against re-trigger
    // when the user later commits a different take (handled by handleCommitTake).
    useEffect(() => {
        if (autoLoadDoneRef.current) return;
        if (!strip || !committedTakeId) return;
        const take = takes.find(t => t.id === committedTakeId);
        if (!take) return;
        autoLoadDoneRef.current = true;
        strip.loadBlob(take.blob).then(() => {
            requestAnimationFrame(drawWave);
        }).catch(err => {
            console.warn(`Channel ${index + 1} auto-load failed:`, err);
            autoLoadDoneRef.current = false;
        });
    }, [strip, committedTakeId, takes, drawWave, index]);

    // One-shot: push restored knob/loop values into the audio strip when it
    // first becomes available, so the persisted session matches what's heard.
    // Mute/solo applies through the parent's mix handler so the panel can
    // recompute the "any-soloed" cross-channel state.
    const stripStateAppliedRef = useRef(false);
    useEffect(() => {
        if (!strip || stripStateAppliedRef.current) return;
        stripStateAppliedRef.current = true;
        strip.setUserGain(gainDbToLinear(knobs.gain));
        strip.setPan(knobs.pan);
        {
            const { lpf, hpf } = bipolarToFilterFreqs(knobs.filter);
            strip.setFilter(lpf);
            strip.setHighpass(hpf);
        }
        strip.setDelayMix(knobs.delay);
        strip.setReverbMix(knobs.reverb);
        strip.setLoop(looping);
        if (muted || soloed) {
            onMuteSoloChange?.(index, { mute: muted, solo: soloed });
        }
        // Initial values are intentionally only applied once; subsequent edits
        // flow through the normal handlers below.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [strip]);

    useEffect(() => {
        setDuration(prev => Math.min(prev, maxDuration));
    }, [maxDuration]);

    useEffect(() => {
        if (!availableBars.includes(bars)) {
            setBars(availableBars[availableBars.length - 1]);
        }
    }, [availableBars, bars]);

    const handleGenerate = async () => {
        if (!prompt.trim() || generating) return;
        const inBarsMode = durationMode === 'bars';
        const effectiveDuration = inBarsMode ? secondsFromBars : duration;
        setGenerating(true);
        // Stop any in-flight cue audition so the old preview doesn't keep
        // playing while we generate the new take.
        stopCue();
        setAuditioningTakeId(null);

        const promptSnap = prompt.trim();

        try {
            await onGenerate({
                prompt,
                duration: effectiveDuration,
                batchSize,
                // Only forward alignment params in bars mode — seconds mode
                // generates raw audio with no post-processing.
                ...(inBarsMode ? { alignBars: bars, alignBpm: bpm } : {}),
                // Phase 7: bars-mode + channel-looping ⇒ ask the backend
                // to wrap-inpaint the seam so the clip loops seamlessly.
                ...(inBarsMode && looping ? { loopStitch: 'inpaint' } : {}),
                // Streamed handler — fires per take as the backend returns
                // each blob. Take #0 auto-loads into the strip so the user
                // can start auditioning while takes #1..N are still rendering.
                onBlob: async (blob, i) => {
                    const takeNumber = nextTakeNumberRef.current;
                    nextTakeNumberRef.current = takeNumber + 1;
                    const take = {
                        id: `${Date.now()}_${i}`,
                        blob,
                        audioUrl: URL.createObjectURL(blob),
                        prompt: promptSnap,
                        duration: effectiveDuration,
                        createdAt: Date.now(),
                        starred: false,
                        number: takeNumber,
                    };

                    // Persist the blob to IndexedDB so it survives reload.
                    // Fire-and-forget — the in-memory take is usable now;
                    // a failed write just means the take is lost on reload.
                    putTakeBlob(scope, take.id, blob).catch((err) => {
                        console.warn(`Channel ${index + 1} take persist failed:`, err);
                    });

                    // Append to history with TAKE_CAP eviction. Order is
                    // chronological: oldest at the top, newest at the bottom.
                    // Starred takes survive until everything is starred,
                    // then oldest go first regardless.
                    setTakes((prev) => {
                        const combined = [...prev, take];
                        if (combined.length <= TAKE_CAP) return combined;
                        const trimmed = combined.slice();
                        while (trimmed.length > TAKE_CAP) {
                            let idx = -1;
                            // Scan from the start (oldest first) for the
                            // first unstarred take to drop.
                            for (let j = 0; j < trimmed.length; j++) {
                                if (!trimmed[j].starred) { idx = j; break; }
                            }
                            if (idx < 0) idx = 0;  // all starred → drop oldest
                            const dying = trimmed[idx];
                            if (dying.audioUrl?.startsWith('blob:')) {
                                try { URL.revokeObjectURL(dying.audioUrl); } catch { /* ignore */ }
                            }
                            // Drop the evicted blob from IDB too, otherwise
                            // it stays around as orphaned storage.
                            deleteTakeBlob(scope, dying.id).catch(() => { /* ignore */ });
                            trimmed.splice(idx, 1);
                        }
                        return trimmed;
                    });

                    if (i === 0) {
                        await strip.loadBlob(blob);
                        setCommittedTakeId(take.id);
                        setLoaded(true);
                        onStateChange?.(index, { loaded: true });
                        requestAnimationFrame(drawWave);
                    }
                },
            });
        } catch (err) {
            console.error(`Channel ${index + 1} generate failed:`, err);
        } finally {
            setGenerating(false);
        }
    };

    // Take history actions — toggle audition through cue, commit a take to
    // the channel buffer, star/unstar, delete one, or clear the whole list.
    const handleAuditionTake = async (takeId) => {
        const take = takes.find((t) => t.id === takeId);
        if (!take) return;
        if (auditioningTakeId === takeId) {
            stopCue();
            setAuditioningTakeId(null);
            return;
        }
        setAuditioningTakeId(takeId);
        try {
            await playCueBlob(take.blob, {
                onEnded: () => setAuditioningTakeId((prev) => (prev === takeId ? null : prev)),
            });
        } catch (err) {
            console.warn(`Channel ${index + 1} audition failed:`, err);
            setAuditioningTakeId(null);
        }
    };

    const handleCommitTake = async (takeId) => {
        const take = takes.find((t) => t.id === takeId);
        if (!take || committedTakeId === takeId) return;
        // Stop the live channel before swapping the buffer so we don't get a
        // glitch in the middle of a loop iteration.
        try { strip.stop(); } catch { /* not playing */ }
        onStateChange?.(index, { playing: false });
        await strip.loadBlob(take.blob);
        setCommittedTakeId(takeId);
        // Mark loaded so the play button enables. Required for the case where
        // the channel had hydrated takes but no committedTakeId (preset load,
        // or any flow where the first commit is via this code path rather
        // than the generate flow that sets loaded itself).
        if (!loaded) {
            setLoaded(true);
            onStateChange?.(index, { loaded: true });
        }
        requestAnimationFrame(drawWave);
    };

    // Drag-and-drop: a take row from this channel's history can be dropped
    // onto the waveform monitor to load it (same effect as the row's commit
    // ✓ button). The MIME type is channel-scoped, so a row from channel 1
    // won't even highlight channel 2's waveform — the browser filters at
    // dragOver level via dataTransfer.types matching.
    const dragMime = `application/x-fragmenta-take-ch${index}`;
    const [dropActive, setDropActive] = useState(false);
    // Counter pattern — dragenter/leave also fire when the cursor crosses
    // into child elements (canvas, overlay). Without the counter, dropActive
    // would flicker false whenever the cursor moved over a child.
    const dragCounterRef = useRef(0);

    const handleWaveDragEnter = (e) => {
        if (!e.dataTransfer.types.includes(dragMime)) return;
        e.preventDefault();
        dragCounterRef.current += 1;
        if (dragCounterRef.current === 1) setDropActive(true);
    };
    const handleWaveDragOver = (e) => {
        if (!e.dataTransfer.types.includes(dragMime)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    };
    const handleWaveDragLeave = () => {
        dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
        if (dragCounterRef.current === 0) setDropActive(false);
    };
    const handleWaveDrop = (e) => {
        e.preventDefault();
        dragCounterRef.current = 0;
        setDropActive(false);
        const takeId = e.dataTransfer.getData(dragMime);
        if (takeId) handleCommitTake(takeId);
    };

    const handleToggleStar = (takeId) => {
        setTakes((prev) => prev.map((t) =>
            t.id === takeId ? { ...t, starred: !t.starred } : t,
        ));
    };

    const handleDeleteTake = (takeId) => {
        const target = takes.find((t) => t.id === takeId);
        if (target?.audioUrl?.startsWith('blob:')) {
            try { URL.revokeObjectURL(target.audioUrl); } catch { /* ignore */ }
        }
        deleteTakeBlob(scope, takeId).catch(() => { /* ignore */ });
        setTakes((prev) => prev.filter((t) => t.id !== takeId));
        if (committedTakeId === takeId) setCommittedTakeId(null);
        if (auditioningTakeId === takeId) {
            stopCue();
            setAuditioningTakeId(null);
        }
    };

    const handleClearTakes = () => {
        // Stop any in-flight audition and revoke every blob URL before
        // dropping references — otherwise the URLs leak until reload.
        stopCue();
        setAuditioningTakeId(null);
        takes.forEach((t) => {
            if (t.audioUrl?.startsWith('blob:')) {
                try { URL.revokeObjectURL(t.audioUrl); } catch { /* ignore */ }
            }
        });
        clearTakeScope(scope).catch(() => { /* ignore */ });
        setTakes([]);
        setCommittedTakeId(null);
    };

    const handlePlay = () => {
        if (!loaded) return;
        if (engine) engine.playChannel(index, looping);
        else strip.play(looping);
        onStateChange?.(index, { playing: true });
    };

    const handleStop = () => {
        strip.stop();
        onStateChange?.(index, { playing: false });
    };

    const handleLoopToggle = () => {
        setLooping(prev => {
            const next = !prev;
            strip.setLoop(next);
            return next;
        });
    };

    const handleMuteToggle = () => {
        const next = !muted;
        setMuted(next);
        onMuteSoloChange(index, { mute: next });
    };

    const handleSoloToggle = () => {
        const next = !soloed;
        setSoloed(next);
        onMuteSoloChange(index, { solo: next });
    };

    const handleKnob = (key, value) => {
        setKnobs(prev => ({ ...prev, [key]: value }));
        if (key === 'gain') strip.setUserGain(gainDbToLinear(value));
        else if (key === 'pan') strip.setPan(value);
        else if (key === 'filter') {
            const { lpf, hpf } = bipolarToFilterFreqs(value);
            strip.setFilter(lpf);
            strip.setHighpass(hpf);
        }
        else if (key === 'delay') strip.setDelayMix(value);
        else if (key === 'reverb') strip.setReverbMix(value);
    };

    const handlePan = (v) => {
        const snapped = Math.abs(v) < PAN_CENTER_SNAP ? 0 : v;
        handleKnob('pan', snapped);
    };

    const handleTransportToggle = () => {
        if (!loaded) return;
        if (playing) handleStop();
        else handlePlay();
    };

    const ctrlId = (suffix) => `channel.${index}.${suffix}`;
    const ctrlLabel = (name) => `Ch ${index + 1} · ${name}`;

    return (
        <Box sx={styles.strip(color, playing)}>
            <Box sx={styles.stripHeader(color)}>
                <Box sx={styles.channelBadge(color)}>{String(index + 1).padStart(2, '0')}</Box>
                <Box sx={styles.muteSoloRow}>
                    <MidiMappable id={ctrlId('mute')} label={ctrlLabel('Mute')} kind="trigger" onChange={handleMuteToggle}>
                        <Tooltip title="Mute">
                            <IconButton size="small" onClick={handleMuteToggle} sx={styles.muteBtn(muted)}>M</IconButton>
                        </Tooltip>
                    </MidiMappable>
                    <MidiMappable id={ctrlId('solo')} label={ctrlLabel('Solo')} kind="trigger" onChange={handleSoloToggle}>
                        <Tooltip title="Solo">
                            <IconButton size="small" onClick={handleSoloToggle} sx={styles.soloBtn(soloed)}>S</IconButton>
                        </Tooltip>
                    </MidiMappable>
                </Box>
            </Box>

            <Box sx={styles.promptBox}>
                <TextField
                    placeholder="Prompt…"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    multiline
                    minRows={2}
                    maxRows={2}
                    size="small"
                    fullWidth
                    sx={styles.promptField}
                    disabled={generating}
                />
                <Box sx={{ ...styles.durationRow, minHeight: 26, height: 26 }}>
                    {durationMode === 'seconds' ? (
                        <>
                            <Typography sx={styles.durationLabel}>{duration.toFixed(0)}s</Typography>
                            <Slider
                                value={duration}
                                onChange={(_, v) => setDuration(v)}
                                min={2}
                                max={maxDuration}
                                step={1}
                                size="small"
                                sx={styles.durationSlider(color)}
                            />
                        </>
                    ) : (
                        <Select
                            value={availableBars.includes(bars) ? bars : availableBars[availableBars.length - 1]}
                            onChange={(e) => setBars(Number(e.target.value))}
                            size="small"
                            sx={{ ...panelStyles.pillControl, flex: 1 }}
                        >
                            {availableBars.map(b => (
                                <MenuItem key={b} value={b} sx={{ fontSize: perfTokens.fontSize.sm }}>
                                    {b} {b === 1 ? 'bar' : 'bars'}
                                </MenuItem>
                            ))}
                        </Select>
                    )}

                    {/* Sec/Bars mode toggle — moved to the right of the row so
                        it mirrors the Generate row layout (content fills left,
                        modifier sits right). Width matches the ×N selector so
                        the right column reads as a uniform stack. */}
                    <Box
                        sx={{
                            display: 'inline-flex',
                            border: '1px solid',
                            borderColor: 'divider',
                            borderRadius: 0.75,
                            overflow: 'hidden',
                            height: '100%',
                            flexShrink: 0,
                        }}
                    >
                        {[
                            { mode: 'sec',  label: 'Sec'  },
                            { mode: 'bars', label: 'Bars' },
                        ].map(({ mode, label }) => {
                            const value = mode === 'sec' ? 'seconds' : 'bars';
                            const active = durationMode === value;
                            return (
                                <ButtonBase
                                    key={mode}
                                    onClick={() => setDurationMode(value)}
                                    sx={{
                                        fontSize: perfTokens.fontSize.sm,
                                        px: 0.7,
                                        minWidth: 36,
                                        bgcolor: active ? color : 'transparent',
                                        color: active ? 'rgba(0,0,0,0.88)' : 'text.disabled',
                                        fontWeight: active ? perfTokens.weight.bold : perfTokens.weight.regular,
                                        transition: 'background-color 120ms, color 120ms',
                                        '&:hover': {
                                            bgcolor: active ? color : 'action.hover',
                                            color: active ? 'rgba(0,0,0,0.88)' : 'text.secondary',
                                        },
                                    }}
                                >
                                    {label}
                                </ButtonBase>
                            );
                        })}
                    </Box>
                </Box>
                <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    mt: 0.5,
                    width: '100%',
                }}>
                    {/* Generate pill — wide CTA on the left so the eye lands
                        on the primary action first. Fills left-to-right with
                        live progress while generating; resets when complete. */}
                    <MidiMappable id={ctrlId('generate')} label={ctrlLabel('Generate')} kind="trigger" onChange={handleGenerate}>
                        <Tooltip
                            title={
                                generating
                                    ? ''
                                    : !canGenerate
                                        ? 'Pick a model in the Generation tab first'
                                        : !prompt.trim()
                                            ? 'Enter a prompt to generate'
                                            : ''
                            }
                            placement="top"
                        >
                            <span style={{ display: 'inline-flex', flex: 1, minWidth: 0 }}>
                                <ButtonBase
                                    onClick={handleGenerate}
                                    disabled={!canGenerate || !prompt.trim() || generating}
                                    sx={styles.generatePill(color, {
                                        generating,
                                        disabled: !canGenerate || !prompt.trim(),
                                    })}
                                >
                                    {generating && (
                                        <Box sx={styles.generatePillFill(color, progress)} />
                                    )}
                                    <Box component="span" sx={styles.generatePillLabel}>
                                        {generating
                                            ? `Generating · ${Math.round(progress)}%`
                                            : 'Generate'}
                                        {!generating && <GenerateArrowIcon size={14} strokeWidth={2.25} />}
                                    </Box>
                                </ButtonBase>
                            </span>
                        </Tooltip>
                    </MidiMappable>

                    {/* Takes selector — sits right of Generate so the row
                        reads "Generate × 4" (action then modifier). Width
                        matches the Sec/Bars toggle above so the right column
                        reads as a uniform stack of modifiers. */}
                    <Tooltip
                        title="Number of candidates to generate. Each take lands in the channel's take history."
                        placement="top"
                        enterDelay={500}
                    >
                        <Select
                            value={batchSize}
                            onChange={(e) => setBatchSize(Number(e.target.value))}
                            disabled={generating}
                            size="small"
                            sx={{ ...styles.channelPillControl, width: 74, flexShrink: 0 }}
                            renderValue={(v) => `×${v}`}
                        >
                            {BATCH_OPTIONS.map((n) => (
                                <MenuItem
                                    key={n}
                                    value={n}
                                    sx={{ fontSize: perfTokens.fontSize.sm, fontVariantNumeric: 'tabular-nums' }}
                                >
                                    ×{n}
                                </MenuItem>
                            ))}
                        </Select>
                    </Tooltip>
                </Box>
            </Box>

            <Box
                onDragEnter={handleWaveDragEnter}
                onDragOver={handleWaveDragOver}
                onDragLeave={handleWaveDragLeave}
                onDrop={handleWaveDrop}
                sx={[
                    styles.waveformWrap,
                    dropActive && {
                        borderColor: color,
                        boxShadow: `inset 0 0 0 2px ${color}`,
                        backgroundColor: `${color}1F`,
                        transition: 'border-color 120ms, box-shadow 120ms, background-color 120ms',
                    },
                ]}
            >
                <canvas
                    ref={canvasRef}
                    width={140}
                    height={42}
                    style={{ width: '100%', height: 42, display: 'block', pointerEvents: 'none' }}
                />
                {!loaded && (
                    <Typography sx={styles.waveformPlaceholder}>
                        {dropActive ? 'Drop to load' : 'Waveform'}
                    </Typography>
                )}
            </Box>

            {/* Per-channel rolling take history. Always rendered (empty
                state included). Star/keep, delete, audition, load — all
                inline per row. Capped at TAKE_CAP via FIFO with star
                priority. */}
            <ChannelTakeHistory
                takes={takes}
                color={color}
                channelIndex={index}
                auditioningId={auditioningTakeId}
                committedId={committedTakeId}
                maxTakes={TAKE_CAP}
                onAudition={handleAuditionTake}
                onCommit={handleCommitTake}
                onToggleStar={handleToggleStar}
                onDelete={handleDeleteTake}
                onClearAll={handleClearTakes}
            />

            <Box sx={{ px: 1, py: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Box component="span" sx={{ ...perfTokens.caps, color: 'text.secondary', minWidth: 28 }}>PAN</Box>
                    <MidiMappable
                        id={ctrlId('pan')}
                        label={ctrlLabel('Pan')}
                        kind="continuous"
                        min={-1}
                        max={1}
                        value={knobs.pan ?? 0}
                        onChange={handlePan}
                        sx={{ flex: 1, flexDirection: 'row' }}
                    >
                        <Slider
                            value={knobs.pan ?? 0}
                            onChange={(_, v) => handlePan(v)}
                            min={-1}
                            max={1}
                            step={0.01}
                            size="small"
                            track={false}
                            marks={[{ value: 0 }]}
                            sx={{
                                flex: 1,
                                '& .MuiSlider-mark': {
                                    width: 2,
                                    height: 10,
                                    borderRadius: 1,
                                    backgroundColor: 'text.secondary',
                                    opacity: 0.8,
                                },
                                '& .MuiSlider-markActive': {
                                    backgroundColor: 'text.secondary',
                                    opacity: 0.8,
                                },
                            }}
                        />
                    </MidiMappable>
                </Box>
            </Box>

            <Box sx={styles.knobsGrid}>
                {KNOB_DEFS.map((k) => {
                    const isLog = k.scale === 'log';
                    const isBipolar = k.scale === 'bipolar';
                    // For log knobs, the slider drives a 0..1 position and we
                    // convert to/from the underlying value (Hz) on the audio
                    // boundary. The knob value stored in state stays in the
                    // domain unit (Hz here) so persistence and MIDI keep working.
                    const valueToPos = isLog
                        ? (v) => Math.log(Math.max(v, k.min) / k.min) / Math.log(k.max / k.min)
                        : (v) => v;
                    const posToValue = isLog
                        ? (p) => k.min * Math.pow(k.max / k.min, p)
                        : (v) => v;
                    return (
                        <Box key={k.key} sx={styles.knobCell}>
                            <MidiMappable
                                id={ctrlId(k.key)}
                                label={ctrlLabel(k.label)}
                                kind="continuous"
                                curve={isLog ? 'log' : 'linear'}
                                min={k.min}
                                max={k.max}
                                value={knobs[k.key]}
                                onChange={(v) => handleKnob(k.key, v)}
                                sx={{ alignItems: 'center' }}
                            >
                                <Slider
                                    orientation="vertical"
                                    value={valueToPos(knobs[k.key])}
                                    onChange={(_, v) => handleKnob(k.key, posToValue(v))}
                                    min={isLog ? 0 : k.min}
                                    max={isLog ? 1 : k.max}
                                    step={isLog ? 0.001 : k.step}
                                    size="small"
                                    track={isBipolar ? false : undefined}
                                    marks={isBipolar ? [{ value: 0 }] : undefined}
                                    sx={{
                                        ...styles.knobSlider(color, k.key === 'gain'),
                                        ...(isBipolar && {
                                            '& .MuiSlider-mark': {
                                                width: 10,
                                                height: 2,
                                                borderRadius: 1,
                                                backgroundColor: 'text.secondary',
                                                opacity: 0.7,
                                            },
                                            '& .MuiSlider-markActive': {
                                                backgroundColor: 'text.secondary',
                                                opacity: 0.7,
                                            },
                                        }),
                                    }}
                                />
                            </MidiMappable>
                            <Box component="span" sx={styles.knobLabel}>{k.label}</Box>
                        </Box>
                    );
                })}
            </Box>

            <Box sx={styles.transportRow}>
                <MidiMappable id={ctrlId('transport')} label={ctrlLabel('Play/Stop')} kind="trigger" onChange={handleTransportToggle}>
                    <IconButton
                        onClick={playing ? handleStop : handlePlay}
                        disabled={!loaded}
                        sx={styles.transportBtn(color, playing)}
                        size="small"
                    >
                        {playing ? <StopIcon size={16} /> : <PlayIcon size={16} />}
                    </IconButton>
                </MidiMappable>
                <MidiMappable id={ctrlId('loop')} label={ctrlLabel('Loop')} kind="trigger" onChange={handleLoopToggle}>
                    <Tooltip
                        title={
                            looping
                                ? (durationMode === 'bars'
                                    ? 'Seamless loop — next generation will be inpaint-stitched at the bar boundary'
                                    : 'Playback loop on')
                                : 'Loop off'
                        }
                        placement="top"
                        enterDelay={400}
                    >
                        <IconButton
                            onClick={handleLoopToggle}
                            sx={styles.loopBtn(color, looping)}
                            size="small"
                            aria-label={looping ? 'Loop on' : 'Loop off'}
                        >
                            L
                        </IconButton>
                    </Tooltip>
                </MidiMappable>
                <Box sx={styles.meterTrack}>
                    <Box ref={meterRef} sx={styles.meterFill(color)} />
                </Box>
            </Box>

        </Box>
    );
}
