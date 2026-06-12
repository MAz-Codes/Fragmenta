import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import {
    Box,
    Typography,
    TextField,
    IconButton,
    Slider,
    Select,
    MenuItem,
    ButtonBase,
} from '@mui/material';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import {
    Play as PlayIcon,
    Square as StopIcon,
    ArrowRight as GenerateArrowIcon,
    Volume2 as VolumeIcon,
    VolumeX as MuteIcon,
    Shuffle as VariationIcon,
    Repeat as LoopIcon,
} from 'lucide-react';
import { performanceChannelStyles as styles, performancePanelStyles as panelStyles, perfTokens, SHEEN_DARK, RAISE_DARK } from '../theme';
import { MidiMappable } from './MidiContext';
import { playBlob as playCueBlob, stopCue, isCueSupported } from '../utils/cueAudio';
import { extractError } from '../utils/errors';
import { faderPosToDb, faderDbToPos, FADER_MAX_DB } from '../utils/faderLaw';
import {
    channelScope,
    putFragmentBlob,
    getFragmentBlob,
    deleteFragmentBlob,
    clearScope as clearFragmentScope,
} from '../utils/fragmentStorage';
import api from '../api';
import ChannelFragmentHistory from './ChannelFragmentHistory';

const CHANNEL_COLORS = [
    // Original introduction palette. Ch1 teal, ch2 violet, ch3 green, ch4 amber.
    // Light enough that the black label text on active toggles stays legible.
    // Slots 4–7 are spares (only 4 channels render).
    '#35C2D4', '#9F8AE6', '#53C18A', '#E3A34B',
    '#E36C61', '#F08AD2', '#5BA0F0', '#A8D86B',
];

// Channel gain runs on the same dBFS scale as the master fader so the two
// scales line up: -60 dB floor, 0 dB ceiling, default at -6 dB. The knob's
// dB value is converted to linear before reaching the audio graph.
const GAIN_DB_MIN = -60;
// +6 dB boost headroom — the fader taper puts unity at ~80% of throw.
const GAIN_DB_MAX = FADER_MAX_DB;
const GAIN_DB_DEFAULT = 0;
const gainDbToLinear = (db) => (db <= GAIN_DB_MIN ? 0 : Math.pow(10, db / 20));

const KNOB_DEFS = [
    // scale 'fader' → console-style position↔dB taper (faderLaw), same as
    // the master fader. Without it the gain knob was linear-in-dB and a
    // quarter-turn down already dropped to ~-15 dB.
    { key: 'gain', label: 'GAIN', min: GAIN_DB_MIN, max: GAIN_DB_MAX, step: 0.5, default: GAIN_DB_DEFAULT, scale: 'fader' },
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
// Per-channel rolling fragment history cap. Starred fragments survive eviction.
const FRAGMENT_CAP = 200;

export default function PerformanceChannel({
    index,
    strip,
    engine,
    playing = false,
    onGenerate,
    canGenerate,
    onMuteSoloChange,
    // Index of the channel that owns sidechain ducking, or null. Exclusive:
    // when set, every other channel's sidechain button is grayed out.
    sidechainOwner = null,
    onStateChange,
    onFormStateChange,
    initialFormState,
    maxDuration = 380,
    bpm = 120,
    // False while the Performance tab is hidden (panel stays mounted via
    // keepMounted) — pauses this channel's meter RAF loop.
    panelActive = true,
}) {
    const color = CHANNEL_COLORS[index % CHANNEL_COLORS.length];
    const canvasRef = useRef(null);
    const meterRef = useRef(null);
    const meterRafRef = useRef(null);
    // IDB scope key for this channel's fragment blobs. Stable across the
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
    const [durationMode, setDurationMode] = useState(init.durationMode ?? 'bars');
    const [bars, setBars] = useState(init.bars ?? 4);
    const [generating, setGenerating] = useState(false);
    const [loaded, setLoaded] = useState(false);
    // Surfaced in the strip (item-level toast); generation/variation
    // failures used to go to the console only — a live performer staring at
    // the channel had no idea why nothing arrived.
    const [channelError, setChannelError] = useState(null);
    // Live mirror of `loaded` for async callbacks: makeOnBlob is created at
    // generation START, so reading `loaded` directly inside it sees a stale
    // false even after the user manually commits a fragment mid-generation —
    // and the arriving blob would silently override their choice.
    const loadedRef = useRef(false);
    useEffect(() => { loadedRef.current = loaded; }, [loaded]);
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

    // DJ-style in/out points over the loaded clip, normalized 0..1.
    // Mirrored into the strip (which drives the actual loop bounds / start
    // offset) and persisted with the channel form state so a trimmed clip
    // survives reload. Reset to full whenever DIFFERENT audio is loaded.
    const [trim, setTrimState] = useState(() => {
        const ts = Number(init.trimStart);
        const te = Number(init.trimEnd);
        return (Number.isFinite(ts) && Number.isFinite(te) && te > ts)
            ? { start: Math.max(0, ts), end: Math.min(1, te) }
            : { start: 0, end: 1 };
    });
    const trimRef = useRef(trim);
    useEffect(() => { trimRef.current = trim; }, [trim]);
    const applyTrim = useCallback((next) => {
        setTrimState(next);
        strip?.setRegion(next.start, next.end);
    }, [strip]);
    const resetTrim = useCallback(() => {
        // loadBufferFromBlob already reset the strip's region; this re-syncs
        // the UI (and is harmless when called redundantly).
        applyTrim({ start: 0, end: 1 });
    }, [applyTrim]);
    // Min handle gap: at least 50 ms of audio (or 2% when nothing is loaded).
    const minTrimGap = useCallback(() => {
        const dur = strip?.buffer?.duration || 0;
        return dur > 0 ? Math.max(0.005, Math.min(0.5, 0.05 / dur)) : 0.02;
    }, [strip]);
    const waveWrapRef = useRef(null);
    const startTrimDrag = (which) => (e) => {
        if (!loaded) return;
        e.preventDefault();
        e.stopPropagation();
        const wrap = waveWrapRef.current;
        if (!wrap) return;
        const rect = wrap.getBoundingClientRect();
        const move = (ev) => {
            const x = (ev.clientX - rect.left) / Math.max(1, rect.width);
            const gap = minTrimGap();
            const cur = trimRef.current;
            const next = which === 'start'
                ? { start: Math.max(0, Math.min(x, cur.end - gap)), end: cur.end }
                : { start: cur.start, end: Math.min(1, Math.max(x, cur.start + gap)) };
            applyTrim(next);
        };
        const up = () => {
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
        };
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', up);
    };

    // Per-channel rolling fragment history. Each fragment:
    //   { id, blob, audioUrl, prompt, duration, createdAt, starred, number }
    // Oldest-first. Capped at FRAGMENT_CAP via FIFO eviction with star
    // priority (starred fragments survive until everything is starred, then
    // oldest go first regardless). `nextFragmentNumberRef` provides a stable
    // F# even after deletes — so F1 stays F1.
    const [fragments, setFragments] = useState([]);
    const [auditioningFragmentId, setAuditioningFragmentId] = useState(null);
    const [committedFragmentId, setCommittedFragmentId] = useState(null);
    const nextFragmentNumberRef = useRef(1);
    const cueSupported = useMemo(() => isCueSupported(), []);

    // Stop any active cue audition when the channel unmounts.
    useEffect(() => () => stopCue(), []);

    // Revoke this channel's fragment object URLs on unmount. Preset loads and
    // Fresh Start force a full panel remount, and the remounted channel mints
    // NEW URLs while re-hydrating from IndexedDB — without this cleanup every
    // remount leaked the previous generation of URLs (and their buffers).
    const fragmentsRef = useRef(fragments);
    useEffect(() => { fragmentsRef.current = fragments; }, [fragments]);
    useEffect(() => () => {
        fragmentsRef.current.forEach((f) => {
            if (f.audioUrl?.startsWith('blob:')) {
                try { URL.revokeObjectURL(f.audioUrl); } catch { /* ignore */ }
            }
        });
    }, []);

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
    // Fragments mirror as metadata only — the Blob bodies live in IndexedDB
    // and get rehydrated on mount by the effect below.
    const initialReportSkippedRef = useRef(false);
    useEffect(() => {
        if (!initialReportSkippedRef.current) {
            initialReportSkippedRef.current = true;
            return;
        }
        const fragmentsMeta = fragments.map(({ blob, audioUrl, ...rest }) => rest);
        onFormStateChange?.(index, {
            prompt, duration, durationMode, bars, looping, muted, soloed, batchSize, knobs,
            fragments: fragmentsMeta,
            committedFragmentId,
            trimStart: trim.start,
            trimEnd: trim.end,
        });
    }, [prompt, duration, durationMode, bars, looping, muted, soloed, batchSize, knobs,
        fragments, committedFragmentId, trim, index, onFormStateChange]);

    // Hydrate fragments on mount from the session metadata + IDB blobs. Runs
    // once, tolerates missing blobs (skips the entry), and rewinds the
    // fragment numbering counter so newly generated fragments don't collide
    // with the restored ones.
    const hydrationRef = useRef(false);
    useEffect(() => {
        if (hydrationRef.current) return;
        hydrationRef.current = true;
        // Backward compat: pre-rename saves used `takes`/`committedTakeId`.
        // The session loader migrates them into `fragments`/`committedFragmentId`,
        // but we also fall back here defensively in case `initialFormState`
        // came from somewhere unmigrated.
        const meta = initialFormState?.fragments
            ?? initialFormState?.takes
            ?? [];
        const persistedCommittedId = initialFormState?.committedFragmentId
            ?? initialFormState?.committedTakeId
            ?? null;
        if (meta.length === 0) {
            if (persistedCommittedId) setCommittedFragmentId(null);
            return;
        }

        let cancelled = false;
        (async () => {
            const hydrated = [];
            for (const m of meta) {
                try {
                    const blob = await getFragmentBlob(scope, m.id);
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
            nextFragmentNumberRef.current = maxNumber + 1;
            setFragments(hydrated);
            if (persistedCommittedId && hydrated.some(t => t.id === persistedCommittedId)) {
                setCommittedFragmentId(persistedCommittedId);
                loadedRef.current = true;
                setLoaded(true);
                onStateChange?.(index, { loaded: true });
            }
        })();

        return () => { cancelled = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // When the audio strip becomes available AND we have a hydrated committed
    // fragment, load that fragment's blob into the strip so the channel comes back
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
        if (!panelActive) return undefined;
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
    }, [strip, panelActive]);

    const drawWave = useCallback(() => {
        if (strip && canvasRef.current) {
            // Each channel's waveform is drawn in that channel's own color.
            strip.drawWaveform(canvasRef.current, color);
        }
    }, [strip, color]);

    useEffect(() => { drawWave(); }, [drawWave, loaded]);

    // Auto-load the persisted committed fragment into the strip once Tone.js
    // is ready. Runs at most once per mount; the ref guards against re-trigger
    // when the user later commits a different fragment (handled by
    // handleCommitFragment).
    useEffect(() => {
        if (autoLoadDoneRef.current) return;
        if (!strip || !committedFragmentId) return;
        const fragment = fragments.find(f => f.id === committedFragmentId);
        if (!fragment) return;
        autoLoadDoneRef.current = true;
        strip.loadBlob(fragment.blob).then(() => {
            // Restore the persisted in/out points for the restored clip —
            // the decode reset the strip's region to full.
            strip.setRegion(trimRef.current.start, trimRef.current.end);
            requestAnimationFrame(drawWave);
        }).catch(err => {
            console.warn(`Channel ${index + 1} auto-load failed:`, err);
            autoLoadDoneRef.current = false;
        });
    }, [strip, committedFragmentId, fragments, drawWave, index]);

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

    // Per-fragment handler factory — fires as each blob returns. Fragment #0
    // auto-loads into the strip so the user can audition while #1..N render.
    // Shared by Generate and Variation so both feed channel history identically.
    const makeOnBlob = (promptSnap, effectiveDuration) => async (blob, i) => {
        const fragmentNumber = nextFragmentNumberRef.current;
        nextFragmentNumberRef.current = fragmentNumber + 1;
        const fragment = {
            id: `${Date.now()}_${i}`,
            blob,
            audioUrl: URL.createObjectURL(blob),
            prompt: promptSnap,
            duration: effectiveDuration,
            createdAt: Date.now(),
            starred: false,
            number: fragmentNumber,
        };

        // Persist the blob to IndexedDB so it survives reload. Fire-and-forget.
        putFragmentBlob(scope, fragment.id, blob).catch((err) => {
            console.warn(`Channel ${index + 1} fragment persist failed:`, err);
        });

        // Append to history with FRAGMENT_CAP eviction (oldest unstarred first).
        setFragments((prev) => {
            const combined = [...prev, fragment];
            if (combined.length <= FRAGMENT_CAP) return combined;
            const trimmed = combined.slice();
            while (trimmed.length > FRAGMENT_CAP) {
                let idx = -1;
                for (let j = 0; j < trimmed.length; j++) {
                    if (!trimmed[j].starred) { idx = j; break; }
                }
                if (idx < 0) idx = 0;  // all starred → drop oldest
                const dying = trimmed[idx];
                if (dying.audioUrl?.startsWith('blob:')) {
                    try { URL.revokeObjectURL(dying.audioUrl); } catch { /* ignore */ }
                }
                deleteFragmentBlob(scope, dying.id).catch(() => { /* ignore */ });
                trimmed.splice(idx, 1);
            }
            return trimmed;
        });

        // Generating must never disturb playback: a playing channel keeps
        // looping its current clip while new fragments just pile into the
        // history list. Only auto-load when the channel has nothing loaded yet
        // (first-ever fragment) — harmless since nothing is playing — so the
        // user still gets a ready-to-play clip on a fresh channel. To start a
        // newly generated fragment, pick it from the list (handleCommitFragment).
        if (i === 0 && !loadedRef.current) {
            await strip.loadBlob(blob);
            resetTrim();
            setCommittedFragmentId(fragment.id);
            loadedRef.current = true;
            setLoaded(true);
            onStateChange?.(index, { loaded: true });
            requestAnimationFrame(drawWave);
        }
    };

    const handleGenerate = async () => {
        if (!prompt.trim() || generating) return;
        const inBarsMode = durationMode === 'bars';
        const effectiveDuration = inBarsMode ? secondsFromBars : duration;
        setGenerating(true);
        setChannelError(null);
        // Stop any in-flight cue audition so the old preview doesn't keep
        // playing while we generate the new fragment.
        stopCue();
        setAuditioningFragmentId(null);

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
                onBlob: makeOnBlob(promptSnap, effectiveDuration),
            });
        } catch (err) {
            console.error(`Channel ${index + 1} generate failed:`, err);
            setChannelError(extractError(err, 'Generation failed'));
        } finally {
            setGenerating(false);
        }
    };

    // Phase 8 "Variation": re-roll the channel using its current fragment as
    // init_audio at a high noise level — gives a related-but-different take
    // (A/B/A/C/A live sets). Uploads the source blob to get a server path,
    // then routes through the same generate flow.
    const handleVariation = async () => {
        if (generating) return;
        const src = fragments.find((f) => f.id === committedFragmentId)
            || fragments[fragments.length - 1];
        if (!src?.blob) return;
        const inBarsMode = durationMode === 'bars';
        const effectiveDuration = inBarsMode ? secondsFromBars : duration;
        const promptSnap = (prompt || '').trim() || src.prompt || 'variation';
        setGenerating(true);
        setChannelError(null);
        stopCue();
        setAuditioningFragmentId(null);
        try {
            const form = new FormData();
            form.append('file', new File([src.blob], `${scope}_variation_src.wav`, { type: 'audio/wav' }));
            const up = await api.post('/api/audio/upload', form);
            await onGenerate({
                prompt: promptSnap,
                duration: effectiveDuration,
                batchSize: 1,
                initAudioPath: up.data.path,
                initNoiseLevel: 0.9,
                onBlob: makeOnBlob(promptSnap, effectiveDuration),
            });
        } catch (err) {
            console.error(`Channel ${index + 1} variation failed:`, err);
            setChannelError(extractError(err, 'Variation failed'));
        } finally {
            setGenerating(false);
        }
    };

    // Fragment history actions — toggle audition through cue, commit a
    // fragment to the channel buffer, star/unstar, delete one, or clear
    // the whole list.
    const handleAuditionFragment = async (fragmentId) => {
        const fragment = fragments.find((f) => f.id === fragmentId);
        if (!fragment) return;
        if (auditioningFragmentId === fragmentId) {
            stopCue();
            setAuditioningFragmentId(null);
            return;
        }
        setAuditioningFragmentId(fragmentId);
        try {
            await playCueBlob(fragment.blob, {
                onEnded: () => setAuditioningFragmentId((prev) => (prev === fragmentId ? null : prev)),
            });
        } catch (err) {
            console.warn(`Channel ${index + 1} audition failed:`, err);
            setAuditioningFragmentId(null);
        }
    };

    // Choosing a fragment launches it from the beginning. The currently
    // playing clip (if any) keeps sounding until the launch point: immediately
    // in seconds mode or when launch quantization is None, otherwise at the
    // next launch-quantization bar. The buffer is decoded WITHOUT stopping the
    // live source, so the swap is gapless (the engine schedules the handoff).
    const handleCommitFragment = async (fragmentId) => {
        const fragment = fragments.find((f) => f.id === fragmentId);
        if (!fragment) return;
        const sameFragment = committedFragmentId === fragmentId;
        // Already looping this exact clip → nothing to (re)launch.
        if (sameFragment && playing) return;

        // Decode the new clip without cutting the live source; skip the decode
        // when this fragment's buffer is already loaded.
        if (!sameFragment) {
            await strip.loadBufferFromBlob(fragment.blob);
            resetTrim();
            setCommittedFragmentId(fragmentId);
        }
        // Mark loaded so the play button enables (covers preset/hydrated flows
        // where the first commit happens here rather than via generate).
        if (!loaded) {
            loadedRef.current = true;
            setLoaded(true);
            onStateChange?.(index, { loaded: true });
        }

        // Launch from the top. Seconds mode is always immediate; bars mode
        // defers to the engine's launch-quantization (ASAP when quantum=None).
        const immediate = durationMode === 'seconds';
        if (engine) engine.relaunchChannel(index, looping, immediate);
        else strip.playAt(looping, 0);
        onStateChange?.(index, { playing: true });
        requestAnimationFrame(drawWave);
    };

    // Drag-and-drop: a fragment row from this channel's history can be
    // dropped onto the waveform monitor to load it (same effect as the row's
    // commit ✓ button). The MIME type is channel-scoped, so a row from
    // channel 1 won't even highlight channel 2's waveform — the browser
    // filters at dragOver level via dataTransfer.types matching.
    const dragMime = `application/x-fragmenta-fragment-ch${index}`;
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
        const fragmentId = e.dataTransfer.getData(dragMime);
        if (fragmentId) handleCommitFragment(fragmentId);
    };

    const handleToggleStar = (fragmentId) => {
        setFragments((prev) => prev.map((f) =>
            f.id === fragmentId ? { ...f, starred: !f.starred } : f,
        ));
    };

    const handleDeleteFragment = (fragmentId) => {
        const target = fragments.find((f) => f.id === fragmentId);
        if (target?.audioUrl?.startsWith('blob:')) {
            try { URL.revokeObjectURL(target.audioUrl); } catch { /* ignore */ }
        }
        deleteFragmentBlob(scope, fragmentId).catch(() => { /* ignore */ });
        setFragments((prev) => prev.filter((f) => f.id !== fragmentId));
        if (committedFragmentId === fragmentId) setCommittedFragmentId(null);
        if (auditioningFragmentId === fragmentId) {
            stopCue();
            setAuditioningFragmentId(null);
        }
    };

    const handleClearFragments = () => {
        // Stop any in-flight audition and revoke every blob URL before
        // dropping references — otherwise the URLs leak until reload.
        stopCue();
        setAuditioningFragmentId(null);
        fragments.forEach((f) => {
            if (f.audioUrl?.startsWith('blob:')) {
                try { URL.revokeObjectURL(f.audioUrl); } catch { /* ignore */ }
            }
        });
        clearFragmentScope(scope).catch(() => { /* ignore */ });
        setFragments([]);
        setCommittedFragmentId(null);
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

    // Sidechain state is owned by the panel (it's exclusive across channels),
    // so unlike mute/solo there is no local mirror — just derive from the prop.
    const sidechained = sidechainOwner === index;
    const sidechainLocked = sidechainOwner !== null && sidechainOwner !== index;

    const handleSidechainToggle = () => {
        if (sidechainLocked) return;
        onMuteSoloChange(index, { sidechain: !sidechained });
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
                {/* Transport (Play / Loop) on the left, Mute / Solo on the
                    right — replaces the old "01" channel badge so the channel
                    number isn't using up that slot. */}
                <Box sx={styles.muteSoloRow}>
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
                            title={TIPS.channel.loop(looping, durationMode)}
                            placement="top"
                            enterDelay={400}
                        >
                            <IconButton
                                onClick={handleLoopToggle}
                                sx={styles.loopBtn(color, looping)}
                                size="small"
                                aria-label={looping ? 'Loop on' : 'Loop off'}
                            >
                                <LoopIcon size={14} strokeWidth={2.25} />
                            </IconButton>
                        </Tooltip>
                    </MidiMappable>
                </Box>
                <Box sx={styles.muteSoloRow}>
                    <MidiMappable id={ctrlId('mute')} label={ctrlLabel('Mute')} kind="trigger" onChange={handleMuteToggle}>
                        <Tooltip title={TIPS.channel.mute}>
                            <IconButton size="small" onClick={handleMuteToggle} sx={styles.muteBtn(muted)}>M</IconButton>
                        </Tooltip>
                    </MidiMappable>
                    <MidiMappable id={ctrlId('solo')} label={ctrlLabel('Solo')} kind="trigger" onChange={handleSoloToggle}>
                        <Tooltip title={TIPS.channel.solo}>
                            <IconButton size="small" onClick={handleSoloToggle} sx={styles.soloBtn(soloed)}>S</IconButton>
                        </Tooltip>
                    </MidiMappable>
                    <MidiMappable id={ctrlId('sidechain')} label={ctrlLabel('Sidechain')} kind="trigger" onChange={handleSidechainToggle}>
                        <Tooltip title={TIPS.channel.sidechain(sidechained, sidechainLocked)}>
                            {/* span — MUI tooltips need a focusable wrapper around disabled buttons */}
                            <span>
                                <IconButton
                                    size="small"
                                    onClick={handleSidechainToggle}
                                    disabled={sidechainLocked}
                                    sx={styles.sidechainBtn(sidechained)}
                                    aria-label={sidechained ? 'Sidechain on' : 'Sidechain off'}
                                >
                                    SC
                                </IconButton>
                            </span>
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
                                        backgroundImage: active ? SHEEN_DARK : 'none',
                                        boxShadow: active ? RAISE_DARK : 'none',
                                        color: active ? 'rgba(0,0,0,0.88)' : 'text.disabled',
                                        fontWeight: active ? perfTokens.weight.bold : perfTokens.weight.regular,
                                        transition: 'background-color 120ms, color 120ms, box-shadow 120ms',
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
                            title={TIPS.channel.generateDisabled(generating, canGenerate, prompt.trim())}
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

                    {/* Batch selector — sits right of Generate so the row
                        reads "Generate × 4" (action then modifier). Sized to
                        its content (×1…×8 + dropdown arrow); no need to match
                        the wider Sec/Bars toggle above. */}
                    <Tooltip
                        title={TIPS.channel.batch}
                        placement="top"
                        enterDelay={500}
                    >
                        <Select
                            value={batchSize}
                            onChange={(e) => setBatchSize(Number(e.target.value))}
                            disabled={generating}
                            size="small"
                            sx={{ ...styles.channelPillControl, width: 54, flexShrink: 0 }}
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

                    {/* Variation — re-roll from the current fragment as
                        init_audio (Phase 8). Disabled until a fragment exists. */}
                    <MidiMappable id={ctrlId('variation')} label={ctrlLabel('Variation')} kind="trigger" onChange={handleVariation}>
                        <Tooltip
                            title={TIPS.channel.variation(loaded)}
                            placement="top"
                        >
                            <span style={{ display: 'inline-flex', flexShrink: 0 }}>
                                <ButtonBase
                                    onClick={handleVariation}
                                    disabled={!loaded || generating}
                                    sx={{
                                        ...styles.channelPillControl,
                                        width: 40,
                                        justifyContent: 'center',
                                        '&.Mui-disabled': { opacity: 0.4, color: 'text.disabled' },
                                    }}
                                    aria-label="Variation"
                                >
                                    <VariationIcon size={15} strokeWidth={2.25} />
                                </ButtonBase>
                            </span>
                        </Tooltip>
                    </MidiMappable>
                </Box>

                {channelError && (
                    <Typography
                        onClick={() => setChannelError(null)}
                        title="Dismiss"
                        sx={{
                            fontSize: perfTokens.fontSize.xs,
                            color: 'error.main',
                            lineHeight: 1.3,
                            mt: 0.5,
                            cursor: 'pointer',
                            overflow: 'hidden',
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                        }}
                    >
                        {channelError}
                    </Typography>
                )}
            </Box>

            <Box
                onDragEnter={handleWaveDragEnter}
                ref={waveWrapRef}
                onDragOver={handleWaveDragOver}
                onDragLeave={handleWaveDragLeave}
                onDrop={handleWaveDrop}
                onDoubleClick={() => loaded && resetTrim()}
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
                {/* DJ-style in/out points: drag the brackets to choose where
                    the clip starts and stops (loops cycle the region —
                    bounds retarget live while playing). Double-click the
                    waveform to reset to the full clip. */}
                {loaded && (
                    <>
                        <Box sx={{
                            position: 'absolute', top: 0, bottom: 0, left: 0,
                            width: `${trim.start * 100}%`,
                            backgroundColor: 'rgba(0, 0, 0, 0.55)',
                            pointerEvents: 'none',
                        }} />
                        <Box sx={{
                            position: 'absolute', top: 0, bottom: 0, right: 0,
                            width: `${(1 - trim.end) * 100}%`,
                            backgroundColor: 'rgba(0, 0, 0, 0.55)',
                            pointerEvents: 'none',
                        }} />
                        <Tooltip title="Start point — drag to set where the clip launches. Double-click the waveform to reset." placement="top" enterDelay={600}>
                            <Box
                                onPointerDown={startTrimDrag('start')}
                                sx={{
                                    position: 'absolute', top: 0, bottom: 0,
                                    left: `calc(${trim.start * 100}% - 5px)`,
                                    width: 10,
                                    cursor: 'ew-resize',
                                    touchAction: 'none',
                                    zIndex: 2,
                                    display: 'flex', justifyContent: 'center',
                                    '&::before': {
                                        content: '""', width: 2,
                                        height: '100%', backgroundColor: color,
                                    },
                                    '&::after': {
                                        content: '""', position: 'absolute',
                                        top: 0, left: 4, width: 5, height: 7,
                                        backgroundColor: color,
                                        borderRadius: '0 0 2px 0',
                                    },
                                }}
                            />
                        </Tooltip>
                        <Tooltip title="End point — drag to set where the clip stops (loops wrap back to the start point)." placement="top" enterDelay={600}>
                            <Box
                                onPointerDown={startTrimDrag('end')}
                                sx={{
                                    position: 'absolute', top: 0, bottom: 0,
                                    left: `calc(${trim.end * 100}% - 5px)`,
                                    width: 10,
                                    cursor: 'ew-resize',
                                    touchAction: 'none',
                                    zIndex: 2,
                                    display: 'flex', justifyContent: 'center',
                                    '&::before': {
                                        content: '""', width: 2,
                                        height: '100%', backgroundColor: color,
                                    },
                                    '&::after': {
                                        content: '""', position: 'absolute',
                                        bottom: 0, right: 4, width: 5, height: 7,
                                        backgroundColor: color,
                                        borderRadius: '2px 0 0 0',
                                    },
                                }}
                            />
                        </Tooltip>
                    </>
                )}
            </Box>

            {/* Per-channel rolling fragment history. Always rendered (empty
                state included). Star/keep, delete, audition, load — all
                inline per row. Capped at FRAGMENT_CAP via FIFO with star
                priority. */}
            <ChannelFragmentHistory
                fragments={fragments}
                color={color}
                channelIndex={index}
                auditioningId={auditioningFragmentId}
                committedId={committedFragmentId}
                maxFragments={FRAGMENT_CAP}
                onAudition={handleAuditionFragment}
                onCommit={handleCommitFragment}
                onToggleStar={handleToggleStar}
                onDelete={handleDeleteFragment}
                onClearAll={handleClearFragments}
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
                                // Match the channel main color (the global
                                // MuiSlider override is amber; the vertical
                                // knobs already pass `color` via knobSlider).
                                color,
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
                    const isFader = k.scale === 'fader';
                    // log + fader knobs drive a 0..1 slider position and
                    // convert to/from the underlying value at the boundary;
                    // the value stored in state stays in the domain unit
                    // (Hz for log, dB for fader) so persistence + MIDI keep
                    // working unchanged.
                    const usesPos = isLog || isFader;
                    const valueToPos = isLog
                        ? (v) => Math.log(Math.max(v, k.min) / k.min) / Math.log(k.max / k.min)
                        : isFader
                            ? faderDbToPos
                            : (v) => v;
                    const posToValue = isLog
                        ? (p) => k.min * Math.pow(k.max / k.min, p)
                        : isFader
                            ? faderPosToDb
                            : (v) => v;
                    return (
                        <Box key={k.key} sx={styles.knobCell}>
                            <MidiMappable
                                id={ctrlId(k.key)}
                                label={ctrlLabel(k.label)}
                                kind="continuous"
                                curve={isLog ? 'log' : isFader ? 'fader' : 'linear'}
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
                                    min={usesPos ? 0 : k.min}
                                    max={usesPos ? 1 : k.max}
                                    step={usesPos ? 0.001 : k.step}
                                    size="small"
                                    track={isBipolar ? false : undefined}
                                    marks={isBipolar ? [{ value: 0 }]
                                        : isFader ? [{ value: faderDbToPos(0) }] : undefined}
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
                                        // Subtle unity (0 dB) tick on the gain knob.
                                        // height must be > 1: numeric <= 1 in
                                        // MUI sx means a percentage (1 = 100%),
                                        // which drew a full-height bar past the
                                        // top of the knob lane.
                                        ...(isFader && {
                                            '& .MuiSlider-mark': {
                                                width: 10,
                                                height: 2,
                                                borderRadius: 1,
                                                backgroundColor: 'text.disabled',
                                                opacity: 0.7,
                                            },
                                            '& .MuiSlider-markActive': {
                                                backgroundColor: 'text.disabled',
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

            {/* Bottom row is now just the channel level meter — Play and Loop
                moved to the top header so the channel reads "controls on top,
                signal flow below". */}
            <Box sx={styles.transportRow}>
                <Box sx={styles.meterTrack}>
                    <Box ref={meterRef} sx={styles.meterFill(color)} />
                </Box>
            </Box>

        </Box>
    );
}
