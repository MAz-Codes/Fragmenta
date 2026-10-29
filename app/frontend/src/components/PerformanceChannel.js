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
import { useTheme } from '@mui/material/styles';
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
import { performanceChannelStyles as styles, performancePanelStyles as panelStyles, perfTokens, RADIUS } from '../theme';
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
import { PERFORMANCE_VARIATION } from '../features';
import ChannelFragmentHistory from './ChannelFragmentHistory';
import RotaryKnob from './RotaryKnob';

// Channel and cue-point colours come from the palette (theme.palette.perf):
// ch1–ch4 are its four pastels, labels on them in ink; the cue marker is the
// deepened "hot" accent, which no channel uses, so it stays visible on every
// strip. Paired with a dark drop-shadow for legibility over waveform peaks.
const CUE_SHADOW = 'drop-shadow(0 0 1.5px rgba(0, 0, 0, 0.9))';

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
    { key: 'gain', label: 'Gain', min: GAIN_DB_MIN, max: GAIN_DB_MAX, step: 0.5, default: GAIN_DB_DEFAULT, scale: 'fader' },
    // Bipolar "DJ-filter" knob. -1..+1 with 0 = bypass. Negative side drives
    // the LPF cutoff down from 20 kHz → 20 Hz (kills highs). Positive side
    // drives the HPF cutoff up from 20 Hz → 20 kHz (kills lows). The two
    // biquads sit in series in the engine; only one side ever cuts at a time.
    { key: 'filter', label: 'Filter', min: -1, max: 1, step: 0.001, default: 0, scale: 'bipolar' },
    { key: 'delay', label: 'Delay', min: 0, max: 1.0, step: 0.01, default: 0.0 },
    { key: 'reverb', label: 'Reverb', min: 0, max: 1.0, step: 0.01, default: 0.0 },
];

// Pitch (rotary knob in the Generate row), whole semitones ±1 octave. Output
// only: the engine plays a same-length shifted copy (utils/pitchShift.js), so
// loops, cue and Link timing don't move. Stored in `knobs.pitch`.
const PITCH_RANGE = 12;
// Settle time before a pitch change is rendered: a drag or MIDI sweep
// updates the knob at once but renders only where it comes to rest.
const PITCH_SETTLE_MS = 120;
const fmtSemitones = (v) => (v > 0 ? `+${v}` : `${v}`);
const fmtPan = (v) => (Math.abs(v) < 0.005 ? 'C' : `${v < 0 ? 'L' : 'R'} ${Math.round(Math.abs(v) * 100)}`);

// Map a bipolar filter position (-1..+1) to the (LPF, HPF) frequencies that
// the engine's two biquads need. 20 Hz / 20 kHz are the bypass anchors on
// each side; log-scaled so each octave gets equal slider travel.
function bipolarToFilterFreqs(pos) {
    const lpf = pos <= 0 ? 20 * Math.pow(1000, 1 + pos) : 20000;
    const hpf = pos >= 0 ? 20 * Math.pow(1000, pos) : 20;
    return { lpf, hpf };
}

const PAN_CENTER_SNAP = 0.06;

// Lane waveform height: the most expressive element on the lane, so it gets
// room to breathe.
const WAVE_HEIGHT = 96;

const fmtHz = (hz) => (hz < 1000 ? `${Math.round(hz)}` : `${(hz / 1000).toFixed(hz < 10000 ? 1 : 0)}k`);
// Knob readouts under each effect knob.
const fmtFilter = (pos) => {
    if (Math.abs(pos) < 0.005) return 'Off';
    const { lpf, hpf } = bipolarToFilterFreqs(pos);
    return pos < 0 ? `LP ${fmtHz(lpf)}` : `HP ${fmtHz(hpf)}`;
};
const fmtPercent = (v) => `${Math.round(v * 100)}%`;

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
    const perfColors = useTheme().palette.perf;
    const color = perfColors.channels[index % perfColors.channels.length];
    // Deeper shade of the lane colour for fine marks (waveform, arcs,
    // brackets) so they read on the lane's pale tint.
    const deep = perfColors.channelsDeep[index % perfColors.channelsDeep.length];
    // Soft shade for the loop toggle and the loaded fragment row.
    const soft = perfColors.channelsSoft[index % perfColors.channelsSoft.length];
    const canvasRef = useRef(null);
    const meterRef = useRef(null);
    const meterRafRef = useRef(null);
    const playheadRef = useRef(null);
    // IDB scope key for this channel's fragment blobs. Stable across the
    // component's lifetime since the channel index doesn't change.
    const scope = channelScope(index);

    const init = initialFormState || {};
    const initKnobs = init.knobs || {};
    const defaultKnobs = (() => {
        const d = Object.fromEntries(KNOB_DEFS.map(k => [k.key, k.default]));
        d.pan = 0;
        d.pitch = 0;
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

    // DJ cue point over the loaded clip, normalized 0..1, or null when pinned
    // to the in-point. The first pass and every retrigger enter here; repeats
    // cycle the trim region as before. Constrained to sit inside the region.
    const [launchStart, setLaunchStartState] = useState(() => {
        const ls = Number(init.launchStart);
        return Number.isFinite(ls) && ls > 0 ? Math.min(1, ls) : null;
    });
    const launchRef = useRef(launchStart);
    useEffect(() => { launchRef.current = launchStart; }, [launchStart]);

    const applyTrim = useCallback((next) => {
        setTrimState(next);
        strip?.setRegion(next.start, next.end);
        // setRegion may have evicted the cue (either bracket can squeeze it
        // out) — mirror whatever the strip settled on back into the UI.
        if (strip && launchRef.current !== strip.launchStart) {
            setLaunchStartState(strip.launchStart);
        }
    }, [strip]);
    const applyLaunch = useCallback((next) => {
        strip?.setLaunchStart(next);
        // Read back rather than trusting `next`: the strip clamps into the
        // region and collapses a cue sitting on the in-point back to null.
        setLaunchStartState(strip ? strip.launchStart : next);
    }, [strip]);
    const resetTrim = useCallback(() => {
        // loadBufferFromBlob already reset the strip's region and cue; this
        // re-syncs the UI (and is harmless when called redundantly).
        applyTrim({ start: 0, end: 1 });
        applyLaunch(null);
    }, [applyTrim, applyLaunch]);
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
            if (which === 'cue') {
                // The strip clamps into [regionStart, regionEnd - gap] and
                // nulls a cue that lands on the in-point, so pass x raw.
                applyLaunch(x);
                return;
            }
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
    //   { id, blob, audioUrl, prompt, duration, createdAt, starred, number,
    //     label? }
    // Oldest-first until the user reorders it by hand. Capped at FRAGMENT_CAP
    // via eviction with star priority (starred fragments survive until
    // everything is starred, then the oldest goes regardless — by createdAt,
    // never by list position). `nextFragmentNumberRef` provides a stable F#
    // even after deletes — so F1 stays F1; `label` overrides it for display.
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
            launchStart,
        });
    }, [prompt, duration, durationMode, bars, looping, muted, soloed, batchSize, knobs,
        fragments, committedFragmentId, trim, launchStart, index, onFormStateChange]);

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
                // The gradient spans the whole track; the clip reveals it.
                el.style.clipPath = `inset(0 ${100 - Math.min(100, level * 140)}% 0 0 round 999px)`;
            }
            // Playhead rides the same frame as the meter, written straight to
            // style so a 60 Hz cursor never re-renders the strip.
            const ph = playheadRef.current;
            if (ph && strip) {
                const pos = strip.getPlayPosition();
                if (pos === null) {
                    ph.style.opacity = '0';
                } else {
                    ph.style.opacity = '1';
                    ph.style.left = `${pos * 100}%`;
                }
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
            // Each channel's waveform is drawn in that channel's own colour.
            strip.drawWaveform(canvasRef.current, deep, soft);
        }
    }, [strip, deep, soft]);

    useEffect(() => { drawWave(); }, [drawWave, loaded]);

    // The canvas is sized from its layout box now, so a width change (window
    // resize, strip reflow, DPR change on monitor switch) leaves the backing
    // store stale until it's redrawn.
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || typeof ResizeObserver === 'undefined') return undefined;
        const ro = new ResizeObserver(() => drawWave());
        ro.observe(canvas);
        return () => ro.disconnect();
    }, [drawWave]);

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
            // Restore the persisted in/out points and cue for the restored
            // clip — the decode reset both to their defaults. Region first:
            // setLaunchStart clamps against the region bounds.
            strip.setRegion(trimRef.current.start, trimRef.current.end);
            strip.setLaunchStart(launchRef.current);
            setLaunchStartState(strip.launchStart);
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
        if (knobs.pitch) strip.setPitch(knobs.pitch);
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
        // "Oldest" is decided by createdAt, not by list position: the user can
        // reorder rows by hand, so position no longer implies age and picking
        // index 0 would silently delete whatever they dragged to the top.
        setFragments((prev) => {
            const combined = [...prev, fragment];
            if (combined.length <= FRAGMENT_CAP) return combined;
            const trimmed = combined.slice();
            const olderThan = (a, b) => (a.createdAt || 0) < (b.createdAt || 0);
            while (trimmed.length > FRAGMENT_CAP) {
                let idx = -1;
                for (let j = 0; j < trimmed.length; j++) {
                    if (trimmed[j].starred) continue;
                    if (idx < 0 || olderThan(trimmed[j], trimmed[idx])) idx = j;
                }
                if (idx < 0) {
                    // Everything is starred → drop the oldest regardless.
                    idx = 0;
                    for (let j = 1; j < trimmed.length; j++) {
                        if (olderThan(trimmed[j], trimmed[idx])) idx = j;
                    }
                }
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

    // Rename — a display label that replaces the F# ordinal in the history
    // row. Empty (or whitespace) clears it and the ordinal comes back. Stored
    // on the fragment metadata, so it rides the session mirror into
    // localStorage and into presets; the IDB blob is untouched.
    const handleRenameFragment = (fragmentId, label) => {
        const trimmed = (label || '').trim().slice(0, 40);
        setFragments((prev) => prev.map((f) => {
            if (f.id !== fragmentId) return f;
            const { label: _prev, ...rest } = f;
            return trimmed ? { ...rest, label: trimmed } : rest;
        }));
    };

    // Manual reordering. Purely presentational — the committed clip, the cue,
    // and every fragment id stay put, so MIDI mappings and the strip's audio
    // are unaffected. Cap eviction picks by createdAt (see makeOnBlob), not by
    // position, so moving a row to the top can't make it the next one dropped.
    const handleMoveFragment = (fragmentId, delta) => {
        setFragments((prev) => {
            const from = prev.findIndex((f) => f.id === fragmentId);
            if (from < 0) return prev;
            const to = from + delta;
            if (to < 0 || to >= prev.length) return prev;
            const next = prev.slice();
            const [moved] = next.splice(from, 1);
            next.splice(to, 0, moved);
            return next;
        });
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

    // Pitch renders take a moment (a few hundred ms per clip), so the knob
    // shows the new value at once, renders once it settles, and dims its
    // label until the shifted copy is what plays.
    const [pitchBusy, setPitchBusy] = useState(false);
    const pitchTimerRef = useRef(null);
    const pitchReqRef = useRef(0);
    const applyPitch = useCallback((semitones) => {
        const req = ++pitchReqRef.current;
        setPitchBusy(true);
        strip.setPitch(semitones)
            .catch((err) => console.warn('Pitch change failed:', err))
            .finally(() => { if (req === pitchReqRef.current) setPitchBusy(false); });
    }, [strip]);
    useEffect(() => () => window.clearTimeout(pitchTimerRef.current), []);

    const handleKnob = (key, value) => {
        if (key === 'pitch') {
            const semitones = Math.round(value);
            setKnobs(prev => ({ ...prev, pitch: semitones }));
            window.clearTimeout(pitchTimerRef.current);
            pitchTimerRef.current = window.setTimeout(() => applyPitch(semitones), PITCH_SETTLE_MS);
            return;
        }
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
        <Box data-lane={index} sx={styles.strip(index, playing)}>
            <Box sx={styles.stripHeader}>
                {/* Lane number in a circle of the lane colour, then Play and
                    Loop; Mute / Solo / Sidechain on the right. */}
                <Box sx={styles.muteSoloRow}>
                    <Box sx={styles.laneIndicator(color, playing)} aria-hidden>{index + 1}</Box>
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
                                sx={styles.loopBtn(soft, looping)}
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
                            sx={[panelStyles.pillControl, styles.laneSelect]}
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
                    <Box sx={styles.segmented}>
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
                                    sx={styles.segment(color, active)}
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
                            sx={[styles.channelPillControl, { width: 54, flexShrink: 0 }]}
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
                        init_audio (Phase 8). Disabled until a fragment exists.

                        Hidden while PERFORMANCE_VARIATION is false. See
                        features.js. */}
                    {PERFORMANCE_VARIATION && (
                    <MidiMappable id={ctrlId('variation')} label={ctrlLabel('Variation')} kind="trigger" onChange={handleVariation}>
                        <Tooltip
                            title={TIPS.channel.variation(loaded)}
                            placement="top"
                        >
                            <span style={{ display: 'inline-flex', flexShrink: 0 }}>
                                <ButtonBase
                                    onClick={handleVariation}
                                    disabled={!loaded || generating}
                                    sx={[styles.channelPillControl, {
                                        width: 40,
                                        justifyContent: 'center',
                                        '&.Mui-disabled': { opacity: 0.4, color: 'text.disabled' },
                                    }]}
                                    aria-label="Variation"
                                >
                                    <VariationIcon size={15} strokeWidth={2.25} />
                                </ButtonBase>
                            </span>
                        </Tooltip>
                    </MidiMappable>
                    )}

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
                onDragOver={handleWaveDragOver}
                onDragLeave={handleWaveDragLeave}
                onDrop={handleWaveDrop}
                onDoubleClick={() => loaded && resetTrim()}
                sx={[
                    styles.waveformWrap,
                    dropActive && {
                        boxShadow: `inset 0 0 0 2px ${deep}`,
                        backgroundColor: `${color}66`,
                        transition: 'box-shadow 120ms, background-color 120ms',
                    },
                    generating && styles.waveformGenerating(color),
                ]}
            >
                {/* Inset from the capsule's round ends: the canvas, the
                    trim maths (waveWrapRef) and every overlay share this box. */}
                <Box ref={waveWrapRef} sx={styles.waveformInner}>
                    <canvas
                        ref={canvasRef}
                        width={140}
                        height={WAVE_HEIGHT}
                        style={{ width: '100%', height: WAVE_HEIGHT, display: 'block', pointerEvents: 'none' }}
                    />
                    {!loaded && (
                        <Typography sx={styles.waveformPlaceholder}>
                            {dropActive ? 'Drop to load' : 'Waveform'}
                        </Typography>
                    )}
                    {/* DJ-style in/out points: drag the handles to choose where
                        the clip loops (bounds retarget live while playing). The
                        cue triangle near the bottom sets where a launch/retrigger
                        enters; repeats still come back to the in-point. Double-
                        click the waveform to reset to the full clip. */}
                    {loaded && (
                        <>
                            <Box sx={{
                                position: 'absolute', top: 0, bottom: 0, left: 0,
                                width: `${trim.start * 100}%`,
                                ...styles.trimVeil,
                                borderRadius: '999px 0 0 999px',
                            }} />
                            <Box sx={{
                                position: 'absolute', top: 0, bottom: 0, right: 0,
                                width: `${(1 - trim.end) * 100}%`,
                                ...styles.trimVeil,
                                borderRadius: '0 999px 999px 0',
                            }} />
                            {/* In-point → cue: inside the loop, so heard on every
                                repeat, but skipped on the launch pass. Tinted so
                                that skip is visible rather than baffling. */}
                            {launchStart !== null && (
                                <Box sx={{
                                    position: 'absolute', top: 0, bottom: 0,
                                    left: `${trim.start * 100}%`,
                                    width: `${Math.max(0, launchStart - trim.start) * 100}%`,
                                    ...styles.trimVeil,
                                    opacity: 0.5,
                                }} />
                            )}
                            <Tooltip title="Loop start — drag to set where each repeat begins. Double-click the waveform to reset." placement="top" enterDelay={600}>
                                <Box onPointerDown={startTrimDrag('start')} sx={styles.trimHandle(trim.start, deep)} />
                            </Tooltip>
                            <Tooltip title="End point — drag to set where the clip stops (loops wrap back to the start point)." placement="top" enterDelay={600}>
                                <Box onPointerDown={startTrimDrag('end')} sx={styles.trimHandle(trim.end, deep)} />
                            </Tooltip>
                            {/* Cue guide — only drawn once the cue is off the
                                in-point, where the loop handle already marks it. */}
                            {launchStart !== null && (
                                <Box sx={(t) => ({
                                    position: 'absolute', top: 14, bottom: 14,
                                    left: `calc(${launchStart * 100}% - 1px)`,
                                    width: '2px',
                                    borderRadius: RADIUS.pill,
                                    backgroundColor: t.palette.perf.cue,
                                    filter: CUE_SHADOW,
                                    pointerEvents: 'none',
                                    zIndex: 2,
                                })} />
                            )}
                            {/* Cue handle: a right-pointing triangle near the
                                bottom, in the cue colour; its hit zone is only
                                the bottom strip, so the loop handles keep the rest. */}
                            <Tooltip title="Cue point — drag to set where a launch or retrigger enters the clip. Repeats still return to the loop start. Double-click to pin it back." placement="top" enterDelay={600}>
                                <Box
                                    onPointerDown={startTrimDrag('cue')}
                                    onDoubleClick={(e) => { e.stopPropagation(); applyLaunch(null); }}
                                    sx={(t) => ({
                                        position: 'absolute', bottom: 6, height: 14,
                                        left: `calc(${(launchStart !== null ? launchStart : trim.start) * 100}% - 4px)`,
                                        width: 12,
                                        cursor: 'ew-resize',
                                        touchAction: 'none',
                                        zIndex: 3,
                                        filter: CUE_SHADOW,
                                        '&::before': {
                                            content: '""', position: 'absolute',
                                            bottom: 2, left: 3,
                                            borderTop: '5px solid transparent',
                                            borderBottom: '5px solid transparent',
                                            borderLeft: `8px solid ${t.palette.perf.cue}`,
                                            opacity: launchStart !== null ? 1 : 0.85,
                                        },
                                    })}
                                />
                            </Tooltip>
                            {/* Playhead: a line with a dot cap. Positioned by the
                                meter rAF loop writing style directly; hidden until
                                the strip reports a position. */}
                            <Box ref={playheadRef} sx={styles.playhead} />
                        </>
                    )}
                </Box>
            </Box>

            {/* Per-channel rolling fragment history. Always rendered (empty
                state included). Star/keep, delete, audition, load — all
                inline per row. Capped at FRAGMENT_CAP via FIFO with star
                priority. */}
            <ChannelFragmentHistory
                fragments={fragments}
                color={color}
                soft={soft}
                deep={deep}
                channelIndex={index}
                auditioningId={auditioningFragmentId}
                committedId={committedFragmentId}
                maxFragments={FRAGMENT_CAP}
                onAudition={handleAuditionFragment}
                onCommit={handleCommitFragment}
                onToggleStar={handleToggleStar}
                onDelete={handleDeleteFragment}
                onRename={handleRenameFragment}
                onMove={handleMoveFragment}
                onClearAll={handleClearFragments}
            />

            {/* Pan and Tune: knobs in the same style and grid as the strip
                below. Both have a centre detent; Tune is whole semitones. */}
            <Box sx={styles.knobsGrid}>
                <MidiMappable
                    id={ctrlId('pan')}
                    label={ctrlLabel('Pan')}
                    kind="continuous"
                    min={-1}
                    max={1}
                    value={knobs.pan ?? 0}
                    onChange={handlePan}
                    sx={{ alignItems: 'center' }}
                >
                    <RotaryKnob
                        value={knobs.pan ?? 0}
                        min={-1}
                        max={1}
                        step={0.01}
                        center={0}
                        detent={0.06}
                        size={52}
                        color={deep}
                        label="Pan"
                        format={fmtPan}
                        ariaLabel={`Pan · ${ctrlLabel('Pan')}`}
                        onChange={handlePan}
                    />
                </MidiMappable>
                <MidiMappable
                    id={ctrlId('pitch')}
                    label={ctrlLabel('Pitch')}
                    kind="continuous"
                    curve="linear"
                    min={-PITCH_RANGE}
                    max={PITCH_RANGE}
                    value={knobs.pitch ?? 0}
                    onChange={(v) => handleKnob('pitch', v)}
                    sx={{ alignItems: 'center' }}
                >
                    <RotaryKnob
                        value={knobs.pitch ?? 0}
                        min={-PITCH_RANGE}
                        max={PITCH_RANGE}
                        size={52}
                        color={deep}
                        busy={pitchBusy}
                        label="Tune"
                        format={(v) => `${fmtSemitones(v)} st`}
                        ariaLabel="Tune in semitones"
                        onChange={(v) => handleKnob('pitch', v)}
                    />
                </MidiMappable>
            </Box>

            {/* Gain and effects: one strip of compact knobs, each with a fine
                arc in the lane colour and its value underneath. Gain turns in
                fader-taper position space (faderLaw), like the master fader,
                so the stored value stays in dB for persistence and MIDI. */}
            <Box sx={styles.knobsGrid}>
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
                            id={ctrlId(k.key)}
                            label={ctrlLabel(k.label)}
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
                                size={52}
                                color={deep}
                                label={k.label}
                                ariaLabel={`${k.label} · ${ctrlLabel(k.label)}`}
                            />
                        </MidiMappable>
                    );
                })}
            </Box>

            {/* Bottom row is now just the channel level meter — Play and Loop
                moved to the top header so the channel reads "controls on top,
                signal flow below". */}
            <Box sx={styles.transportRow}>
                <Box sx={styles.meterTrack}>
                    <Box ref={meterRef} sx={styles.meterFill(color, deep)} />
                </Box>
            </Box>

        </Box>
    );
}
