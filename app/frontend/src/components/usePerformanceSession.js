import { useCallback, useEffect, useRef, useState } from 'react';
import {
    SESSION_KEY,
    deletePresetFile,
    readPresets,
    savePresetFile,
    saveState,
    saveStateNow,
} from '../utils/performanceStore';

// Session and presets are saved in the data folder; the window keeps a copy
// for the synchronous reads here (utils/performanceStore.js).
const STORAGE_KEY = SESSION_KEY;

export function clearPerformanceSession() {
    saveState('session', null);
}

export function listPresetNames() {
    return Object.keys(readPresets()).sort((a, b) => a.localeCompare(b));
}

// False when the window couldn't keep it (quota), so callers can tell the
// user their preset was NOT saved instead of silently claiming success.
export function savePreset(name, sessionData) {
    const trimmed = (name || '').trim();
    if (!trimmed) return false;
    return savePresetFile(trimmed, sessionData);
}

export function deletePreset(name) {
    if (!(name in readPresets())) return false;
    deletePresetFile(name);
    return true;
}

// Replace the live session with a preset's snapshot. Caller is expected to
// force-remount the panel afterward so its useState mirrors pick up the new
// shape; storage alone won't reset mounted state.
export function loadPresetIntoSession(name) {
    const preset = readPresets()[name];
    if (!preset) return false;
    return saveState('session', preset);
}

const CHANNEL_DEFAULT = {
    prompt: '',
    duration: 8,
    durationMode: 'bars',
    bars: 4,
    looping: true,
    muted: false,
    soloed: false,
    batchSize: 1,
    knobs: { gain: -6, pan: 0, filter: 0, delay: 0, reverb: 0 },
    // Fragment history metadata (id, prompt, duration, createdAt, starred,
    // number). The Blob audio bodies live in IndexedDB under the
    // `session-ch{N}` scope — see utils/fragmentStorage.js. Cleared on
    // Fresh Start and overwritten on preset load.
    fragments: [],
    // Which fragment was loaded into the channel strip last; restored on
    // reload so the channel comes back ready to play instead of empty.
    committedFragmentId: null,
    // DJ-style in/out points over the committed clip (normalized 0..1).
    // Full clip by default; reset whenever different audio is loaded.
    trimStart: 0,
    trimEnd: 1,
    // DJ cue point — where the first pass and every retrigger enter the clip.
    // null = pinned to trimStart (pre-cue-marker behaviour), so older saves
    // and presets keep playing exactly as they did.
    launchStart: null,
};

function defaultSession(channelCount) {
    return {
        bpm: 120,
        launchQuantum: 4,
        masterDb: 0,
        injectBpm: true,
        linkEnabled: false,
        selectedModel: '',
        selectedUnwrappedModel: '',
        steps: 50,
        randomSeed: true,
        seedValue: '',
        cueDeviceId: '',
        // Master FX defaults — the FX are always-on; the wet level on the
        // master bus is determined entirely by per-channel DLY/REV send
        // levels. We only persist the IR choice and the delay division.
        masterReverbIR: 'hall',
        masterDelayDivision: '1/4',
        // Prompt auto-inject fields. Each is appended (comma-separated) to
        // every generated prompt when set. Key and Time accept any text;
        // empty = no injection. BPM is a toggle that, when on, grabs the
        // live master BPM (top-bar value) at generation time.
        promptKey: '',
        promptInjectBpm: false,
        promptTimeSig: '',
        channels: Array.from({ length: channelCount }, () => ({
            ...CHANNEL_DEFAULT,
            knobs: { ...CHANNEL_DEFAULT.knobs },
        })),
    };
}

function loadSession(channelCount) {
    const fallback = defaultSession(channelCount);
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return fallback;
        const parsed = JSON.parse(raw);
        // Merge against defaults so older saves don't crash on missing fields.
        // Length shifts (channel count change between releases) are absorbed
        // by always producing exactly `channelCount` channels.
        //
        // Migration: pre-rename saves used `takes` / `committedTakeId`. Copy
        // those into the new `fragments` / `committedFragmentId` slots when
        // present, so users' existing generations carry over after the
        // "Takes → Fragments" rename. Old fields are left in place but unused.
        const channels = Array.from({ length: channelCount }, (_, i) => {
            const ch = parsed.channels?.[i] || {};
            return {
                ...CHANNEL_DEFAULT,
                ...ch,
                fragments: ch.fragments ?? ch.takes ?? [],
                committedFragmentId: ch.committedFragmentId ?? ch.committedTakeId ?? null,
                knobs: { ...CHANNEL_DEFAULT.knobs, ...(ch.knobs || {}) },
            };
        });
        return { ...fallback, ...parsed, channels };
    } catch {
        return fallback;
    }
}

export function usePerformanceSession(channelCount = 4) {
    const [session, setSession] = useState(() => loadSession(channelCount));
    const persistTimerRef = useRef(null);
    const sessionRef = useRef(session);

    // Knobs and sliders fire many times per second; debounce writes so we don't
    // hammer storage. Last-write-wins is fine for session continuity.
    useEffect(() => {
        sessionRef.current = session;
        if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
        persistTimerRef.current = setTimeout(() => {
            persistTimerRef.current = null;
            saveState('session', session);
        }, 250);
        return () => {
            if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
        };
    }, [session]);

    // Drop a queued (debounced) write without persisting it. Preset load and
    // Restore Defaults replace the storage key directly and then force a
    // remount — but they await IndexedDB blob copies in between, which is
    // plenty of time for a pending 250 ms persist to fire and overwrite the
    // freshly written payload with the OLD session. They must cancel first.
    const cancelPendingPersist = useCallback(() => {
        if (persistTimerRef.current) {
            clearTimeout(persistTimerRef.current);
            persistTimerRef.current = null;
        }
    }, []);

    // Write the latest session right now (used on beforeunload so up to
    // 250 ms of final tweaks aren't lost when the window closes).
    const flushPersist = useCallback(() => {
        cancelPendingPersist();
        saveStateNow('session', sessionRef.current);
    }, [cancelPendingPersist]);

    useEffect(() => {
        window.addEventListener('beforeunload', flushPersist);
        return () => window.removeEventListener('beforeunload', flushPersist);
    }, [flushPersist]);

    const updateGlobal = useCallback((key, value) => {
        setSession(prev => (prev[key] === value ? prev : { ...prev, [key]: value }));
    }, []);

    const updateChannel = useCallback((index, partial) => {
        setSession(prev => {
            const channels = prev.channels.slice();
            channels[index] = { ...channels[index], ...partial };
            return { ...prev, channels };
        });
    }, []);

    return { session, updateGlobal, updateChannel, cancelPendingPersist, flushPersist };
}
