import { useCallback, useEffect, useRef, useState } from 'react';

export const PERFORMANCE_SESSION_STORAGE_KEY = 'fragmenta.performance.session.v1';

const STORAGE_KEY = PERFORMANCE_SESSION_STORAGE_KEY;

export function clearPerformanceSession() {
    try { localStorage.removeItem(STORAGE_KEY); }
    catch { /* non-fatal */ }
}

const PRESETS_STORAGE_KEY = 'fragmenta.performance.presets.v1';

function readPresetBag() {
    try {
        const raw = localStorage.getItem(PRESETS_STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
        return {};
    }
}

function writePresetBag(bag) {
    try { localStorage.setItem(PRESETS_STORAGE_KEY, JSON.stringify(bag)); }
    catch { /* quota — non-fatal */ }
}

export function listPresetNames() {
    return Object.keys(readPresetBag()).sort((a, b) => a.localeCompare(b));
}

export function savePreset(name, sessionData) {
    const trimmed = (name || '').trim();
    if (!trimmed) return false;
    const bag = readPresetBag();
    bag[trimmed] = sessionData;
    writePresetBag(bag);
    return true;
}

export function deletePreset(name) {
    const bag = readPresetBag();
    if (!(name in bag)) return false;
    delete bag[name];
    writePresetBag(bag);
    return true;
}

// Replace the live session storage with a preset's snapshot. Caller is
// expected to force-remount the panel afterward so its useState mirrors
// pick up the new shape; localStorage alone won't reset mounted state.
export function loadPresetIntoSession(name) {
    const bag = readPresetBag();
    const preset = bag[name];
    if (!preset) return false;
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(preset));
        return true;
    } catch {
        return false;
    }
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
    // Take history metadata (id, prompt, duration, createdAt, starred,
    // number). The Blob audio bodies live in IndexedDB under the
    // `session-ch{N}` scope — see utils/takeStorage.js. Cleared on Fresh
    // Start and overwritten on preset load.
    takes: [],
    // Which take was loaded into the channel strip last; restored on reload
    // so the channel comes back ready to play instead of empty.
    committedTakeId: null,
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
        steps: 250,
        randomSeed: true,
        seedValue: '',
        cueDeviceId: '',
        // Master FX defaults — both wet sends at 0 (silent) so the master is
        // bit-exact dry until the user dials in. Reverb IR and delay division
        // come up at the engine's defaults (`hall`, `1/4`).
        masterReverbIR: 'hall',
        masterReverbMix: 0,
        masterDelayDivision: '1/4',
        masterDelayMix: 0,
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
        const channels = Array.from({ length: channelCount }, (_, i) => ({
            ...CHANNEL_DEFAULT,
            ...(parsed.channels?.[i] || {}),
            knobs: { ...CHANNEL_DEFAULT.knobs, ...(parsed.channels?.[i]?.knobs || {}) },
        }));
        return { ...fallback, ...parsed, channels };
    } catch {
        return fallback;
    }
}

export function usePerformanceSession(channelCount = 4) {
    const [session, setSession] = useState(() => loadSession(channelCount));
    const persistTimerRef = useRef(null);

    // Knobs and sliders fire many times per second; debounce writes so we don't
    // hammer localStorage. Last-write-wins is fine for session continuity.
    useEffect(() => {
        if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
        persistTimerRef.current = setTimeout(() => {
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
            } catch { /* quota or serialization — non-fatal */ }
        }, 250);
        return () => {
            if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
        };
    }, [session]);

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

    return { session, updateGlobal, updateChannel };
}
