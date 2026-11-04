// The Performance page's state (session, presets, MIDI settings, channel
// audio) lives in the data folder, so a new install, Settings > Import and
// Uninstall all see it. The window's localStorage keeps a copy for the
// synchronous reads at mount: loadPerformanceState() fills it from the files
// before the app renders, and every save goes to both. Whatever the window
// has that the folder doesn't (an older Fragmenta's state) moves over once.
// On the Space, where visitors share one backend, it all stays in the window.
import api from '../api';

export const SESSION_KEY = 'fragmenta.performance.session.v1';
export const PRESETS_KEY = 'fragmenta.performance.presets.v1';
export const MIDI_KEY = 'fragmenta.midi.config.v1';
const STATE_KEYS = { session: SESSION_KEY, midi: MIDI_KEY };

let filesAvailable = false;
export const performanceFilesAvailable = () => filesAvailable;

function readLocal(key) {
    try {
        const raw = localStorage.getItem(key);
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
}

function writeLocal(key, value) {
    try {
        if (value === null) localStorage.removeItem(key);
        else localStorage.setItem(key, JSON.stringify(value));
        return true;
    } catch {
        return false; // quota or serialization
    }
}

const presetUrl = (name) => `/api/performance/presets/${encodeURIComponent(name)}`;

// One request in flight per file; a newer value waits and replaces any older
// one still waiting, so the file ends on the last save.
const waiting = {};
const sending = {};
function send(name) {
    if (sending[name] || !(name in waiting)) return;
    const value = waiting[name];
    delete waiting[name];
    sending[name] = true;
    const url = `/api/performance/state/${name}`;
    (value === null ? api.delete(url) : api.put(url, value))
        .catch((err) => console.warn(`[performanceStore] saving ${name} failed`, err))
        .finally(() => {
            sending[name] = false;
            send(name);
        });
}

/** Save `session` or `midi` (null deletes it). Returns false if the window
 *  couldn't keep its copy (quota). */
export function saveState(name, value) {
    const ok = writeLocal(STATE_KEYS[name], value);
    if (filesAvailable) {
        waiting[name] = value;
        send(name);
    }
    return ok;
}

/** For the window closing: send what's waiting right away. The request
 *  outlives the page only while it's under the browser's 64 KB keepalive cap. */
export function saveStateNow(name, value) {
    writeLocal(STATE_KEYS[name], value);
    if (!filesAvailable) return;
    delete waiting[name];
    fetch(`/api/performance/state/${name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(value),
        keepalive: true,
    }).catch(() => send(name));
}

export function readPresets() {
    const bag = readLocal(PRESETS_KEY);
    return bag && typeof bag === 'object' ? bag : {};
}

export function savePresetFile(name, value) {
    const bag = readPresets();
    bag[name] = value;
    if (!writeLocal(PRESETS_KEY, bag)) return false;
    if (filesAvailable) {
        api.put(presetUrl(name), value)
            .catch((err) => console.warn('[performanceStore] saving preset failed', err));
    }
    return true;
}

export function deletePresetFile(name) {
    const bag = readPresets();
    delete bag[name];
    writeLocal(PRESETS_KEY, bag);
    if (filesAvailable) api.delete(presetUrl(name)).catch(() => {});
}

/** Before the app renders: fill the window's copy from the data folder. */
export async function loadPerformanceState() {
    let state;
    try {
        state = (await api.get('/api/performance/state')).data;
    } catch {
        return; // backend unreachable: run on the window's copy
    }
    if (!state?.available) return;
    filesAvailable = true;
    const moves = [];
    for (const [name, key] of Object.entries(STATE_KEYS)) {
        if (state[name] != null) {
            writeLocal(key, state[name]);
        } else {
            const local = readLocal(key);
            if (local != null) moves.push(api.put(`/api/performance/state/${name}`, local));
        }
    }
    if (state.presets != null) {
        writeLocal(PRESETS_KEY, state.presets);
    } else {
        Object.entries(readPresets()).forEach(([name, value]) => {
            moves.push(api.put(presetUrl(name), value));
        });
    }
    await Promise.allSettled(moves);
}
