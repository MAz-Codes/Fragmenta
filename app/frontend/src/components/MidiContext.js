import React, {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import { Box } from '@mui/material';
import api from '../api';
import { faderPosToDb, faderDbToPos } from '../utils/faderLaw';

const STORAGE_KEY = 'fragmenta.midi.config.v1';

const DEFAULT_CONFIG = {
    deviceIds: [],              // every input listened to, by port name
    channelFilter: 0,           
    takeover: 'jump',           
    mappings: [],              
};

const MIDI_MODE = {
    NOTE_ON: 0x90,
    NOTE_OFF: 0x80,
    CC: 0xb0,
};

function loadConfig() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return { ...DEFAULT_CONFIG, mappings: [] };
        const { deviceId, deviceName, ...parsed } = JSON.parse(raw);
        // Before several inputs could be on at once there was one device;
        // its port name is the id now.
        const deviceIds = Array.isArray(parsed.deviceIds)
            ? parsed.deviceIds
            : (deviceName || deviceId ? [deviceName || deviceId] : []);
        return {
            ...DEFAULT_CONFIG,
            ...parsed,
            deviceIds,
            mappings: Array.isArray(parsed.mappings) ? parsed.mappings : [],
        };
    } catch {
        return { ...DEFAULT_CONFIG, mappings: [] };
    }
}

function saveConfig(config) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    } catch { /* quota or serialization — non-fatal */ }
}

// Wipes the persisted MIDI device + mappings. The provider will pick up the
// reset on its next mount (caller is expected to remount).
export function clearMidiConfig() {
    try { localStorage.removeItem(STORAGE_KEY); }
    catch { /* non-fatal */ }
}

// Do two mappings answer the same messages? A mapping learned before
// several inputs could be on has no port and answers every input.
function sameSource(a, b) {
    return a.type === b.type && a.channel === b.channel && a.number === b.number
        && (!a.port || !b.port || a.port === b.port);
}

function formatMidi(midi) {
    if (!midi) return '';
    const t = midi.type === 'cc' ? 'CC' : 'Note';
    return `${t} ${midi.number} · ch.${midi.channel}${midi.port ? ` · ${midi.port}` : ''}`;
}

const MidiContext = createContext(null);

export function MidiProvider({ children }) {
    const [config, setConfig] = useState(loadConfig);
    const [inputs, setInputs] = useState([]);
    const [supported, setSupported] = useState(true);
    const [permissionError, setPermissionError] = useState(null);
    const [learnMode, setLearnMode] = useState(false);
    const [learnTarget, setLearnTarget] = useState(null);
    // Inputs that are plugged in but wouldn't open (another app holding them).
    const [failedInputs, setFailedInputs] = useState([]);

    const subscribersRef = useRef(new Map());
    const pickupArmedRef = useRef(new Map());
    // Last value seen per CC, so a CC button fires once per press and Learn
    // can tell a press from a turn.
    const ccValuesRef = useRef(new Map());
    const configRef = useRef(config);
    const learnTargetRef = useRef(learnTarget);

    useEffect(() => {
        configRef.current = config;
        saveConfig(config);
    }, [config]);

    useEffect(() => { learnTargetRef.current = learnTarget; }, [learnTarget]);

    // Device list comes from the native backend (python-rtmidi) instead of
    // Web MIDI, so it works in every web engine.
    const refreshInputs = useCallback(async () => {
        try {
            const { data } = await api.get('/api/midi/devices');
            setSupported(!!data.available);
            setInputs(Array.isArray(data.inputs) ? data.inputs : []);
            setPermissionError(data.available
                ? null
                : 'Native MIDI is unavailable (python-rtmidi not installed).');
        } catch (err) {
            setSupported(false);
            setPermissionError(err?.message || 'Could not reach the MIDI backend.');
        }
    }, []);

    useEffect(() => { refreshInputs(); }, [refreshInputs]);

    const captureLearn = useCallback((controlId, midi) => {
        setConfig((prev) => {
            const subOpts = subscribersRef.current.get(controlId)?.opts || {};
            const newMapping = {
                controlId,
                label: subOpts.label || controlId,
                kind: subOpts.kind || 'continuous',
                curve: subOpts.curve || 'linear',
                min: subOpts.min ?? 0,
                max: subOpts.max ?? 1,
                midi,
            };
            const filtered = prev.mappings.filter(
                (m) => m.controlId !== controlId && !sameSource(m.midi, midi),
            );
            return { ...prev, mappings: [...filtered, newMapping] };
        });
        pickupArmedRef.current.delete(controlId);
        setLearnTarget(null);
    }, []);

    const dispatchMessage = useCallback((event) => {
        const data = event.data;
        if (!data || data.length < 2) return;
        const status = data[0];
        const data1 = data[1];
        const data2 = data.length > 2 ? data[2] : 0;
        const type = status & 0xf0;
        const channel = (status & 0x0f) + 1;
        const cfg = configRef.current;

        if (cfg.channelFilter && channel !== cfg.channelFilter) return;

        const isCC = type === MIDI_MODE.CC;
        const isNoteOn = type === MIDI_MODE.NOTE_ON && data2 > 0;
        const isNoteOff = type === MIDI_MODE.NOTE_OFF || (type === MIDI_MODE.NOTE_ON && data2 === 0);
        if (!isCC && !isNoteOn && !isNoteOff) return;

        const incomingType = isCC ? 'cc' : 'note';
        const port = event.port || null;
        let prevCC;
        if (isCC) {
            const ccKey = `${port}:${channel}:${data1}`;
            prevCC = ccValuesRef.current.get(ccKey);
            ccValuesRef.current.set(ccKey, data2);
        }
        const target = learnTargetRef.current;
        if (target) {
            // Learn only what fits the control. Encoders without detents (MIDI
            // Fighter Twister) turn a little when pressed, so the first message
            // alone can't be trusted: knobs take a CC that moved off its ends,
            // buttons a note or a CC pressed and released.
            const kind = subscribersRef.current.get(target)?.opts?.kind || 'continuous';
            const fits = kind === 'trigger'
                ? isNoteOn || (isCC && data2 === 0 && prevCC >= 64)
                : isCC && data2 > 0 && data2 < 127;
            if (fits) captureLearn(target, { type: incomingType, channel, number: data1, port });
            return;
        }

        for (const m of cfg.mappings) {
            if (m.midi.channel !== channel || m.midi.number !== data1) continue;
            if (m.midi.type !== incomingType) continue;
            if (m.midi.port && port && m.midi.port !== port) continue;
            const sub = subscribersRef.current.get(m.controlId);
            if (!sub) continue;

            if (sub.opts.kind === 'continuous') {
                if (!isCC) continue;
                applyContinuous(sub, m, data2, cfg.takeover);
            } else if (sub.opts.kind === 'trigger') {
                if (isNoteOn) sub.handler();
                // Once per press, not on every message a turning knob sends.
                else if (isCC && data2 >= 64 && !(prevCC >= 64)) sub.handler();
            }
        }
    }, [captureLearn]);

    // Tell the backend which ports to open. The stream only carries the open
    // ports' events, so device selection happens server-side.
    const selectInputs = useCallback(() => {
        api.post('/api/midi/select', { port_ids: configRef.current?.deviceIds || [] })
            .then(({ data }) => setFailedInputs(data?.failed || []))
            .catch(() => { /* non-fatal */ });
    }, []);

    // Stream incoming MIDI from the backend over SSE. Each event is the
    // {data:[status,d1,d2]} shape Web MIDI gave us, plus the port it came in
    // on. EventSource auto-reconnects on drop.
    useEffect(() => {
        if (typeof EventSource === 'undefined') {
            setSupported(false);
            return undefined;
        }
        const es = new EventSource('/api/midi/stream');
        es.onopen = () => {
            // Fires on the initial connect AND on every EventSource auto-
            // reconnect — i.e. after a backend restart. The new backend
            // process starts with no MIDI port open, so re-assert the saved
            // selection (idempotent when the port is already open). Without
            // this, mappings stay dead after a restart until a page reload.
            selectInputs();
        };
        es.onmessage = (e) => {
            try { dispatchMessage(JSON.parse(e.data)); }
            catch { /* malformed line — ignore */ }
        };
        pickupArmedRef.current = new Map();
        return () => es.close();
    }, [dispatchMessage, selectInputs]);

    // Runs again when a saved device reappears, so plugging it back in
    // reconnects its mappings.
    const wanted = config.deviceIds.join('\n');
    const wantedPresent = config.deviceIds.filter((id) => inputs.some((i) => i.id === id)).join('\n');
    useEffect(() => {
        if (supported) selectInputs();
    }, [wanted, wantedPresent, supported, selectInputs]);

    function applyContinuous(sub, mapping, midiValue, takeover) {
        const norm = midiValue / 127;
        // Prefer the LIVE subscriber's curve/range over the snapshot stored
        // in the mapping at learn time — when a control's law changes (e.g.
        // the gain faders gained a console taper + boost headroom), existing
        // learned mappings follow it without needing a re-learn.
        const curve = sub.opts?.curve ?? mapping.curve;
        const min = sub.opts?.min ?? mapping.min;
        const max = sub.opts?.max ?? mapping.max;
        let target;
        if (curve === 'fader') {
            // Same console taper as the on-screen fader throw, so a hardware
            // fader at 80% sits at unity just like the UI.
            target = Math.min(max, Math.max(min, faderPosToDb(norm)));
        } else if (curve === 'log' && min > 0 && max > 0) {
            target = min * Math.pow(max / min, norm);
        } else {
            target = min + norm * (max - min);
        }

        if (takeover === 'pickup') {
            const armed = pickupArmedRef.current.get(mapping.controlId);
            if (!armed) {
                const current = typeof sub.getValue === 'function' ? sub.getValue() : sub.value;
                const span = max - min;
                if (span === 0 || !isFinite(current)) {
                    pickupArmedRef.current.set(mapping.controlId, true);
                } else {
                    // Compare on the same curve we used to compute target.
                    let currentNorm;
                    if (curve === 'fader') {
                        currentNorm = faderDbToPos(current);
                    } else if (curve === 'log' && min > 0 && current > 0) {
                        currentNorm = Math.log(current / min) / Math.log(max / min);
                    } else {
                        currentNorm = (current - min) / span;
                    }
                    if (Math.abs(norm - currentNorm) < 0.02) {
                        pickupArmedRef.current.set(mapping.controlId, true);
                    } else {
                        return;
                    }
                }
            }
        }
        sub.handler(target);
    }

    const beginLearn = useCallback((controlId) => {
        setLearnMode(true);
        setLearnTarget(controlId);
    }, []);

    const cancelLearn = useCallback(() => setLearnTarget(null), []);

    const clearMapping = useCallback((controlId) => {
        setConfig((prev) => ({
            ...prev,
            mappings: prev.mappings.filter((m) => m.controlId !== controlId),
        }));
        pickupArmedRef.current.delete(controlId);
    }, []);

    const clearAll = useCallback(() => {
        setConfig((prev) => ({ ...prev, mappings: [] }));
        pickupArmedRef.current.clear();
    }, []);

    const setDeviceEnabled = useCallback((deviceId, on) => {
        setConfig((prev) => {
            const rest = prev.deviceIds.filter((id) => id !== deviceId);
            return { ...prev, deviceIds: on ? [...rest, deviceId] : rest };
        });
    }, []);

    const setChannelFilter = useCallback((channel) => {
        setConfig((prev) => ({ ...prev, channelFilter: channel }));
    }, []);

    const setTakeover = useCallback((mode) => {
        setConfig((prev) => ({ ...prev, takeover: mode }));
        pickupArmedRef.current.clear();
    }, []);

    const exitLearnMode = useCallback(() => {
        setLearnMode(false);
        setLearnTarget(null);
    }, []);

    const toggleLearnMode = useCallback(() => {
        setLearnMode((prev) => {
            if (prev) setLearnTarget(null);
            return !prev;
        });
    }, []);

    const registerSubscriber = useCallback((id, sub) => {
        subscribersRef.current.set(id, sub);
    }, []);

    const unregisterSubscriber = useCallback((id) => {
        subscribersRef.current.delete(id);
    }, []);

    const value = useMemo(() => ({
        config,
        inputs,
        failedInputs,
        supported,
        permissionError,
        learnMode,
        learnTarget,
        refreshInputs,
        setDeviceEnabled,
        setChannelFilter,
        setTakeover,
        beginLearn,
        cancelLearn,
        clearMapping,
        clearAll,
        toggleLearnMode,
        exitLearnMode,
        registerSubscriber,
        unregisterSubscriber,
    }), [
        config, inputs, failedInputs, supported, permissionError, learnMode, learnTarget,
        refreshInputs, setDeviceEnabled, setChannelFilter, setTakeover, beginLearn,
        cancelLearn, clearMapping, clearAll, toggleLearnMode, exitLearnMode,
        registerSubscriber, unregisterSubscriber,
    ]);

    return <MidiContext.Provider value={value}>{children}</MidiContext.Provider>;
}

export function useMidi() {
    const ctx = useContext(MidiContext);
    return ctx;
}

export { formatMidi };


export function MidiMappable({
    id,
    label,
    kind = 'continuous',
    curve = 'linear',
    min = 0,
    max = 1,
    value,
    onChange,
    sx,
    children,
}) {
    const ctx = useMidi();
    const valueRef = useRef(value);
    useEffect(() => { valueRef.current = value; }, [value]);
    const handlerRef = useRef(onChange);
    useEffect(() => { handlerRef.current = onChange; }, [onChange]);

    useEffect(() => {
        if (!ctx) return undefined;
        ctx.registerSubscriber(id, {
            opts: { kind, curve, min, max, label },
            handler: (v) => handlerRef.current?.(v),
            getValue: () => valueRef.current,
        });
        return () => ctx.unregisterSubscriber(id);
    }, [ctx, id, kind, curve, min, max, label]);

    if (!ctx) return <>{children}</>;

    const mapping = ctx.config.mappings.find((m) => m.controlId === id);
    const isLearningThis = ctx.learnTarget === id;
    const showOverlay = ctx.learnMode;

    return (
        <Box sx={{ position: 'relative', display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0, ...sx }}>
            {children}
            {mapping && !showOverlay && (
                <Box
                    sx={(t) => ({
                        position: 'absolute',
                        top: 1,
                        right: 1,
                        fontSize: '0.5rem',
                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                        bgcolor: `rgba(${t.palette.perf.midiMappedRgb}, 0.85)`,
                        color: '#FFFFFF',
                        px: 0.4,
                        py: 0.05,
                        borderRadius: 0.5,
                        pointerEvents: 'none',
                        letterSpacing: '0.04em',
                        zIndex: 5,
                    })}
                >
                    {mapping.midi.type === 'cc' ? 'CC' : 'N'}{mapping.midi.number}
                </Box>
            )}
            {showOverlay && (
                <Box
                    onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (isLearningThis) ctx.cancelLearn();
                        else ctx.beginLearn(id);
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (mapping) ctx.clearMapping(id);
                    }}
                    title={
                        isLearningThis
                            ? `${label}: move a hardware control to bind (right-click to clear)`
                            : mapping
                                ? `${label}: ${formatMidi(mapping.midi)} (click to re-learn, right-click to clear)`
                                : `${label}: click then move a hardware control to bind`
                    }
                    sx={(t) => ({
                        position: 'absolute',
                        inset: 0,
                        cursor: 'pointer',
                        zIndex: 20,
                        bgcolor: isLearningThis
                            ? `rgba(${t.palette.perf.midiLearnRgb}, 0.32)`
                            : mapping
                                ? `rgba(${t.palette.perf.midiMappedRgb}, 0.18)`
                                : `rgba(${t.palette.perf.midiLearnRgb}, 0.10)`,
                        border: '1px dashed',
                        borderColor: isLearningThis || !mapping
                            ? t.palette.warm.main
                            : t.palette.success.main,
                        borderRadius: 1,
                        animation: isLearningThis ? 'midiPulse 900ms ease-in-out infinite' : 'none',
                        '@keyframes midiPulse': {
                            '0%, 100%': { opacity: 0.5 },
                            '50%': { opacity: 1 },
                        },
                    })}
                >
                    {(mapping || isLearningThis) && (
                        <Box
                            sx={{
                                position: 'absolute',
                                top: 2,
                                left: 2,
                                fontSize: '0.5rem',
                                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                                color: isLearningThis ? 'warm.main' : 'success.main',
                                letterSpacing: '0.04em',
                                pointerEvents: 'none',
                            }}
                        >
                            {isLearningThis ? 'learn…' : `${mapping.midi.type === 'cc' ? 'CC' : 'N'}${mapping.midi.number}`}
                        </Box>
                    )}
                </Box>
            )}
        </Box>
    );
}
