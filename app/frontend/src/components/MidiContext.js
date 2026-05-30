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

const STORAGE_KEY = 'fragmenta.midi.config.v1';

const DEFAULT_CONFIG = {
    deviceId: null,
    deviceName: null,
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
        const parsed = JSON.parse(raw);
        return {
            ...DEFAULT_CONFIG,
            ...parsed,
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

function midiKey(midi) {
    return `${midi.type}:${midi.channel}:${midi.number}`;
}

function formatMidi(midi) {
    if (!midi) return '';
    const t = midi.type === 'cc' ? 'CC' : 'Note';
    return `${t} ${midi.number} · ch.${midi.channel}`;
}

const MidiContext = createContext(null);

export function MidiProvider({ children }) {
    const [config, setConfig] = useState(loadConfig);
    const [inputs, setInputs] = useState([]);
    const [supported, setSupported] = useState(true);
    const [permissionError, setPermissionError] = useState(null);
    const [learnMode, setLearnMode] = useState(false);
    const [learnTarget, setLearnTarget] = useState(null);

    const subscribersRef = useRef(new Map());
    const pickupArmedRef = useRef(new Map());
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

    useEffect(() => {
        if (!inputs.length || !config.deviceName) return;
        const stillThere = config.deviceId && inputs.some(i => i.id === config.deviceId);
        if (stillThere) return;
        const byName = inputs.find(i => i.name === config.deviceName);
        if (byName) {
            setConfig(prev => ({ ...prev, deviceId: byName.id }));
        }
    }, [inputs, config.deviceId, config.deviceName]);

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
            const targetKey = midiKey(midi);
            const filtered = prev.mappings.filter(
                (m) => m.controlId !== controlId && midiKey(m.midi) !== targetKey,
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
        const target = learnTargetRef.current;
        if (target && (isCC || isNoteOn)) {
            captureLearn(target, { type: incomingType, channel, number: data1 });
            return;
        }

        for (const m of cfg.mappings) {
            if (m.midi.channel !== channel || m.midi.number !== data1) continue;
            if (m.midi.type !== incomingType) continue;
            const sub = subscribersRef.current.get(m.controlId);
            if (!sub) continue;

            if (sub.opts.kind === 'continuous') {
                if (!isCC) continue;
                applyContinuous(sub, m, data2, cfg.takeover);
            } else if (sub.opts.kind === 'trigger') {
                if (isNoteOn) sub.handler();
                else if (isCC && data2 >= 64) sub.handler();
            }
        }
    }, [captureLearn]);

    // Stream incoming MIDI from the backend over SSE. Each event is the same
    // {data:[status,d1,d2]} shape Web MIDI gave us, so dispatchMessage is
    // unchanged. EventSource auto-reconnects on drop.
    useEffect(() => {
        if (typeof EventSource === 'undefined') {
            setSupported(false);
            return undefined;
        }
        const es = new EventSource('/api/midi/stream');
        es.onmessage = (e) => {
            try { dispatchMessage(JSON.parse(e.data)); }
            catch { /* malformed line — ignore */ }
        };
        pickupArmedRef.current = new Map();
        return () => es.close();
    }, [dispatchMessage]);

    // Tell the backend which port to open. The stream only carries the open
    // port's events, so device selection happens server-side.
    useEffect(() => {
        if (!supported) return;
        api.post('/api/midi/select', { port_id: config.deviceId || null })
            .catch(() => { /* non-fatal */ });
    }, [config.deviceId, supported]);

    function applyContinuous(sub, mapping, midiValue, takeover) {
        const norm = midiValue / 127;
        let target;
        if (mapping.curve === 'log' && mapping.min > 0 && mapping.max > 0) {
            target = mapping.min * Math.pow(mapping.max / mapping.min, norm);
        } else {
            target = mapping.min + norm * (mapping.max - mapping.min);
        }

        if (takeover === 'pickup') {
            const armed = pickupArmedRef.current.get(mapping.controlId);
            if (!armed) {
                const current = typeof sub.getValue === 'function' ? sub.getValue() : sub.value;
                const span = mapping.max - mapping.min;
                if (span === 0 || !isFinite(current)) {
                    pickupArmedRef.current.set(mapping.controlId, true);
                } else {
                    // Compare on the same curve we used to compute target.
                    let currentNorm;
                    if (mapping.curve === 'log' && mapping.min > 0 && current > 0) {
                        currentNorm = Math.log(current / mapping.min) / Math.log(mapping.max / mapping.min);
                    } else {
                        currentNorm = (current - mapping.min) / span;
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

    const setDevice = useCallback((deviceId) => {
        const found = inputs.find((i) => i.id === deviceId);
        setConfig((prev) => ({
            ...prev,
            deviceId: deviceId || null,
            deviceName: found ? found.name : null,
        }));
    }, [inputs]);

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
        supported,
        permissionError,
        learnMode,
        learnTarget,
        setDevice,
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
        config, inputs, supported, permissionError, learnMode, learnTarget,
        setDevice, setChannelFilter, setTakeover, beginLearn, cancelLearn,
        clearMapping, clearAll, toggleLearnMode, exitLearnMode,
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
                    sx={{
                        position: 'absolute',
                        top: 1,
                        right: 1,
                        fontSize: '0.5rem',
                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                        bgcolor: 'rgba(83, 193, 138, 0.85)',
                        color: '#000',
                        px: 0.4,
                        py: 0.05,
                        borderRadius: 0.5,
                        pointerEvents: 'none',
                        letterSpacing: '0.04em',
                        zIndex: 5,
                    }}
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
                    sx={{
                        position: 'absolute',
                        inset: 0,
                        cursor: 'pointer',
                        zIndex: 20,
                        bgcolor: isLearningThis
                            ? 'rgba(245, 197, 66, 0.32)'
                            : mapping
                                ? 'rgba(83, 193, 138, 0.18)'
                                : 'rgba(245, 197, 66, 0.10)',
                        border: '1px dashed',
                        borderColor: isLearningThis
                            ? '#F5C542'
                            : mapping
                                ? 'rgba(83, 193, 138, 0.7)'
                                : 'rgba(245, 197, 66, 0.65)',
                        borderRadius: 1,
                        animation: isLearningThis ? 'midiPulse 900ms ease-in-out infinite' : 'none',
                        '@keyframes midiPulse': {
                            '0%, 100%': { opacity: 0.5 },
                            '50%': { opacity: 1 },
                        },
                    }}
                >
                    {(mapping || isLearningThis) && (
                        <Box
                            sx={{
                                position: 'absolute',
                                top: 2,
                                left: 2,
                                fontSize: '0.5rem',
                                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                                color: isLearningThis ? '#F5C542' : 'rgba(83, 193, 138, 0.95)',
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
