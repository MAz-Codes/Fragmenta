import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Box,
    Typography,
    Paper,
    Slider,
    Button,
    Alert,
    Divider,
    FormControl,
    FormControlLabel,
    Switch,
    Select,
    MenuItem,
    TextField,
    IconButton,
    Tooltip,
    ButtonBase,
    Menu,
    ListItemText,
    ListSubheader,
} from '@mui/material';
import {
    Play as PlayAllIcon,
    Square as StopAllIcon,
    Trash2 as DeleteIcon,
    Settings as SettingsIcon,
    Save as SaveIcon,
    X as CloseXIcon,
    Headphones as CueIcon,
    Volume2 as AudioSetupIcon,
    RotateCcw as RestoreIcon,
    Download as DownloadIcon,
} from 'lucide-react';
import api from '../api';
import PerformanceChannel from './PerformanceChannel';
import { PerformanceEngine, IMPULSE_RESPONSES, MASTER_DELAY_DIVISIONS } from '../utils/performanceAudio';
import { performancePanelStyles as styles, perfTokens } from '../theme';
import { MidiProvider, MidiMappable, useMidi, clearMidiConfig } from './MidiContext';
import MidiConfigMenu from './MidiConfigMenu';
import { isCueSupported, listOutputDevices, setCueDevice, setCueOutputPair } from '../utils/cueAudio';
import { filterLorasForModel } from '../utils/loraMatch';
import {
    usePerformanceSession,
    listPresetNames,
    savePreset,
    deletePreset,
    loadPresetIntoSession,
    clearPerformanceSession,
} from './usePerformanceSession';
import {
    channelScope,
    presetChannelScope,
    copyScope,
    clearScope as clearTakeScope,
} from '../utils/takeStorage';

const CHANNEL_COUNT = 4;
const MASTER_COLOR = '#35C2D4';
const MASTER_DB_MIN = -60;
const MASTER_DB_MAX = 0;
const MASTER_DB_DEFAULT = -6;
const METER_FLOOR_DB = -60;
const BPM_MIN = 20;
const BPM_MAX = 300;
const BPM_DEFAULT = 120;


const LAUNCH_QUANTIZE_OPTIONS = [
    { value: 0,     label: 'None' },
    { value: 32,    label: '8 Bars' },
    { value: 16,    label: '4 Bars' },
    { value: 8,     label: '2 Bars' },
    { value: 4,     label: '1 Bar' },
    { value: 2,     label: '1/2' },
    { value: 1,     label: '1/4' },
    { value: 0.5,   label: '1/8' },
    { value: 0.25,  label: '1/16' },
    { value: 0.125, label: '1/32' },
];
const LAUNCH_Q_DEFAULT = 4;

const dbToGain = (db) => (db <= MASTER_DB_MIN ? 0 : Math.pow(10, db / 20));
const ampToDb = (amp) => (amp <= 0 ? -Infinity : 20 * Math.log10(amp));
const formatDb = (db) => {
    if (!isFinite(db) || db <= METER_FLOOR_DB) return '−∞';
    if (Math.abs(db) < 0.05) return '0.0';
    return db.toFixed(1);
};

export default function PerformancePanel(props) {
    // Reset key for the inner panel. Bumping it forces a full remount of
    // PerformancePanelInner, which makes usePerformanceSession re-read from
    // localStorage and every PerformanceChannel re-hydrate from IDB. The
    // MidiProvider sits outside so MIDI mappings survive a reset (they
    // have their own clearMidiConfig pathway when needed).
    const [resetKey, setResetKey] = useState(0);
    const triggerReset = useCallback(() => setResetKey((k) => k + 1), []);
    return (
        <MidiProvider>
            <PerformancePanelInner
                key={resetKey}
                {...props}
                onPresetLoaded={triggerReset}
            />
        </MidiProvider>
    );
}

function PerformancePanelInner({
    selectedModel,
    availableModels = [],
    baseModels = [],
    availableLoras = [],
    selectedLora = '',
    loraMultiplier = 1.0,
    onSelectModel,
    onRefreshModels,
    onSelectLora,
    onLoraMultiplierChange,
    steps = 250,
    onStepsChange,
    randomSeed = true,
    seedValue = '',
    onRandomSeedChange,
    onSeedValueChange,
    onPresetLoaded,
    onOpenCheckpointManager,
}) {
    const { session, updateGlobal, updateChannel } = usePerformanceSession(CHANNEL_COUNT);

    const engineRef = useRef(null);
    const meterFillRef = useRef(null);
    const peakHoldRef = useRef({ db: METER_FLOOR_DB, decayedAt: performance.now() });
    const meterRafRef = useRef(null);
    const [engineReady, setEngineReady] = useState(false);
    const [masterDb, setMasterDb] = useState(session.masterDb ?? MASTER_DB_DEFAULT);
    const [bpm, setBpm] = useState(session.bpm ?? BPM_DEFAULT);
    const [bpmInput, setBpmInput] = useState(String(session.bpm ?? BPM_DEFAULT));
    const bpmInputFocusedRef = useRef(false);
    const [error, setError] = useState(null);
    const [linkAvailable, setLinkAvailable] = useState(false);
    const [linkEnabled, setLinkEnabled] = useState(session.linkEnabled ?? false);
    const [linkPeers, setLinkPeers] = useState(0);
    const [linkInstalling, setLinkInstalling] = useState(false);
    const [launchQuantum, setLaunchQuantum] = useState(session.launchQuantum ?? LAUNCH_Q_DEFAULT);
    const wasPlayingRef = useRef(false);
    const bpmOriginRef = useRef('user');
    const [peakLabelDb, setPeakLabelDb] = useState(METER_FLOOR_DB);
    const [channelStates, setChannelStates] = useState(() =>
        Array.from({ length: CHANNEL_COUNT }, () => ({ loaded: false, playing: false }))
    );
    const [promptKey, setPromptKey] = useState(session.promptKey ?? '');
    const [promptInjectBpm, setPromptInjectBpm] = useState(session.promptInjectBpm ?? false);
    const [promptTimeSig, setPromptTimeSig] = useState(session.promptTimeSig ?? '');
    const [masterReverbIR, setMasterReverbIR] = useState(session.masterReverbIR ?? 'hall');
    const [masterDelayDivision, setMasterDelayDivision] = useState(session.masterDelayDivision ?? '1/4');

    // Audio output device. setSinkId requires Chromium ≥ 110 (cueSupported
    // is the runtime check). One device drives BOTH main and cue. Per-pair
    // channel routing within the device is Stage 2 — pair selections are
    // tracked here but the merger plumbing comes later.
    const cueSupported = useMemo(() => isCueSupported(), []);
    const [outputDeviceId, setOutputDeviceId] = useState(session.outputDeviceId ?? session.cueDeviceId ?? '');
    const [audioDevices, setAudioDevices] = useState([]);
    const [audioMenuAnchor, setAudioMenuAnchor] = useState(null);
    const [maxChannelCount, setMaxChannelCount] = useState(2);
    const [mainOutputPair, setMainOutputPair] = useState(session.mainOutputPair ?? 0);
    const [cueOutputPair, setCueOutputPair] = useState(session.cueOutputPair ?? 0);

    useEffect(() => { updateGlobal('outputDeviceId', outputDeviceId); }, [outputDeviceId, updateGlobal]);
    useEffect(() => {
        updateGlobal('mainOutputPair', mainOutputPair);
        engineRef.current?.setMainOutputPair?.(mainOutputPair);
    }, [mainOutputPair, updateGlobal]);
    useEffect(() => {
        updateGlobal('cueOutputPair', cueOutputPair);
        setCueOutputPair(cueOutputPair);
    }, [cueOutputPair, updateGlobal]);

    // When the chosen device changes, bind both the main engine context and
    // the (separate) cue context to it. Re-read maxChannelCount afterwards so
    // the pair selectors populate with everything the device exposes.
    useEffect(() => {
        if (!cueSupported) return;
        let cancelled = false;
        (async () => {
            const engine = engineRef.current;
            if (engine?.setOutputDevice) {
                const max = await engine.setOutputDevice(outputDeviceId);
                if (!cancelled) setMaxChannelCount(max);
            }
            await setCueDevice(outputDeviceId).catch(() => { /* logged in cueAudio */ });
        })();
        return () => { cancelled = true; };
    }, [cueSupported, outputDeviceId]);

    const refreshAudioDevices = useCallback(async () => {
        if (!cueSupported) return;
        try {
            const devices = await listOutputDevices();
            setAudioDevices(devices);
        } catch (err) {
            console.warn('[PerformancePanel] device enumeration failed', err);
        }
    }, [cueSupported]);

    const handleOpenAudioMenu = (e) => {
        setAudioMenuAnchor(e.currentTarget);
        refreshAudioDevices();
    };
    const handleCloseAudioMenu = () => setAudioMenuAnchor(null);
    const handlePickAudioDevice = (id) => {
        setOutputDeviceId(id);
        handleCloseAudioMenu();
    };

    // Restore App-level state (model, steps, seed) once on mount via the setter
    // props the panel was given. The panel doesn't own those, so this is the
    // only point at which we push session into App state. Subsequent changes
    // flow normally through the prop callbacks.
    const appStateRestoredRef = useRef(false);
    useEffect(() => {
        if (appStateRestoredRef.current) return;
        appStateRestoredRef.current = true;
        if (session.selectedModel && session.selectedModel !== selectedModel) {
            onSelectModel?.(session.selectedModel);
        }
        if (typeof session.steps === 'number' && session.steps !== steps) {
            onStepsChange?.(session.steps);
        }
        if (typeof session.randomSeed === 'boolean' && session.randomSeed !== randomSeed) {
            onRandomSeedChange?.(session.randomSeed);
        }
        if (typeof session.seedValue === 'string' && session.seedValue !== seedValue) {
            onSeedValueChange?.(session.seedValue);
        }
        // Intentionally only run on first mount.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Push panel + App-level state into the session whenever any of it changes.
    useEffect(() => { updateGlobal('bpm', bpm); }, [bpm, updateGlobal]);
    useEffect(() => { updateGlobal('launchQuantum', launchQuantum); }, [launchQuantum, updateGlobal]);
    useEffect(() => { updateGlobal('masterDb', masterDb); }, [masterDb, updateGlobal]);
    useEffect(() => { updateGlobal('promptKey', promptKey); }, [promptKey, updateGlobal]);
    useEffect(() => { updateGlobal('promptInjectBpm', promptInjectBpm); }, [promptInjectBpm, updateGlobal]);
    useEffect(() => { updateGlobal('promptTimeSig', promptTimeSig); }, [promptTimeSig, updateGlobal]);
    useEffect(() => {
        updateGlobal('masterReverbIR', masterReverbIR);
        engineRef.current?.setMasterReverbIR?.(masterReverbIR);
    }, [masterReverbIR, updateGlobal]);
    useEffect(() => {
        updateGlobal('masterDelayDivision', masterDelayDivision);
        engineRef.current?.setMasterDelayDivision?.(masterDelayDivision);
    }, [masterDelayDivision, updateGlobal]);
    useEffect(() => { updateGlobal('linkEnabled', linkEnabled); }, [linkEnabled, updateGlobal]);
    useEffect(() => { updateGlobal('selectedModel', selectedModel || ''); }, [selectedModel, updateGlobal]);
    useEffect(() => { updateGlobal('steps', steps); }, [steps, updateGlobal]);
    useEffect(() => { updateGlobal('randomSeed', randomSeed); }, [randomSeed, updateGlobal]);
    useEffect(() => { updateGlobal('seedValue', seedValue); }, [seedValue, updateGlobal]);

    const handleChannelFormChange = useCallback((index, partial) => {
        updateChannel(index, partial);
    }, [updateChannel]);

    // ---- Preset menu state ----
    const [presetMenuAnchor, setPresetMenuAnchor] = useState(null);
    const [presetNames, setPresetNames] = useState(() => listPresetNames());
    const [saveAsName, setSaveAsName] = useState('');
    const [restoreArmed, setRestoreArmed] = useState(false);
    const restoreArmTimerRef = useRef(null);

    const refreshPresetNames = useCallback(() => {
        setPresetNames(listPresetNames());
    }, []);

    const openPresetMenu = (e) => {
        refreshPresetNames();
        setSaveAsName('');
        setRestoreArmed(false);
        setPresetMenuAnchor(e.currentTarget);
    };
    const closePresetMenu = () => {
        setPresetMenuAnchor(null);
        setRestoreArmed(false);
        if (restoreArmTimerRef.current) {
            clearTimeout(restoreArmTimerRef.current);
            restoreArmTimerRef.current = null;
        }
    };

    const handleRestoreDefaults = async () => {
        if (!restoreArmed) {
            // First click arms; second click within 3 s commits. Disarms
            // automatically so the destructive path is never one accidental
            // click away.
            setRestoreArmed(true);
            if (restoreArmTimerRef.current) clearTimeout(restoreArmTimerRef.current);
            restoreArmTimerRef.current = setTimeout(() => setRestoreArmed(false), 3000);
            return;
        }
        clearPerformanceSession();
        clearMidiConfig();
        // Drop every channel's take blobs from IDB so a fresh start is
        // actually fresh. Presets keep their own scopes and survive.
        await Promise.all(
            Array.from({ length: CHANNEL_COUNT }, (_, i) =>
                clearTakeScope(channelScope(i)).catch(() => { /* ignore */ })
            )
        );
        closePresetMenu();
        onPresetLoaded?.();
    };

    const handleSaveAs = async () => {
        const name = saveAsName.trim();
        if (!name) return;
        savePreset(name, session);
        // Copy each channel's session-scope blobs into the preset-scope so
        // the preset's takes survive overwrites of the live session. Done
        // after the metadata save so a quota failure here still leaves a
        // recoverable (if blob-less) preset entry.
        await Promise.all(
            Array.from({ length: CHANNEL_COUNT }, async (_, i) => {
                const dst = presetChannelScope(name, i);
                // Replace, don't merge — a re-save of the same preset name
                // should reflect the current session exactly.
                await clearTakeScope(dst).catch(() => { /* ignore */ });
                await copyScope(channelScope(i), dst).catch(() => { /* ignore */ });
            })
        );
        setSaveAsName('');
        refreshPresetNames();
    };

    const handleLoadPreset = async (name) => {
        if (!loadPresetIntoSession(name)) return;
        // Swap the IDB session-scope blobs to match the loaded preset's
        // metadata. MUST complete before onPresetLoaded triggers remount —
        // otherwise the new channels hydrate from a stale session scope.
        await Promise.all(
            Array.from({ length: CHANNEL_COUNT }, async (_, i) => {
                const dst = channelScope(i);
                await clearTakeScope(dst).catch(() => { /* ignore */ });
                await copyScope(presetChannelScope(name, i), dst).catch(() => { /* ignore */ });
            })
        );
        closePresetMenu();
        // Force-remount via the App-level reset key. Same pathway as Fresh
        // Start, just with a different localStorage payload pre-loaded.
        onPresetLoaded?.();
    };

    const handleDeletePreset = async (name, e) => {
        e?.stopPropagation();
        deletePreset(name);
        await Promise.all(
            Array.from({ length: CHANNEL_COUNT }, (_, i) =>
                clearTakeScope(presetChannelScope(name, i)).catch(() => { /* ignore */ })
            )
        );
        refreshPresetNames();
    };

    // Resolve which SA3 base the current selection maps to. Direct picks of
    // `sa3-*` models are themselves the base; fine-tuned models carry their
    // base_model in training_metadata.json (exposed via /api/models).
    const resolvedBaseModel = (() => {
        if (!selectedModel) return null;
        if (selectedModel.startsWith('sa3-')) return selectedModel;
        const model = availableModels.find((m) => m.name === selectedModel);
        return model?.base_model || null;
    })();

    const isSmallModel = !!resolvedBaseModel && resolvedBaseModel.startsWith('sa3-small-');
    // Distilled (post-trained) SA3 variants — names that DON'T end in `-base`.
    // The Steps dropdown only locks at 8 for these; the *-base checkpoints let
    // the user pick a real step count.
    const isDistilledSA3 = !!resolvedBaseModel
        && resolvedBaseModel.startsWith('sa3-')
        && !resolvedBaseModel.endsWith('-base');

    // Split baseModels by `kind` for the model-picker grouping. The render
    // helper is also hoisted to component scope so its MenuItems land as
    // direct children of <Select>, not nested inside a Fragment.
    const distilledBaseModels = baseModels.filter(m => m.kind !== 'base');
    const trueBaseModels = baseModels.filter(m => m.kind === 'base');
    const renderBaseModelRow = (model) => (
        <MenuItem
            key={model.name}
            value={String(model.name)}
            disabled={!model.downloaded}
            sx={{
                fontSize: perfTokens.fontSize.sm,
                display: 'flex',
                justifyContent: 'space-between',
                gap: 1,
                pr: 0.5,
            }}
        >
            <Box component="span" sx={{
                flex: 1,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            }}>
                {model.displayName || model.name}
            </Box>
            {!model.downloaded && (
                <Tooltip title="Not downloaded — open Checkpoint Manager">
                    <IconButton
                        size="small"
                        onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                        onClick={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            onOpenCheckpointManager?.();
                        }}
                        sx={{
                            ...styles.compactIconBtn('md'),
                            // Override the MenuItem-disabled pointer-events:none
                            // so the download button stays clickable, and the
                            // dimmed opacity so it reads as actionable.
                            pointerEvents: 'auto',
                            opacity: 1,
                        }}
                    >
                        <DownloadIcon size={perfTokens.icon.md} />
                    </IconButton>
                </Tooltip>
            )}
        </MenuItem>
    );

    if (!engineRef.current) {
        engineRef.current = new PerformanceEngine(CHANNEL_COUNT);
        engineRef.current.setMasterGain(dbToGain(MASTER_DB_DEFAULT));
    }

    useEffect(() => {
        setEngineReady(true);
        const engine = engineRef.current;

        const tick = () => {
            const now = performance.now();
            const peakAmp = engine ? engine.getMasterPeak() : 0;
            const instantDb = ampToDb(peakAmp);
            const hold = peakHoldRef.current;
            const elapsed = now - hold.decayedAt;
            let displayDb = hold.db - (elapsed / 1000) * 24;
            if (instantDb > displayDb) displayDb = instantDb;
            if (displayDb < METER_FLOOR_DB) displayDb = METER_FLOOR_DB;
            hold.db = displayDb;
            hold.decayedAt = now;

            const fill = meterFillRef.current;
            if (fill) {
                const pct = ((displayDb - METER_FLOOR_DB) / -METER_FLOOR_DB) * 100;
                fill.style.height = `${Math.max(0, Math.min(100, pct))}%`;
            }
            if (Math.abs(displayDb - peakLabelDb) > 0.5) {
                setPeakLabelDb(displayDb);
            }
            meterRafRef.current = requestAnimationFrame(tick);
        };
        meterRafRef.current = requestAnimationFrame(tick);

        return () => {
            if (meterRafRef.current) cancelAnimationFrame(meterRafRef.current);
            engine.dispose();
            engineRef.current = null;
        };
    }, []);

    const handleMasterChange = (_, value) => {
        setMasterDb(value);
        engineRef.current?.setMasterGain(dbToGain(value));
    };

    useEffect(() => {
        engineRef.current?.setBpm(bpm);
    }, [bpm]);

    const handleBpmChange = (event) => {
        const raw = event.target.value;
        setBpmInput(raw);
        const parsed = Number(raw);
        if (Number.isFinite(parsed) && parsed >= BPM_MIN && parsed <= BPM_MAX) {
            bpmOriginRef.current = 'user';
            setBpm(Math.round(parsed));
        }
    };

    const handleBpmBlur = () => {
        bpmInputFocusedRef.current = false;
        const parsed = Number(bpmInput);
        if (!Number.isFinite(parsed) || bpmInput.trim() === '') {
            // Non-numeric or empty — restore the committed value.
            setBpmInput(String(bpm));
            return;
        }
        const clamped = Math.max(BPM_MIN, Math.min(BPM_MAX, Math.round(parsed)));
        bpmOriginRef.current = 'user';
        setBpm(clamped);
        setBpmInput(String(clamped));
    };

    const handleBpmFocus = () => {
        bpmInputFocusedRef.current = true;
    };

    useEffect(() => {
        if (!bpmInputFocusedRef.current) setBpmInput(String(bpm));
    }, [bpm]);

    useEffect(() => {
        api.get('/api/link/state')
            .then((r) => {
                setLinkAvailable(Boolean(r.data?.available));
                setLinkEnabled(Boolean(r.data?.enabled));
            })
            .catch(() => setLinkAvailable(false));
    }, []);


    useEffect(() => {
        if (!linkEnabled) {
            engineRef.current?.setLinkSnapshot(null);
            wasPlayingRef.current = false;
            return undefined;
        }
        let cancelled = false;
        const poll = async () => {
            const capturedAt = performance.now();
            try {
                const r = await api.get('/api/link/state');
                if (cancelled || !r.data?.enabled) return;
                const serverBpm = Math.round(r.data.bpm);
                if (serverBpm >= BPM_MIN && serverBpm <= BPM_MAX) {
                    setBpm((prev) => {
                        if (prev === serverBpm) return prev;
                        bpmOriginRef.current = 'link';
                        return serverBpm;
                    });
                }
                setLinkPeers(Number(r.data.num_peers || 0));

                const isPlaying = Boolean(r.data.is_playing);
                const beat = Number(r.data.beat) || 0;
                const bpmFloat = Number(r.data.bpm) || 120;
                engineRef.current?.setLinkSnapshot({
                    beat,
                    bpm: bpmFloat,
                    isPlaying,
                    capturedAt,
                });
                
                if (wasPlayingRef.current && !isPlaying) {
                    engineRef.current?.stopAll();
                    setChannelStates(prev => prev.map(s => ({ ...s, playing: false })));
                }
                wasPlayingRef.current = isPlaying;
            } catch {
                /* transient network blip — next tick will retry */
            }
        };
        poll();
        const timer = setInterval(poll, 500);
        return () => { cancelled = true; clearInterval(timer); };
    }, [linkEnabled]);

    useEffect(() => {
        engineRef.current?.setLaunchQuantum(launchQuantum);
    }, [launchQuantum]);

    useEffect(() => {
        if (!linkEnabled) return;
        if (bpmOriginRef.current === 'link') {
            bpmOriginRef.current = 'user';
            return;
        }
        api.post('/api/link/bpm', { bpm }).catch(() => {});
    }, [bpm, linkEnabled]);

    const handleToggleLink = useCallback(async () => {
        // First click when Link isn't installed: offer to install it.
        if (!linkAvailable) {
            const confirmed = window.confirm(
                'Ableton Link requires the LinkPython-extern package (~1–2 MB, ~30s install).\n\n'
                + 'Install it now? You\'ll only need to do this once.'
            );
            if (!confirmed) return;
            setLinkInstalling(true);
            try {
                await api.post('/api/link/install');
                setLinkAvailable(true);
                await api.post('/api/link/enable');
                setLinkEnabled(true);
            } catch (err) {
                const msg = err?.response?.data?.error || err?.response?.data?.detail || err.message || 'Install failed';
                setError(`Link install failed: ${msg}`);
            } finally {
                setLinkInstalling(false);
            }
            return;
        }

        try {
            if (linkEnabled) {
                await api.post('/api/link/disable');
                setLinkEnabled(false);
                setLinkPeers(0);
            } else {
                await api.post('/api/link/enable');
                setLinkEnabled(true);
            }
        } catch (err) {
            const status = err?.response?.status;
            const msg = err?.response?.data?.error || err.message || 'Link toggle failed';
            if (status === 503) setLinkAvailable(false);
            setError(msg);
        }
    }, [linkEnabled, linkAvailable]);

    const handlePlayAll = () => {
        engineRef.current?.playAll(true);
        setChannelStates(prev => prev.map(s => (s.loaded ? { ...s, playing: true } : s)));
    };
    const handleStopAll = () => {
        engineRef.current?.stopAll();
        setChannelStates(prev => prev.map(s => ({ ...s, playing: false })));
    };

    const applyExternalBpm = useCallback((value) => {
        const next = Math.max(BPM_MIN, Math.min(BPM_MAX, Math.round(value)));
        bpmOriginRef.current = 'user';
        setBpm(next);
    }, []);

    const midi = useMidi();
    const [midiMenuAnchor, setMidiMenuAnchor] = useState(null);

    useEffect(() => {
        if (!midi?.learnMode) return undefined;
        const onKey = (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                midi.exitLearnMode();
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [midi?.learnMode, midi?.exitLearnMode]);

    const generateForChannel = async ({ prompt, duration, alignBars, alignBpm, loopStitch, batchSize = 1, onBlob }) => {
        setError(null);
        if (!selectedModel) {
            const msg = 'Pick a model first.';
            setError(msg);
            throw new Error(msg);
        }

        // Auto-inject Key / BPM / Time sig when populated. Skip a field if
        // its value already appears in the user's prompt so we don't double
        // up. BPM is a toggle — when on we grab the live master BPM from
        // the top bar (so it tracks tempo changes).
        const trimmed = (prompt || '').trim();
        const lower = trimmed.toLowerCase();
        const additions = [];
        if (promptKey.trim() && !lower.includes(promptKey.trim().toLowerCase())) {
            additions.push(promptKey.trim());
        }
        if (promptInjectBpm && !/\b\d{2,3}\s*bpm\b/i.test(trimmed)) {
            additions.push(`${Math.round(bpm)} BPM`);
        }
        if (promptTimeSig.trim() && !lower.includes(promptTimeSig.trim().toLowerCase())) {
            additions.push(promptTimeSig.trim());
        }
        const finalPrompt = additions.length > 0
            ? `${trimmed}${trimmed ? ', ' : ''}${additions.join(', ')}`
            : trimmed;

        let baseSeed;
        if (randomSeed) {
            baseSeed = Math.floor(Math.random() * 0xffffffff);
        } else {
            const parsed = parseInt(seedValue, 10);
            if (Number.isNaN(parsed) || parsed < 0) {
                const msg = 'Enter a valid seed (0 or greater) or enable Random.';
                setError(msg);
                throw new Error(msg);
            }
            baseSeed = parsed;
        }

        const count = Math.max(1, Math.min(4, batchSize | 0));
        for (let i = 0; i < count; i++) {
            // Sequential rather than parallel — the backend serves one
            // generation at a time anyway, and parallelizing would just
            // queue them server-side with no time saved. Each take gets a
            // distinct seed so the batch produces actual variations rather
            // than the same audio repeated.
            const seed = (baseSeed + i * 0x9e3779b1) >>> 0;
            const isSA3Base = baseModels.some(m => m.name === selectedModel);
            const isDistilled = isSA3Base && !selectedModel.endsWith('-base');
            const requestData = {
                prompt: finalPrompt,
                duration,
                seed,
                model_id: selectedModel,
                // CFG is only meaningful for *-base; backend ignores it on
                // distilled models. Steps default to 8 for distilled, 50 for
                // base — only send the override when we're on base.
                ...(isDistilled ? {} : { steps, cfg_scale: 7.0 }),
                // LoRA stacking (Phase 4): only attach when the user picked
                // a LoRA AND the active model is a *-base variant (the only
                // architecturally valid target).
                ...(selectedLora && isSA3Base && !isDistilled
                    ? { loras: [{ path: selectedLora, strength: loraMultiplier }] }
                    : {}),
                ...(alignBars && alignBpm ? { align_bars: alignBars, align_bpm: alignBpm } : {}),
                // Phase 7: seamless looping. Bars-mode + channel-looping
                // signals "I want a loop"; backend wrap-inpaints the seam.
                ...(loopStitch && alignBars && alignBpm ? { loop_stitch: loopStitch } : {}),
            };
            const response = await api.post('/api/generate', requestData, { responseType: 'blob' });
            // Stream: hand each blob to the caller as it arrives so the take
            // can land in the channel's history (and the first one can
            // auto-load) without waiting for the rest of the batch. Awaited
            // so the callback's async work (blob load, setState) finishes
            // before the next backend round-trip starts.
            await onBlob?.(response.data, i);
        }
    };

    const handleChannelStateChange = (index, change) => {
        setChannelStates(prev => {
            const next = [...prev];
            next[index] = { ...next[index], ...change };
            return next;
        });
    };

    const anyLoaded = channelStates.some(s => s.loaded);
    const anyPlaying = channelStates.some(s => s.playing);
    // SA3 max-duration limits (match _MODEL_INFO in audio_generator.py and
    // max_duration_sec in the Checkpoint Manager catalog).
    const maxDuration = isSmallModel ? 120 : 380;

    const handleMuteSoloChange = (index, change) => {
        const engine = engineRef.current;
        if (!engine) return;
        if ('mute' in change) engine.setMute(index, change.mute);
        if ('solo' in change) engine.setSolo(index, change.solo);
    };

    const channels = useMemo(() => {
        if (!engineReady || !engineRef.current) return [];
        return engineRef.current.channels;
    }, [engineReady]);

    const handleModelChange = (event) => {
        const value = event.target.value;
        if (onSelectModel) onSelectModel(value);
    };

    const handleDeleteFineTuned = async (modelName) => {
        const confirmed = window.confirm(
            `Delete fine-tuned model "${modelName}"? This removes the directory and all its checkpoints. This cannot be undone.`
        );
        if (!confirmed) return;
        try {
            await api.delete(`/api/models/fine-tuned/${encodeURIComponent(modelName)}`);
            if (selectedModel === modelName) {
                onSelectModel?.('');
            }
            onRefreshModels?.();
        } catch (err) {
            const msg = err?.response?.data?.error || err.message || 'Delete failed';
            setError(`Failed to delete "${modelName}": ${msg}`);
        }
    };

    // LoRAs share the same on-disk layout as fine-tuned models (one dir under
    // models/fine_tuned/<name>/), so the same DELETE endpoint handles both.
    const handleDeleteLora = async (loraName) => {
        const confirmed = window.confirm(
            `Delete LoRA "${loraName}"? This removes the directory and all its checkpoints. This cannot be undone.`
        );
        if (!confirmed) return;
        try {
            await api.delete(`/api/models/fine-tuned/${encodeURIComponent(loraName)}`);
            // Clear the active LoRA if it points anywhere inside the deleted dir.
            const deleted = availableLoras.find(l => l.name === loraName);
            const paths = deleted ? (deleted.all_checkpoints || [deleted.path]) : [];
            if (paths.includes(selectedLora)) onSelectLora?.('');
            onRefreshModels?.();
        } catch (err) {
            const msg = err?.response?.data?.error || err.message || 'Delete failed';
            setError(`Failed to delete LoRA "${loraName}": ${msg}`);
        }
    };

    return (
        <Box sx={styles.root}>
            <Paper sx={{
                ...styles.barCard,
                gap: 1,
                // Top-only drop shadow — keeps the bar visually lifted from
                // the app surface above it without casting darkness down onto
                // the channels grid below. Negative Y inverts the direction
                // of MUI's default Paper elevation.
                boxShadow: '0 -2px 6px rgba(0, 0, 0, 0.12)',
            }}>
                {/* Link — compact rectangle, Ableton-style */}
                <Tooltip
                    title={
                        linkInstalling
                            ? 'Installing LinkPython-extern…'
                            : !linkAvailable
                                ? 'Click to install Ableton Link script'
                                : linkEnabled
                                    ? `Link on — ${linkPeers} peer${linkPeers === 1 ? '' : 's'} (click to disable)`
                                    : 'Click to sync BPM with other Link-enabled apps on this network'
                    }
                >
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                        <ButtonBase
                            onClick={handleToggleLink}
                            disabled={linkInstalling}
                            sx={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: perfTokens.fontSize.sm,
                                fontWeight: 600,
                                px: 0.5,
                                minWidth: 38,
                                height: perfTokens.height.compact,
                                borderRadius: '2px',
                                // Three states matching Ableton Live's button:
                                // off → gray, on/no peers → yellow (broadcasting,
                                // alone), on/peers → teal (sync'd with at least
                                // one other app).
                                bgcolor: !linkEnabled
                                    ? '#6e6e6e'
                                    : linkPeers > 0
                                        ? MASTER_COLOR
                                        : '#F5C542',
                                color: linkEnabled ? '#000' : '#2a2a2a',
                                opacity: linkInstalling ? 0.55 : 1,
                                transition: 'background-color 120ms',
                                '&:hover': {
                                    bgcolor: !linkEnabled
                                        ? '#7d7d7d'
                                        : linkPeers > 0
                                            ? '#4DD0DE'
                                            : '#FFD54F',
                                },
                                '&.Mui-disabled': {
                                    color: linkEnabled ? '#000' : '#2a2a2a',
                                },
                            }}
                        >
                            {linkInstalling
                                ? 'Installing…'
                                : linkEnabled && linkPeers > 0
                                    ? `${linkPeers} Link`
                                    : 'Link'}
                        </ButtonBase>
                    </span>
                </Tooltip>

                {/* MIDI learn toggle — same compact rectangle style as Link. */}
                <Tooltip
                    title={
                        !midi?.supported
                            ? (midi?.permissionError || 'Web MIDI is not available')
                            : midi.learnMode
                                ? 'Exit MIDI mode (Esc)'
                                : 'Enter MIDI mode — click a control then move a hardware knob/button to bind'
                    }
                >
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                        <ButtonBase
                            onClick={() => midi?.toggleLearnMode()}
                            disabled={!midi?.supported}
                            sx={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: perfTokens.fontSize.sm,
                                fontWeight: 600,
                                px: 0.5,
                                minWidth: 38,
                                height: perfTokens.height.compact,
                                borderRadius: '2px',
                                bgcolor: midi?.learnMode ? '#F5C542' : '#6e6e6e',
                                color: midi?.learnMode ? '#000' : '#2a2a2a',
                                opacity: midi?.supported ? 1 : 0.45,
                                transition: 'background-color 120ms',
                                '&:hover': {
                                    bgcolor: midi?.learnMode ? '#FFD54F' : '#7d7d7d',
                                },
                                '&.Mui-disabled': {
                                    color: '#2a2a2a',
                                },
                            }}
                        >
                            MIDI
                        </ButtonBase>
                    </span>
                </Tooltip>
                <Tooltip title="MIDI settings & mappings">
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                        <IconButton
                            size="small"
                            onClick={(e) => setMidiMenuAnchor(e.currentTarget)}
                            sx={styles.compactIconBtn('lg')}
                        >
                            <SettingsIcon size={perfTokens.icon.md} />
                        </IconButton>
                    </span>
                </Tooltip>
                <MidiConfigMenu
                    anchorEl={midiMenuAnchor}
                    open={Boolean(midiMenuAnchor)}
                    onClose={() => setMidiMenuAnchor(null)}
                />

                <Tooltip
                    title={cueSupported
                        ? 'Audio setup — choose output device'
                        : 'Audio device selection requires Chrome/Edge (AudioContext.setSinkId). Output falls back to system default.'}
                >
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                        <IconButton
                            size="small"
                            onClick={handleOpenAudioMenu}
                            disabled={!cueSupported}
                            sx={styles.compactIconBtn('lg')}
                        >
                            <AudioSetupIcon size={perfTokens.icon.md} />
                        </IconButton>
                    </span>
                </Tooltip>
                <Menu
                    anchorEl={audioMenuAnchor}
                    open={Boolean(audioMenuAnchor)}
                    onClose={handleCloseAudioMenu}
                    MenuListProps={{ sx: { py: 0 } }}
                    PaperProps={{
                        sx: {
                            width: 300,
                            borderRadius: 2,
                            border: '1px solid',
                            borderColor: 'divider',
                        },
                    }}
                >
                    {/* Title bar — matches the Presets / MIDI menus. */}
                    <Box sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        px: 1.5,
                        pt: 1.25,
                        pb: 1,
                    }}>
                        <Typography sx={{ ...perfTokens.caps, color: 'text.secondary' }}>
                            Audio Output
                        </Typography>
                        <IconButton onClick={handleCloseAudioMenu} sx={styles.compactIconBtn('md')}>
                            <CloseXIcon size={perfTokens.icon.sm} />
                        </IconButton>
                    </Box>

                    <Divider />

                    <Box sx={{ px: 1.5, pt: 1.25, pb: 0.5 }}>
                        <Typography sx={{ ...perfTokens.labelMuted, display: 'block' }}>
                            Output device
                        </Typography>
                    </Box>
                    <Box sx={{ pb: 0.75, px: 0.75 }}>
                        <MenuItem
                            onClick={() => handlePickAudioDevice('')}
                            selected={outputDeviceId === ''}
                            sx={{
                                borderRadius: 1,
                                py: 0.5,
                                px: 1,
                                fontSize: perfTokens.fontSize.sm,
                                fontWeight: 500,
                            }}
                        >
                            <Box component="span" sx={{
                                flex: 1,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                            }}>
                                System device (default)
                            </Box>
                        </MenuItem>
                        {audioDevices.length === 0 && (
                            <Box sx={{ px: 1, py: 0.5 }}>
                                <Typography sx={{
                                    fontSize: perfTokens.fontSize.xs,
                                    color: 'text.disabled',
                                    fontStyle: 'italic',
                                }}>
                                    No additional output devices detected
                                </Typography>
                            </Box>
                        )}
                        {audioDevices.map(d => (
                            <MenuItem
                                key={d.deviceId}
                                onClick={() => handlePickAudioDevice(d.deviceId)}
                                selected={outputDeviceId === d.deviceId}
                                sx={{
                                    borderRadius: 1,
                                    py: 0.5,
                                    px: 1,
                                    fontSize: perfTokens.fontSize.sm,
                                    fontWeight: 500,
                                }}
                            >
                                <Box component="span" sx={{
                                    flex: 1,
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    whiteSpace: 'nowrap',
                                }}>
                                    {d.label || `Output (${d.deviceId.slice(0, 6)}…)`}
                                </Box>
                            </MenuItem>
                        ))}
                    </Box>
                </Menu>

                <Tooltip title="Save / load presets">
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                        <IconButton
                            size="small"
                            onClick={openPresetMenu}
                            sx={styles.compactIconBtn('lg')}
                        >
                            <SaveIcon size={perfTokens.icon.md} />
                        </IconButton>
                    </span>
                </Tooltip>
                <Menu
                    anchorEl={presetMenuAnchor}
                    open={Boolean(presetMenuAnchor)}
                    onClose={closePresetMenu}
                    MenuListProps={{ sx: { py: 0 } }}
                    PaperProps={{
                        sx: {
                            width: 300,
                            borderRadius: 2,
                            border: '1px solid',
                            borderColor: 'divider',
                        },
                    }}
                >
                    {/* Title bar — matches the MIDI settings popover so the two
                        top-bar menus speak the same visual language. */}
                    <Box sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        px: 1.5,
                        pt: 1.25,
                        pb: 1,
                    }}>
                        <Typography sx={{ ...perfTokens.caps, color: 'text.secondary' }}>
                            Presets
                        </Typography>
                        <IconButton onClick={closePresetMenu} sx={styles.compactIconBtn('md')}>
                            <CloseXIcon size={perfTokens.icon.sm} />
                        </IconButton>
                    </Box>

                    <Divider />

                    {/* SAVE — type a name, Enter or click Save. */}
                    <Box sx={{ px: 1.5, pt: 1.25, pb: 1.25 }}>
                        <Typography sx={{ ...perfTokens.labelMuted, display: 'block', mb: 0.75 }}>
                            Save current session as
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                            <TextField
                                autoFocus
                                size="small"
                                placeholder="Preset name"
                                value={saveAsName}
                                onChange={(e) => setSaveAsName(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter') handleSaveAs();
                                    e.stopPropagation();
                                }}
                                sx={{
                                    flex: 1,
                                    '& .MuiOutlinedInput-root': {
                                        borderRadius: 1.5,
                                        height: perfTokens.height.compact,
                                        fontSize: perfTokens.fontSize.sm,
                                    },
                                }}
                            />
                            <Button
                                size="small"
                                variant="contained"
                                onClick={handleSaveAs}
                                disabled={!saveAsName.trim()}
                                sx={{
                                    height: perfTokens.height.compact,
                                    fontSize: perfTokens.fontSize.sm,
                                    fontWeight: 600,
                                    borderRadius: 1.5,
                                    minWidth: 60,
                                    px: 1.5,
                                    textTransform: 'none',
                                }}
                            >
                                Save
                            </Button>
                        </Box>
                        {saveAsName.trim() && presetNames.includes(saveAsName.trim()) && (
                            <Typography sx={{
                                fontSize: perfTokens.fontSize.xs,
                                color: 'warning.main',
                                fontStyle: 'italic',
                                mt: 0.5,
                            }}>
                                Will overwrite existing preset.
                            </Typography>
                        )}
                    </Box>

                    <Divider />

                    {/* LOAD — saved presets. */}
                    <Box sx={{
                        px: 1.5,
                        pt: 1.25,
                        pb: presetNames.length === 0 ? 1.25 : 0.5,
                    }}>
                        <Typography sx={{
                            ...perfTokens.labelMuted,
                            display: 'block',
                            mb: presetNames.length === 0 ? 0.5 : 0,
                        }}>
                            Saved presets
                        </Typography>
                        {presetNames.length === 0 && (
                            <Typography sx={{
                                fontSize: perfTokens.fontSize.xs,
                                color: 'text.disabled',
                                fontStyle: 'italic',
                            }}>
                                No presets saved yet
                            </Typography>
                        )}
                    </Box>
                    {presetNames.length > 0 && (
                        <Box sx={{ pb: 0.75, px: 0.75 }}>
                            {presetNames.map((name) => (
                                <MenuItem
                                    key={name}
                                    onClick={() => handleLoadPreset(name)}
                                    sx={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        gap: 1,
                                        borderRadius: 1,
                                        py: 0.5,
                                        px: 1,
                                        fontSize: perfTokens.fontSize.sm,
                                        fontWeight: 500,
                                    }}
                                >
                                    <Box component="span" sx={{
                                        flex: 1,
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap',
                                    }}>
                                        {name}
                                    </Box>
                                    <Tooltip title="Delete preset" placement="left" arrow>
                                        <IconButton
                                            size="small"
                                            onClick={(e) => handleDeletePreset(name, e)}
                                            sx={{ ...styles.compactIconBtn('md', 'danger'), ml: 1 }}
                                        >
                                            <CloseXIcon size={perfTokens.icon.sm} />
                                        </IconButton>
                                    </Tooltip>
                                </MenuItem>
                            ))}
                        </Box>
                    )}

                    <Divider />

                    {/* Danger zone — armed-to-confirm wipes session config, takes,
                        and MIDI mappings. Auto-disarms after 3 s. */}
                    <Box sx={{ px: 0.75, py: 0.75 }}>
                        <Tooltip
                            title={restoreArmed
                                ? 'Click again within 3s to confirm — clears session, takes, and MIDI mappings'
                                : 'Reset all panel settings, clear takes, and clear MIDI mappings'}
                            placement="left"
                            arrow
                        >
                            <MenuItem
                                onClick={handleRestoreDefaults}
                                sx={(theme) => ({
                                    borderRadius: 1,
                                    py: 0.5,
                                    px: 1,
                                    fontSize: perfTokens.fontSize.sm,
                                    fontWeight: 500,
                                    color: restoreArmed ? theme.palette.error.main : 'text.secondary',
                                    bgcolor: restoreArmed ? `${theme.palette.error.main}14` : 'transparent',
                                    '&:hover': {
                                        bgcolor: restoreArmed
                                            ? `${theme.palette.error.main}26`
                                            : 'action.hover',
                                        color: restoreArmed ? theme.palette.error.main : 'text.primary',
                                    },
                                    transition: 'background-color 120ms, color 120ms',
                                })}
                            >
                                <RestoreIcon size={perfTokens.icon.sm} style={{ marginRight: 8, flexShrink: 0 }} />
                                {restoreArmed ? 'Click again to confirm' : 'Restore defaults'}
                            </MenuItem>
                        </Tooltip>
                    </Box>
                </Menu>

                <Tooltip placement="right" title="Launch quantization — match Live's">
                    <FormControl size="small" sx={{ ...styles.pillControl, minWidth: 92 }}>
                        <Select
                            value={launchQuantum}
                            onChange={(e) => setLaunchQuantum(Number(e.target.value))}
                            renderValue={(val) => {
                                const opt = LAUNCH_QUANTIZE_OPTIONS.find((o) => o.value === val);
                                return `Q · ${opt?.label ?? 'None'}`;
                            }}
                        >
                            {LAUNCH_QUANTIZE_OPTIONS.map((opt) => (
                                <MenuItem key={opt.value} value={opt.value} sx={{ fontSize: perfTokens.fontSize.sm }}>
                                    {opt.label}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Tooltip>

                <MidiMappable
                    id="master.bpm"
                    label="Tempo (BPM)"
                    kind="continuous"
                    min={BPM_MIN}
                    max={BPM_MAX}
                    value={bpm}
                    onChange={applyExternalBpm}
                >
                    <TextField
                        size="small"
                        type="number"
                        value={bpmInput}
                        onChange={handleBpmChange}
                        onFocus={handleBpmFocus}
                        onBlur={handleBpmBlur}
                        // inputProps.style wins against MUI's
                        // .MuiInputBase-inputSizeSmall 14px default — needed
                        // because the pillControl theme can't reach the
                        // rendered <input> at the same CSS specificity.
                        inputProps={{
                            step: 1,
                            inputMode: 'numeric',
                            'aria-label': 'Tempo in BPM',
                            style: {
                                textAlign: 'right',
                                fontVariantNumeric: 'tabular-nums',
                                fontSize: perfTokens.fontSize.sm,
                                fontWeight: perfTokens.weight.bold,
                                paddingRight: 0,
                            },
                        }}
                        InputProps={{
                            endAdornment: (
                                <Box component="span" sx={{ ...perfTokens.caps, color: 'text.disabled', pl: 0.5, userSelect: 'none' }}>
                                    BPM
                                </Box>
                            ),
                        }}
                        sx={{
                            ...styles.pillControl,
                            width: 92,
                            '& .MuiOutlinedInput-root': {
                                ...styles.pillControl['& .MuiOutlinedInput-root'],
                                pr: 1,
                            },
                            '& input::-webkit-outer-spin-button, & input::-webkit-inner-spin-button': {
                                WebkitAppearance: 'none',
                                margin: 0,
                            },
                            '& input[type=number]': { MozAppearance: 'textfield' },
                        }}
                    />
                </MidiMappable>

                {/* Model picker + FT-checkpoint + LoRA pickers now live in the
                    bottom bar so the top strip stays just BPM + transport. */}

                <MidiMappable id="master.playAll" label="Play All" kind="trigger" onChange={handlePlayAll}>
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<PlayAllIcon size={14} />}
                        onClick={handlePlayAll}
                        disabled={!anyLoaded}
                        sx={styles.masterBtn(MASTER_COLOR, 'play')}
                    >
                        Play All
                    </Button>
                </MidiMappable>
                <MidiMappable id="master.stopAll" label="Stop All" kind="trigger" onChange={handleStopAll}>
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<StopAllIcon size={14} />}
                        onClick={handleStopAll}
                        disabled={!anyPlaying}
                        sx={styles.masterBtn(MASTER_COLOR, 'stop')}
                    >
                        Stop All
                    </Button>
                </MidiMappable>

                {/* Main / Cue output channel-pair selectors. Pushed to the right
                    edge of the bar via ml: 'auto' so the transport controls keep
                    their cluster on the left. Pairs are derived from the current
                    device's maxChannelCount; Stage 1 stores the selection,
                    Stage 2 wires the ChannelMergerNode routing. */}
                {(() => {
                    const pairCount = Math.max(1, Math.floor(maxChannelCount / 2));
                    const pairs = Array.from({ length: pairCount }, (_, i) => ({
                        value: i,
                        label: `${i * 2 + 1}–${i * 2 + 2}`,
                    }));
                    const fieldGroup = (label, value, onChange) => (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography component="span" sx={perfTokens.labelMuted}>{label}</Typography>
                            <FormControl size="small" sx={{ ...styles.pillControl, width: 84 }}>
                                <Select value={value} onChange={onChange}>
                                    {pairs.map(p => (
                                        <MenuItem key={p.value} value={p.value} sx={{ fontSize: perfTokens.fontSize.sm }}>
                                            Ch {p.label}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Box>
                    );
                    return (
                        <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 1 }}>
                            {fieldGroup(
                                'Main out',
                                Math.min(mainOutputPair, pairCount - 1),
                                (e) => setMainOutputPair(Number(e.target.value)),
                            )}
                            {fieldGroup(
                                'Cue out',
                                Math.min(cueOutputPair, pairCount - 1),
                                (e) => setCueOutputPair(Number(e.target.value)),
                            )}
                        </Box>
                    );
                })()}
            </Paper>

            {error && (
                <Alert severity="warning" sx={styles.errorAlert} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}

            <Box sx={styles.channelsRow}>
                <Box sx={styles.channelsGrid}>
                    {channels.map((strip, i) => (
                        <PerformanceChannel
                            key={i}
                            index={i}
                            strip={strip}
                            engine={engineRef.current}
                            playing={channelStates[i]?.playing || false}
                            onGenerate={generateForChannel}
                            canGenerate={Boolean(selectedModel)}
                            onMuteSoloChange={handleMuteSoloChange}
                            onStateChange={handleChannelStateChange}
                            onFormStateChange={handleChannelFormChange}
                            initialFormState={session.channels[i]}
                            maxDuration={maxDuration}
                            bpm={bpm}
                        />
                    ))}
                </Box>

                <Box sx={styles.masterStrip(MASTER_COLOR)}>
                    <Box sx={styles.masterHeader(MASTER_COLOR)}>
                        <Box sx={styles.masterBadge(MASTER_COLOR)}>Master</Box>
                    </Box>

                    {/* Middle group — fader+meter (fixed height) plus the
                        DBFS/Pk readouts, vertically centered in the remaining
                        space between the header and the FX pickers below. */}
                    <Box sx={{
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 0.5,
                        minHeight: 0,
                    }}>
                        <Box sx={styles.masterFaderWrap}>
                            <Box sx={styles.masterMeterTrack}>
                                <Box ref={meterFillRef} sx={styles.masterMeterFill(MASTER_COLOR)} />
                                <Box sx={styles.masterMeterSegments} />
                            </Box>
                            <MidiMappable
                                id="master.fader"
                                label="Master Fader"
                                kind="continuous"
                                min={MASTER_DB_MIN}
                                max={MASTER_DB_MAX}
                                value={masterDb}
                                onChange={(v) => handleMasterChange(null, v)}
                                sx={{
                                    alignSelf: 'stretch',
                                    // Fixed-width lane wide enough for the 16 px
                                    // thumb plus a touch of hit area. Without an
                                    // explicit width the wrapper would either
                                    // collapse to nothing (no flex parent on the
                                    // slider) or stretch to fill (with flex: 1),
                                    // which pinned the fader to the left edge.
                                    width: 20,
                                    display: 'flex',
                                    justifyContent: 'center',
                                }}
                            >
                                <Slider
                                    orientation="vertical"
                                    value={masterDb}
                                    onChange={handleMasterChange}
                                    min={MASTER_DB_MIN}
                                    max={MASTER_DB_MAX}
                                    step={0.1}
                                    sx={styles.masterFader(MASTER_COLOR)}
                                />
                            </MidiMappable>
                        </Box>

                        <Box sx={styles.masterReadouts}>
                            <Typography sx={styles.masterValue}>
                                <Box component="span" sx={{ ...perfTokens.caps, color: 'inherit' }}>
                                    DBFS
                                </Box>
                                {' '}{formatDb(masterDb)}
                            </Typography>
                            <Typography sx={styles.masterPeakValue}>
                                <Box component="span" sx={{ ...perfTokens.caps, color: 'inherit' }}>
                                    Pk
                                </Box>
                                {' '}{formatDb(peakLabelDb)}
                            </Typography>
                        </Box>
                    </Box>

                    {/* Master FX pickers — Reverb IR and Delay division.
                        No wet sliders: the shared FX run always-on, and the
                        per-channel DLY/REV knobs drive the send amounts. */}
                    <Box sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 0.75,
                        pt: 0.75,
                        borderTop: `1px solid ${MASTER_COLOR}33`,
                    }}>
                        <FormControl size="small" sx={{ ...styles.pillControl, width: '100%' }}>
                            <Select
                                value={masterReverbIR}
                                onChange={(e) => setMasterReverbIR(e.target.value)}
                                renderValue={(value) => {
                                    const ir = IMPULSE_RESPONSES.find((x) => x.id === value);
                                    return `Rev · ${ir?.name ?? '—'}`;
                                }}
                            >
                                {IMPULSE_RESPONSES.map((ir) => (
                                    <MenuItem key={ir.id} value={ir.id} sx={{ fontSize: perfTokens.fontSize.sm }}>
                                        {ir.name}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                        <FormControl size="small" sx={{ ...styles.pillControl, width: '100%' }}>
                            <Select
                                value={masterDelayDivision}
                                onChange={(e) => setMasterDelayDivision(e.target.value)}
                                renderValue={(value) => {
                                    const d = MASTER_DELAY_DIVISIONS.find((x) => x.id === value);
                                    return `Dly · ${d?.label ?? '—'}`;
                                }}
                            >
                                {MASTER_DELAY_DIVISIONS.map((d) => (
                                    <MenuItem key={d.id} value={d.id} sx={{ fontSize: perfTokens.fontSize.sm, fontVariantNumeric: 'tabular-nums' }}>
                                        {d.label}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>
                </Box>
            </Box>

            <Paper sx={{ ...styles.barCard, gap: 2.5, mt: 1 }}>
                {/* Model + artifact pickers — moved here from the top row so the
                    primary strip stays just BPM + transport. Each is gated so
                    they only appear when relevant. */}
                <FormControl size="small" sx={{ ...styles.pillControl, width: 140 }}>
                    <Select
                        value={selectedModel || ''}
                        onChange={handleModelChange}
                        displayEmpty
                        renderValue={(value) => {
                            if (!value) return <em style={{ opacity: 0.6 }}>Model</em>;
                            const SHORT = {
                                'stable-audio-open-1.0': 'SAO Full',
                                'stable-audio-open-small': 'SAO Small',
                            };
                            if (SHORT[value]) return SHORT[value];
                            const base = baseModels.find((m) => m.name === value);
                            if (base) return base.displayName || base.name;
                            return value;
                        }}
                    >
                        {/* Distilled (post-trained) and Base SA3 variants are
                            split by their `kind` field. modelRow is defined
                            inline at component scope so MUI Select sees the
                            generated MenuItems as direct flat children — wrapping
                            them in an IIFE-returned Fragment broke selection
                            (Select couldn't find the option matching `value`). */}
                        {distilledBaseModels.length > 0 && (
                            <MenuItem disabled sx={{ fontSize: perfTokens.fontSize.xs, color: 'text.secondary' }}>
                                ── Distilled ──
                            </MenuItem>
                        )}
                        {distilledBaseModels.map(renderBaseModelRow)}
                        {trueBaseModels.length > 0 && (
                            <MenuItem disabled sx={{ fontSize: perfTokens.fontSize.xs, color: 'text.secondary' }}>
                                ── Base ──
                            </MenuItem>
                        )}
                        {trueBaseModels.map(renderBaseModelRow)}
                        {availableModels.length > 0 && (
                            <MenuItem disabled sx={{ fontSize: perfTokens.fontSize.xs, color: 'text.secondary' }}>
                                ── Fine-tuned ──
                            </MenuItem>
                        )}
                        {availableModels.map((model) => (
                            <MenuItem
                                key={model.name}
                                value={String(model.name)}
                                sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, pr: 0.5, fontSize: perfTokens.fontSize.sm }}
                            >
                                <Box component="span" sx={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {model.name}
                                </Box>
                                <Tooltip title="Delete fine-tuned model">
                                    <IconButton
                                        size="small"
                                        onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            e.preventDefault();
                                            handleDeleteFineTuned(model.name);
                                        }}
                                        sx={styles.compactIconBtn('md', 'danger')}
                                    >
                                        <DeleteIcon size={perfTokens.icon.md} />
                                    </IconButton>
                                </Tooltip>
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>

                {/* Combined LoRA + checkpoint picker — every option is a
                    specific (LoRA, checkpoint) pair. Multi-checkpoint LoRAs
                    show a ListSubheader with the LoRA name + delete, then one
                    MenuItem per checkpoint indented below. Single-checkpoint
                    LoRAs collapse to one MenuItem with the LoRA name and
                    inline delete. Saves the separate Ckpt dropdown's slot. */}
                {(() => {
                    const isBaseModel = baseModels.some(m => m.name === selectedModel);
                    const compatibleLoras = isBaseModel
                        ? filterLorasForModel(availableLoras, selectedModel)
                        : [];
                    const findLoraForPath = (path) => compatibleLoras.find(
                        l => l.path === path || (l.all_checkpoints || []).includes(path)
                    );
                    const parseCheckpointLabel = (filepath) => {
                        const name = (filepath || '').split('/').pop() || filepath || '';
                        const m = name.match(/epoch=(\d+)-step=(\d+)/);
                        if (m) return `Ep ${m[1]} · ${m[2]}`;
                        return name.replace(/\.ckpt$/i, '');
                    };
                    const loraDisabled = compatibleLoras.length === 0;

                    const deleteBtn = (loraName, size = 'md') => (
                        <Tooltip title="Delete LoRA">
                            <IconButton
                                size="small"
                                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    e.preventDefault();
                                    handleDeleteLora(loraName);
                                }}
                                sx={styles.compactIconBtn(size, 'danger')}
                            >
                                <DeleteIcon size={perfTokens.icon[size === 'sm' ? 'sm' : 'md']} />
                            </IconButton>
                        </Tooltip>
                    );

                    return (
                        <FormControl size="small" disabled={loraDisabled} sx={{ ...styles.pillControl, width: 180 }}>
                            <Select
                                value={selectedLora || ''}
                                onChange={(e) => onSelectLora?.(String(e.target.value))}
                                displayEmpty
                                renderValue={(value) => {
                                    if (!value) return <em style={{ opacity: 0.6 }}>No LoRA</em>;
                                    const lora = findLoraForPath(value);
                                    if (!lora) return value;
                                    const ckpts = lora.all_checkpoints || [lora.path];
                                    return ckpts.length === 1
                                        ? lora.name
                                        : `${lora.name} · ${parseCheckpointLabel(value)}`;
                                }}
                                MenuProps={{ PaperProps: { sx: { maxHeight: 360 } } }}
                            >
                                <MenuItem value="">
                                    <em>No LoRA</em>
                                </MenuItem>
                                {compatibleLoras.flatMap((lora) => {
                                    const ckpts = lora.all_checkpoints || [lora.path];
                                    if (ckpts.length <= 1) {
                                        // Single-checkpoint LoRA collapses to one row
                                        // with the LoRA name and inline delete.
                                        return [
                                            <MenuItem
                                                key={`${lora.name}::${ckpts[0]}`}
                                                value={ckpts[0]}
                                                sx={{
                                                    display: 'flex',
                                                    justifyContent: 'space-between',
                                                    gap: 1,
                                                    pr: 0.5,
                                                    fontSize: perfTokens.fontSize.sm,
                                                }}
                                            >
                                                <Box component="span" sx={{
                                                    flex: 1,
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                    whiteSpace: 'nowrap',
                                                }}>
                                                    {lora.name}
                                                </Box>
                                                {deleteBtn(lora.name)}
                                            </MenuItem>,
                                        ];
                                    }
                                    // Multi-checkpoint LoRA — subheader with name +
                                    // delete, then one MenuItem per checkpoint.
                                    return [
                                        <ListSubheader
                                            key={`${lora.name}::header`}
                                            disableSticky
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'space-between',
                                                gap: 1,
                                                pr: 0.5,
                                                lineHeight: 1.6,
                                                fontSize: perfTokens.fontSize.xs,
                                                color: 'text.secondary',
                                                textTransform: 'uppercase',
                                                letterSpacing: perfTokens.letterSpacing.wide,
                                                bgcolor: 'background.paper',
                                            }}
                                        >
                                            <Box component="span" sx={{
                                                flex: 1,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap',
                                            }}>
                                                {lora.name}
                                            </Box>
                                            {deleteBtn(lora.name, 'sm')}
                                        </ListSubheader>,
                                        ...ckpts.map((ckpt, i) => (
                                            <MenuItem
                                                key={`${lora.name}::${ckpt}`}
                                                value={ckpt}
                                                sx={{ fontSize: perfTokens.fontSize.sm, pl: 3 }}
                                            >
                                                {parseCheckpointLabel(ckpt)}
                                                {i === ckpts.length - 1 ? ' (latest)' : ''}
                                            </MenuItem>
                                        )),
                                    ];
                                })}
                            </Select>
                        </FormControl>
                    );
                })()}

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                    <Box component="span" sx={perfTokens.labelMuted}>
                        Steps
                    </Box>
                    <Tooltip
                        placement="right"
                        title={
                            isDistilledSA3
                                ? 'Locked at 8 steps for distilled SA3 models — pick a *-base checkpoint to override'
                                : 'Diffusion steps per generation (more = higher quality, slower)'
                        }
                    >
                        <FormControl size="small" sx={{ ...styles.pillControl, width: 68 }}>

                            <Select
                                value={isDistilledSA3 ? 8 : steps}
                                onChange={(e) => onStepsChange?.(Number(e.target.value))}
                                disabled={isDistilledSA3}
                                renderValue={(value) => `${value}`}
                            >
                                {isDistilledSA3 && (
                                    <MenuItem value={8} sx={{ fontSize: perfTokens.fontSize.sm }}>
                                        8 (locked)
                                    </MenuItem>
                                )}
                                {[50, 100, 150, 200, 250].map((n) => (
                                    <MenuItem key={n} value={n} sx={{ fontSize: perfTokens.fontSize.sm, fontVariantNumeric: 'tabular-nums' }}>
                                        {n}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Tooltip>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box component="span" sx={perfTokens.labelMuted}>
                        Seed
                    </Box>
                    <FormControlLabel
                        sx={{ mr: 0, ml: 0.25 }}
                        control={
                            <Switch
                                size="small"
                                checked={randomSeed}
                                onChange={(e) => onRandomSeedChange?.(e.target.checked)}
                            />
                        }
                        label={
                            <Typography component="span" sx={{ fontSize: perfTokens.fontSize.sm }}>
                                Random
                            </Typography>
                        }
                    />
                    <TextField
                        size="small"
                        type="number"
                        placeholder="42"
                        value={seedValue}
                        onChange={(e) => onSeedValueChange?.(e.target.value)}
                        disabled={randomSeed}
                        // inputProps.style wins against MuiInputBase-inputSizeSmall's 14px default.
                        inputProps={{
                            min: 0,
                            max: 4294967295,
                            step: 1,
                            style: {
                                fontVariantNumeric: 'tabular-nums',
                                fontSize: perfTokens.fontSize.sm,
                                fontWeight: perfTokens.weight.bold,
                            },
                        }}
                        sx={{ ...styles.pillControl, width: 78 }}
                    />
                </Box>

                {/* Prompt auto-inject — Key / Tempo / Time sig fields. Each
                    is appended to every generated prompt when populated; empty
                    fields are skipped. Replaces the old Auto BPM toggle. */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box component="span" sx={perfTokens.labelMuted}>
                        Inject
                    </Box>
                    <Tooltip title="Musical key to auto-inject (e.g. C minor). Leave empty to skip." placement="top" arrow enterDelay={500}>
                        <TextField
                            size="small"
                            placeholder="Key"
                            value={promptKey}
                            onChange={(e) => setPromptKey(e.target.value)}
                            inputProps={{
                                'aria-label': 'Key to inject into prompt',
                                style: {
                                    fontSize: perfTokens.fontSize.sm,
                                    fontWeight: perfTokens.weight.bold,
                                },
                            }}
                            sx={{ ...styles.pillControl, width: 78 }}
                        />
                    </Tooltip>
                    <Tooltip
                        title={promptInjectBpm
                            ? `Injecting master BPM (${Math.round(bpm)}) into prompts — click to disable`
                            : 'Click to auto-inject the master BPM (top bar) into every prompt'}
                        placement="top"
                        arrow
                        enterDelay={500}
                    >
                        <ButtonBase
                            onClick={() => setPromptInjectBpm((p) => !p)}
                            aria-label={promptInjectBpm ? 'Disable master BPM injection' : 'Enable master BPM injection'}
                            aria-pressed={promptInjectBpm}
                            sx={(theme) => ({
                                height: perfTokens.height.compact,
                                minWidth: 62,
                                px: 1,
                                borderRadius: 1.5,
                                border: '1px solid',
                                borderColor: promptInjectBpm ? MASTER_COLOR : theme.palette.divider,
                                backgroundColor: promptInjectBpm ? MASTER_COLOR : 'transparent',
                                color: promptInjectBpm ? '#0c1018' : 'text.disabled',
                                fontSize: perfTokens.fontSize.sm,
                                fontWeight: perfTokens.weight.bold,
                                fontVariantNumeric: 'tabular-nums',
                                transition: 'background-color 120ms, color 120ms, border-color 120ms',
                                '&:hover': {
                                    backgroundColor: promptInjectBpm ? MASTER_COLOR : 'action.hover',
                                    color: promptInjectBpm ? '#0c1018' : 'text.secondary',
                                },
                            })}
                        >
                            {promptInjectBpm ? `${Math.round(bpm)} BPM` : 'BPM'}
                        </ButtonBase>
                    </Tooltip>
                    <Tooltip title="Time signature to auto-inject (e.g. 4/4). Leave empty to skip." placement="top" arrow enterDelay={500}>
                        <TextField
                            size="small"
                            placeholder="Time"
                            value={promptTimeSig}
                            onChange={(e) => setPromptTimeSig(e.target.value)}
                            inputProps={{
                                'aria-label': 'Time signature to inject into prompt',
                                style: {
                                    fontVariantNumeric: 'tabular-nums',
                                    fontSize: perfTokens.fontSize.sm,
                                    fontWeight: perfTokens.weight.bold,
                                },
                            }}
                            sx={{ ...styles.pillControl, width: 62 }}
                        />
                    </Tooltip>
                </Box>
            </Paper>
        </Box>
    );
}
