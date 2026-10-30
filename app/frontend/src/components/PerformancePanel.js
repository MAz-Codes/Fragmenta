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
    ButtonBase,
    Menu,
    ListItemText,
    ListSubheader,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    InputAdornment,
    Popover,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import RotaryKnob from './RotaryKnob';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import {
    Play as PlayAllIcon,
    Square as StopAllIcon,
    Trash2 as DeleteIcon,
    Save as SaveIcon,
    X as CloseXIcon,
    Headphones as CueIcon,
    RotateCcw as RestoreIcon,
    CloudDownload as DownloadIcon,
    Circle as RecordIcon,
    Minus as MinusIcon,
    Plus as PlusIcon,
    ChevronDown as ChevronDownIcon,
} from 'lucide-react';
import api from '../api';
import { extractError } from '../utils/errors';
import { faderPosToDb, faderDbToPos, FADER_MAX_DB, FADER_UNITY_POS } from '../utils/faderLaw';
import PerformanceChannel from './PerformanceChannel';
import { PerformanceEngine, IMPULSE_RESPONSES, MASTER_DELAY_DIVISIONS } from '../utils/performanceAudio';
import { performancePanelStyles as styles, perfTokens, onFill } from '../theme';
import { MidiMappable, useMidi, clearMidiConfig } from './MidiContext';
import { useAudioSettings } from './AudioSettingsContext';
// setCueOutputPair is aliased: the component has a useState setter of the same
// name which would otherwise shadow the import and leave cue routing dead.
import { isCueSupported, listOutputDevices, refreshCueGraph, setCueDevice, setCueOutputPair as applyCueOutputPair } from '../utils/cueAudio';
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
    clearScope as clearFragmentScope,
} from '../utils/fragmentStorage';

// Link fine-tune range (ms). Wide enough for Bluetooth headphones, whose
// 150–300 ms the browser often doesn't report as output latency.
const LINK_OFFSET_MIN = -200;
const LINK_OFFSET_MAX = 500;

const CHANNEL_COUNT = 4;
// Master, transport and Link/MIDI colours come from the palette
// (theme.palette.perf): pastel fills with ink labels.
const MASTER_DB_MIN = -60;
// +6 dB of boost headroom above unity; the fader taper (faderLaw) puts unity
// at ~80% of throw, not the very top.
const MASTER_DB_MAX = FADER_MAX_DB;
const MASTER_DB_DEFAULT = -3;
const METER_FLOOR_DB = -60;
const PEAK_HOLD_MS = 1500;
// Scale beside the master meters (dB).
const METER_SCALE_DB = [0, -6, -12, -24, METER_FLOOR_DB];
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
    // Explicit + in the boost range so "3.0" can't read as a cut.
    return (db > 0 ? '+' : '') + db.toFixed(1);
};

export default function PerformancePanel(props) {
    // Reset key for the inner panel. Bumping it forces a full remount of
    // PerformancePanelInner, which makes usePerformanceSession re-read from
    // localStorage and every PerformanceChannel re-hydrate from IDB. The
    // MidiProvider sits in App (Settings shows the MIDI setup), so MIDI
    // mappings survive a reset.
    const [resetKey, setResetKey] = useState(0);
    const triggerReset = useCallback(() => setResetKey((k) => k + 1), []);
    return (
        <PerformancePanelInner
            key={resetKey}
            {...props}
            onPresetLoaded={triggerReset}
        />
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
    // False while another tab is shown (the panel stays mounted via
    // keepMounted). Used to pause the meter RAF loops, which otherwise run
    // at full frame rate behind every other tab.
    active = true,
}) {
    const perfColors = useTheme().palette.perf;
    const {
        session, updateGlobal, updateChannel, cancelPendingPersist,
    } = usePerformanceSession(CHANNEL_COUNT);

    const engineRef = useRef(null);
    // Left/right master meters: fill and peak-hold marker per channel.
    const meterFillRefs = [useRef(null), useRef(null)];
    const peakMarkerRefs = [useRef(null), useRef(null)];
    // Peak markers hold the loudest level for PEAK_HOLD_MS, then fall.
    const peakMarkRef = useRef([{ db: METER_FLOOR_DB, until: 0, at: 0 }, { db: METER_FLOOR_DB, until: 0, at: 0 }]);
    const peakHoldRef = useRef([0, 1].map(() => ({ db: METER_FLOOR_DB, decayedAt: performance.now() })));
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
    // Link sync: Fragmenta schedules Link launches early by its measured
    // output latency (performanceAudio.js); this per-output-device trim covers
    // what the browser doesn't report. Per-viewer convenience → localStorage.
    const [linkOffsetMs, setLinkOffsetMs] = useState(0);
    const [syncAnchor, setSyncAnchor] = useState(null);
    const [calibrating, setCalibrating] = useState(false);
    const [measuredLatencyMs, setMeasuredLatencyMs] = useState(null);
    const [launchQuantum, setLaunchQuantum] = useState(session.launchQuantum ?? LAUNCH_Q_DEFAULT);
    const wasPlayingRef = useRef(false);
    // Recent Link polls, for placing Link's clock on performance.now() — see
    // the poll below.
    const linkClockRef = useRef([]);
    const bpmOriginRef = useRef('user');
    const [peakLabelDb, setPeakLabelDb] = useState(METER_FLOOR_DB);
    // Mirror of peakLabelDb for the RAF tick's 0.5 dB hysteresis gate. The
    // tick is created once, so reading the state variable directly would
    // compare against a permanently stale closure value — making the gate
    // pass every frame and re-render the whole panel at 60 fps.
    const peakLabelDbRef = useRef(METER_FLOOR_DB);
    const [channelStates, setChannelStates] = useState(() =>
        Array.from({ length: CHANNEL_COUNT }, () => ({ loaded: false, playing: false }))
    );
    const [promptKey, setPromptKey] = useState(session.promptKey ?? '');
    const [promptInjectBpm, setPromptInjectBpm] = useState(session.promptInjectBpm ?? false);
    const [masterReverbIR, setMasterReverbIR] = useState(session.masterReverbIR ?? 'hall');
    const [masterDelayDivision, setMasterDelayDivision] = useState(session.masterDelayDivision ?? '1/4');

    // Master recording. `recording` reflects an active capture; once stopped,
    // `pendingRecording` holds the encoded WAV { blob, durationSec } until the
    // user names + saves it (or discards) via the name dialog.
    const [recording, setRecording] = useState(false);
    const [pendingRecording, setPendingRecording] = useState(null);
    const [recordingName, setRecordingName] = useState('');
    const [savingRecording, setSavingRecording] = useState(false);

    // Audio output device. setSinkId requires Chromium ≥ 110 (cueSupported
    // is the runtime check). One device drives BOTH main and cue. Per-pair
    // channel routing within the device is Stage 2 — pair selections are
    // tracked here but the merger plumbing comes later.
    const cueSupported = useMemo(() => isCueSupported(), []);
    const [outputDeviceId, setOutputDeviceId] = useState(session.outputDeviceId ?? session.cueDeviceId ?? '');
    const [audioDevices, setAudioDevices] = useState([]);
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
        applyCueOutputPair(cueOutputPair);
    }, [cueOutputPair, updateGlobal]);

    // Bind both the main engine context and the (separate) cue context to the
    // chosen device, then re-read maxChannelCount so the pair selectors
    // populate with everything the device exposes. Without setSinkId support
    // we can't pick a device, but the system-default device may still be
    // multichannel — refreshOutputGraph claims its full channel count.
    const bindAudioOutput = useCallback(async () => {
        const engine = engineRef.current;
        if (!engine) return;
        if (cueSupported && engine.setOutputDevice) {
            const max = await engine.setOutputDevice(outputDeviceId);
            setMaxChannelCount(max);
            await setCueDevice(outputDeviceId).catch(() => { /* logged in cueAudio */ });
        } else if (engine.refreshOutputGraph) {
            setMaxChannelCount(engine.refreshOutputGraph());
            refreshCueGraph();
        }
    }, [cueSupported, outputDeviceId]);

    useEffect(() => { bindAudioOutput(); }, [bindAudioOutput]);

    // Chromium reports destination.maxChannelCount=2 while the context is
    // suspended (autoplay policy keeps it suspended until a user gesture), so
    // the mount-time binding above may have seen a stereo device. Re-claim the
    // channel count whenever the context actually starts running.
    useEffect(() => {
        const engine = engineRef.current;
        const ctx = engine?.ctx;
        if (!ctx || typeof ctx.addEventListener !== 'function') return;
        const onStateChange = () => {
            if (ctx.state !== 'running' || !engine.refreshOutputGraph) return;
            const max = engine.refreshOutputGraph();
            setMaxChannelCount(max);
        };
        ctx.addEventListener('statechange', onStateChange);
        return () => ctx.removeEventListener('statechange', onStateChange);
    }, []);

    // macOS (WKWebView, no setSinkId): the backend lists the Mac's output
    // devices and makes the chosen one the default output; the page plays on
    // the default device with all its channels, so the pairs work as usual.
    // null = not available (other platforms, or Chromium on a Mac).
    const [systemOutputs, setSystemOutputs] = useState(null);
    const [systemOutput, setSystemOutput] = useState('');
    const refreshSystemOutputs = useCallback(async () => {
        if (cueSupported) return;
        try {
            const { data } = await api.get('/api/audio/outputs');
            if (!data?.available) return;
            setSystemOutputs(data.outputs || []);
            setSystemOutput(data.current || '');
        } catch { /* backend without the route: leave it unavailable */ }
    }, [cueSupported]);
    useEffect(() => { refreshSystemOutputs(); }, [refreshSystemOutputs]);
    const pickSystemOutput = useCallback(async (uid) => {
        if (!uid) return;
        try {
            const { data } = await api.post('/api/audio/outputs/select', { uid });
            setSystemOutput(data?.current || uid);
        } catch (err) {
            console.warn('[PerformancePanel] output switch failed', err);
            refreshSystemOutputs();
            return;
        }
        // The context follows the new default device; give Core Audio a
        // moment, then re-claim its channel count for the pair selectors.
        setTimeout(() => bindAudioOutput(), 300);
    }, [bindAudioOutput, refreshSystemOutputs]);

    const refreshAudioDevices = useCallback(async () => {
        if (!cueSupported) return;
        try {
            const devices = await listOutputDevices();
            setAudioDevices(devices);
        } catch (err) {
            console.warn('[PerformancePanel] device enumeration failed', err);
        }
    }, [cueSupported]);

    // The output device is picked in the app's Settings dialog; publish what
    // it needs. Opening it is a user gesture, so the context can resume and
    // report the device's real channel count: rebind then, so the pair
    // selectors are populated even if the mount-time binding ran against a
    // suspended context.
    const [genMenuAnchor, setGenMenuAnchor] = useState(null);
    const { setApi: publishAudioSettings } = useAudioSettings();
    const refreshAudioSettings = useCallback(() => {
        refreshAudioDevices();
        refreshSystemOutputs();
        bindAudioOutput();
    }, [refreshAudioDevices, refreshSystemOutputs, bindAudioOutput]);
    // For polling while Settings is open: the lists only, nothing rebinds.
    const rescanAudioDevices = useCallback(() => {
        refreshAudioDevices();
        refreshSystemOutputs();
    }, [refreshAudioDevices, refreshSystemOutputs]);
    useEffect(() => {
        if (!cueSupported && systemOutputs) {
            publishAudioSettings({
                supported: true,
                systemOutput: true,
                deviceId: systemOutput,
                devices: systemOutputs.map((o) => ({ deviceId: o.uid, label: o.name })),
                refresh: refreshAudioSettings,
                rescan: rescanAudioDevices,
                pick: pickSystemOutput,
            });
            return;
        }
        publishAudioSettings({
            supported: cueSupported,
            deviceId: outputDeviceId,
            devices: audioDevices,
            refresh: refreshAudioSettings,
            rescan: rescanAudioDevices,
            pick: setOutputDeviceId,
        });
    }, [publishAudioSettings, cueSupported, outputDeviceId, audioDevices, refreshAudioSettings,
        rescanAudioDevices, systemOutputs, systemOutput, pickSystemOutput]);
    useEffect(() => () => publishAudioSettings(null), [publishAudioSettings]);

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
        // Kill any queued (debounced) session write FIRST — the IDB clears
        // below take long enough for it to fire and re-persist the old
        // session over the just-cleared key, undoing the restore.
        cancelPendingPersist();
        clearPerformanceSession();
        clearMidiConfig();
        // Drop every channel's fragment blobs from IDB so a fresh start is
        // actually fresh. Presets keep their own scopes and survive.
        await Promise.all(
            Array.from({ length: CHANNEL_COUNT }, (_, i) =>
                clearFragmentScope(channelScope(i)).catch(() => { /* ignore */ })
            )
        );
        closePresetMenu();
        onPresetLoaded?.();
    };

    const handleSaveAs = async () => {
        const name = saveAsName.trim();
        if (!name) return;
        if (!savePreset(name, session)) {
            setError(
                `Could not save preset "${name}" — browser storage is full. `
                + 'Delete old presets or clear some fragments and try again.'
            );
            return;
        }
        // Copy each channel's session-scope blobs into the preset-scope so
        // the preset's fragments survive overwrites of the live session. Done
        // after the metadata save so a quota failure here still leaves a
        // recoverable (if blob-less) preset entry.
        await Promise.all(
            Array.from({ length: CHANNEL_COUNT }, async (_, i) => {
                const dst = presetChannelScope(name, i);
                // Replace, don't merge — a re-save of the same preset name
                // should reflect the current session exactly.
                await clearFragmentScope(dst).catch(() => { /* ignore */ });
                await copyScope(channelScope(i), dst).catch(() => { /* ignore */ });
            })
        );
        setSaveAsName('');
        refreshPresetNames();
    };

    const handleLoadPreset = async (name) => {
        // Same race as Restore Defaults: a pending debounced persist firing
        // during the IDB copies below would overwrite the preset payload we
        // just wrote with the old session. Cancel it before touching the key.
        cancelPendingPersist();
        if (!loadPresetIntoSession(name)) return;
        // Swap the IDB session-scope blobs to match the loaded preset's
        // metadata. MUST complete before onPresetLoaded triggers remount —
        // otherwise the new channels hydrate from a stale session scope.
        await Promise.all(
            Array.from({ length: CHANNEL_COUNT }, async (_, i) => {
                const dst = channelScope(i);
                await clearFragmentScope(dst).catch(() => { /* ignore */ });
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
                clearFragmentScope(presetChannelScope(name, i)).catch(() => { /* ignore */ })
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

    // The generation pill's label: model · LoRA · steps.
    const genSummary = (() => {
        if (!selectedModel) return 'Choose a model';
        const base = baseModels.find((m) => m.name === selectedModel);
        const modelName = base ? (base.displayName || base.name) : selectedModel;
        const lora = selectedLora
            ? (availableLoras.find((l) => l.path === selectedLora || (l.all_checkpoints || []).includes(selectedLora))?.name || 'LoRA')
            : 'No LoRA';
        return `${modelName} · ${lora} · ${isDistilledSA3 ? 8 : steps} steps`;
    })();

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
                <Tooltip title={TIPS.perf.notDownloaded}>
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
        return () => {
            engine.dispose();
            engineRef.current = null;
        };
    }, []);

    // Master meter loop. Separate from the engine-lifecycle effect so it can
    // pause while the Performance tab is hidden (keepMounted leaves the panel
    // mounted behind other tabs).
    useEffect(() => {
        if (!active) return undefined;
        const toPct = (db) => Math.max(0, Math.min(100, ((db - METER_FLOOR_DB) / -METER_FLOOR_DB) * 100));
        const tick = () => {
            const now = performance.now();
            const engine = engineRef.current;
            const peaks = engine ? engine.getMasterPeaks() : [0, 0];
            let loudest = METER_FLOOR_DB;
            peaks.forEach((amp, ch) => {
                const instantDb = ampToDb(amp);
                const hold = peakHoldRef.current[ch];
                let displayDb = hold.db - ((now - hold.decayedAt) / 1000) * 24;
                if (instantDb > displayDb) displayDb = instantDb;
                if (displayDb < METER_FLOOR_DB) displayDb = METER_FLOOR_DB;
                hold.db = displayDb;
                hold.decayedAt = now;
                if (displayDb > loudest) loudest = displayDb;
                const fill = meterFillRefs[ch].current;
                // The gradient spans the whole track; the clip reveals it.
                if (fill) fill.style.clipPath = `inset(${100 - toPct(displayDb)}% 0 0 0 round 999px)`;
                const mark = peakMarkRef.current[ch];
                if (instantDb >= mark.db) {
                    mark.db = instantDb;
                    mark.until = now + PEAK_HOLD_MS;
                } else if (now > mark.until) {
                    mark.db = Math.max(METER_FLOOR_DB, mark.db - ((now - mark.at) / 1000) * 20);
                }
                mark.at = now;
                const marker = peakMarkerRefs[ch].current;
                if (marker) {
                    marker.style.bottom = `calc(${toPct(mark.db)}% - 2px)`;
                    marker.style.opacity = mark.db > METER_FLOOR_DB + 1 ? '0.8' : '0';
                }
            });
            if (Math.abs(loudest - peakLabelDbRef.current) > 0.5) {
                peakLabelDbRef.current = loudest;
                setPeakLabelDb(loudest);
            }
            meterRafRef.current = requestAnimationFrame(tick);
        };
        meterRafRef.current = requestAnimationFrame(tick);
        return () => {
            if (meterRafRef.current) cancelAnimationFrame(meterRafRef.current);
        };
    }, [active]);

    const handleMasterChange = (_, value) => {
        setMasterDb(value);
        engineRef.current?.setMasterGain(dbToGain(value));
    };

    // Toggle master capture. Stopping opens the name dialog with the encoded
    // WAV held in pendingRecording.
    const handleToggleRecord = useCallback(async () => {
        const engine = engineRef.current;
        if (!engine) return;
        if (engine.isRecording()) {
            const result = await engine.stopRecording();
            setRecording(false);
            if (result?.blob) {
                const now = new Date();
                const pad = (n) => String(n).padStart(2, '0');
                const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
                setRecordingName(`performance_${stamp}`);
                setPendingRecording(result);
                if (result.truncated) {
                    setError(
                        'The recording hit the in-memory capture limit and was '
                        + 'truncated — everything up to that point is intact.'
                    );
                }
            } else {
                setError('Nothing was captured — the recording was empty.');
            }
        } else {
            try {
                if (await engine.startRecording()) setRecording(true);
            } catch (e) {
                setError(e?.message || 'Recording is not supported in this environment.');
            }
        }
    }, []);

    const handleSaveRecording = useCallback(async () => {
        const name = (recordingName || '').trim();
        if (!pendingRecording?.blob || !name) return;
        setSavingRecording(true);
        try {
            const fd = new FormData();
            fd.append('file', pendingRecording.blob, 'recording.wav');
            fd.append('name', name);
            if (pendingRecording.durationSec != null) {
                fd.append('duration', String(pendingRecording.durationSec));
            }
            await api.post('/api/performance/recording', fd);
            setPendingRecording(null);
            setRecordingName('');
        } catch (e) {
            // extractError digs the message out of both backend error shapes
            // ({error: "str"} and the APIResponse {error: {message, …}}
            // envelope) — setting the raw envelope object here crashed the
            // render ("Objects are not valid as a React child").
            setError(extractError(e, 'Failed to save the recording.'));
        } finally {
            setSavingRecording(false);
        }
    }, [pendingRecording, recordingName]);

    const handleDiscardRecording = useCallback(() => {
        setPendingRecording(null);
        setRecordingName('');
    }, []);

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
            const sentAt = performance.now();
            try {
                const r = await api.get('/api/link/state');
                const receivedAt = performance.now();
                // When the backend read Link's clock, on our clock. The round
                // trip's midpoint is only right when the request didn't wait
                // on the way: a slow answer put single snapshots tens of ms
                // off, which skewed launches and the drift correction. So the
                // midpoints only estimate the offset between the two clocks —
                // NTP-style, from the fastest round trip of the last 5 s, which
                // brackets the read most tightly — and every snapshot is then
                // placed by Link's own timestamp. Bindings without a clock
                // fall back to the midpoint.
                let capturedAt = (sentAt + receivedAt) / 2;
                const micros = Number(r.data?.time_micros) || 0;
                if (micros > 0) {
                    const samples = linkClockRef.current;
                    samples.push({ rtt: receivedAt - sentAt, offset: capturedAt - micros / 1000, at: receivedAt });
                    while (samples.length > 1 && receivedAt - samples[0].at > 5000) samples.shift();
                    const best = samples.reduce((a, b) => (b.rtt < a.rtt ? b : a));
                    capturedAt = micros / 1000 + best.offset;
                }
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
        // 200 ms (was 500): fresher beat for launch extrapolation and a tighter
        // bound on mirroring Ableton's stop. Localhost round-trips are cheap.
        const timer = setInterval(poll, 200);
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

    const linkOffsetKey = `fragmenta.linkOffsetMs:${outputDeviceId || 'default'}`;
    useEffect(() => {
        let stored = 0;
        try { stored = Number(window.localStorage.getItem(linkOffsetKey)) || 0; } catch { /* private mode */ }
        setLinkOffsetMs(stored);
    }, [linkOffsetKey]);
    useEffect(() => { engineRef.current?.setLinkOffsetMs(linkOffsetMs); }, [linkOffsetMs]);
    const updateLinkOffset = useCallback((ms) => {
        ms = Math.max(LINK_OFFSET_MIN, Math.min(LINK_OFFSET_MAX, Math.round(ms)));
        setLinkOffsetMs(ms);
        try { window.localStorage.setItem(linkOffsetKey, String(ms)); } catch { /* private mode */ }
    }, [linkOffsetKey]);

    // Refresh the measured latency while Link is on (it changes with the device).
    useEffect(() => {
        if (!linkEnabled) { setMeasuredLatencyMs(null); return undefined; }
        const read = () => {
            const ms = engineRef.current?.getOutputLatencyMs?.();
            setMeasuredLatencyMs(Number.isFinite(ms) ? Math.round(ms) : null);
        };
        read();
        const t = setInterval(read, 2000);
        return () => clearInterval(t);
    }, [linkEnabled, outputDeviceId]);

    const stopCalibration = useCallback(() => {
        engineRef.current?.stopCalibrationClicks();
        setCalibrating(false);
    }, []);
    const toggleCalibration = useCallback(async () => {
        const engine = engineRef.current;
        if (!engine) return;
        if (calibrating) { stopCalibration(); return; }
        try { await engine.ctx.resume(); } catch { /* already running */ }
        engine.startCalibrationClicks();
        setCalibrating(true);
    }, [calibrating, stopCalibration]);
    useEffect(() => { if (!linkEnabled) stopCalibration(); }, [linkEnabled, stopCalibration]);
    useEffect(() => () => engineRef.current?.stopCalibrationClicks(), []);

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
                setError(`Link install failed: ${extractError(err, 'Install failed')}`);
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
            if (err?.response?.status === 503) setLinkAvailable(false);
            setError(extractError(err, 'Link toggle failed'));
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

    const generateForChannel = async ({ prompt, duration, alignBars, alignBpm, loopStitch, batchSize = 1, initAudioPath, initNoiseLevel, onBlob }) => {
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
            // queue them server-side with no time saved. Each fragment gets a
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
                // LoRA stacking (Phase 4). TRAINING targets *-base only,
                // but at inference a base-trained LoRA also applies to its
                // distilled sibling (same backbone, only the CFG state
                // differs) — the backend gate and the main Generation tab
                // both already work that way. This used to skip distilled
                // models, silently ignoring the LoRA the picker said was
                // active.
                ...(selectedLora && isSA3Base
                    ? { loras: [{ path: selectedLora, strength: loraMultiplier }] }
                    : {}),
                ...(alignBars && alignBpm ? { align_bars: alignBars, align_bpm: alignBpm } : {}),
                // Phase 7: seamless looping. Bars-mode + channel-looping
                // signals "I want a loop"; backend wrap-inpaints the seam.
                ...(loopStitch && alignBars && alignBpm ? { loop_stitch: loopStitch } : {}),
                // Phase 8 Variation: re-roll from a prior clip as init_audio.
                ...(initAudioPath ? { init_audio_path: initAudioPath, init_noise_level: initNoiseLevel ?? 0.9 } : {}),
            };
            const response = await api.post('/api/generate', requestData, { responseType: 'blob' });
            // Stream: hand each blob to the caller as it arrives so the fragment
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

    // Which channel (if any) owns sidechain ducking. Lifted here because it
    // is exclusive across channels: the owner's button is lit, every other
    // channel's button is grayed out. Deliberately not persisted — it's a
    // live performance gesture, so it resets off on reload.
    const [sidechainOwner, setSidechainOwner] = useState(null);

    const handleMuteSoloChange = (index, change) => {
        const engine = engineRef.current;
        if (!engine) return;
        if ('mute' in change) engine.setMute(index, change.mute);
        if ('solo' in change) engine.setSolo(index, change.solo);
        if ('sidechain' in change) {
            // Exclusive ownership. The buttons on other channels are disabled,
            // but a learned MIDI trigger can still fire — ignore those.
            if (change.sidechain) {
                if (sidechainOwner === null || sidechainOwner === index) {
                    engine.setSidechain(index, true);
                    setSidechainOwner(index);
                }
            } else if (sidechainOwner === index) {
                engine.setSidechain(index, false);
                setSidechainOwner(null);
            }
        }
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
            setError(`Failed to delete "${modelName}": ${extractError(err, 'Delete failed')}`);
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
            setError(`Failed to delete LoRA "${loraName}": ${extractError(err, 'Delete failed')}`);
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
                    title={TIPS.perf.link({ installing: linkInstalling, available: linkAvailable, enabled: linkEnabled, peers: linkPeers })}
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
                                px: 1.75,
                                minWidth: 52,
                                height: perfTokens.height.compact,
                                borderRadius: '999px',
                                // Three states matching Ableton Live's button:
                                // off → gray, on/no peers → yellow (broadcasting,
                                // alone), on/peers → teal (sync'd with at least
                                // one other app).
                                bgcolor: !linkEnabled
                                    ? perfColors.toggleOff
                                    : linkPeers > 0
                                        ? perfColors.master
                                        : perfColors.toggleOn,
                                color: onFill(!linkEnabled
                                    ? perfColors.toggleOff
                                    : linkPeers > 0 ? perfColors.master : perfColors.toggleOn),
                                opacity: linkInstalling ? 0.55 : 1,
                                transition: 'background-color 120ms',
                                '&:hover': {
                                    bgcolor: !linkEnabled
                                        ? perfColors.toggleOffHover
                                        : linkPeers > 0
                                            ? perfColors.masterHover
                                            : perfColors.toggleOnHover,
                                },
                                '&.Mui-disabled': {
                                    color: perfColors.label,
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

                {/* Link sync — latency compensation + calibrate by ear. */}
                {linkEnabled && (
                    <>
                        <Tooltip title={TIPS.perf.linkSync}>
                            <ButtonBase
                                onClick={(e) => setSyncAnchor(e.currentTarget)}
                                sx={{
                                    fontSize: perfTokens.fontSize.sm,
                                    fontWeight: 600,
                                    px: 1.5,
                                    height: perfTokens.height.compact,
                                    borderRadius: '999px',
                                    backgroundColor: (t) => t.palette.fragmenta.control,
                                    color: 'text.secondary',
                                    '&:hover': { color: 'text.primary', backgroundColor: (t) => t.palette.fragmenta.controlHover },
                                }}
                            >
                                {measuredLatencyMs === null
                                    ? 'Sync'
                                    : `${measuredLatencyMs + linkOffsetMs} ms`}
                            </ButtonBase>
                        </Tooltip>
                        <Popover
                            open={Boolean(syncAnchor)}
                            anchorEl={syncAnchor}
                            onClose={() => { setSyncAnchor(null); stopCalibration(); }}
                            anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
                            transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
                        >
                            <Box sx={{ p: 2, width: 240 }}>
                                <Typography variant="subtitle2" sx={{ mb: 1 }}>Link sync</Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', minHeight: 30 }}>
                                    <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>Output delay</Typography>
                                    <Typography variant="caption">{measuredLatencyMs ?? '…'} ms</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                                    <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>Offset</Typography>
                                    <Typography variant="caption" sx={{ mr: 0.5 }}>
                                        {linkOffsetMs > 0 ? '+' : ''}{linkOffsetMs} ms
                                    </Typography>
                                    <IconButton size="small" aria-label="1 ms less" onClick={() => updateLinkOffset(linkOffsetMs - 1)}>
                                        <MinusIcon size={14} />
                                    </IconButton>
                                    <IconButton size="small" aria-label="1 ms more" onClick={() => updateLinkOffset(linkOffsetMs + 1)}>
                                        <PlusIcon size={14} />
                                    </IconButton>
                                </Box>
                                <Slider
                                    size="small" min={LINK_OFFSET_MIN} max={LINK_OFFSET_MAX} step={1}
                                    value={linkOffsetMs}
                                    onChange={(_, v) => updateLinkOffset(v)}
                                    marks={[{ value: 0 }]}
                                    aria-label="Link offset in milliseconds"
                                />
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                    <Button size="small" variant={calibrating ? 'contained' : 'outlined'} onClick={toggleCalibration}>
                                        {calibrating ? 'Stop' : 'Calibrate'}
                                    </Button>
                                    <Button size="small" onClick={() => updateLinkOffset(0)} disabled={linkOffsetMs === 0}>
                                        Reset
                                    </Button>
                                </Box>
                                {calibrating && (
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                                        Match the clicks to Live's metronome.
                                    </Typography>
                                )}
                            </Box>
                        </Popover>
                    </>
                )}

                {/* MIDI learn toggle — same compact rectangle style as Link. */}
                <Tooltip
                    title={TIPS.perf.midiMode({ supported: midi?.supported, permissionError: midi?.permissionError, learnMode: midi?.learnMode })}
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
                                px: 1.75,
                                minWidth: 52,
                                height: perfTokens.height.compact,
                                borderRadius: '999px',
                                bgcolor: midi?.learnMode ? perfColors.toggleOn : perfColors.toggleOff,
                                color: onFill(midi?.learnMode ? perfColors.toggleOn : perfColors.toggleOff),
                                opacity: midi?.supported ? 1 : 0.45,
                                transition: 'background-color 120ms',
                                '&:hover': {
                                    bgcolor: midi?.learnMode ? perfColors.toggleOnHover : perfColors.toggleOffHover,
                                },
                                '&.Mui-disabled': {
                                    color: perfColors.label,
                                },
                            }}
                        >
                            MIDI Learn
                        </ButtonBase>
                    </span>
                </Tooltip>


                <Tooltip title={TIPS.perf.presets}>
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
                                    <Tooltip title={TIPS.perf.deletePreset} placement="left" arrow>
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

                    {/* Danger zone — armed-to-confirm wipes session config, fragments,
                        and MIDI mappings. Auto-disarms after 3 s. */}
                    <Box sx={{ px: 0.75, py: 0.75 }}>
                        <Tooltip
                            title={TIPS.perf.restoreDefaults(restoreArmed)}
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

                <Tooltip placement="right" title={TIPS.perf.launchQuant}>
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
                                fontWeight: 600,
                                paddingLeft: 0,
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
                            width: 76,
                            '& .MuiOutlinedInput-root': {
                                ...styles.pillControl['& .MuiOutlinedInput-root'],
                                pl: 1.25,
                                pr: 1.25,
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

                {/* Icon-only transport — titles live in the tooltips. */}
                <MidiMappable id="master.playAll" label="Play All" kind="trigger" onChange={handlePlayAll}>
                    <Tooltip title="Play All">
                        <span style={{ display: 'inline-flex' }}>
                            <IconButton
                                size="small"
                                onClick={handlePlayAll}
                                disabled={!anyLoaded}
                                aria-label="Play All"
                                sx={styles.masterIconBtn(perfColors.play)}
                            >
                                <PlayAllIcon size={14} fill="currentColor" />
                            </IconButton>
                        </span>
                    </Tooltip>
                </MidiMappable>
                <MidiMappable id="master.stopAll" label="Stop All" kind="trigger" onChange={handleStopAll}>
                    <Tooltip title="Stop All">
                        <span style={{ display: 'inline-flex' }}>
                            <IconButton
                                size="small"
                                onClick={handleStopAll}
                                disabled={!anyPlaying}
                                aria-label="Stop All"
                                sx={styles.masterIconBtn(perfColors.stop)}
                            >
                                <StopAllIcon size={14} fill="currentColor" />
                            </IconButton>
                        </span>
                    </Tooltip>
                </MidiMappable>

                {/* Record the master output. Stopping opens a dialog to name
                    the capture, saved as a WAV in the output folder. */}
                <MidiMappable id="master.record" label="Record Performance" kind="trigger" onChange={handleToggleRecord}>
                    <Tooltip title={recording ? 'Stop recording' : 'Record performance'}>
                        <span style={{ display: 'inline-flex' }}>
                            <IconButton
                                size="small"
                                disableRipple
                                onClick={handleToggleRecord}
                                aria-label={recording ? 'Stop recording' : 'Record performance'}
                                sx={styles.masterIconBtn(perfColors.record)}
                            >
                                {recording
                                    ? <StopAllIcon size={14} fill="currentColor" />
                                    : <RecordIcon size={14} fill="currentColor" />}
                            </IconButton>
                        </span>
                    </Tooltip>
                </MidiMappable>

                {/* Generation settings: one summary pill; the model, LoRA,
                    steps and seed live in its popover (set once, rarely
                    changed mid-set). */}
                <ButtonBase
                    onClick={(e) => setGenMenuAnchor(e.currentTarget)}
                    aria-label="Generation settings"
                    sx={(theme) => ({
                        ml: 'auto',
                        height: perfTokens.height.compact,
                        px: 1.75,
                        gap: 0.75,
                        maxWidth: 360,
                        borderRadius: '999px',
                        bgcolor: theme.palette.fragmenta.control,
                        fontSize: perfTokens.fontSize.sm,
                        fontWeight: 600,
                        whiteSpace: 'nowrap',
                        '&:hover': { bgcolor: theme.palette.fragmenta.controlHover },
                    })}
                >
                    <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {genSummary}
                    </Box>
                    <ChevronDownIcon size={14} />
                </ButtonBase>
                <Popover
                    open={Boolean(genMenuAnchor)}
                    anchorEl={genMenuAnchor}
                    onClose={() => setGenMenuAnchor(null)}
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
                    transformOrigin={{ vertical: 'top', horizontal: 'center' }}
                    slotProps={{ paper: { sx: { width: 340, p: 2.5, mt: 1, display: 'flex', flexDirection: 'column', gap: 2 } } }}
                >
                    <Box>
                        <Box component="div" sx={{ ...perfTokens.caps, color: 'text.secondary', mb: 0.75 }}>Model</Box>
                    <FormControl size="small" sx={{ ...styles.pillControl, width: '100%' }}>
                        <Select
                            value={selectedModel || ''}
                            onChange={handleModelChange}
                            displayEmpty
                            renderValue={(value) => {
                                if (!value) return <em style={{ opacity: 0.6 }}>Model</em>;
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
                                    <Tooltip title={TIPS.perf.deleteFineTuned}>
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
                    </Box>
                    <Box>
                        <Box component="div" sx={{ ...perfTokens.caps, color: 'text.secondary', mb: 0.75 }}>LoRA</Box>
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
                            <Tooltip title={TIPS.perf.deleteLora}>
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
                            <FormControl size="small" disabled={loraDisabled} sx={{ ...styles.pillControl, width: '100%' }}>
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
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                        <Box component="span" sx={{ ...perfTokens.caps, color: 'text.secondary', width: 56 }}>
                            Steps
                        </Box>
                        <Tooltip
                            placement="right"
                            title={TIPS.perf.steps(isDistilledSA3)}
                        >
                            <FormControl size="small" sx={{ ...styles.pillControl, width: 90 }}>

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
                        <Box component="span" sx={{ ...perfTokens.caps, color: 'text.secondary', width: 56 }}>
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
                            sx={{ ...styles.pillControl, flex: 1 }}
                        />
                    </Box>
                </Popover>

                {/* Prompt auto-inject — Key and the master tempo, appended to
                    every generated prompt when set. */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box component="span" sx={perfTokens.labelMuted}>
                        Add to prompt
                    </Box>
                    <Tooltip title={TIPS.perf.promptKey} placement="top" arrow enterDelay={500}>
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
                        title={TIPS.perf.bpmInject(promptInjectBpm, bpm)}
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
                                borderRadius: '999px',
                                // Lights up only when on; otherwise a plain control pill.
                                backgroundColor: promptInjectBpm ? perfColors.master : theme.palette.fragmenta.control,
                                color: promptInjectBpm ? onFill(perfColors.master) : 'text.secondary',
                                fontSize: perfTokens.fontSize.sm,
                                fontWeight: perfTokens.weight.bold,
                                fontVariantNumeric: 'tabular-nums',
                                transition: 'background-color 120ms, color 120ms, border-color 120ms',
                                '&:hover': {
                                    backgroundColor: promptInjectBpm ? perfColors.master : theme.palette.fragmenta.controlHover,
                                    color: promptInjectBpm ? onFill(perfColors.master) : 'text.secondary',
                                },
                            })}
                        >
                            BPM
                        </ButtonBase>
                    </Tooltip>
                </Box>
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
                            sidechainOwner={sidechainOwner}
                            onStateChange={handleChannelStateChange}
                            onFormStateChange={handleChannelFormChange}
                            initialFormState={session.channels[i]}
                            maxDuration={maxDuration}
                            bpm={bpm}
                            panelActive={active}
                        />
                    ))}
                </Box>

                <Box sx={styles.masterStrip(perfColors.master)}>
                    <Box sx={styles.masterHeader(perfColors.master)}>
                        <Box sx={styles.masterBadge(perfColors.master)}>Master</Box>
                    </Box>

                    {/* Output level: a big knob on the fader taper (faderLaw), so
                        MIDI and the stored value stay in dB. Double-click: 0 dB. */}
                    <MidiMappable
                        id="master.fader"
                        label="Master Fader"
                        kind="continuous"
                        curve="fader"
                        min={MASTER_DB_MIN}
                        max={MASTER_DB_MAX}
                        value={masterDb}
                        onChange={(v) => handleMasterChange(null, v)}
                    >
                        <RotaryKnob
                            value={faderDbToPos(masterDb)}
                            min={0}
                            max={1}
                            step={0.005}
                            detent={0}
                            center={FADER_UNITY_POS}
                            origin={0}
                            size={96}
                            valueInside={false}
                            ariaLabel="Master output level"
                            format={(pos) => `${formatDb(faderPosToDb(pos))} dB`}
                            onChange={(pos) => handleMasterChange(null, faderPosToDb(pos))}
                        />
                    </MidiMappable>
                    <Box sx={styles.masterReadouts}>
                        <Typography sx={styles.masterValue}>{formatDb(masterDb)} dB</Typography>
                        <Typography sx={styles.masterCaption}>Output</Typography>
                    </Box>

                    {/* Left/right meters on a dB scale, with peak-hold markers. */}
                    <Box sx={styles.masterMeters} role="img" aria-label="Master output meters">
                        <Box sx={styles.masterScale} aria-hidden>
                            {METER_SCALE_DB.map((db) => (
                                <Box key={db} component="span" sx={styles.masterScaleLabel(((db - METER_FLOOR_DB) / -METER_FLOOR_DB) * 100)}>
                                    {db === METER_FLOOR_DB ? '−∞' : db === 0 ? '0' : `−${-db}`}
                                </Box>
                            ))}
                        </Box>
                        {['L', 'R'].map((ch, i) => (
                            <Box key={ch} sx={styles.masterMeterCol}>
                                <Box sx={styles.masterMeterTrack}>
                                    <Box ref={meterFillRefs[i]} sx={styles.masterMeterFill(perfColors.master)} />
                                    <Box ref={peakMarkerRefs[i]} sx={styles.masterPeakMarker} />
                                </Box>
                                <Typography component="span" sx={styles.masterMeterLabel}>{ch}</Typography>
                            </Box>
                        ))}
                    </Box>
                    <Typography sx={styles.masterPeakValue}>Peak {formatDb(peakLabelDb)} dB</Typography>

                    {/* Master FX pickers — Reverb IR and Delay division.
                        No wet sliders: the shared FX run always-on, and the
                        per-channel DLY/REV knobs drive the send amounts. */}
                    <Box sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 0.75,
                        alignSelf: 'stretch',
                        mt: 'auto',
                    }}>
                        {/* Output channel pairs (from the device's channel
                            count), above the shared FX. */}
                        {(() => {
                            const pairCount = Math.max(1, Math.floor(maxChannelCount / 2));
                            const pairs = Array.from({ length: pairCount }, (_, i) => ({
                                value: i,
                                label: `${i * 2 + 1}–${i * 2 + 2}`,
                            }));
                            const picker = (label, value, onChange) => (
                                <FormControl size="small" sx={{ ...styles.pillControl, width: '100%' }}>
                                    <Select
                                        value={Math.min(value, pairCount - 1)}
                                        onChange={(e) => onChange(Number(e.target.value))}
                                        renderValue={(v) => `${label} · ${pairs[v]?.label ?? '—'}`}
                                    >
                                        {pairs.map((p) => (
                                            <MenuItem key={p.value} value={p.value} sx={{ fontSize: perfTokens.fontSize.sm }}>
                                                Ch {p.label}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>
                            );
                            return (
                                <>
                                    {picker('Main', mainOutputPair, setMainOutputPair)}
                                    {picker('Cue', cueOutputPair, setCueOutputPair)}
                                </>
                            );
                        })()}
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


            {/* Name + save dialog for a finished master recording. */}
            <Dialog
                open={Boolean(pendingRecording)}
                onClose={savingRecording ? undefined : handleDiscardRecording}
                maxWidth="xs"
                fullWidth
            >
                <DialogTitle>Save performance recording</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" sx={{ mb: 1.5, color: 'text.secondary' }}>
                        {(() => {
                            const s = Math.round(pendingRecording?.durationSec ?? 0);
                            const mm = Math.floor(s / 60);
                            const ss = String(s % 60).padStart(2, '0');
                            return `Captured ${mm}:${ss}. Name the file — it will be saved to your output folder.`;
                        })()}
                    </Typography>
                    <TextField
                        autoFocus
                        fullWidth
                        size="small"
                        label="Recording name"
                        value={recordingName}
                        onChange={(e) => setRecordingName(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && recordingName.trim() && !savingRecording) {
                                handleSaveRecording();
                            }
                        }}
                        disabled={savingRecording}
                        InputProps={{
                            endAdornment: (
                                <InputAdornment position="end">
                                    <Typography component="span" sx={{ color: 'text.disabled' }}>.wav</Typography>
                                </InputAdornment>
                            ),
                        }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleDiscardRecording} disabled={savingRecording} color="inherit">
                        Discard
                    </Button>
                    <Button
                        onClick={handleSaveRecording}
                        variant="contained"
                        disabled={savingRecording || !recordingName.trim()}
                    >
                        {savingRecording ? 'Saving…' : 'Save'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}
