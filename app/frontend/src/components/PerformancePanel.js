import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Box,
    Typography,
    Paper,
    Slider,
    Button,
    Alert,
    FormControl,
    Select,
    MenuItem,
    TextField,
    IconButton,
    Tooltip,
    ButtonBase,
} from '@mui/material';
import {
    Play as PlayAllIcon,
    Square as StopAllIcon,
    Trash2 as DeleteIcon,
    Settings as SettingsIcon,
} from 'lucide-react';
import api from '../api';
import PerformanceChannel from './PerformanceChannel';
import { PerformanceEngine } from '../utils/performanceAudio';
import { performancePanelStyles as styles } from '../theme';
import { MidiProvider, MidiMappable, useMidi } from './MidiContext';
import MidiConfigMenu from './MidiConfigMenu';

const CHANNEL_COUNT = 4;
const MASTER_COLOR = '#35C2D4';
const MASTER_DB_MIN = -60;
const MASTER_DB_MAX = 0;
const MASTER_DB_DEFAULT = -6;
const METER_FLOOR_DB = -60;
const BPM_MIN = 20;
const BPM_MAX = 300;
const BPM_DEFAULT = 120;

const dbToGain = (db) => (db <= MASTER_DB_MIN ? 0 : Math.pow(10, db / 20));
const ampToDb = (amp) => (amp <= 0 ? -Infinity : 20 * Math.log10(amp));
const formatDb = (db) => {
    if (!isFinite(db) || db <= METER_FLOOR_DB) return '−∞';
    if (Math.abs(db) < 0.05) return '0.0';
    return db.toFixed(1);
};

export default function PerformancePanel(props) {
    return (
        <MidiProvider>
            <PerformancePanelInner {...props} />
        </MidiProvider>
    );
}

function PerformancePanelInner({
    selectedModel,
    selectedUnwrappedModel,
    availableModels = [],
    baseModels = [],
    onSelectModel,
    onSelectUnwrappedModel,
    onRefreshModels,
}) {
    const engineRef = useRef(null);
    const meterFillRef = useRef(null);
    const peakHoldRef = useRef({ db: METER_FLOOR_DB, decayedAt: performance.now() });
    const meterRafRef = useRef(null);
    const [engineReady, setEngineReady] = useState(false);
    const [masterDb, setMasterDb] = useState(MASTER_DB_DEFAULT);
    const [bpm, setBpm] = useState(BPM_DEFAULT);
    // Separate "what the field displays" from "what the app commits". Lets
    // the user type "1" → "11" → "112" without the first keystroke getting
    // clamped up to BPM_MIN mid-typing (which then produced "201", "211", etc).
    const [bpmInput, setBpmInput] = useState(String(BPM_DEFAULT));
    const bpmInputFocusedRef = useRef(false);
    const [error, setError] = useState(null);
    const [linkAvailable, setLinkAvailable] = useState(false);
    const [linkEnabled, setLinkEnabled] = useState(false);
    const [linkPeers, setLinkPeers] = useState(0);
    const [linkInstalling, setLinkInstalling] = useState(false);
    // Tracks whether the most recent bpm change came from Link (poll response)
    // or from the user (typing in the field). Prevents an echo loop where a
    // Link-driven update gets pushed back to Link as if it were a local edit.
    const bpmOriginRef = useRef('user');
    const [peakLabelDb, setPeakLabelDb] = useState(METER_FLOOR_DB);
    const [channelStates, setChannelStates] = useState(() =>
        Array.from({ length: CHANNEL_COUNT }, () => ({ loaded: false, playing: false }))
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
        // Only commit the numeric bpm if what the user has typed so far is a
        // complete, in-range value. Intermediate digits ("1" on the way to
        // "112") are held in bpmInput without disturbing the committed bpm.
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

    // Mirror committed bpm back into the field — but only when the user isn't
    // currently typing, so Link-driven updates don't overwrite a draft.
    useEffect(() => {
        if (!bpmInputFocusedRef.current) setBpmInput(String(bpm));
    }, [bpm]);

    // Probe whether the backend has an Ableton Link binding installed.
    useEffect(() => {
        api.get('/api/link/state')
            .then((r) => {
                setLinkAvailable(Boolean(r.data?.available));
                setLinkEnabled(Boolean(r.data?.enabled));
            })
            .catch(() => setLinkAvailable(false));
    }, []);

    // Poll Link state while enabled; pull BPM and peer count into local state.
    useEffect(() => {
        if (!linkEnabled) return undefined;
        let cancelled = false;
        const poll = async () => {
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
            } catch {
                /* transient network blip — next tick will retry */
            }
        };
        poll();
        const timer = setInterval(poll, 500);
        return () => { cancelled = true; clearInterval(timer); };
    }, [linkEnabled]);

    // User-initiated BPM changes are pushed to the Link session. Changes that
    // originated from a Link poll are suppressed here (see bpmOriginRef).
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
                // Auto-enable after a successful install — the user just asked for Link.
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

    const handlePlayAll = () => engineRef.current?.playAll(true);
    const handleStopAll = () => engineRef.current?.stopAll();

    // Apply a BPM value coming from MIDI (or any non-typing source). Clamps,
    // rounds, and routes through the same origin tracking the input field uses.
    const applyExternalBpm = useCallback((value) => {
        const next = Math.max(BPM_MIN, Math.min(BPM_MAX, Math.round(value)));
        bpmOriginRef.current = 'user';
        setBpm(next);
    }, []);

    const midi = useMidi();
    const [midiMenuAnchor, setMidiMenuAnchor] = useState(null);

    // Esc exits MIDI learn mode without dismissing the panel.
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

    const generateForChannel = async ({ prompt, duration }) => {
        setError(null);
        if (!selectedModel) {
            const msg = 'Pick a model first.';
            setError(msg);
            throw new Error(msg);
        }
        const requestData = {
            prompt,
            duration,
            cfg_scale: 7.0,
            seed: Math.floor(Math.random() * 0xffffffff),
            model_name: selectedModel,
            ...(selectedUnwrappedModel ? { unwrapped_model_path: selectedUnwrappedModel } : {}),
        };
        const response = await api.post('/api/generate', requestData, { responseType: 'blob' });
        return response.data;
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
    const maxDuration = selectedModel && selectedModel.includes('small') ? 10 : 47;

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
        if (onSelectUnwrappedModel) onSelectUnwrappedModel('');
    };

    const handleCheckpointChange = (event) => {
        if (onSelectUnwrappedModel) onSelectUnwrappedModel(String(event.target.value));
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
                onSelectUnwrappedModel?.('');
            }
            onRefreshModels?.();
        } catch (err) {
            const msg = err?.response?.data?.error || err.message || 'Delete failed';
            setError(`Failed to delete "${modelName}": ${msg}`);
        }
    };

    const selectedFineTuned = selectedModel
        ? availableModels.find((m) => m.name === selectedModel)
        : null;
    const unwrappedModels = selectedFineTuned?.unwrapped_models || [];
    const checkpointValue = unwrappedModels
        .map((u) => String(u.path))
        .includes(selectedUnwrappedModel)
        ? selectedUnwrappedModel
        : '';

    return (
        <Box sx={styles.root}>
            <Paper sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                px: 1.25,
                py: 0.75,
                borderRadius: 2,
                border: '1px solid',
                borderColor: 'divider',
                background: 'linear-gradient(135deg, rgba(53, 194, 212, 0.05) 0%, rgba(159, 138, 230, 0.04) 100%)',
                flexWrap: { xs: 'wrap', md: 'nowrap' },
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
                                fontFamily: 'inherit',
                                fontSize: '0.72rem',
                                fontWeight: 600,
                                px: 1,
                                minWidth: 46,
                                height: 26,
                                borderRadius: '2px',
                                bgcolor: linkEnabled ? '#F5C542' : '#6e6e6e',
                                color: linkEnabled ? '#000' : '#2a2a2a',
                                opacity: linkInstalling ? 0.55 : 1,
                                transition: 'background-color 120ms',
                                '&:hover': {
                                    bgcolor: linkEnabled ? '#FFD54F' : '#7d7d7d',
                                },
                                '&.Mui-disabled': {
                                    color: linkEnabled ? '#000' : '#2a2a2a',
                                },
                            }}
                        >
                            {linkInstalling
                                ? 'installing…'
                                : `Link${linkEnabled && linkPeers > 0 ? ` · ${linkPeers}` : ''}`}
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
                                fontFamily: 'inherit',
                                fontSize: '0.72rem',
                                fontWeight: 600,
                                px: 1,
                                minWidth: 46,
                                height: 26,
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
                            sx={{ width: 26, height: 26, color: 'text.secondary' }}
                        >
                            <SettingsIcon size={14} />
                        </IconButton>
                    </span>
                </Tooltip>
                <MidiConfigMenu
                    anchorEl={midiMenuAnchor}
                    open={Boolean(midiMenuAnchor)}
                    onClose={() => setMidiMenuAnchor(null)}
                />

                {/* Tempo — outlined field matching other inputs. The floating-label
                    notch looked awkward at this width, so the unit label lives
                    inline as an endAdornment instead. */}
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
                        inputProps={{ step: 1, inputMode: 'numeric', 'aria-label': 'Tempo in BPM' }}
                        InputProps={{
                            endAdornment: (
                                <Typography
                                    component="span"
                                    sx={{
                                        fontSize: '0.62rem',
                                        letterSpacing: '0.08em',
                                        color: 'text.disabled',
                                        pl: 0.5,
                                        userSelect: 'none',
                                    }}
                                >
                                    BPM
                                </Typography>
                            ),
                        }}
                        sx={{
                            width: 96,
                            '& .MuiOutlinedInput-root': { borderRadius: 1.5, pr: 1 },
                            '& input': {
                                textAlign: 'right',
                                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                                fontVariantNumeric: 'tabular-nums',
                                pr: 0,
                            },
                            '& input::-webkit-outer-spin-button, & input::-webkit-inner-spin-button': {
                                WebkitAppearance: 'none',
                                margin: 0,
                            },
                            '& input[type=number]': { MozAppearance: 'textfield' },
                        }}
                    />
                </MidiMappable>

                {/* Model picker — half the old width */}
                <FormControl size="small" sx={{
                    flex: 1,
                    minWidth: 110,
                    '& .MuiOutlinedInput-root': { borderRadius: 1.5 },
                }}>
                    <Select
                        value={selectedModel || ''}
                        onChange={handleModelChange}
                        displayEmpty
                        renderValue={(value) => {
                            if (!value) return <em style={{ opacity: 0.6 }}>Select a model</em>;
                            const base = baseModels.find((m) => m.name === value);
                            if (base) return base.displayName || base.name;
                            return value;
                        }}
                    >
                        {baseModels.length > 0 && (
                            <MenuItem disabled>
                                <Typography variant="caption" color="textSecondary">
                                    ── Base Models ──
                                </Typography>
                            </MenuItem>
                        )}
                        {baseModels.map((model) => (
                            <MenuItem
                                key={model.name}
                                value={String(model.name)}
                                disabled={!model.downloaded}
                            >
                                <Typography variant="body2">
                                    {model.displayName || model.name}
                                </Typography>
                            </MenuItem>
                        ))}
                        {availableModels.length > 0 && (
                            <MenuItem disabled>
                                <Typography variant="caption" color="textSecondary">
                                    ── Fine-tuned Models ──
                                </Typography>
                            </MenuItem>
                        )}
                        {availableModels.map((model) => (
                            <MenuItem
                                key={model.name}
                                value={String(model.name)}
                                sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, pr: 0.5 }}
                            >
                                <Typography variant="body2" sx={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {model.name}
                                </Typography>
                                <Tooltip title="Delete fine-tuned model">
                                    <IconButton
                                        size="small"
                                        onMouseDown={(e) => {
                                            e.stopPropagation();
                                            e.preventDefault();
                                        }}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            e.preventDefault();
                                            handleDeleteFineTuned(model.name);
                                        }}
                                        sx={{
                                            color: 'text.disabled',
                                            '&:hover': { color: 'error.main', bgcolor: 'action.hover' },
                                        }}
                                    >
                                        <DeleteIcon size={14} />
                                    </IconButton>
                                </Tooltip>
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>

                {/* Checkpoint picker — also halved */}
                {unwrappedModels.length > 0 && (
                    <FormControl size="small" sx={{
                        flex: 1,
                        minWidth: 100,
                        '& .MuiOutlinedInput-root': { borderRadius: 1.5 },
                    }}>
                        <Select
                            value={checkpointValue}
                            onChange={handleCheckpointChange}
                            displayEmpty
                            renderValue={(value) => {
                                if (!value) return <em style={{ opacity: 0.6 }}>Checkpoint</em>;
                                const found = unwrappedModels.find((u) => String(u.path) === value);
                                return found ? found.name : value;
                            }}
                        >
                            {unwrappedModels.map((unwrapped, index) => (
                                <MenuItem key={index} value={String(unwrapped.path)}>
                                    <Box>
                                        <Typography variant="body2">{unwrapped.name}</Typography>
                                        {unwrapped.size_mb && (
                                            <Typography variant="caption" color="textSecondary">
                                                {unwrapped.size_mb} MB
                                            </Typography>
                                        )}
                                    </Box>
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                )}

                {/* Transport */}
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
                            onGenerate={generateForChannel}
                            canGenerate={Boolean(selectedModel)}
                            onMuteSoloChange={handleMuteSoloChange}
                            onStateChange={handleChannelStateChange}
                            maxDuration={maxDuration}
                            bpm={bpm}
                        />
                    ))}
                </Box>

                <Box sx={styles.masterStrip(MASTER_COLOR)}>
                    <Box sx={styles.masterHeader(MASTER_COLOR)}>
                        <Box sx={styles.masterBadge(MASTER_COLOR)}>MASTER</Box>
                    </Box>

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
                            sx={{ flex: 1, alignSelf: 'stretch' }}
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
                        <Typography variant="caption" sx={styles.masterValue}>
                            dBFS {formatDb(masterDb)}
                        </Typography>
                        <Typography variant="caption" sx={styles.masterPeakValue}>
                            pk {formatDb(peakLabelDb)}
                        </Typography>
                    </Box>
                </Box>
            </Box>
        </Box>
    );
}
