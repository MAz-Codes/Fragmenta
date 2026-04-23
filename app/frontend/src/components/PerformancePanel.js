import React, { useEffect, useMemo, useRef, useState } from 'react';
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
} from '@mui/material';
import {
    Piano as PerformanceIcon,
    Play as PlayAllIcon,
    Square as StopAllIcon,
    Trash2 as DeleteIcon,
} from 'lucide-react';
import api from '../api';
import PerformanceChannel from './PerformanceChannel';
import { PerformanceEngine } from '../utils/performanceAudio';
import { performancePanelStyles as styles } from '../theme';

const CHANNEL_COUNT = 4;
const MASTER_COLOR = '#35C2D4';
const MASTER_DB_MIN = -60;
const MASTER_DB_MAX = 0;
const MASTER_DB_DEFAULT = -1;
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

export default function PerformancePanel({
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
    const [error, setError] = useState(null);
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
        const raw = Number(event.target.value);
        if (!Number.isFinite(raw)) return;
        setBpm(Math.max(BPM_MIN, Math.min(BPM_MAX, Math.round(raw))));
    };

    const handlePlayAll = () => engineRef.current?.playAll(true);
    const handleStopAll = () => engineRef.current?.stopAll();

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
            <Paper sx={styles.headerCard}>
                <Box sx={styles.headerLeft}>
                    <Box sx={styles.titleRow}>
                        <PerformanceIcon size={22} />
                        <Typography variant="h6" sx={styles.title}>Fragmenta Performance</Typography>
                    </Box>
                    <Typography variant="caption" sx={styles.subtitle}>
                        4-voice diffusion sampler
                    </Typography>
                </Box>

                <Box sx={styles.headerPickers}>
                    <TextField
                        size="small"
                        type="number"
                        label="BPM"
                        value={bpm}
                        onChange={handleBpmChange}
                        inputProps={{ min: BPM_MIN, max: BPM_MAX, step: 1 }}
                        sx={{ width: 90, '& .MuiOutlinedInput-root': { borderRadius: 2 } }}
                    />
                    <FormControl size="small" sx={styles.headerModelPicker}>
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
                                            // Prevent the MenuItem's select handler from firing when
                                            // the user clicks delete. onMouseDown beats Select's onChange.
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

                    {unwrappedModels.length > 0 && (
                        <FormControl size="small" sx={styles.headerCheckpointPicker}>
                            <Select
                                value={checkpointValue}
                                onChange={handleCheckpointChange}
                                displayEmpty
                                renderValue={(value) => {
                                    if (!value) return <em style={{ opacity: 0.6 }}>Select checkpoint</em>;
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
                        <Slider
                            orientation="vertical"
                            value={masterDb}
                            onChange={handleMasterChange}
                            min={MASTER_DB_MIN}
                            max={MASTER_DB_MAX}
                            step={0.1}
                            sx={styles.masterFader(MASTER_COLOR)}
                        />
                    </Box>

                    <Box sx={styles.masterReadouts}>
                        <Typography variant="caption" sx={styles.masterValue}>
                            dBFS {formatDb(masterDb)}
                        </Typography>
                        <Typography variant="caption" sx={styles.masterPeakValue}>
                            pk {formatDb(peakLabelDb)}
                        </Typography>
                    </Box>

                    <Box sx={styles.masterTransport}>
                        <Button
                            size="small"
                            variant="outlined"
                            startIcon={<PlayAllIcon size={14} />}
                            onClick={handlePlayAll}
                            disabled={!anyLoaded}
                            sx={styles.masterBtn(MASTER_COLOR, 'play')}
                            fullWidth
                        >
                            Play All
                        </Button>
                        <Button
                            size="small"
                            variant="outlined"
                            startIcon={<StopAllIcon size={14} />}
                            onClick={handleStopAll}
                            disabled={!anyPlaying}
                            sx={styles.masterBtn(MASTER_COLOR, 'stop')}
                            fullWidth
                        >
                            Stop All
                        </Button>
                    </Box>
                </Box>
            </Box>
        </Box>
    );
}
