import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import {
    Box,
    Typography,
    TextField,
    IconButton,
    Slider,
    CircularProgress,
    Tooltip,
    Select,
    MenuItem,
    ButtonBase,
} from '@mui/material';
import {
    Play as PlayIcon,
    Square as StopIcon,
    Repeat as LoopIcon,
    Sparkles as GenerateIcon,
    Volume2 as VolumeIcon,
    VolumeX as MuteIcon,
    Headphones as CueIcon,
    Check as CommitIcon,
} from 'lucide-react';
import { performanceChannelStyles as styles, perfTokens } from '../theme';
import { MidiMappable } from './MidiContext';
import { playBlob as playCueBlob, stopCue, isCueSupported } from '../utils/cueAudio';

const CHANNEL_COLORS = [
    '#35C2D4', '#9F8AE6', '#53C18A', '#E3A34B',
    '#E36C61', '#F08AD2', '#5BA0F0', '#A8D86B',
];

// Channel gain runs on the same dBFS scale as the master fader so the two
// scales line up: -60 dB floor, 0 dB ceiling, default at -6 dB. The knob's
// dB value is converted to linear before reaching the audio graph.
const GAIN_DB_MIN = -60;
const GAIN_DB_MAX = 0;
const GAIN_DB_DEFAULT = -6;
const gainDbToLinear = (db) => (db <= GAIN_DB_MIN ? 0 : Math.pow(10, db / 20));

const KNOB_DEFS = [
    { key: 'gain', label: 'GAIN', min: GAIN_DB_MIN, max: GAIN_DB_MAX, step: 0.5, default: GAIN_DB_DEFAULT },
    // LPF range goes from 20 Hz (full kill) to 20 kHz (bypass). We render the
    // slider on a log axis so each octave gets equal travel — without this
    // the bottom 5% of the knob does all the audible work.
    { key: 'filter', label: 'LPF', min: 20, max: 20000, step: 1, default: 20000, scale: 'log' },
    { key: 'delay', label: 'DLY', min: 0, max: 1.0, step: 0.01, default: 0.0 },
    { key: 'reverb', label: 'REV', min: 0, max: 1.0, step: 0.01, default: 0.0 },
];

const PAN_CENTER_SNAP = 0.06;

const BARS_OPTIONS = [1, 2, 4, 8, 16];
const BEATS_PER_BAR = 4;
const BATCH_OPTIONS = [1, 2, 3, 4];

export default function PerformanceChannel({
    index,
    strip,
    engine,
    playing = false,
    onGenerate,
    canGenerate,
    onMuteSoloChange,
    onStateChange,
    onFormStateChange,
    initialFormState,
    maxDuration = 380,
    bpm = 120,
}) {
    const color = CHANNEL_COLORS[index % CHANNEL_COLORS.length];
    const canvasRef = useRef(null);
    const meterRef = useRef(null);
    const meterRafRef = useRef(null);

    const init = initialFormState || {};
    const initKnobs = init.knobs || {};
    const defaultKnobs = (() => {
        const d = Object.fromEntries(KNOB_DEFS.map(k => [k.key, k.default]));
        d.pan = 0;
        return d;
    })();

    const [prompt, setPrompt] = useState(init.prompt ?? '');
    const [duration, setDuration] = useState(init.duration ?? 8);
    const [durationMode, setDurationMode] = useState(init.durationMode ?? 'seconds');
    const [bars, setBars] = useState(init.bars ?? 4);
    const [generating, setGenerating] = useState(false);
    const [loaded, setLoaded] = useState(false);
    const [looping, setLooping] = useState(init.looping ?? true);
    const [muted, setMuted] = useState(init.muted ?? false);
    const [soloed, setSoloed] = useState(init.soloed ?? false);
    const [batchSize, setBatchSize] = useState(init.batchSize ?? 1);
    const [knobs, setKnobs] = useState(() => ({ ...defaultKnobs, ...initKnobs }));

    // Candidates from the latest batch generation. Held in component state
    // because they don't survive a page reload — the blob URLs would be dead.
    // `committedIndex` tracks which one is currently loaded into the strip.
    const [candidates, setCandidates] = useState([]);
    const [auditioningIndex, setAuditioningIndex] = useState(null);
    const [committedIndex, setCommittedIndex] = useState(null);
    const cueSupported = useMemo(() => isCueSupported(), []);

    // Stop any active cue audition when the channel unmounts.
    useEffect(() => () => stopCue(), []);

    // Mirror form state up to the panel so it can persist the session. Skip the
    // first render so we don't re-write what we just loaded from localStorage.
    const initialReportSkippedRef = useRef(false);
    useEffect(() => {
        if (!initialReportSkippedRef.current) {
            initialReportSkippedRef.current = true;
            return;
        }
        onFormStateChange?.(index, {
            prompt, duration, durationMode, bars, looping, muted, soloed, batchSize, knobs,
        });
    }, [prompt, duration, durationMode, bars, looping, muted, soloed, batchSize, knobs, index, onFormStateChange]);

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
        const tick = () => {
            const el = meterRef.current;
            if (el && strip) {
                const level = strip.getLevel();
                el.style.width = `${Math.min(100, level * 140)}%`;
            }
            meterRafRef.current = requestAnimationFrame(tick);
        };
        meterRafRef.current = requestAnimationFrame(tick);
        return () => {
            if (meterRafRef.current) cancelAnimationFrame(meterRafRef.current);
        };
    }, [strip]);

    const drawWave = useCallback(() => {
        if (strip && canvasRef.current) {
            strip.drawWaveform(canvasRef.current, color);
        }
    }, [strip, color]);

    useEffect(() => { drawWave(); }, [drawWave, loaded]);

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
        strip.setFilter(knobs.filter);
        strip.setDelayMix(knobs.delay);
        strip.setReverbMix(knobs.reverb);
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

    const handleGenerate = async () => {
        if (!prompt.trim() || generating) return;
        const inBarsMode = durationMode === 'bars';
        const effectiveDuration = inBarsMode ? secondsFromBars : duration;
        setGenerating(true);
        // Stop any in-flight cue audition and clear stale candidate state so
        // the audition strip doesn't keep playing the old generation.
        stopCue();
        setAuditioningIndex(null);
        try {
            const result = await onGenerate({
                prompt,
                duration: effectiveDuration,
                batchSize,
                // Only forward alignment params in bars mode — seconds mode
                // generates raw audio with no post-processing.
                ...(inBarsMode ? { alignBars: bars, alignBpm: bpm } : {}),
                // Phase 7: bars-mode + channel-looping ⇒ ask the backend
                // to wrap-inpaint the seam so the clip loops seamlessly.
                ...(inBarsMode && looping ? { loopStitch: 'inpaint' } : {}),
            });
            const blobs = Array.isArray(result) ? result : [result];
            const next = blobs.map((b, i) => ({ index: i, blob: b }));
            setCandidates(next);
            // First candidate auto-loads into the channel strip; the rest sit
            // in the audition row until the user commits a different one.
            await strip.loadBlob(blobs[0]);
            setCommittedIndex(0);
            setLoaded(true);
            onStateChange?.(index, { loaded: true });
            requestAnimationFrame(drawWave);
        } catch (err) {
            console.error(`Channel ${index + 1} generate failed:`, err);
        } finally {
            setGenerating(false);
        }
    };

    const handleAudition = async (i) => {
        const candidate = candidates[i];
        if (!candidate) return;
        if (auditioningIndex === i) {
            stopCue();
            setAuditioningIndex(null);
            return;
        }
        setAuditioningIndex(i);
        try {
            await playCueBlob(candidate.blob, {
                onEnded: () => setAuditioningIndex(prev => (prev === i ? null : prev)),
            });
        } catch (err) {
            console.warn(`Channel ${index + 1} audition failed:`, err);
            setAuditioningIndex(null);
        }
    };

    const handleCommit = async (i) => {
        const candidate = candidates[i];
        if (!candidate || committedIndex === i) return;
        // Stop the live channel before swapping the buffer so we don't get a
        // glitch in the middle of a loop iteration.
        try { strip.stop(); } catch { /* not playing */ }
        onStateChange?.(index, { playing: false });
        await strip.loadBlob(candidate.blob);
        setCommittedIndex(i);
        requestAnimationFrame(drawWave);
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

    const handleKnob = (key, value) => {
        setKnobs(prev => ({ ...prev, [key]: value }));
        if (key === 'gain') strip.setUserGain(gainDbToLinear(value));
        else if (key === 'pan') strip.setPan(value);
        else if (key === 'filter') strip.setFilter(value);
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
        <Box sx={styles.strip(color, playing)}>
            <Box sx={styles.stripHeader(color)}>
                <Box sx={styles.channelBadge(color)}>{String(index + 1).padStart(2, '0')}</Box>
                <Box sx={styles.muteSoloRow}>
                    <MidiMappable id={ctrlId('mute')} label={ctrlLabel('Mute')} kind="trigger" onChange={handleMuteToggle}>
                        <Tooltip title="Mute">
                            <IconButton size="small" onClick={handleMuteToggle} sx={styles.muteBtn(muted)}>M</IconButton>
                        </Tooltip>
                    </MidiMappable>
                    <MidiMappable id={ctrlId('solo')} label={ctrlLabel('Solo')} kind="trigger" onChange={handleSoloToggle}>
                        <Tooltip title="Solo">
                            <IconButton size="small" onClick={handleSoloToggle} sx={styles.soloBtn(soloed)}>S</IconButton>
                        </Tooltip>
                    </MidiMappable>
                </Box>
            </Box>

            <Box sx={styles.promptBox}>
                <TextField
                    placeholder="prompt…"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    multiline
                    minRows={2}
                    maxRows={3}
                    size="small"
                    fullWidth
                    sx={styles.promptField}
                    disabled={generating}
                />
                <Box sx={{ ...styles.durationRow, minHeight: 26, height: 26 }}>
                    <Box
                        sx={{
                            display: 'inline-flex',
                            border: '1px solid',
                            borderColor: 'divider',
                            borderRadius: 0.75,
                            overflow: 'hidden',
                            height: '100%',
                        }}
                    >
                        {['sec', 'bars'].map((mode) => {
                            const value = mode === 'sec' ? 'seconds' : 'bars';
                            const active = durationMode === value;
                            return (
                                <ButtonBase
                                    key={mode}
                                    onClick={() => setDurationMode(value)}
                                    sx={{
                                        fontSize: perfTokens.fontSize.small,
                                        letterSpacing: perfTokens.letterSpacing.wide,
                                        textTransform: 'uppercase',
                                        px: 0.7,
                                        minWidth: 30,
                                        bgcolor: active ? color : 'transparent',
                                        color: active ? 'rgba(0,0,0,0.88)' : 'text.disabled',
                                        fontWeight: active ? 600 : 400,
                                        transition: 'background-color 120ms, color 120ms',
                                        '&:hover': {
                                            bgcolor: active ? color : 'action.hover',
                                            color: active ? 'rgba(0,0,0,0.88)' : 'text.secondary',
                                        },
                                    }}
                                >
                                    {mode}
                                </ButtonBase>
                            );
                        })}
                    </Box>

                    {durationMode === 'seconds' ? (
                        <>
                            <Typography variant="caption" sx={styles.durationLabel}>{duration.toFixed(0)}s</Typography>
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
                            sx={{
                                flex: 1,
                                fontSize: perfTokens.fontSize.body,
                                height: '100%',
                                '& .MuiOutlinedInput-input': {
                                    py: 0,
                                    pl: 1,
                                    minHeight: 'unset',
                                },
                                '& .MuiSelect-select': {
                                    py: 0,
                                    pl: 1,
                                    minHeight: 'unset',
                                },
                            }}
                        >
                            {availableBars.map(b => (
                                <MenuItem key={b} value={b} sx={{ fontSize: perfTokens.fontSize.body }}>
                                    {b} {b === 1 ? 'bar' : 'bars'}
                                </MenuItem>
                            ))}
                        </Select>
                    )}
                </Box>
                <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 1.5,
                    mt: 0.5,
                    width: '100%',
                }}>
                    <Tooltip
                        title="Batch generation: produce N candidates and audition them through the cue output before committing one to this channel."
                        placement="top"
                        disableFocusListener
                        disableTouchListener
                        enterDelay={500}
                    >
                        <Select
                            value={batchSize}
                            onChange={(e) => setBatchSize(Number(e.target.value))}
                            size="small"
                            disabled={generating}
                            sx={{
                                fontSize: perfTokens.fontSize.body,
                                height: 32,
                                minWidth: 64,
                                '& .MuiOutlinedInput-input': { py: 0, pl: 1.25, pr: '28px !important', minHeight: 'unset' },
                                '& .MuiSelect-select': { py: 0, pl: 1.25, pr: '28px !important', minHeight: 'unset' },
                            }}
                        >
                            {BATCH_OPTIONS.map(n => (
                                <MenuItem key={n} value={n} sx={{ fontSize: perfTokens.fontSize.body }}>
                                    ×{n}
                                </MenuItem>
                            ))}
                        </Select>
                    </Tooltip>
                    <MidiMappable id={ctrlId('generate')} label={ctrlLabel('Generate')} kind="trigger" onChange={handleGenerate}>
                        <IconButton
                            onClick={handleGenerate}
                            disabled={!canGenerate || !prompt.trim() || generating}
                            sx={styles.generateBtn(color)}
                            size="small"
                        >
                            {generating ? <CircularProgress size={16} sx={{ color }} /> : <GenerateIcon size={16} />}
                        </IconButton>
                    </MidiMappable>
                </Box>
            </Box>

            <Box sx={styles.waveformWrap}>
                <canvas
                    ref={canvasRef}
                    width={140}
                    height={42}
                    style={{ width: '100%', height: 42, display: 'block' }}
                />
                {!loaded && (
                    <Typography variant="caption" sx={styles.waveformPlaceholder}>
                        empty
                    </Typography>
                )}
            </Box>

            {candidates.length > 1 && (
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 0.5,
                        px: 1,
                        py: 0.5,
                        flexWrap: 'wrap',
                    }}
                >
                    {candidates.map((c, i) => {
                        const isAuditioning = auditioningIndex === i;
                        const isCommitted = committedIndex === i;
                        return (
                            <Box
                                key={c.index}
                                sx={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    border: '1px solid',
                                    borderColor: isCommitted ? color : 'divider',
                                    borderRadius: 0.75,
                                    overflow: 'hidden',
                                    bgcolor: isCommitted ? `${color}1a` : 'transparent',
                                }}
                            >
                                <Tooltip
                                    title={
                                        cueSupported
                                            ? (isAuditioning ? 'Stop cue audition' : 'Audition this take through cue output')
                                            : 'Cue audition requires Chrome/Edge. Plays through main output.'
                                    }
                                >
                                    <IconButton
                                        onClick={() => handleAudition(i)}
                                        size="small"
                                        sx={{
                                            color: isAuditioning ? color : 'text.secondary',
                                            px: 0.5,
                                            borderRadius: 0,
                                        }}
                                    >
                                        <CueIcon size={12} />
                                        <Box
                                            component="span"
                                            sx={{
                                                ml: 0.4,
                                                fontSize: perfTokens.fontSize.small,
                                                fontWeight: isAuditioning ? 700 : 500,
                                            }}
                                        >
                                            {i + 1}
                                        </Box>
                                    </IconButton>
                                </Tooltip>
                                <Tooltip title={isCommitted ? 'Currently in channel' : 'Use this take in the channel'}>
                                    <span>
                                        <IconButton
                                            onClick={() => handleCommit(i)}
                                            size="small"
                                            disabled={isCommitted}
                                            sx={{
                                                color: isCommitted ? color : 'text.disabled',
                                                px: 0.4,
                                                borderRadius: 0,
                                                borderLeft: '1px solid',
                                                borderColor: 'divider',
                                            }}
                                        >
                                            <CommitIcon size={12} />
                                        </IconButton>
                                    </span>
                                </Tooltip>
                            </Box>
                        );
                    })}
                </Box>
            )}

            <Box sx={{ px: 1, py: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Box component="span" sx={{ fontSize: perfTokens.fontSize.knob, color: 'text.secondary', letterSpacing: perfTokens.letterSpacing.wide, minWidth: 28 }}>PAN</Box>
                    <MidiMappable
                        id={ctrlId('pan')}
                        label={ctrlLabel('Pan')}
                        kind="continuous"
                        min={-1}
                        max={1}
                        value={knobs.pan ?? 0}
                        onChange={handlePan}
                        sx={{ flex: 1, flexDirection: 'row' }}
                    >
                        <Slider
                            value={knobs.pan ?? 0}
                            onChange={(_, v) => handlePan(v)}
                            min={-1}
                            max={1}
                            step={0.01}
                            size="small"
                            track={false}
                            marks={[{ value: 0 }]}
                            sx={{
                                flex: 1,
                                '& .MuiSlider-mark': {
                                    width: 2,
                                    height: 10,
                                    borderRadius: 1,
                                    backgroundColor: 'text.secondary',
                                    opacity: 0.8,
                                },
                                '& .MuiSlider-markActive': {
                                    backgroundColor: 'text.secondary',
                                    opacity: 0.8,
                                },
                            }}
                        />
                    </MidiMappable>
                </Box>
            </Box>

            <Box sx={styles.knobsGrid}>
                {KNOB_DEFS.map((k) => {
                    const isLog = k.scale === 'log';
                    // For log knobs, the slider drives a 0..1 position and we
                    // convert to/from the underlying value (Hz) on the audio
                    // boundary. The knob value stored in state stays in the
                    // domain unit (Hz here) so persistence and MIDI keep working.
                    const valueToPos = isLog
                        ? (v) => Math.log(Math.max(v, k.min) / k.min) / Math.log(k.max / k.min)
                        : (v) => v;
                    const posToValue = isLog
                        ? (p) => k.min * Math.pow(k.max / k.min, p)
                        : (v) => v;
                    return (
                        <Box key={k.key} sx={styles.knobCell}>
                            <MidiMappable
                                id={ctrlId(k.key)}
                                label={ctrlLabel(k.label)}
                                kind="continuous"
                                curve={isLog ? 'log' : 'linear'}
                                min={k.min}
                                max={k.max}
                                value={knobs[k.key]}
                                onChange={(v) => handleKnob(k.key, v)}
                                sx={{ alignItems: 'center' }}
                            >
                                <Slider
                                    orientation="vertical"
                                    value={valueToPos(knobs[k.key])}
                                    onChange={(_, v) => handleKnob(k.key, posToValue(v))}
                                    min={isLog ? 0 : k.min}
                                    max={isLog ? 1 : k.max}
                                    step={isLog ? 0.001 : k.step}
                                    size="small"
                                    sx={styles.knobSlider(color, k.key === 'gain')}
                                />
                            </MidiMappable>
                            <Box component="span" sx={styles.knobLabel}>{k.label}</Box>
                        </Box>
                    );
                })}
            </Box>

            <Box sx={styles.transportRow}>
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
                        title={
                            looping
                                ? (durationMode === 'bars'
                                    ? 'Seamless loop — next generation will be inpaint-stitched at the bar boundary'
                                    : 'Playback loop on')
                                : 'Loop off'
                        }
                        placement="top"
                        enterDelay={400}
                    >
                        <IconButton
                            onClick={handleLoopToggle}
                            sx={styles.loopBtn(color, looping)}
                            size="small"
                        >
                            <LoopIcon size={14} />
                        </IconButton>
                    </Tooltip>
                </MidiMappable>
                <Box sx={styles.meterTrack}>
                    <Box ref={meterRef} sx={styles.meterFill(color)} />
                </Box>
            </Box>

        </Box>
    );
}
