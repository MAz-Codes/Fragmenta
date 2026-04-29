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
} from 'lucide-react';
import { performanceChannelStyles as styles, perfTokens } from '../theme';
import { MidiMappable } from './MidiContext';

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
    { key: 'filter', label: 'LPF', min: 200, max: 18000, step: 1, default: 18000, log: true },
    { key: 'delay', label: 'DLY', min: 0, max: 1.0, step: 0.01, default: 0.0 },
    { key: 'reverb', label: 'REV', min: 0, max: 1.0, step: 0.01, default: 0.0 },
];

const PAN_CENTER_SNAP = 0.06;

const BARS_OPTIONS = [1, 2, 4, 8, 16];
const BEATS_PER_BAR = 4;

export default function PerformanceChannel({
    index,
    strip,
    engine,
    playing = false,
    onGenerate,
    canGenerate,
    onMuteSoloChange,
    onStateChange,
    maxDuration = 47,
    bpm = 120,
}) {
    const color = CHANNEL_COLORS[index % CHANNEL_COLORS.length];
    const canvasRef = useRef(null);
    const meterRef = useRef(null);
    const meterRafRef = useRef(null);

    const [prompt, setPrompt] = useState('');
    const [duration, setDuration] = useState(8);
    const [durationMode, setDurationMode] = useState('seconds');
    const [bars, setBars] = useState(4);
    const [generating, setGenerating] = useState(false);
    const [loaded, setLoaded] = useState(false);
    const [looping, setLooping] = useState(true);
    const [muted, setMuted] = useState(false);
    const [soloed, setSoloed] = useState(false);
    const [knobs, setKnobs] = useState(() => {
        const initial = Object.fromEntries(KNOB_DEFS.map(k => [k.key, k.default]));
        initial.pan = 0;
        return initial;
    });

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
        const effectiveDuration = durationMode === 'bars' ? secondsFromBars : duration;
        setGenerating(true);
        try {
            const blob = await onGenerate({ prompt, duration: effectiveDuration });
            await strip.loadBlob(blob);
            setLoaded(true);
            onStateChange?.(index, { loaded: true });
            requestAnimationFrame(drawWave);
        } catch (err) {
            console.error(`Channel ${index + 1} generate failed:`, err);
        } finally {
            setGenerating(false);
        }
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
                                        fontFamily: 'inherit',
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
                {KNOB_DEFS.map((k) => (
                    <Box key={k.key} sx={styles.knobCell}>
                        <MidiMappable
                            id={ctrlId(k.key)}
                            label={ctrlLabel(k.label)}
                            kind="continuous"
                            curve={k.log ? 'log' : 'linear'}
                            min={k.min}
                            max={k.max}
                            value={knobs[k.key]}
                            onChange={(v) => handleKnob(k.key, v)}
                            sx={{ alignItems: 'center' }}
                        >
                            <Slider
                                orientation="vertical"
                                value={knobs[k.key]}
                                onChange={(_, v) => handleKnob(k.key, v)}
                                min={k.min}
                                max={k.max}
                                step={k.step}
                                size="small"
                                sx={styles.knobSlider(color, k.key === 'gain')}
                            />
                        </MidiMappable>
                        <Box component="span" sx={styles.knobLabel}>{k.label}</Box>
                    </Box>
                ))}
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
                    <IconButton
                        onClick={handleLoopToggle}
                        sx={styles.loopBtn(color, looping)}
                        size="small"
                    >
                        <LoopIcon size={14} />
                    </IconButton>
                </MidiMappable>
                <Box sx={styles.meterTrack}>
                    <Box ref={meterRef} sx={styles.meterFill(color)} />
                </Box>
            </Box>

        </Box>
    );
}
