import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Typography,
    Button,
    Stack,
    TextField,
    ToggleButton,
    ToggleButtonGroup,
    Slider,
    Alert,
    LinearProgress,
    IconButton,
    Switch,
    FormControlLabel,
} from '@mui/material';
import { Upload as UploadIcon, X as ClearIcon, Play as PlayIcon, Square as StopIcon } from 'lucide-react';
import api from '../api';
import AudioWaveform from './AudioWaveform';

/**
 * SA3 audio-to-audio + inpainting UI.
 *
 * Three modes:
 *   - Style transfer: feed a source clip + new prompt, init_noise_level
 *     controls how much character is preserved (0 = source-faithful,
 *     1 = prompt-only).
 *   - Inpaint: regenerate a region of the source clip, keeping the rest.
 *   - Extend: append N seconds of new audio to the end of the source.
 *
 * All three send to /api/generate using SA3's init_audio / inpaint_audio
 * params. The backend handles file resolution; this panel just uploads
 * the source clip to /api/audio/upload and posts the returned path.
 *
 * Props:
 *   model_id:        active SA3 model id
 *   negativePrompt:  optional, passed through
 *   loraStack:       [{path, strength, bypassed}] from the Generation panel —
 *                    applied to the edit so style/inpaint/extend inherit the
 *                    same LoRA character as plain generation.
 *   steps:           sampler step count from the Generation panel.
 *   cfgScale:        CFG from the Generation panel (only sent for *-base models;
 *                    distilled models bake CFG at 1.0).
 *   onGenerated(blob, filename, params): called with the resulting WAV
 */
export default function EditPanel({ model_id, negativePrompt, loraStack, steps, cfgScale, onGenerated }) {
    const [mode, setMode] = useState('style');   // 'style' | 'inpaint' | 'extend'
    const [sourcePath, setSourcePath] = useState('');
    const [sourceName, setSourceName] = useState('');
    const [sourceFile, setSourceFile] = useState(null);  // kept for in-browser decode (waveform)
    const [sourceUploading, setSourceUploading] = useState(false);
    const [dropActive, setDropActive] = useState(false);
    const [prompt, setPrompt] = useState('');
    const [duration, setDuration] = useState(8);
    // Seed: random by default, mirroring the rest of the app. When off, the
    // numeric field is honoured (0 included — a legitimate seed).
    const [randomSeed, setRandomSeed] = useState(true);
    const [seedValue, setSeedValue] = useState('');

    // sa3-medium generates up to 380s; small models cap at 120s. Matches the
    // generator's _MODEL_INFO so the slider can't request past the model max.
    const maxDuration = (model_id || '').includes('medium') ? 380 : 120;
    // Distilled (post-trained) models bake CFG at 1.0 and ignore cfg_scale; only
    // *-base variants honour it. Same rule the Generation panel uses.
    const isDistilledBase =
        !!model_id && model_id.startsWith('sa3-') && !model_id.endsWith('-base');

    // style transfer
    const [initNoiseLevel, setInitNoiseLevel] = useState(0.7);

    // inpaint
    const [maskStart, setMaskStart] = useState(2.0);
    const [maskEnd, setMaskEnd] = useState(4.0);

    // extend
    const [extendSeconds, setExtendSeconds] = useState(4.0);
    const [sourceDurationSec, setSourceDurationSec] = useState(null);

    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    // Inpaint region audition — a hidden <audio> set to the source clip, played
    // from maskStart and auto-stopped at maskEnd, so users can hear the segment
    // they're about to regenerate before committing.
    const regionAudioRef = useRef(null);
    const regionStopRef = useRef(null);   // removes the active timeupdate guard
    const [regionUrl, setRegionUrl] = useState(null);
    const [regionPlaying, setRegionPlaying] = useState(false);

    useEffect(() => {
        if (!sourceFile) { setRegionUrl(null); return undefined; }
        const url = URL.createObjectURL(sourceFile);
        setRegionUrl(url);
        return () => URL.revokeObjectURL(url);
    }, [sourceFile]);

    // Stop any in-flight preview when the source changes or the mode switches
    // away from inpaint (don't auto-stop on every region drag — the end is
    // captured per play, so dragging mid-play just runs to the old boundary).
    useEffect(() => {
        const a = regionAudioRef.current;
        if (a) { try { a.pause(); } catch { /* ignore */ } }
        regionStopRef.current?.();
        regionStopRef.current = null;
        setRegionPlaying(false);
    }, [regionUrl, mode]);

    const toggleRegionPreview = () => {
        const a = regionAudioRef.current;
        if (!a || !regionUrl) return;
        if (regionPlaying) {
            a.pause();
            regionStopRef.current?.();
            regionStopRef.current = null;
            setRegionPlaying(false);
            return;
        }
        const start = Math.max(0, Number(maskStart) || 0);
        const end = Math.max(start + 0.05, Number(maskEnd) || 0);
        const onTime = () => {
            if (a.currentTime >= end) {
                a.pause();
                a.removeEventListener('timeupdate', onTime);
                regionStopRef.current = null;
                setRegionPlaying(false);
            }
        };
        try { a.currentTime = start; } catch { /* ignore */ }
        a.addEventListener('timeupdate', onTime);
        regionStopRef.current = () => a.removeEventListener('timeupdate', onTime);
        a.play()
            .then(() => setRegionPlaying(true))
            .catch(() => {
                a.removeEventListener('timeupdate', onTime);
                regionStopRef.current = null;
                setRegionPlaying(false);
            });
    };

    // --- source upload ---------------------------------------------------
    const onPickFile = () => fileInputRef.current?.click();
    const uploadFile = async (f) => {
        if (!f) return;
        setSourceUploading(true);
        setError(null);
        try {
            const form = new FormData();
            form.append('file', f);
            const r = await api.post('/api/audio/upload', form);
            setSourcePath(r.data.path);
            setSourceName(r.data.name);
            setSourceFile(f);  // keep for in-browser waveform decode
            // Probe duration via a temp object URL → <audio>.
            const url = URL.createObjectURL(f);
            const a = new Audio(url);
            a.addEventListener('loadedmetadata', () => {
                if (Number.isFinite(a.duration)) {
                    setSourceDurationSec(a.duration);
                    // Default the output length to the source length (clamped to
                    // the model max). For inpaint this is mandatory — the mask is
                    // measured in source seconds, so the output must be the same
                    // length or the masked region drifts off the audio you see.
                    setDuration(Math.max(1, Math.min(maxDuration, Math.round(a.duration))));
                    // Seed inpaint region to the middle quarter so the
                    // waveform shows something sensible without a 4 s default
                    // landing past the end of short clips.
                    const q = a.duration / 4;
                    setMaskStart(Math.max(0, q));
                    setMaskEnd(Math.min(a.duration, q * 3));
                }
                URL.revokeObjectURL(url);
            }, { once: true });
        } catch (err) {
            setError(err.response?.data?.error?.message || err.message || 'Upload failed');
        } finally {
            setSourceUploading(false);
        }
    };
    const onFileChange = async (e) => {
        const f = e.target.files?.[0];
        e.target.value = '';
        await uploadFile(f);
    };
    // Pull a fragment already on disk (dragged in from the Generated
    // Fragments window) and run it through the same upload path so it gets a
    // server path + waveform + duration probe, exactly like a picked file.
    const loadFragmentByName = async (filename) => {
        if (!filename) return;
        setSourceUploading(true);
        setError(null);
        try {
            const r = await api.get(`/api/fragments/${encodeURIComponent(filename)}`, { responseType: 'blob' });
            const file = new File([r.data], filename, { type: r.data.type || 'audio/wav' });
            await uploadFile(file);
        } catch (err) {
            setError(err.response?.data?.error?.message || err.message || 'Could not load fragment');
            setSourceUploading(false);
        }
    };

    const onDrop = async (e) => {
        e.preventDefault();
        setDropActive(false);
        // In-app drag from the Generated Fragments window carries the
        // fragment filename; OS file drags carry dataTransfer.files. Read the
        // custom payload synchronously before any await.
        const fragName = e.dataTransfer.getData('application/x-fragmenta-fragment');
        if (fragName) { await loadFragmentByName(fragName); return; }
        const f = e.dataTransfer.files?.[0];
        await uploadFile(f);
    };
    const onDragOver = (e) => { e.preventDefault(); setDropActive(true); };
    const onDragLeave = (e) => { e.preventDefault(); setDropActive(false); };
    const clearSource = () => {
        setSourcePath('');
        setSourceName('');
        setSourceFile(null);
        setSourceDurationSec(null);
    };

    // --- generate --------------------------------------------------------
    const generate = async () => {
        if (!model_id) {
            setError('Pick a model in the Generation tab first.');
            return;
        }
        if (!sourcePath) {
            setError('Upload a source clip first.');
            return;
        }
        if (!prompt.trim() && mode !== 'extend') {
            setError('Enter a prompt describing the change.');
            return;
        }

        setGenerating(true);
        setError(null);
        try {
            // Seed: -1 lets the backend pick (and record) a random one; an
            // explicit value is parsed with parseInt so 0 stays 0 rather than
            // collapsing to random via `|| -1`.
            let seedToSend = -1;
            if (!randomSeed) {
                const parsed = parseInt(seedValue, 10);
                if (Number.isNaN(parsed) || parsed < 0) {
                    setError('Enter a non-negative integer seed, or switch Seed to Random.');
                    setGenerating(false);
                    return;
                }
                seedToSend = parsed;
            }

            const body = {
                model_id,
                prompt: prompt.trim() || 'continue',
                duration,
                seed: seedToSend,
                steps,
            };
            if (negativePrompt) body.negative_prompt = negativePrompt;
            // Only base models honour CFG; sending it on a distilled model is
            // harmless (backend forces 1.0) but we keep the UI honest.
            if (!isDistilledBase) body.cfg_scale = cfgScale;
            // Inherit the Generation panel's LoRA stack. Bypassed slots stay in
            // load order but contribute strength 0 (same as plain generation).
            const activeLoras = (loraStack || [])
                .filter((s) => s.path)
                .map((s) => ({ path: s.path, strength: s.bypassed ? 0 : s.strength }));
            if (activeLoras.length) body.loras = activeLoras;

            if (mode === 'style') {
                body.init_audio_path = sourcePath;
                body.init_noise_level = initNoiseLevel;
            } else if (mode === 'inpaint') {
                // Pin output length to the source so the mask (measured in
                // source seconds) maps onto the same timeline the user sees.
                if (!Number.isFinite(sourceDurationSec)) {
                    setError("Couldn't read source duration — re-upload the file.");
                    setGenerating(false);
                    return;
                }
                body.duration = sourceDurationSec;
                body.inpaint_audio_path = sourcePath;
                body.inpaint_starts = [Number(maskStart)];
                body.inpaint_ends = [Number(maskEnd)];
            } else if (mode === 'extend') {
                // Extend = inpaint where the mask is the new tail. Total clip
                // duration = source length + extendSeconds; mask covers
                // [source_length, source_length + extendSeconds].
                if (!Number.isFinite(sourceDurationSec)) {
                    setError("Couldn't read source duration — re-upload the file.");
                    setGenerating(false);
                    return;
                }
                body.duration = sourceDurationSec + extendSeconds;
                body.inpaint_audio_path = sourcePath;
                body.inpaint_starts = [sourceDurationSec];
                body.inpaint_ends = [sourceDurationSec + extendSeconds];
            }

            const resp = await api.post('/api/generate', body, { responseType: 'blob' });
            // Use the backend's real on-disk name (header) so the fragment in
            // the list resolves to an actual file for reveal/delete; only fall
            // back to a synthetic name if the header is absent.
            const fname = resp.headers?.['x-fragment-filename'] || `${mode}_${Date.now()}.wav`;
            // Record the resolved seed (the backend picks a concrete one when we
            // sent -1) so the fragment shows the real value, not "random".
            const resolvedSeed = parseInt(resp.headers?.['x-fragment-seed'], 10);
            const params = Number.isFinite(resolvedSeed) ? { ...body, seed: resolvedSeed } : body;
            onGenerated?.(resp.data, fname, params);
        } catch (err) {
            setError(err.response?.data?.error?.message || err.message || 'Generation failed');
        } finally {
            setGenerating(false);
        }
    };

    // --- render ----------------------------------------------------------
    return (
        <Box sx={{ p: 2 }}>
            {/* Source picker (drag-and-drop or click) */}
            <Box
                sx={{ mb: 2 }}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
            >
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                    Source clip
                </Typography>
                {sourcePath ? (
                    <Stack
                        direction="row"
                        alignItems="center"
                        spacing={1}
                        sx={{
                            p: 1,
                            border: '1px dashed',
                            borderColor: dropActive ? 'primary.main' : 'divider',
                            borderRadius: 1,
                            transition: 'border-color 120ms',
                        }}
                    >
                        <Typography variant="body2" sx={{ flex: 1, fontFamily: 'monospace', fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {sourceName}
                            {sourceDurationSec && ` · ${sourceDurationSec.toFixed(2)}s`}
                        </Typography>
                        <IconButton size="small" onClick={clearSource} aria-label="Remove source"><ClearIcon size={14} /></IconButton>
                    </Stack>
                ) : (
                    <Button
                        variant="outlined"
                        startIcon={<UploadIcon size={14} />}
                        onClick={onPickFile}
                        disabled={sourceUploading}
                        fullWidth
                        sx={{
                            borderStyle: 'dashed',
                            borderColor: dropActive ? 'primary.main' : undefined,
                            bgcolor: dropActive ? 'action.hover' : undefined,
                            transition: 'border-color 120ms, background-color 120ms',
                        }}
                    >
                        {sourceUploading ? 'Uploading…' : 'Drop a clip here, or click to pick a file'}
                    </Button>
                )}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".wav,.mp3,.flac,.m4a,.ogg,.opus,audio/*"
                    style={{ display: 'none' }}
                    onChange={onFileChange}
                />
            </Box>

            {/* Mode selector */}
            <ToggleButtonGroup
                value={mode}
                exclusive
                size="small"
                onChange={(_, v) => v && setMode(v)}
                sx={{ mb: 2 }}
            >
                <ToggleButton value="style">Style transfer</ToggleButton>
                <ToggleButton value="inpaint">Inpaint region</ToggleButton>
                <ToggleButton value="extend">Extend</ToggleButton>
            </ToggleButtonGroup>

            {/* Mode-specific controls */}
            {mode === 'style' && (
                <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                        Preserve source character ←→ follow prompt
                    </Typography>
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Slider
                            value={initNoiseLevel}
                            onChange={(_, v) => setInitNoiseLevel(v)}
                            min={0}
                            max={1}
                            step={0.05}
                            valueLabelDisplay="auto"
                            marks={[
                                { value: 0, label: '0' },
                                { value: 0.5, label: '0.5' },
                                { value: 1, label: '1' },
                            ]}
                            sx={{ flex: 1 }}
                        />
                        <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                            {initNoiseLevel.toFixed(2)}
                        </Typography>
                    </Stack>
                </Box>
            )}

            {mode === 'inpaint' && (
                <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                        Drag the highlighted region to mask the segment SA3 should regenerate
                    </Typography>
                    <AudioWaveform
                        file={sourceFile}
                        duration={sourceDurationSec || 0}
                        start={maskStart}
                        end={maskEnd}
                        onRegionChange={(s, e) => { setMaskStart(s); setMaskEnd(e); }}
                    />
                    <Stack direction="row" alignItems="center" spacing={2} sx={{ mt: 1 }}>
                        <TextField
                            label="Start (s)"
                            type="number"
                            size="small"
                            value={maskStart.toFixed(2)}
                            onChange={(e) => setMaskStart(parseFloat(e.target.value) || 0)}
                            inputProps={{ min: 0, max: sourceDurationSec || 999, step: 0.05 }}
                            sx={{ width: 96 }}
                        />
                        <TextField
                            label="End (s)"
                            type="number"
                            size="small"
                            value={maskEnd.toFixed(2)}
                            onChange={(e) => setMaskEnd(parseFloat(e.target.value) || 0)}
                            inputProps={{ min: 0, max: sourceDurationSec || 999, step: 0.05 }}
                            sx={{ width: 96 }}
                        />
                        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Button
                                size="small"
                                variant="outlined"
                                startIcon={regionPlaying ? <StopIcon size={14} /> : <PlayIcon size={14} />}
                                onClick={toggleRegionPreview}
                                disabled={!regionUrl || (maskEnd - maskStart) < 0.05}
                                // Fixed width so swapping "Preview" ↔ "Stop" doesn't
                                // resize the button. Sized to fit "Preview" + icon.
                                sx={{ width: 108, flexShrink: 0 }}
                            >
                                {regionPlaying ? 'Stop' : 'Preview'}
                            </Button>
                            <Typography variant="caption" color="text.secondary">
                                {(maskEnd - maskStart).toFixed(2)} s
                            </Typography>
                        </Box>
                    </Stack>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                        Output stays {sourceDurationSec ? `${sourceDurationSec.toFixed(2)} s` : 'the source length'} —
                        only the masked region is regenerated.
                    </Typography>
                </Box>
            )}

            {mode === 'extend' && (
                <Box sx={{ mb: 2 }}>
                    <TextField
                        label="Seconds to add at the end"
                        type="number"
                        size="small"
                        value={extendSeconds}
                        onChange={(e) => setExtendSeconds(parseFloat(e.target.value) || 0)}
                        inputProps={{ min: 0.5, max: 60, step: 0.5 }}
                        fullWidth
                    />
                    <Typography variant="caption" color="text.secondary">
                        Source is {sourceDurationSec ? sourceDurationSec.toFixed(2) : '—'} s; final clip will be{' '}
                        {sourceDurationSec ? (sourceDurationSec + Number(extendSeconds || 0)).toFixed(2) : '—'} s.
                    </Typography>
                </Box>
            )}

            {/* Shared inputs */}
            <TextField
                label={mode === 'inpaint' ? 'Prompt for the masked region' : 'Prompt for the edit'}
                placeholder={
                    mode === 'style' ? 'How the source should sound now…' :
                    mode === 'inpaint' ? 'What goes in the gap…' :
                    'What the continuation should sound like (optional)'
                }
                multiline
                minRows={1}
                maxRows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                fullWidth
                sx={{ mb: 2 }}
            />

            {mode === 'style' && (
                <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ minWidth: 80 }}>
                        Duration
                    </Typography>
                    <Slider
                        value={Math.min(duration, maxDuration)}
                        onChange={(_, v) => setDuration(v)}
                        min={1}
                        max={maxDuration}
                        step={1}
                        valueLabelDisplay="auto"
                        sx={{ flex: 1 }}
                    />
                    <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                        {duration}s
                    </Typography>
                </Stack>
            )}

            {/* Seed — random by default, mirrors the Generation panel */}
            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" sx={{ minWidth: 80 }}>
                    Seed
                </Typography>
                <FormControlLabel
                    control={
                        <Switch
                            size="small"
                            checked={randomSeed}
                            onChange={(e) => setRandomSeed(e.target.checked)}
                        />
                    }
                    label="Random"
                    sx={{ mr: 0 }}
                />
                <TextField
                    size="small"
                    type="number"
                    value={seedValue}
                    disabled={randomSeed}
                    onChange={(e) => setSeedValue(e.target.value)}
                    placeholder={randomSeed ? 'Randomized each run (recorded)' : 'e.g. 42'}
                    inputProps={{ min: 0, step: 1 }}
                    sx={{ flex: 1 }}
                />
            </Stack>

            {/* Hidden element backing the inpaint region preview */}
            <audio
                ref={regionAudioRef}
                src={regionUrl || undefined}
                preload="auto"
                style={{ display: 'none' }}
                onEnded={() => setRegionPlaying(false)}
            />

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            {generating && <LinearProgress sx={{ mb: 2 }} />}

            <Button
                variant="contained"
                fullWidth
                onClick={generate}
                disabled={generating || !sourcePath}
            >
                {generating
                    ? 'Generating…'
                    : mode === 'style' ? 'Apply style'
                    : mode === 'inpaint' ? 'Inpaint region'
                    : 'Extend clip'}
            </Button>
        </Box>
    );
}
