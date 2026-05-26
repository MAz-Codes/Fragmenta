import React, { useState, useRef } from 'react';
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
    Tooltip,
} from '@mui/material';
import { Upload as UploadIcon, X as ClearIcon } from 'lucide-react';
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
 *   onGenerated(blob, filename, params): called with the resulting WAV
 */
export default function EditPanel({ model_id, negativePrompt, onGenerated }) {
    const [mode, setMode] = useState('style');   // 'style' | 'inpaint' | 'extend'
    const [sourcePath, setSourcePath] = useState('');
    const [sourceName, setSourceName] = useState('');
    const [sourceFile, setSourceFile] = useState(null);  // kept for in-browser decode (waveform)
    const [sourceUploading, setSourceUploading] = useState(false);
    const [dropActive, setDropActive] = useState(false);
    const [prompt, setPrompt] = useState('');
    const [duration, setDuration] = useState(8);
    const [seed, setSeed] = useState(-1);

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
    const onDrop = async (e) => {
        e.preventDefault();
        setDropActive(false);
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
            const body = {
                model_id,
                prompt: prompt.trim() || 'continue',
                duration,
                seed: Number(seed) || -1,
            };
            if (negativePrompt) body.negative_prompt = negativePrompt;

            if (mode === 'style') {
                body.init_audio_path = sourcePath;
                body.init_noise_level = initNoiseLevel;
            } else if (mode === 'inpaint') {
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
            const fname = `${mode}_${Date.now()}.wav`;
            onGenerated?.(resp.data, fname, body);
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
                        <Tooltip title="Remove source">
                            <IconButton size="small" onClick={clearSource}><ClearIcon size={14} /></IconButton>
                        </Tooltip>
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
                    <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                        <TextField
                            label="Start (s)"
                            type="number"
                            size="small"
                            value={maskStart.toFixed(2)}
                            onChange={(e) => setMaskStart(parseFloat(e.target.value) || 0)}
                            inputProps={{ min: 0, max: sourceDurationSec || 999, step: 0.05 }}
                            sx={{ flex: 1 }}
                        />
                        <TextField
                            label="End (s)"
                            type="number"
                            size="small"
                            value={maskEnd.toFixed(2)}
                            onChange={(e) => setMaskEnd(parseFloat(e.target.value) || 0)}
                            inputProps={{ min: 0, max: sourceDurationSec || 999, step: 0.05 }}
                            sx={{ flex: 1 }}
                        />
                        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                            <Typography variant="caption" color="text.secondary">
                                {(maskEnd - maskStart).toFixed(2)} s region
                            </Typography>
                        </Box>
                    </Stack>
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

            {mode !== 'extend' && (
                <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ minWidth: 80 }}>
                        Duration
                    </Typography>
                    <Slider
                        value={duration}
                        onChange={(_, v) => setDuration(v)}
                        min={1}
                        max={120}
                        step={1}
                        valueLabelDisplay="auto"
                        sx={{ flex: 1 }}
                    />
                    <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                        {duration}s
                    </Typography>
                </Stack>
            )}

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
