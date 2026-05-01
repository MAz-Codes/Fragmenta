import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box, Paper, Typography, TextField, Button, MenuItem, Select, FormControl,
    InputLabel, LinearProgress, Alert, Table, TableHead, TableRow, TableCell,
    TableBody, TableContainer, Checkbox, Tooltip, CircularProgress,
    Accordion, AccordionSummary, AccordionDetails, Autocomplete, Chip,
} from '@mui/material';
import {
    Tags as TagsIcon,
    CloudDownload as CloudDownloadIcon,
    Save as SaveIcon,
    FolderOpen as FolderOpenIcon,
    Upload as UploadIcon,
    ChevronDown as ExpandMoreIcon,
    RotateCcw as ResetIcon,
} from 'lucide-react';
import api from '../api';

const POLL_INTERVAL_MS = 800;

export default function BulkAnnotatePanel({ onCommitted }) {
    const [folderPath, setFolderPath] = useState('');
    const [tier, setTier] = useState('basic');
    const [status, setStatus] = useState(null);
    const [results, setResults] = useState([]);
    const [selected, setSelected] = useState({});
    const [copyFiles, setCopyFiles] = useState(true);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [committing, setCommitting] = useState(false);
    const [isDocker, setIsDocker] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [labels, setLabels] = useState({ genre: [], mood: [], instruments: [] });
    const [labelsOverridden, setLabelsOverridden] = useState(false);
    const [labelsLoading, setLabelsLoading] = useState(false);
    const [labelsSaving, setLabelsSaving] = useState(false);
    const [labelsMessage, setLabelsMessage] = useState('');
    const [labelsError, setLabelsError] = useState('');
    const pollRef = useRef(null);
    const folderInputRef = useRef(null);

    useEffect(() => {
        api.get('/api/environment')
            .then(({ data }) => setIsDocker(!!data?.docker))
            .catch(() => {});
    }, []);

    const loadLabels = useCallback(async () => {
        setLabelsLoading(true);
        try {
            const { data } = await api.get('/api/annotator-labels');
            setLabels({
                genre: data?.labels?.genre || [],
                mood: data?.labels?.mood || [],
                instruments: data?.labels?.instruments || [],
            });
            setLabelsOverridden(!!data?.overridden);
            setLabelsError('');
        } catch (exc) {
            setLabelsError(exc.response?.data?.error || exc.message);
        } finally {
            setLabelsLoading(false);
        }
    }, []);

    useEffect(() => { loadLabels(); }, [loadLabels]);

    const updateLabelCategory = (category, value) => {
        const cleaned = Array.from(new Set(
            (value || []).map((s) => String(s).trim()).filter(Boolean)
        ));
        setLabels((prev) => ({ ...prev, [category]: cleaned }));
        setLabelsMessage('');
    };

    const saveLabels = async () => {
        setLabelsSaving(true);
        setLabelsMessage('');
        setLabelsError('');
        try {
            const { data } = await api.put('/api/annotator-labels', labels);
            setLabels(data?.labels || labels);
            setLabelsOverridden(!!data?.overridden);
            setLabelsMessage('Annotator labels saved.');
        } catch (exc) {
            setLabelsError(exc.response?.data?.error || exc.message);
        } finally {
            setLabelsSaving(false);
        }
    };

    const resetLabels = async () => {
        setLabelsSaving(true);
        setLabelsMessage('');
        setLabelsError('');
        try {
            const { data } = await api.delete('/api/annotator-labels');
            setLabels(data?.labels || { genre: [], mood: [], instruments: [] });
            setLabelsOverridden(false);
            setLabelsMessage('Reverted to built-in default labels.');
        } catch (exc) {
            setLabelsError(exc.response?.data?.error || exc.message);
        } finally {
            setLabelsSaving(false);
        }
    };

    const stopPolling = useCallback(() => {
        if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
        }
    }, []);

    const fetchStatus = useCallback(async () => {
        let data;
        try {
            const resp = await api.get('/api/bulk-annotate/status');
            data = resp.data;
        } catch (exc) {
            return;
        }
        setStatus(data);

        const annotationState = data.state;
        const downloadState = data.clap_download?.state;

        if (annotationState === 'done') {
            try {
                const resp = await api.get('/api/bulk-annotate/results');
                setResults(resp.data.results || []);
                const sel = {};
                (resp.data.results || []).forEach((r, i) => { sel[i] = !r.error; });
                setSelected(sel);
            } catch {
                return;
            }
        } else if (annotationState === 'error') {
            setError(data.error || 'Annotation failed.');
        }

        const annotationInactive = annotationState !== 'running';
        const downloadInactive = downloadState !== 'running';
        if (annotationInactive && downloadInactive) {
            stopPolling();
        }
    }, [stopPolling]);

    const startPolling = useCallback(() => {
        stopPolling();
        pollRef.current = setInterval(fetchStatus, POLL_INTERVAL_MS);
    }, [fetchStatus, stopPolling]);

    useEffect(() => {
        fetchStatus();
        return () => stopPolling();
    }, [fetchStatus, stopPolling]);

    const startAnnotation = async () => {
        setError('');
        setMessage('');
        setResults([]);
        try {
            await api.post('/api/bulk-annotate', { folder_path: folderPath, tier });
            startPolling();
        } catch (exc) {
            setError(exc.response?.data?.error || exc.message);
        }
    };

    const pickFolder = async () => {
        setError('');
        try {
            const { data } = await api.post('/api/pick-folder', { start_dir: folderPath || undefined });
            if (data?.path) setFolderPath(data.path);
        } catch (exc) {
            setError(exc.response?.data?.error || exc.message);
        }
    };

    const openFolderUpload = () => {
        setError('');
        if (folderInputRef.current) {
            folderInputRef.current.value = '';
            folderInputRef.current.click();
        }
    };

    const handleFolderSelected = async (event) => {
        const fileList = Array.from(event.target.files || []);
        if (fileList.length === 0) return;

        setError('');
        setUploading(true);
        try {
            const form = new FormData();
            fileList.forEach((file) => {
                form.append('files', file);
                form.append('rel_paths', file.webkitRelativePath || file.name);
            });
            const { data } = await api.post('/api/upload-folder', form, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            if (data?.path) setFolderPath(data.path);
        } catch (exc) {
            setError(exc.response?.data?.error || exc.message);
        } finally {
            setUploading(false);
        }
    };

    const downloadClap = async () => {
        setError('');
        try {
            await api.post('/api/bulk-annotate/download-clap', {});
            startPolling();
        } catch (exc) {
            setError(exc.response?.data?.error || exc.message);
        }
    };

    const updatePrompt = (idx, value) => {
        setResults(prev => prev.map((r, i) => (i === idx ? { ...r, prompt: value } : r)));
    };

    const toggleSelected = (idx) => {
        setSelected(prev => ({ ...prev, [idx]: !prev[idx] }));
    };

    const toggleAll = () => {
        const allSelected = results.every((_, i) => selected[i]);
        const next = {};
        results.forEach((_, i) => { next[i] = !allSelected; });
        setSelected(next);
    };

    const commit = async () => {
        setError('');
        setMessage('');
        setCommitting(true);
        try {
            const entries = results
                .filter((_, i) => selected[i])
                .map(r => ({ file_name: r.file_name, prompt: r.prompt, path: r.path }));
            const { data } = await api.post('/api/bulk-annotate/commit', { entries, copy_files: copyFiles });
            setMessage(data.message || 'Committed.');
            if (onCommitted) onCommitted();
        } catch (exc) {
            setError(exc.response?.data?.error || exc.message);
        } finally {
            setCommitting(false);
        }
    };

    const isRunning = status?.state === 'running';
    const clapDownload = status?.clap_download;
    const clapAvailable = !!status?.clap_available;
    const clapDownloading = clapDownload?.state === 'running';
    const richBlocked = tier === 'rich' && !clapAvailable;
    const progressPct = status?.total ? Math.round((status.current / status.total) * 100) : 0;

    return (
        <Paper sx={{ p: 2, mt: 3 }} variant="outlined">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                <TagsIcon size={20} />
                <Typography variant="h6">Bulk Auto-Annotation</Typography>
            </Box>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Point at a folder of audio files and auto-generate prompts.
                Basic uses librosa (tempo + key). Rich adds CLAP tagging (genre, mood, instruments).
            </Typography>
            
            <Accordion sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon size={18} />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle1">Annotator Labels</Typography>
                        {labelsOverridden && (
                            <Chip size="small" color="primary" variant="outlined" label="Custom" />
                        )}
                    </Box>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                        These replace the built-in label sets used by Rich (CLAP) auto-annotation.
                        Type a label and press Enter to add it; click the × on a chip to remove it.
                    </Typography>
                    {labelsError && <Alert severity="error" sx={{ mb: 2 }}>{labelsError}</Alert>}
                    {labelsMessage && <Alert severity="success" sx={{ mb: 2 }}>{labelsMessage}</Alert>}
                    {['genre', 'mood', 'instruments'].map((category) => (
                        <Box key={category} sx={{ mb: 2 }}>
                            <Autocomplete
                                multiple
                                freeSolo
                                options={[]}
                                value={labels[category] || []}
                                onChange={(_, value) => updateLabelCategory(category, value)}
                                disabled={labelsLoading || labelsSaving}
                                renderTags={(value, getTagProps) =>
                                    value.map((option, index) => (
                                        <Chip
                                            variant="outlined"
                                            size="small"
                                            label={option}
                                            {...getTagProps({ index })}
                                        />
                                    ))
                                }
                                renderInput={(params) => (
                                    <TextField
                                        {...params}
                                        label={category.charAt(0).toUpperCase() + category.slice(1)}
                                        placeholder="Add a label…"
                                        size="small"
                                    />
                                )}
                            />
                        </Box>
                    ))}
                    <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                        <Button
                            variant="contained"
                            onClick={saveLabels}
                            startIcon={labelsSaving ? <CircularProgress size={16} /> : <SaveIcon size={16} />}
                            disabled={labelsLoading || labelsSaving}
                        >
                            {labelsSaving ? 'Saving…' : 'Save labels'}
                        </Button>
                        <Tooltip title={labelsOverridden ? '' : 'No custom labels to reset'}>
                            <span>
                                <Button
                                    variant="outlined"
                                    onClick={resetLabels}
                                    startIcon={<ResetIcon size={16} />}
                                    disabled={labelsSaving || !labelsOverridden}
                                >
                                    Reset to defaults
                                </Button>
                            </span>
                        </Tooltip>
                    </Box>
                </AccordionDetails>
            </Accordion>

            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap', mb: 2 }}>
                <TextField
                    label="Folder path"
                    size="small"
                    value={folderPath}
                    onChange={(e) => setFolderPath(e.target.value)}
                    placeholder="Click Browse to choose a folder…"
                    sx={{ flexGrow: 1, minWidth: 260 }}
                    disabled={isRunning}
                    InputProps={{ readOnly: true }}
                />
                {isDocker ? (
                    <>
                        <input
                            ref={folderInputRef}
                            type="file"
                            webkitdirectory=""
                            directory=""
                            multiple
                            style={{ display: 'none' }}
                            onChange={handleFolderSelected}
                        />
                        <Button
                            variant="outlined"
                            onClick={openFolderUpload}
                            startIcon={uploading ? <CircularProgress size={16} /> : <UploadIcon size={16} />}
                            disabled={isRunning || uploading}
                        >
                            {uploading ? 'Uploading…' : 'Upload Folder'}
                        </Button>
                    </>
                ) : (
                    <Button
                        variant="outlined"
                        onClick={pickFolder}
                        startIcon={<FolderOpenIcon size={16} />}
                        disabled={isRunning}
                    >
                        Browse
                    </Button>
                )}
                <FormControl size="small" sx={{ minWidth: 140 }} disabled={isRunning}>
                    <InputLabel id="tier-label">Tier</InputLabel>
                    <Select
                        labelId="tier-label"
                        value={tier}
                        label="Tier"
                        onChange={(e) => setTier(e.target.value)}
                    >
                        <MenuItem value="basic">Basic (no download)</MenuItem>
                        <MenuItem value="rich">Rich (CLAP, ~2.35 GB)</MenuItem>
                    </Select>
                </FormControl>
                <Tooltip title={richBlocked ? 'Download CLAP to enable Rich tier' : ''}>
                    <span>
                        <Button
                            variant="contained"
                            onClick={startAnnotation}
                            startIcon={isRunning ? <CircularProgress size={16} /> : <TagsIcon size={16} />}
                            disabled={isRunning || !folderPath || richBlocked}
                        >
                            {isRunning ? 'Annotating…' : 'Annotate'}
                        </Button>
                    </span>
                </Tooltip>
                {tier === 'rich' && !clapAvailable && (
                    <Button
                        variant="outlined"
                        onClick={downloadClap}
                        startIcon={clapDownloading ? <CircularProgress size={16} /> : <CloudDownloadIcon size={16} />}
                        disabled={clapDownloading}
                    >
                        {clapDownloading ? 'Downloading…' : 'Download CLAP'}
                    </Button>
                )}
            </Box>

            {isRunning && (
                <Box sx={{ mb: 2 }}>
                    <LinearProgress variant={status?.total ? 'determinate' : 'indeterminate'} value={progressPct} />
                    <Typography variant="caption" color="textSecondary">
                        {status?.current}/{status?.total} — {status?.current_file}
                    </Typography>
                </Box>
            )}
            {clapDownload?.state === 'running' && (
                <Alert severity="info" sx={{ mb: 2 }}>{clapDownload.message || 'Downloading CLAP checkpoint…'}</Alert>
            )}
            {clapDownload?.state === 'error' && (
                <Alert severity="error" sx={{ mb: 2 }}>CLAP download failed: {clapDownload.error}</Alert>
            )}
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            {message && <Alert severity="success" sx={{ mb: 2 }}>{message}</Alert>}

            {results.length > 0 && (
                <>
                    <TableContainer sx={{ maxHeight: 420, mb: 2 }}>
                        <Table size="small" stickyHeader>
                            <TableHead>
                                <TableRow>
                                    <TableCell padding="checkbox">
                                        <Checkbox
                                            checked={results.every((_, i) => selected[i])}
                                            indeterminate={
                                                results.some((_, i) => selected[i]) && !results.every((_, i) => selected[i])
                                            }
                                            onChange={toggleAll}
                                        />
                                    </TableCell>
                                    <TableCell>File</TableCell>
                                    <TableCell>Prompt (editable)</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {results.map((row, idx) => (
                                    <TableRow key={row.file_name + idx} hover>
                                        <TableCell padding="checkbox">
                                            <Checkbox
                                                checked={!!selected[idx]}
                                                onChange={() => toggleSelected(idx)}
                                                disabled={!!row.error}
                                            />
                                        </TableCell>
                                        <TableCell sx={{ maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            <Tooltip title={row.path || row.file_name}>
                                                <span>{row.file_name}</span>
                                            </Tooltip>
                                            {row.error && (
                                                <Typography variant="caption" color="error" display="block">
                                                    {row.error}
                                                </Typography>
                                            )}
                                        </TableCell>
                                        <TableCell>
                                            <TextField
                                                fullWidth
                                                multiline
                                                size="small"
                                                value={row.prompt || ''}
                                                onChange={(e) => updatePrompt(idx, e.target.value)}
                                                disabled={!!row.error}
                                            />
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>

                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                        <FormControl size="small">
                            <Select
                                value={copyFiles ? 'copy' : 'link'}
                                onChange={(e) => setCopyFiles(e.target.value === 'copy')}
                            >
                                <MenuItem value="copy">Copy files into data/</MenuItem>
                                <MenuItem value="link">Symlink files into data/</MenuItem>
                            </Select>
                        </FormControl>
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={commit}
                            startIcon={committing ? <CircularProgress size={16} /> : <SaveIcon size={16} />}
                            disabled={committing || !Object.values(selected).some(Boolean)}
                        >
                            {committing ? 'Saving…' : `Save ${Object.values(selected).filter(Boolean).length} to dataset`}
                        </Button>
                    </Box>
                </>
            )}
        </Paper>
    );
}
