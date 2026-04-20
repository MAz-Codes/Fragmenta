import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box, Paper, Typography, TextField, Button, MenuItem, Select, FormControl,
    InputLabel, LinearProgress, Alert, Table, TableHead, TableRow, TableCell,
    TableBody, TableContainer, Checkbox, Tooltip, CircularProgress,
} from '@mui/material';
import {
    Tags as TagsIcon,
    CloudDownload as CloudDownloadIcon,
    Save as SaveIcon,
    FolderOpen as FolderOpenIcon,
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
    const pollRef = useRef(null);

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
            // Transient errors (e.g. Flask auto-reload) must not kill polling —
            // the download/annotation keeps running on the backend side.
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
                <Typography variant="h6">Bulk Auto-Annotate</Typography>
            </Box>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Point at a folder of audio files and auto-generate prompts.
                Basic uses librosa (tempo + key). Rich adds CLAP tagging (genre, mood, instruments).
            </Typography>

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
                <Button
                    variant="outlined"
                    onClick={pickFolder}
                    startIcon={<FolderOpenIcon size={16} />}
                    disabled={isRunning}
                >
                    Browse
                </Button>
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
                                <MenuItem value="link">Leave files in place</MenuItem>
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
