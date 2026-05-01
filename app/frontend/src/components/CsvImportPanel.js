import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box, Paper, Typography, TextField, Button, MenuItem, Select, FormControl,
    InputLabel, Alert, Table, TableHead, TableRow, TableCell, TableBody,
    TableContainer, Checkbox, Tooltip, CircularProgress, Chip,
} from '@mui/material';
import {
    FileSpreadsheet as CsvIcon,
    FolderOpen as FolderOpenIcon,
    Upload as UploadIcon,
    Save as SaveIcon,
    FileText as FileTextIcon,
} from 'lucide-react';
import api from '../api';

const CONFLICT_POLICIES = [
    { value: 'skip', label: 'Skip', help: 'Keep the existing entry; ignore the CSV row.' },
    { value: 'overwrite', label: 'Overwrite', help: "Replace the existing entry's prompt and audio." },
    { value: 'rename', label: 'Rename', help: 'Add as a new entry with a numeric suffix (e.g. track_2.wav). Always copies into data/.' },
];

export default function CsvImportPanel({ onCommitted }) {
    const [folderPath, setFolderPath] = useState('');
    const [csvFile, setCsvFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [previewError, setPreviewError] = useState('');
    const [previewLoading, setPreviewLoading] = useState(false);
    const [selected, setSelected] = useState({});
    const [conflictPolicy, setConflictPolicy] = useState('skip');
    const [copyFiles, setCopyFiles] = useState(true);
    const [committing, setCommitting] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [isDocker, setIsDocker] = useState(false);
    const [uploadingFolder, setUploadingFolder] = useState(false);
    const folderInputRef = useRef(null);
    const csvInputRef = useRef(null);

    useEffect(() => {
        api.get('/api/environment')
            .then(({ data }) => setIsDocker(!!data?.docker))
            .catch(() => {});
    }, []);

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
        setUploadingFolder(true);
        try {
            const form = new FormData();
            fileList.forEach((file) => {
                form.append('files', file);
                form.append('rel_paths', file.webkitRelativePath || file.name);
            });
            const { data } = await api.post('/api/upload-folder', form);
            if (data?.path) setFolderPath(data.path);
        } catch (exc) {
            setError(exc.response?.data?.error || exc.message);
        } finally {
            setUploadingFolder(false);
        }
    };

    const openCsvPicker = () => {
        if (csvInputRef.current) {
            csvInputRef.current.value = '';
            csvInputRef.current.click();
        }
    };

    const handleCsvSelected = (event) => {
        const file = event.target.files?.[0];
        if (file) {
            setCsvFile(file);
            setPreview(null);
            setSelected({});
            setMessage('');
            setError('');
        }
    };

    const runPreview = useCallback(async () => {
        if (!csvFile || !folderPath) return;
        setPreviewLoading(true);
        setPreviewError('');
        setPreview(null);
        setMessage('');
        setError('');
        try {
            const form = new FormData();
            form.append('csv', csvFile);
            form.append('audio_folder', folderPath);
            const { data } = await api.post('/api/import-csv/preview', form);
            setPreview(data);
            const sel = {};
            (data.rows || []).forEach((r, i) => {
                sel[i] = r.audio_found && r.errors.length === 0;
            });
            setSelected(sel);
        } catch (exc) {
            setPreviewError(exc.response?.data?.error || exc.message);
        } finally {
            setPreviewLoading(false);
        }
    }, [csvFile, folderPath]);

    const toggleSelected = (idx) => {
        setSelected((prev) => ({ ...prev, [idx]: !prev[idx] }));
    };

    const toggleAll = () => {
        if (!preview) return;
        const eligible = preview.rows
            .map((r, i) => ({ r, i }))
            .filter(({ r }) => r.audio_found && r.errors.length === 0);
        const allSelected = eligible.every(({ i }) => selected[i]);
        const next = { ...selected };
        eligible.forEach(({ i }) => { next[i] = !allSelected; });
        setSelected(next);
    };

    const commit = async () => {
        if (!preview) return;
        setError('');
        setMessage('');
        setCommitting(true);
        try {
            const entries = preview.rows
                .filter((_, i) => selected[i])
                .map((r) => ({
                    file_name: r.file_name,
                    prompt: r.prompt,
                    src_path: r.src_path,
                }));
            if (entries.length === 0) {
                setError('No rows selected.');
                setCommitting(false);
                return;
            }
            const { data } = await api.post('/api/import-csv/commit', {
                entries,
                conflict_policy: conflictPolicy,
                copy_files: copyFiles,
            });
            setMessage(data.message || 'Imported.');
            if (onCommitted) onCommitted();
            setPreview(null);
            setSelected({});
            setCsvFile(null);
        } catch (exc) {
            setError(exc.response?.data?.error || exc.message);
        } finally {
            setCommitting(false);
        }
    };

    const selectedCount = Object.values(selected).filter(Boolean).length;
    const selectedConflictCount = preview
        ? preview.rows.filter((r, i) => selected[i] && r.conflict).length
        : 0;
    const policyHelp = CONFLICT_POLICIES.find((p) => p.value === conflictPolicy)?.help || '';

    return (
        <Paper sx={{ p: 2, mt: 3 }} variant="outlined">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                <CsvIcon size={20} />
                <Typography variant="h6">Import CSV + Audio Folder</Typography>
            </Box>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Upload a CSV with <code>file_name</code> and <code>prompt</code> columns plus the
                folder containing those audio files. Rows merge into the same dataset as the
                other modes — duplicate filenames follow the conflict policy you choose.
            </Typography>

            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap', mb: 2 }}>
                <TextField
                    label="Audio folder"
                    size="small"
                    value={folderPath}
                    onChange={(e) => setFolderPath(e.target.value)}
                    placeholder="Click Browse to choose a folder…"
                    sx={{ flexGrow: 1, minWidth: 260 }}
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
                            startIcon={uploadingFolder ? <CircularProgress size={16} /> : <UploadIcon size={16} />}
                            disabled={uploadingFolder}
                        >
                            {uploadingFolder ? 'Uploading…' : 'Upload Folder'}
                        </Button>
                    </>
                ) : (
                    <Button
                        variant="outlined"
                        onClick={pickFolder}
                        startIcon={<FolderOpenIcon size={16} />}
                    >
                        Browse
                    </Button>
                )}
            </Box>

            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap', mb: 2 }}>
                <input
                    ref={csvInputRef}
                    type="file"
                    accept=".csv,text/csv"
                    style={{ display: 'none' }}
                    onChange={handleCsvSelected}
                />
                <Button
                    variant="outlined"
                    onClick={openCsvPicker}
                    startIcon={<FileTextIcon size={16} />}
                >
                    Choose CSV
                </Button>
                <Typography variant="body2" color="textSecondary" sx={{ flexGrow: 1 }}>
                    {csvFile ? csvFile.name : 'No CSV chosen.'}
                </Typography>
                <Button
                    variant="contained"
                    onClick={runPreview}
                    startIcon={previewLoading ? <CircularProgress size={16} /> : <CsvIcon size={16} />}
                    disabled={!csvFile || !folderPath || previewLoading}
                >
                    {previewLoading ? 'Parsing…' : 'Preview'}
                </Button>
            </Box>

            {previewError && <Alert severity="error" sx={{ mb: 2 }}>{previewError}</Alert>}
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            {message && <Alert severity="success" sx={{ mb: 2 }}>{message}</Alert>}

            {preview && (
                <>
                    <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                        <Chip size="small" label={`${preview.total} rows`} />
                        <Chip
                            size="small"
                            color={preview.conflicts > 0 ? 'warning' : 'default'}
                            label={`${preview.conflicts} conflict${preview.conflicts === 1 ? '' : 's'}`}
                        />
                        <Chip
                            size="small"
                            color={preview.missing_audio > 0 ? 'error' : 'default'}
                            label={`${preview.missing_audio} missing audio`}
                        />
                    </Box>

                    <TableContainer sx={{ maxHeight: 420, mb: 2 }}>
                        <Table size="small" stickyHeader>
                            <TableHead>
                                <TableRow>
                                    <TableCell padding="checkbox">
                                        <Checkbox
                                            checked={
                                                preview.rows.length > 0 &&
                                                preview.rows.every((r, i) =>
                                                    !r.audio_found || r.errors.length > 0 || selected[i]
                                                )
                                            }
                                            indeterminate={
                                                preview.rows.some((_, i) => selected[i]) &&
                                                !preview.rows.every((r, i) =>
                                                    !r.audio_found || r.errors.length > 0 || selected[i]
                                                )
                                            }
                                            onChange={toggleAll}
                                        />
                                    </TableCell>
                                    <TableCell>File</TableCell>
                                    <TableCell>Prompt</TableCell>
                                    <TableCell>Status</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {preview.rows.map((row, idx) => {
                                    const blocked = !row.audio_found || row.errors.length > 0;
                                    return (
                                        <TableRow key={`${row.file_name}-${idx}`} hover>
                                            <TableCell padding="checkbox">
                                                <Checkbox
                                                    checked={!!selected[idx]}
                                                    onChange={() => toggleSelected(idx)}
                                                    disabled={blocked}
                                                />
                                            </TableCell>
                                            <TableCell sx={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                <Tooltip title={row.csv_path || row.file_name}>
                                                    <span>{row.file_name || '(empty)'}</span>
                                                </Tooltip>
                                            </TableCell>
                                            <TableCell sx={{ maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                <Tooltip title={row.prompt}>
                                                    <span>{row.prompt}</span>
                                                </Tooltip>
                                            </TableCell>
                                            <TableCell>
                                                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                                    {row.conflict && (
                                                        <Chip size="small" color="warning" label="conflict" />
                                                    )}
                                                    {!row.audio_found && (
                                                        <Chip size="small" color="error" label="no audio" />
                                                    )}
                                                    {row.errors.map((err, i) => (
                                                        <Chip key={i} size="small" color="error" variant="outlined" label={err} />
                                                    ))}
                                                    {!row.conflict && row.audio_found && row.errors.length === 0 && (
                                                        <Chip size="small" color="success" variant="outlined" label="ok" />
                                                    )}
                                                </Box>
                                            </TableCell>
                                        </TableRow>
                                    );
                                })}
                            </TableBody>
                        </Table>
                    </TableContainer>

                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                        <FormControl size="small" sx={{ minWidth: 180 }}>
                            <InputLabel id="conflict-policy-label">On conflict</InputLabel>
                            <Select
                                labelId="conflict-policy-label"
                                value={conflictPolicy}
                                label="On conflict"
                                onChange={(e) => setConflictPolicy(e.target.value)}
                            >
                                {CONFLICT_POLICIES.map((p) => (
                                    <MenuItem key={p.value} value={p.value}>{p.label}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
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
                            disabled={committing || selectedCount === 0}
                        >
                            {committing ? 'Saving…' : `Save ${selectedCount} to dataset`}
                        </Button>
                    </Box>
                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1 }}>
                        {policyHelp}
                        {selectedConflictCount > 0 && (
                            <> {selectedConflictCount} of the selected rows conflict with existing entries.</>
                        )}
                    </Typography>
                </>
            )}
        </Paper>
    );
}
