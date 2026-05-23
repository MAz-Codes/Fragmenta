import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    Alert,
    Box,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    FormControl,
    FormControlLabel,
    IconButton,
    InputLabel,
    LinearProgress,
    MenuItem,
    Paper,
    Radio,
    RadioGroup,
    Select,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField,
    Tooltip,
    Typography,
} from '@mui/material';
import {
    FolderOpenIcon,
    PlusIcon,
    SparklesIcon,
    Trash2 as TrashIcon,
} from 'lucide-react';
import api from '../../api';

/**
 * DatasetPrep — the SA3 sidecar-native unified dataset surface.
 *
 * One page, no modes. Pick or create a project; the dataset is the folder
 * under `<user_data_dir>/projects/<name>/`. Audio + .txt sidecars live there
 * directly, so training can point at the same folder unchanged. Editing a
 * prompt writes its sidecar immediately. See DATASET_PREP_REDESIGN.md.
 */
export default function DatasetPrep() {
    const [projects, setProjects] = useState([]);
    const [selectedName, setSelectedName] = useState(() => {
        try {
            return window.localStorage.getItem('fragmenta.datasetPrep.lastProject') || '';
        } catch {
            return '';
        }
    });
    const [project, setProject] = useState(null);
    const [createOpen, setCreateOpen] = useState(false);
    const [ingestOpen, setIngestOpen] = useState(false);
    const [error, setError] = useState('');

    // Annotation progress, polled while a job is running.
    const [annotateJob, setAnnotateJob] = useState(null); // { state, current, total, current_file, ... }
    const pollHandleRef = useRef(null);

    const isAnnotating = annotateJob?.state === 'running';

    const refreshProjects = useCallback(async () => {
        try {
            const { data } = await api.get('/api/projects');
            setProjects(data.projects || []);
        } catch (e) {
            setError(extractError(e, 'Failed to list projects'));
        }
    }, []);

    const refreshProject = useCallback(async (name) => {
        if (!name) {
            setProject(null);
            return;
        }
        try {
            const { data } = await api.get(`/api/projects/${encodeURIComponent(name)}`);
            setProject(data);
        } catch (e) {
            if (e?.response?.status === 404) {
                setSelectedName('');
                setProject(null);
                await refreshProjects();
                return;
            }
            setError(extractError(e, 'Failed to load project'));
        }
    }, [refreshProjects]);

    // Initial load + selection persistence.
    useEffect(() => { refreshProjects(); }, [refreshProjects]);
    useEffect(() => {
        if (selectedName) {
            try { window.localStorage.setItem('fragmenta.datasetPrep.lastProject', selectedName); } catch {}
            refreshProject(selectedName);
        } else {
            setProject(null);
        }
    }, [selectedName, refreshProject]);

    // Cleanup any in-flight poll on unmount.
    useEffect(() => () => {
        if (pollHandleRef.current) {
            window.clearTimeout(pollHandleRef.current);
            pollHandleRef.current = null;
        }
    }, []);

    async function pollAnnotateStatus(name) {
        try {
            const { data } = await api.get(`/api/projects/${encodeURIComponent(name)}/annotate/status`);
            setAnnotateJob(data.job);
            if (data.job.state === 'done') {
                await refreshProject(name);
                return;
            }
            if (data.job.state === 'error') {
                setError(data.job.error || 'Annotation failed');
                return;
            }
            pollHandleRef.current = window.setTimeout(() => pollAnnotateStatus(name), 500);
        } catch (e) {
            setError(extractError(e, 'Status poll failed'));
        }
    }

    async function handleAnnotate(scope /* "all" | [file_names] */) {
        if (!project) return;
        setError('');
        try {
            await api.post(`/api/projects/${encodeURIComponent(project.name)}/annotate`, {
                tier: 'basic',
                scope: scope ?? 'all',
            });
            pollAnnotateStatus(project.name);
        } catch (e) {
            setError(extractError(e, 'Failed to start annotation'));
        }
    }

    async function handleClipPromptChange(fileName, newPrompt) {
        if (!project) return;
        try {
            // Manually editing a prompt locks it so a later bulk auto-annotate
            // doesn't silently overwrite the user's text.
            await api.patch(
                `/api/projects/${encodeURIComponent(project.name)}/clip/${encodeURIComponent(fileName)}`,
                { prompt: newPrompt, locked: true },
            );
            await refreshProject(project.name);
        } catch (e) {
            setError(extractError(e, 'Failed to save prompt'));
        }
    }

    async function handleClipDelete(fileName) {
        if (!project) return;
        if (!window.confirm(`Remove ${fileName} from this project?`)) return;
        try {
            await api.delete(
                `/api/projects/${encodeURIComponent(project.name)}/clip/${encodeURIComponent(fileName)}`,
            );
            await refreshProject(project.name);
        } catch (e) {
            setError(extractError(e, 'Failed to delete clip'));
        }
    }

    return (
        <Stack spacing={2.5}>
            <Typography variant="body2" color="text.secondary">
                Each project is a folder on disk holding audio + matching .txt sidecars —
                the format SA3 trains against directly.
            </Typography>

            <ProjectSelector
                projects={projects}
                selectedName={selectedName}
                onSelect={setSelectedName}
                onCreateClick={() => setCreateOpen(true)}
            />

            {error && <Alert severity="error" onClose={() => setError('')}>{error}</Alert>}

            {project && (
                <Stack spacing={2}>
                    <ProjectHeader project={project} />

                    <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
                        <Button
                            variant="outlined"
                            startIcon={<FolderOpenIcon size={18} />}
                            onClick={() => setIngestOpen(true)}
                            disabled={isAnnotating}
                        >
                            Add audio
                        </Button>
                        <Button
                            variant="contained"
                            startIcon={<SparklesIcon size={18} />}
                            onClick={() => handleAnnotate('all')}
                            disabled={isAnnotating || project.clip_count === 0}
                        >
                            Auto-annotate all
                        </Button>
                    </Box>

                    {isAnnotating && annotateJob && (
                        <Box>
                            <LinearProgress
                                variant={annotateJob.total > 0 ? 'determinate' : 'indeterminate'}
                                value={annotateJob.total > 0 ? (annotateJob.current / annotateJob.total) * 100 : undefined}
                            />
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.75, display: 'block' }}>
                                Annotating {annotateJob.current} / {annotateJob.total}
                                {annotateJob.current_file ? ` · ${annotateJob.current_file}` : ''}
                            </Typography>
                        </Box>
                    )}

                    <ClipTable
                        clips={project.clips}
                        onPromptChange={handleClipPromptChange}
                        onAnnotate={(fname) => handleAnnotate([fname])}
                        onDelete={handleClipDelete}
                        disabled={isAnnotating}
                    />
                </Stack>
            )}

            <CreateProjectDialog
                open={createOpen}
                existingNames={projects.map((p) => p.name)}
                onClose={() => setCreateOpen(false)}
                onCreated={async (name) => {
                    setCreateOpen(false);
                    await refreshProjects();
                    setSelectedName(name);
                }}
            />

            <IngestDialog
                open={ingestOpen}
                projectName={project?.name}
                onClose={() => setIngestOpen(false)}
                onIngested={async () => {
                    setIngestOpen(false);
                    if (project) await refreshProject(project.name);
                    await refreshProjects();
                }}
            />
        </Stack>
    );
}

// ---------- subcomponents --------------------------------------------------

function ProjectSelector({ projects, selectedName, onSelect, onCreateClick }) {
    return (
        <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap">
            <FormControl size="small" sx={{ minWidth: 240 }}>
                <InputLabel id="project-select-label">Project</InputLabel>
                <Select
                    labelId="project-select-label"
                    label="Project"
                    value={selectedName}
                    onChange={(e) => onSelect(e.target.value)}
                    displayEmpty
                >
                    <MenuItem value="">
                        <em>None selected</em>
                    </MenuItem>
                    {projects.map((p) => (
                        <MenuItem key={p.name} value={p.name}>
                            {p.name} · {p.clip_count} clip{p.clip_count === 1 ? '' : 's'}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <Button variant="outlined" startIcon={<PlusIcon size={18} />} onClick={onCreateClick}>
                New project
            </Button>
        </Stack>
    );
}

function ProjectHeader({ project }) {
    return (
        <Box>
            <Typography variant="h6">{project.name}</Typography>
            <Typography variant="body2" color="text.secondary">
                {project.clip_count} clip{project.clip_count === 1 ? '' : 's'} · ingest mode: {project.ingest_mode}
            </Typography>
        </Box>
    );
}

function ClipTable({ clips, onPromptChange, onAnnotate, onDelete, disabled }) {
    if (!clips || clips.length === 0) {
        return (
            <Box sx={{ py: 4, textAlign: 'center', color: 'text.secondary' }}>
                <Typography variant="body2">
                    No clips yet. Use “Add audio” to bring in a folder.
                </Typography>
            </Box>
        );
    }
    return (
        <TableContainer component={Paper} variant="outlined">
            <Table size="small">
                <TableHead>
                    <TableRow>
                        <TableCell sx={{ width: '30%' }}>File</TableCell>
                        <TableCell>Prompt</TableCell>
                        <TableCell sx={{ width: 120, textAlign: 'right' }}>Actions</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {clips.map((c) => (
                        <ClipRow
                            key={c.file_name}
                            clip={c}
                            onPromptChange={onPromptChange}
                            onAnnotate={onAnnotate}
                            onDelete={onDelete}
                            disabled={disabled}
                        />
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
}

function ClipRow({ clip, onPromptChange, onAnnotate, onDelete, disabled }) {
    const [draft, setDraft] = useState(clip.prompt);
    // Keep local draft in sync with server-side updates (e.g. after auto-annotate).
    useEffect(() => { setDraft(clip.prompt); }, [clip.prompt]);

    const dirty = draft !== clip.prompt;

    return (
        <TableRow hover>
            <TableCell sx={{ wordBreak: 'break-all' }}>{clip.file_name}</TableCell>
            <TableCell>
                <TextField
                    fullWidth
                    size="small"
                    variant="standard"
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={() => { if (dirty) onPromptChange(clip.file_name, draft); }}
                    placeholder="(empty — write a prompt or auto-annotate)"
                    disabled={disabled}
                />
            </TableCell>
            <TableCell sx={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                <Tooltip title="Auto-annotate this clip">
                    <span>
                        <IconButton
                            size="small"
                            onClick={() => onAnnotate(clip.file_name)}
                            disabled={disabled}
                        >
                            <SparklesIcon size={16} />
                        </IconButton>
                    </span>
                </Tooltip>
                <Tooltip title="Remove this clip from the project">
                    <span>
                        <IconButton
                            size="small"
                            onClick={() => onDelete(clip.file_name)}
                            disabled={disabled}
                        >
                            <TrashIcon size={16} />
                        </IconButton>
                    </span>
                </Tooltip>
            </TableCell>
        </TableRow>
    );
}

function CreateProjectDialog({ open, existingNames, onClose, onCreated }) {
    const [name, setName] = useState('');
    const [busy, setBusy] = useState(false);
    const [dialogError, setDialogError] = useState('');

    useEffect(() => {
        if (open) { setName(''); setDialogError(''); }
    }, [open]);

    const duplicate = existingNames.includes(name.trim());

    async function submit() {
        setDialogError('');
        setBusy(true);
        try {
            const { data } = await api.post('/api/projects', { name: name.trim() });
            await onCreated(data.name);
        } catch (e) {
            setDialogError(extractError(e, 'Failed to create project'));
        } finally {
            setBusy(false);
        }
    }

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle>New project</DialogTitle>
            <DialogContent>
                <Stack spacing={2} sx={{ pt: 1 }}>
                    <TextField
                        autoFocus
                        label="Project name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        helperText="Letters, digits, spaces, dashes, underscores, dots. Becomes a folder name on disk."
                        error={duplicate}
                    />
                    {duplicate && (
                        <Typography variant="caption" color="error">
                            A project with this name already exists.
                        </Typography>
                    )}
                    {dialogError && <Alert severity="error">{dialogError}</Alert>}
                </Stack>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} disabled={busy}>Cancel</Button>
                <Button
                    variant="contained"
                    onClick={submit}
                    disabled={busy || !name.trim() || duplicate}
                >
                    Create
                </Button>
            </DialogActions>
        </Dialog>
    );
}

function IngestDialog({ open, projectName, onClose, onIngested }) {
    const [folder, setFolder] = useState('');
    const [mode, setMode] = useState('copy');
    const [busy, setBusy] = useState(false);
    const [dialogError, setDialogError] = useState('');

    useEffect(() => {
        if (open) { setFolder(''); setMode('copy'); setDialogError(''); }
    }, [open]);

    async function pick() {
        try {
            const { data } = await api.post('/api/pick-folder', {});
            if (data?.path) setFolder(data.path);
        } catch (e) {
            setDialogError(extractError(e, 'Folder picker failed'));
        }
    }

    async function submit() {
        if (!projectName) return;
        setBusy(true);
        setDialogError('');
        try {
            await api.post(
                `/api/projects/${encodeURIComponent(projectName)}/ingest`,
                { folder_path: folder, mode },
            );
            await onIngested();
        } catch (e) {
            setDialogError(extractError(e, 'Ingest failed'));
        } finally {
            setBusy(false);
        }
    }

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle>Add audio to {projectName}</DialogTitle>
            <DialogContent>
                <Stack spacing={2} sx={{ pt: 1 }}>
                    <Stack direction="row" spacing={1.5} alignItems="center">
                        <Button variant="outlined" startIcon={<FolderOpenIcon size={18} />} onClick={pick}>
                            Pick folder
                        </Button>
                        <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-all' }}>
                            {folder || 'No folder selected'}
                        </Typography>
                    </Stack>

                    <FormControl>
                        <Typography variant="body2" gutterBottom>How to bring the audio in:</Typography>
                        <RadioGroup value={mode} onChange={(e) => setMode(e.target.value)}>
                            <FormControlLabel
                                value="copy"
                                control={<Radio size="small" />}
                                label={<Typography variant="body2">Copy — duplicates audio into the project (safe, originals untouched)</Typography>}
                            />
                            <FormControlLabel
                                value="symlink"
                                control={<Radio size="small" />}
                                label={<Typography variant="body2">Symlink — points at the originals (saves disk, breaks if you move them)</Typography>}
                            />
                        </RadioGroup>
                    </FormControl>

                    {dialogError && <Alert severity="error">{dialogError}</Alert>}
                </Stack>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} disabled={busy}>Cancel</Button>
                <Button variant="contained" onClick={submit} disabled={busy || !folder}>
                    {busy ? 'Adding…' : 'Add'}
                </Button>
            </DialogActions>
        </Dialog>
    );
}

// ---------- utils ----------------------------------------------------------

function extractError(e, fallback) {
    return e?.response?.data?.error || e?.message || fallback;
}
