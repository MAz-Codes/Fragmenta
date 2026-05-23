import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    Alert,
    Box,
    Button,
    Checkbox,
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
    SaveIcon,
    CheckCircle2 as CommitIcon,
    Undo2 as DiscardIcon,
    Square as StopIcon,
    Trash2 as TrashIcon,
} from 'lucide-react';
import api from '../../api';

/**
 * DatasetPrep — sidecar-native dataset surface with a buffered editing model.
 *
 * One page, no modes. Pick or create a project. The dataset folder on disk
 * is the *committed* state. Edits, auto-annotate output, and just-ingested
 * audio all live in an in-memory session until the user explicitly hits
 * Save (writes a draft) or Commit (writes .txt sidecars).
 */
export default function DatasetPrep() {
    const [projects, setProjects] = useState([]);
    const [selectedName, setSelectedName] = useState(() => {
        try { return window.localStorage.getItem('fragmenta.datasetPrep.lastProject') || ''; }
        catch { return ''; }
    });
    const [project, setProject] = useState(null);
    const [createOpen, setCreateOpen] = useState(false);
    const [ingestOpen, setIngestOpen] = useState(false);
    const [error, setError] = useState('');

    const [annotateJob, setAnnotateJob] = useState(null);
    const [tier, setTier] = useState(() => {
        try { return window.localStorage.getItem('fragmenta.datasetPrep.tier') || 'basic'; }
        catch { return 'basic'; }
    });
    const [skipExisting, setSkipExisting] = useState(true);

    const pollHandleRef = useRef(null);
    const isAnnotating = annotateJob?.state === 'running';

    const refreshProjects = useCallback(async () => {
        try {
            const { data } = await api.get('/api/projects');
            setProjects(data.projects || []);
        } catch (e) { setError(extractError(e, 'Failed to list projects')); }
    }, []);

    const refreshProject = useCallback(async (name) => {
        if (!name) { setProject(null); return; }
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

    useEffect(() => { refreshProjects(); }, [refreshProjects]);

    useEffect(() => {
        if (selectedName) {
            try { window.localStorage.setItem('fragmenta.datasetPrep.lastProject', selectedName); } catch {}
            refreshProject(selectedName);
        } else {
            setProject(null);
        }
    }, [selectedName, refreshProject]);

    useEffect(() => () => {
        if (pollHandleRef.current) {
            window.clearTimeout(pollHandleRef.current);
            pollHandleRef.current = null;
        }
    }, []);

    function changeTier(value) {
        setTier(value);
        try { window.localStorage.setItem('fragmenta.datasetPrep.tier', value); } catch {}
    }

    function trySelectProject(nextName) {
        // Confirm before switching if there are unsaved or uncommitted edits.
        if (project && (project.dirty || project.has_unsaved_changes) && nextName !== project.name) {
            const ok = window.confirm(
                `“${project.name}” has unsaved or uncommitted changes. Switch anyway? They'll stay in memory until you reload the project — but a backend restart will lose them.`,
            );
            if (!ok) return;
        }
        setSelectedName(nextName);
    }

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
        } catch (e) { setError(extractError(e, 'Status poll failed')); }
    }

    async function handleAnnotate(scope /* "all" | [file_names] */, opts = {}) {
        if (!project) return;
        setError('');
        try {
            await api.post(`/api/projects/${encodeURIComponent(project.name)}/annotate`, {
                tier,
                scope: scope ?? 'all',
                skip_existing: opts.skip_existing ?? skipExisting,
            });
            pollAnnotateStatus(project.name);
        } catch (e) { setError(extractError(e, 'Failed to start annotation')); }
    }

    async function handleCancelAnnotate() {
        if (!project) return;
        try {
            await api.post(`/api/projects/${encodeURIComponent(project.name)}/annotate/cancel`);
        } catch (e) { setError(extractError(e, 'Cancel failed')); }
    }

    async function handleSave() {
        if (!project) return;
        setError('');
        try {
            const { data } = await api.post(`/api/projects/${encodeURIComponent(project.name)}/save`);
            setProject(data);
        } catch (e) { setError(extractError(e, 'Save failed')); }
    }

    async function handleCommit() {
        if (!project) return;
        setError('');
        try {
            const { data } = await api.post(`/api/projects/${encodeURIComponent(project.name)}/commit`);
            setProject(data);
            await refreshProjects();
        } catch (e) { setError(extractError(e, 'Commit failed')); }
    }

    async function handleDiscard() {
        if (!project) return;
        const ok = window.confirm(
            `Discard all uncommitted work in “${project.name}”? Audio files added since the last commit will be deleted. This cannot be undone.`,
        );
        if (!ok) return;
        setError('');
        try {
            const { data } = await api.post(`/api/projects/${encodeURIComponent(project.name)}/discard`);
            setProject(data);
            await refreshProjects();
        } catch (e) { setError(extractError(e, 'Discard failed')); }
    }

    async function handleClipPromptChange(fileName, newPrompt) {
        if (!project) return;
        try {
            await api.patch(
                `/api/projects/${encodeURIComponent(project.name)}/clip/${encodeURIComponent(fileName)}`,
                { prompt: newPrompt },
            );
            // Reload to pick up dirty-state flip in the header.
            await refreshProject(project.name);
        } catch (e) { setError(extractError(e, 'Failed to save prompt')); }
    }

    async function handleClipDelete(fileName) {
        if (!project) return;
        if (!window.confirm(`Remove ${fileName} from this project? (Deletes the audio file from disk immediately — cannot be discarded back.)`)) return;
        try {
            await api.delete(
                `/api/projects/${encodeURIComponent(project.name)}/clip/${encodeURIComponent(fileName)}`,
            );
            await refreshProject(project.name);
        } catch (e) { setError(extractError(e, 'Failed to delete clip')); }
    }

    return (
        <Paper variant="outlined" sx={{ p: { xs: 2.25, sm: 3 }, borderRadius: 2.5 }}>
        <Stack spacing={2.5}>
            <Box>
                <Typography variant="h6">Dataset</Typography>
                <Typography variant="body2" color="text.secondary">
                    Each project is a folder on disk holding audio + matching .txt sidecars —
                    the format SA3 trains against directly. Edits live in memory; Save persists a draft,
                    Commit writes the final sidecars.
                </Typography>
            </Box>

            <ProjectSelector
                projects={projects}
                selectedName={selectedName}
                onSelect={trySelectProject}
                onCreateClick={() => setCreateOpen(true)}
            />

            {error && <Alert severity="error" onClose={() => setError('')}>{error}</Alert>}

            {project && (
                <Stack spacing={2}>
                    <ProjectHeader
                        project={project}
                        onSave={handleSave}
                        onCommit={handleCommit}
                        onDiscard={handleDiscard}
                        disabled={isAnnotating}
                    />

                    <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
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
                        <FormControl size="small" sx={{ minWidth: 220 }}>
                            <InputLabel id="annotate-tier-label">Annotation tier</InputLabel>
                            <Select
                                labelId="annotate-tier-label"
                                label="Annotation tier"
                                value={tier}
                                onChange={(e) => changeTier(e.target.value)}
                                disabled={isAnnotating}
                            >
                                <MenuItem value="basic">Basic — librosa only</MenuItem>
                                <MenuItem value="rich">Rich — LAION-CLAP</MenuItem>
                            </Select>
                        </FormControl>
                        <FormControlLabel
                            control={
                                <Checkbox
                                    size="small"
                                    checked={skipExisting}
                                    onChange={(e) => setSkipExisting(e.target.checked)}
                                    disabled={isAnnotating}
                                />
                            }
                            label={<Typography variant="caption" color="text.secondary">Skip clips that already have a prompt</Typography>}
                        />
                    </Box>

                    {isAnnotating && annotateJob && (
                        <Box>
                            <LinearProgress
                                variant={annotateJob.total > 0 ? 'determinate' : 'indeterminate'}
                                value={annotateJob.total > 0 ? (annotateJob.current / annotateJob.total) * 100 : undefined}
                            />
                            <Box sx={{ mt: 0.75, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
                                    Annotating {annotateJob.current} / {annotateJob.total}
                                    {annotateJob.current_file ? ` · ${annotateJob.current_file}` : ''}
                                </Typography>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    color="error"
                                    startIcon={<StopIcon size={14} />}
                                    onClick={handleCancelAnnotate}
                                >
                                    Stop
                                </Button>
                            </Box>
                        </Box>
                    )}

                    <ClipTable
                        clips={project.clips}
                        onPromptChange={handleClipPromptChange}
                        onAnnotate={(fname) => handleAnnotate([fname], { skip_existing: false })}
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
        </Paper>
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
                            {p.has_draft ? ' · draft' : ''}
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

function ProjectHeader({ project, onSave, onCommit, onDiscard, disabled }) {
    const stateLabel = (() => {
        if (project.dirty && project.has_unsaved_changes) return 'Unsaved changes';
        if (project.dirty && !project.has_unsaved_changes) return 'Draft saved · not committed';
        if (!project.dirty) return 'All changes committed';
        return '';
    })();
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Box sx={{ flex: 1, minWidth: 240 }}>
                <Typography variant="h6">{project.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                    {project.clip_count} clip{project.clip_count === 1 ? '' : 's'}
                    {' · '}{stateLabel}
                </Typography>
            </Box>
            <Stack direction="row" spacing={1}>
                <Tooltip title="Discard uncommitted work — deletes audio added since the last commit">
                    <span>
                        <Button
                            variant="text"
                            color="error"
                            size="small"
                            startIcon={<DiscardIcon size={16} />}
                            onClick={onDiscard}
                            disabled={disabled || !project.dirty}
                        >
                            Discard
                        </Button>
                    </span>
                </Tooltip>
                <Tooltip title="Save a draft — persists across app restarts but isn't the SA3 sidecar form">
                    <span>
                        <Button
                            variant="outlined"
                            size="small"
                            startIcon={<SaveIcon size={16} />}
                            onClick={onSave}
                            disabled={disabled || !project.has_unsaved_changes}
                        >
                            Save
                        </Button>
                    </span>
                </Tooltip>
                <Tooltip title="Commit — writes .txt sidecars (overwrites the previous commit)">
                    <span>
                        <Button
                            variant="contained"
                            size="small"
                            startIcon={<CommitIcon size={16} />}
                            onClick={onCommit}
                            disabled={disabled || !project.dirty}
                        >
                            Commit
                        </Button>
                    </span>
                </Tooltip>
            </Stack>
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
                <Tooltip title="Auto-annotate this clip (overwrites any current prompt)">
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
                <Tooltip title="Remove this clip from the project (immediate)">
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
