import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Alert,
    Autocomplete,
    Box,
    Button,
    Checkbox,
    Chip,
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
    ChevronDown as ChevronDownIcon,
    FolderOpenIcon,
    PlusIcon,
    SparklesIcon,
    SaveIcon,
    Upload as UploadIcon,
    CheckCircle2 as CommitIcon,
    Undo2 as DiscardIcon,
    Square as StopIcon,
    Trash2 as TrashIcon,
} from 'lucide-react';
import api from '../../api';
import { appStyles } from '../../theme';

/**
 * DatasetPrep — sidecar-native dataset surface with a buffered editing model.
 *
 * One page, no modes. Pick or create a project. The dataset folder on disk
 * is the *committed* state. Edits, auto-annotate output, and just-ingested
 * audio all live in an in-memory session until the user explicitly hits
 * Save (writes a draft) or Commit (writes .txt sidecars).
 */
export default function DatasetPrep({ onOpenCheckpointManager }) {
    const [projects, setProjects] = useState([]);
    const [selectedName, setSelectedName] = useState(() => {
        try { return window.localStorage.getItem('fragmenta.datasetPrep.lastProject') || ''; }
        catch { return ''; }
    });
    const [project, setProject] = useState(null);
    const [createOpen, setCreateOpen] = useState(false);
    const [loadOpen, setLoadOpen] = useState(false);
    const [ingestOpen, setIngestOpen] = useState(false);
    const [error, setError] = useState('');

    const [errorCode, setErrorCode] = useState('');
    const [errorExtra, setErrorExtra] = useState(null);
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
        setError(''); setErrorCode(''); setErrorExtra(null);
        try {
            await api.post(`/api/projects/${encodeURIComponent(project.name)}/annotate`, {
                tier,
                scope: scope ?? 'all',
                skip_existing: opts.skip_existing ?? skipExisting,
            });
            pollAnnotateStatus(project.name);
        } catch (e) {
            const body = e?.response?.data || {};
            setError(extractError(e, 'Failed to start annotation'));
            setErrorCode(body.code || '');
            setErrorExtra(body.install_command ? { install_command: body.install_command } : null);
        }
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
                <Box sx={{ ...appStyles.sectionCardHeader, mb: 0.5 }}>
                    <Box component="span" sx={appStyles.sectionCardIcon}>
                        <UploadIcon size={20} />
                    </Box>
                    <Typography variant="h6" sx={appStyles.sectionCardTitle}>
                        Dataset Workbench
                    </Typography>
                    <Box sx={{ flex: 1 }} />
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<FolderOpenIcon size={16} />}
                        onClick={() => setLoadOpen(true)}
                        disabled={projects.length === 0}
                    >
                        Load project
                    </Button>
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<PlusIcon size={16} />}
                        onClick={() => setCreateOpen(true)}
                    >
                        New project
                    </Button>
                </Box>
                <Typography variant="body2" color="text.secondary">
                    Create a new dataset or load and edit one. 
                </Typography>
                <Typography variant="body2" color="text.secondary" paddingBottom={2}>
                     You can auto-annotate using Librosa and CLAP or annotate everything manually.
                </Typography>
            </Box>

            {error && (
                <Alert
                    severity={(errorCode === 'clap_not_available' || errorCode === 'clap_package_missing') ? 'warning' : 'error'}
                    onClose={() => { setError(''); setErrorCode(''); setErrorExtra(null); }}
                    action={
                        errorCode === 'clap_not_available' && onOpenCheckpointManager ? (
                            <Button
                                color="inherit"
                                size="small"
                                onClick={() => { setError(''); setErrorCode(''); setErrorExtra(null); onOpenCheckpointManager(); }}
                            >
                                Open Model Management
                            </Button>
                        ) : null
                    }
                >
                    <Box>
                        <Typography variant="body2">{error}</Typography>
                        {errorCode === 'clap_package_missing' && errorExtra?.install_command && (
                            <Box
                                component="pre"
                                sx={{
                                    mt: 1,
                                    mb: 0,
                                    p: 1,
                                    borderRadius: 1,
                                    bgcolor: 'action.hover',
                                    fontSize: '0.8rem',
                                    fontFamily: 'monospace',
                                    overflowX: 'auto',
                                }}
                            >
                                {errorExtra.install_command}
                            </Box>
                        )}
                    </Box>
                </Alert>
            )}

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

                    {tier === 'rich' && (
                        <ClapVocabAccordion disabled={isAnnotating} />
                    )}

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

            <LoadProjectDialog
                open={loadOpen}
                projects={projects}
                currentName={selectedName}
                onClose={() => setLoadOpen(false)}
                onLoad={(name) => {
                    setLoadOpen(false);
                    trySelectProject(name);
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

function ClapVocabAccordion({ disabled }) {
    const [labels, setLabels] = useState({ genre: [], mood: [], instruments: [] });
    const [overridden, setOverridden] = useState(false);
    const [dirty, setDirty] = useState(false);
    const [busy, setBusy] = useState(false);
    const [vocabError, setVocabError] = useState('');

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const { data } = await api.get('/api/annotator-labels');
                if (cancelled) return;
                setLabels(data.labels || { genre: [], mood: [], instruments: [] });
                setOverridden(!!data.overridden);
                setDirty(false);
            } catch (e) {
                if (!cancelled) setVocabError(extractError(e, 'Failed to load vocabulary'));
            }
        })();
        return () => { cancelled = true; };
    }, []);

    function setCategory(cat, values) {
        setLabels((prev) => ({ ...prev, [cat]: values }));
        setDirty(true);
    }

    async function save() {
        setBusy(true);
        setVocabError('');
        try {
            await api.put('/api/annotator-labels', labels);
            setDirty(false);
            setOverridden(true);
        } catch (e) {
            setVocabError(extractError(e, 'Failed to save vocabulary'));
        } finally {
            setBusy(false);
        }
    }

    async function reset() {
        if (!window.confirm('Reset vocabulary to the built-in defaults? Your custom tags will be lost.')) return;
        setBusy(true);
        setVocabError('');
        try {
            await api.delete('/api/annotator-labels');
            const { data } = await api.get('/api/annotator-labels');
            setLabels(data.labels || { genre: [], mood: [], instruments: [] });
            setOverridden(false);
            setDirty(false);
        } catch (e) {
            setVocabError(extractError(e, 'Failed to reset vocabulary'));
        } finally {
            setBusy(false);
        }
    }

    const tagCount = (labels.genre?.length || 0) + (labels.mood?.length || 0) + (labels.instruments?.length || 0);

    return (
        <Accordion sx={{ '&, &.Mui-expanded': { mt: 0, mb: 0 } }}>
            <AccordionSummary expandIcon={<ChevronDownIcon size={18} />}>
                <Typography variant="subtitle1">CLAP Vocabulary</Typography>
                <Typography variant="caption" color="text.secondary" sx={{ ml: 1.5, alignSelf: 'center' }}>
                    {overridden ? 'custom' : 'defaults'} · {tagCount} tags
                </Typography>
            </AccordionSummary>
            <AccordionDetails>
                <Stack spacing={2}>
                    <Typography variant="body2" color="text.secondary">
                        Words CLAP scores each clip against. Empty categories are ignored. Tweak to match your dataset's territory.
                    </Typography>
                    <VocabCategory
                        label="Genre"
                        values={labels.genre || []}
                        onChange={(v) => setCategory('genre', v)}
                        disabled={disabled || busy}
                    />
                    <VocabCategory
                        label="Mood"
                        values={labels.mood || []}
                        onChange={(v) => setCategory('mood', v)}
                        disabled={disabled || busy}
                    />
                    <VocabCategory
                        label="Instruments"
                        values={labels.instruments || []}
                        onChange={(v) => setCategory('instruments', v)}
                        disabled={disabled || busy}
                    />
                    {vocabError && <Alert severity="error" onClose={() => setVocabError('')}>{vocabError}</Alert>}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <Button
                            variant="text"
                            size="small"
                            onClick={reset}
                            disabled={disabled || busy || !overridden}
                        >
                            Reset to defaults
                        </Button>
                        <Box sx={{ flex: 1 }} />
                        <Button
                            variant="contained"
                            size="small"
                            onClick={save}
                            disabled={disabled || busy || !dirty}
                        >
                            Save vocabulary
                        </Button>
                    </Box>
                </Stack>
            </AccordionDetails>
        </Accordion>
    );
}

function VocabCategory({ label, values, onChange, disabled }) {
    return (
        <Autocomplete
            multiple
            freeSolo
            options={[]}
            value={values}
            onChange={(_e, newValues) => onChange(newValues)}
            disabled={disabled}
            renderTags={(value, getTagProps) =>
                value.map((option, index) => {
                    const tagProps = getTagProps({ index });
                    return (
                        <Chip
                            variant="outlined"
                            size="small"
                            label={option}
                            {...tagProps}
                            key={`${option}-${index}`}
                        />
                    );
                })
            }
            renderInput={(params) => (
                <TextField
                    {...params}
                    label={label}
                    placeholder="Add tag, press Enter"
                    size="small"
                />
            )}
        />
    );
}

function LoadProjectDialog({ open, projects, currentName, onClose, onLoad }) {
    const [picked, setPicked] = useState(currentName || '');

    useEffect(() => {
        if (open) setPicked(currentName || (projects[0]?.name ?? ''));
    }, [open, currentName, projects]);

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle>Load project</DialogTitle>
            <DialogContent>
                {projects.length === 0 ? (
                    <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                        No projects yet. Create one first.
                    </Typography>
                ) : (
                    <RadioGroup value={picked} onChange={(e) => setPicked(e.target.value)}>
                        {projects.map((p) => (
                            <FormControlLabel
                                key={p.name}
                                value={p.name}
                                control={<Radio size="small" />}
                                label={
                                    <Box>
                                        <Typography variant="body2">{p.name}</Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            {p.clip_count} clip{p.clip_count === 1 ? '' : 's'}
                                            {p.has_draft ? ' · has unsaved draft' : ''}
                                        </Typography>
                                    </Box>
                                }
                                sx={{ alignItems: 'flex-start', py: 0.5 }}
                            />
                        ))}
                    </RadioGroup>
                )}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Button
                    variant="contained"
                    onClick={() => onLoad(picked)}
                    disabled={!picked || projects.length === 0}
                >
                    Load
                </Button>
            </DialogActions>
        </Dialog>
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
