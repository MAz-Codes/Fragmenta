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
    Portal,
    Radio,
    RadioGroup,
    Select,
    Snackbar,
    Stack,
    Switch,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField,
    Typography,
    useTheme,
} from '@mui/material';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import {
    ChevronDown as ChevronDownIcon,
    FolderOpenIcon,
    PlusIcon,
    WandSparkles,
    SaveIcon,
    Database as Database,
    DatabaseZap as DatasetIcon,
    Square as StopIcon,
    Trash2 as TrashIcon,
    Play as PlayIcon,
    Pause as PauseIcon,
    Scissors as ScissorsIcon,
    Music as MusicIcon,
    Activity as HealthIcon,
    X as CloseIcon,
} from 'lucide-react';
import api from '../api';
import { extractError } from '../utils/errors';
import { appStyles } from '../theme';

/**
 * DatasetPrep — sidecar-native dataset surface with a buffered editing model.
 *
 * One page, no modes. Pick or create a project. The dataset folder on disk
 * is the *committed* state. Edits, auto-annotate output, and just-ingested
 * audio all live in an in-memory session until the user explicitly hits
 * Save (writes a draft) or Commit (writes .txt sidecars).
 */
export default function DatasetPrep({ onOpenCheckpointManager, isDocker = false }) {
    const [projects, setProjects] = useState([]);
    const [selectedName, setSelectedName] = useState(() => {
        try { return window.localStorage.getItem('fragmenta.datasetPrep.lastProject') || ''; }
        catch { return ''; }
    });
    const [project, setProject] = useState(null);
    const [createOpen, setCreateOpen] = useState(false);
    const [loadOpen, setLoadOpen] = useState(false);
    const [ingestOpen, setIngestOpen] = useState(false);
    const [sliceTarget, setSliceTarget] = useState(null);  // file_name or null
    // Single confirm-dialog state powering destructive actions. Mirrors the
    // Free GPU / Start Fresh confirm style from App.js — replaces the
    // browser-native window.confirm() prompts so the UX is consistent.
    const [confirm, setConfirm] = useState(null);
    const [confirmBusy, setConfirmBusy] = useState(false);
    const [error, setError] = useState('');

    const [errorCode, setErrorCode] = useState('');
    const [errorExtra, setErrorExtra] = useState(null);
    const [annotateJob, setAnnotateJob] = useState(null);
    const [notice, setNotice] = useState(null);  // { severity, message } | null
    // Phase 6 — pre-encoded latents
    const [preEncodeJob, setPreEncodeJob] = useState(null);
    const [preEncodeOffer, setPreEncodeOffer] = useState(false); // post-commit dialog
    const [tier, setTier] = useState(() => {
        try { return window.localStorage.getItem('fragmenta.datasetPrep.tier') || 'basic'; }
        catch { return 'basic'; }
    });
    const [skipExisting, setSkipExisting] = useState(true);

    const pollHandleRef = useRef(null);
    const preEncodePollRef = useRef(null);
    // Cancellation token for the self-rescheduling poll chains. Clearing the
    // pending timeout on project switch isn't enough: a poll request that is
    // already in flight resolves AFTER the cleanup, setState's the OLD
    // project's job onto the new project's UI, and re-arms its chain with the
    // captured old name — two pollers then run forever. Bumping the epoch
    // makes the resolved callback drop its result instead.
    const pollEpochRef = useRef(0);
    const isAnnotating = annotateJob?.state === 'running';
    const isPreEncoding = preEncodeJob?.state === 'running' || preEncodeJob?.state === 'queued';

    // --- Multi-row selection (for bulk Slice) -----------------------------
    // Set<string> of clip file_names. Reset whenever the active project
    // changes, since selections from a different project are meaningless.
    const [selectedFiles, setSelectedFiles] = useState(() => new Set());
    useEffect(() => { setSelectedFiles(new Set()); }, [selectedName]);

    const toggleSelected = useCallback((fileName) => {
        setSelectedFiles((prev) => {
            const next = new Set(prev);
            if (next.has(fileName)) next.delete(fileName);
            else next.add(fileName);
            return next;
        });
    }, []);
    const toggleSelectAll = useCallback((clips) => {
        setSelectedFiles((prev) => {
            const allNames = clips.map((c) => c.file_name);
            const allSelected = allNames.length > 0 && allNames.every((n) => prev.has(n));
            return allSelected ? new Set() : new Set(allNames);
        });
    }, []);
    const clearSelection = useCallback(() => setSelectedFiles(new Set()), []);

    // --- Per-row audio preview --------------------------------------------
    // One <audio> for the whole table. Rows just say "play me" / "pause";
    // the parent reconciles which file is loaded and where the playhead is.
    const audioRef = useRef(null);
    const [playingFile, setPlayingFile] = useState(null);
    const [playProgress, setPlayProgress] = useState(0);  // 0..1

    const stopPlayback = useCallback(() => {
        const audio = audioRef.current;
        if (audio) { audio.pause(); }
        setPlayingFile(null);
        setPlayProgress(0);
    }, []);

    const handlePlayToggle = useCallback((fileName) => {
        if (!selectedName) return;
        const audio = audioRef.current;
        if (!audio) return;
        if (playingFile === fileName) {
            audio.pause();
            setPlayingFile(null);
            return;
        }
        const url = `/api/projects/${encodeURIComponent(selectedName)}/clip/${encodeURIComponent(fileName)}/audio`;
        audio.src = url;
        setPlayProgress(0);
        setPlayingFile(fileName);
        audio.play().catch(() => {
            setPlayingFile(null);
        });
    }, [selectedName, playingFile]);

    // Stop playback when the project changes — the audio element's src would
    // suddenly refer to a different project's file.
    useEffect(() => { stopPlayback(); }, [selectedName, stopPlayback]);

    const refreshProjects = useCallback(async () => {
        try {
            const { data } = await api.get('/api/projects');
            setProjects(data.projects || []);
        } catch (e) { setError(extractError(e, 'Failed to list projects')); }
    }, []);

    const [health, setHealth] = useState(null);
    const refreshHealth = useCallback(async (name) => {
        if (!name) { setHealth(null); return; }
        try {
            const { data } = await api.get(`/api/projects/${encodeURIComponent(name)}/health`);
            setHealth(data);
        } catch {
            // Non-fatal — strip just hides until next refresh.
            setHealth(null);
        }
    }, []);

    const refreshProject = useCallback(async (name) => {
        if (!name) { setProject(null); setHealth(null); return; }
        try {
            const { data } = await api.get(`/api/projects/${encodeURIComponent(name)}`);
            setProject(data);
            refreshHealth(name);
        } catch (e) {
            if (e?.response?.status === 404) {
                setSelectedName('');
                setProject(null);
                setHealth(null);
                await refreshProjects();
                return;
            }
            setError(extractError(e, 'Failed to load project'));
        }
    }, [refreshProjects, refreshHealth]);

    useEffect(() => { refreshProjects(); }, [refreshProjects]);

    const pollAnnotateStatus = useCallback(async function poll(name) {
        const epoch = pollEpochRef.current;
        try {
            const { data } = await api.get(`/api/projects/${encodeURIComponent(name)}/annotate/status`);
            if (epoch !== pollEpochRef.current) return;  // project switched mid-flight
            setAnnotateJob(data.job);
            if (data.job.state === 'done') {
                await refreshProject(name);
                return;
            }
            if (data.job.state === 'error') {
                setError(data.job.error || 'Annotation failed');
                return;
            }
            // Only keep polling while the backend is actively annotating. Other
            // states ('idle', 'cancelled', missing) terminate the loop so a
            // freshly-mounted tab doesn't poll forever for a non-existent job.
            if (data.job.state === 'running') {
                pollHandleRef.current = window.setTimeout(() => poll(name), 500);
            }
        } catch (e) {
            if (epoch === pollEpochRef.current) setError(extractError(e, 'Status poll failed'));
        }
    }, [refreshProject]);

    // Phase 6 — pre-encode polling. Same survives-tab-switch shape as the
    // annotate poller above.
    const pollPreEncodeStatus = useCallback(async function poll(name) {
        const epoch = pollEpochRef.current;
        try {
            const { data } = await api.get(`/api/projects/${encodeURIComponent(name)}/pre-encode/status`);
            if (epoch !== pollEpochRef.current) return;  // project switched mid-flight
            setPreEncodeJob(data.job);
            if (data.job.state === 'complete') {
                refreshProject(name);
                return;
            }
            if (data.job.state === 'failed') {
                setError(data.job.error || 'Pre-encoding failed');
                return;
            }
            if (data.job.state === 'running' || data.job.state === 'queued') {
                preEncodePollRef.current = window.setTimeout(() => poll(name), 750);
            }
        } catch (e) { /* non-fatal — bar just freezes */ }
    }, [refreshProject]);

    useEffect(() => {
        if (selectedName) {
            try { window.localStorage.setItem('fragmenta.datasetPrep.lastProject', selectedName); } catch {}
            refreshProject(selectedName);
            // Re-bootstrap progress polling on (re)mount or project switch, so
            // the progress strip survives tab changes while a job runs.
            pollAnnotateStatus(selectedName);
            pollPreEncodeStatus(selectedName);
        } else {
            setProject(null);
            setAnnotateJob(null);
            setPreEncodeJob(null);
        }
        return () => {
            // Invalidate in-flight poll promises (they check the epoch on
            // resolve) in addition to clearing the scheduled timeouts.
            pollEpochRef.current += 1;
            if (pollHandleRef.current) {
                window.clearTimeout(pollHandleRef.current);
                pollHandleRef.current = null;
            }
            if (preEncodePollRef.current) {
                window.clearTimeout(preEncodePollRef.current);
                preEncodePollRef.current = null;
            }
        };
    }, [selectedName, refreshProject, pollAnnotateStatus, pollPreEncodeStatus]);

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

    function handleCloseProject() {
        // Unload the current project — nothing on disk is touched. The
        // workbench returns to the empty no-project state.
        if (!project) return;
        if (project.dirty || project.has_unsaved_changes) {
            const ok = window.confirm(
                `“${project.name}” has unsaved or uncommitted changes. Close anyway? They'll stay in memory until you reload the project — but a backend restart will lose them.`,
            );
            if (!ok) return;
        }
        stopPlayback();
        setSelectedName('');
        try { window.localStorage.removeItem('fragmenta.datasetPrep.lastProject'); } catch {}
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
            setNotice({ severity: 'success', message: `Draft saved · ${data.clip_count} clips` });
        } catch (e) { setError(extractError(e, 'Save failed')); }
    }

    async function handleStartPreEncode() {
        if (!project) return;
        setError('');
        try {
            const { data } = await api.post(`/api/projects/${encodeURIComponent(project.name)}/pre-encode`);
            setPreEncodeJob(data.job);
            pollPreEncodeStatus(project.name);
        } catch (e) { setError(extractError(e, 'Pre-encode failed to start')); }
    }

    async function handleCancelPreEncode() {
        if (!project) return;
        try {
            await api.post(`/api/projects/${encodeURIComponent(project.name)}/pre-encode/cancel`);
        } catch (e) { setError(extractError(e, 'Cancel failed')); }
    }

    async function persistPreEncodeSuppression(suppress) {
        if (!project) return;
        try {
            const { data } = await api.patch(
                `/api/projects/${encodeURIComponent(project.name)}/pre-encode/prompt`,
                { suppress: !!suppress },
            );
            setProject(data);
        } catch (e) { /* non-fatal — dialog still closes */ }
    }

    async function handleCommit() {
        if (!project) return;
        setError('');
        try {
            const { data } = await api.post(`/api/projects/${encodeURIComponent(project.name)}/commit`);
            setProject(data);
            await refreshProjects();
            // Phase 6 — post-commit pre-encode prompt.
            // Open the dialog unless: (a) latents already present (re-commit
            // wiped them but we still avoid re-asking immediately), or
            // (b) the user previously chose "Don't ask again".
            if (!data.suppress_pre_encode_prompt && !data.latents_present && data.clip_count > 0) {
                setPreEncodeOffer(true);
            }
            setNotice({
                severity: 'success',
                message: `Dataset created · ${data.clip_count} clips written to disk`,
            });
        } catch (e) { setError(extractError(e, 'Create Dataset failed')); }
    }

    function handleDiscard() {
        if (!project) return;
        setConfirm({
            title: 'Delete unsaved changes',
            body: `Delete all changes in “${project.name}” since the last created dataset? Audio files added since then will be removed.`,
            warning: 'This cannot be undone.',
            confirmLabel: 'Delete',
            busyLabel: 'Deleting…',
            danger: true,
            onConfirm: async () => {
                setError('');
                try {
                    const { data } = await api.post(`/api/projects/${encodeURIComponent(project.name)}/discard`);
                    setProject(data);
                    await refreshProjects();
                    setNotice({ severity: 'info', message: 'Unsaved changes discarded' });
                } catch (e) { setError(extractError(e, 'Delete failed')); }
            },
        });
    }

    function handleDeleteProject(name) {
        if (!name) return;
        setConfirm({
            title: 'Delete project',
            body: `Permanently delete project “${name}”? The project folder (audio copies/symlinks, sidecars, drafts) will be removed from disk. Original source files outside the project are never touched.`,
            warning: 'This cannot be undone.',
            confirmLabel: 'Delete',
            busyLabel: 'Deleting…',
            danger: true,
            onConfirm: async () => {
                setError('');
                try {
                    await api.delete(`/api/projects/${encodeURIComponent(name)}`);
                    if (selectedName === name) {
                        stopPlayback();
                        setSelectedName('');
                        setProject(null);
                        try { window.localStorage.removeItem('fragmenta.datasetPrep.lastProject'); } catch {}
                    }
                    await refreshProjects();
                } catch (e) { setError(extractError(e, 'Delete project failed')); }
            },
        });
    }

    async function handleChangeTemplatePreset(presetId) {
        if (!project) return;
        try {
            const { data } = await api.patch(
                `/api/projects/${encodeURIComponent(project.name)}/template`,
                { preset: presetId },
            );
            setProject(data);
        } catch (e) {
            setError(extractError(e, 'Could not update annotation style'));
        }
    }

    function handleClearSelectedAnnotations() {
        if (!project || selectedFiles.size === 0) return;
        const count = selectedFiles.size;
        const files = Array.from(selectedFiles);
        setConfirm({
            title: 'Clear',
            body: `Clear annotations on ${count} clip${count === 1 ? '' : 's'}? Buffered in memory until you Save or Create Dataset.`,
            warning: 'Use the Delete button to revert; this action itself can’t be undone in place.',
            confirmLabel: `Clear (${count})`,
            busyLabel: 'Clearing…',
            danger: true,
            onConfirm: async () => {
                setError('');
                try {
                    for (const f of files) {
                        await api.patch(
                            `/api/projects/${encodeURIComponent(project.name)}/clip/${encodeURIComponent(f)}`,
                            { prompt: '' },
                        );
                    }
                    clearSelection();
                    await refreshProject(project.name);
                } catch (e) { setError(extractError(e, 'Clear annotations failed')); }
            },
        });
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
        if (!window.confirm(`Remove ${fileName} from this project? Only the project's copy (or symlink) is deleted — the original source file is never touched. This is immediate and cannot be discarded back.`)) return;
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
                        <Database size={20} />
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
                        onAddAudio={() => setIngestOpen(true)}
                        onClose={handleCloseProject}
                        disabled={isAnnotating}
                    />

                    <HealthStrip
                        health={health}
                        onSelectFiles={(files) => setSelectedFiles(new Set(files))}
                    />

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

                    {isPreEncoding && preEncodeJob && (
                        <Box>
                            <LinearProgress
                                variant={preEncodeJob.total > 0 ? 'determinate' : 'indeterminate'}
                                value={preEncodeJob.total > 0 ? (preEncodeJob.current / preEncodeJob.total) * 100 : undefined}
                            />
                            <Box sx={{ mt: 0.75, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
                                    Pre-encoding latents · {preEncodeJob.current} / {preEncodeJob.total}
                                    {preEncodeJob.autoencoder ? ` · ${preEncodeJob.autoencoder}` : ''}
                                </Typography>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    color="error"
                                    startIcon={<StopIcon size={14} />}
                                    onClick={handleCancelPreEncode}
                                >
                                    Stop
                                </Button>
                            </Box>
                        </Box>
                    )}

                    <ClipTable
                        projectName={selectedName}
                        clips={project.clips}
                        playingFile={playingFile}
                        playProgress={playProgress}
                        onPlayToggle={handlePlayToggle}
                        onPromptChange={handleClipPromptChange}
                        onAnnotate={(fname) => handleAnnotate([fname], { skip_existing: false })}
                        onDelete={(fname) => {
                            if (playingFile === fname) stopPlayback();
                            return handleClipDelete(fname);
                        }}
                        onSlice={(fname) => {
                            if (playingFile === fname) stopPlayback();
                            setSliceTarget(fname);
                        }}
                        selectedFiles={selectedFiles}
                        onToggleSelected={toggleSelected}
                        onToggleSelectAll={() => toggleSelectAll(project.clips)}
                        disabled={isAnnotating}
                        toolbar={
                            <Stack spacing={1}>
                                <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 1.5 }}>
                                    <Tooltip title={TIPS.dataset.autoAnnotateAll}>
                                    <Button
                                        variant="contained"
                                        color="warm"
                                        size="small"
                                        startIcon={<WandSparkles size={16} />}
                                        onClick={() => handleAnnotate('all')}
                                        disabled={isAnnotating || project.clip_count === 0}
                                    >
                                        Auto-annotate all
                                    </Button>
                                    </Tooltip>
                                    <Tooltip title={TIPS.dataset.templatePreset}>
                                    <FormControl size="small" sx={{ minWidth: 180 }}>
                                        <Select
                                            value={project.prompt_template_preset || 'music'}
                                            onChange={(e) => handleChangeTemplatePreset(e.target.value)}
                                            disabled={isAnnotating}
                                            renderValue={(v) => {
                                                const p = (project.prompt_template_presets || []).find((x) => x.id === v);
                                                return p ? p.label : v;
                                            }}
                                        >
                                            {(project.prompt_template_presets || []).map((p) => (
                                                <MenuItem key={p.id} value={p.id}>
                                                    <Box>
                                                        <Typography variant="body2">{p.label}</Typography>
                                                        <Typography variant="caption" color="text.secondary">
                                                            {p.description}
                                                        </Typography>
                                                    </Box>
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                    </Tooltip>
                                    <Tooltip title={TIPS.dataset.richAnnotate}>
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    size="small"
                                                    checked={tier === 'rich'}
                                                    onChange={(e) => changeTier(e.target.checked ? 'rich' : 'basic')}
                                                    disabled={isAnnotating}
                                                />
                                            }
                                            label={<Typography variant="caption" color="text.secondary">Rich annotation</Typography>}
                                            sx={{ mr: 0 }}
                                        />
                                    </Tooltip>
                                    <Tooltip title={TIPS.dataset.skipAnnotated}>
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    size="small"
                                                    checked={skipExisting}
                                                    onChange={(e) => setSkipExisting(e.target.checked)}
                                                    disabled={isAnnotating}
                                                />
                                            }
                                            label={<Typography variant="caption" color="text.secondary">Skip already annotated</Typography>}
                                            sx={{ mr: 0 }}
                                        />
                                    </Tooltip>
                                    <Box sx={{ flex: 1 }} />
                                    {selectedFiles.size > 0 && (
                                        <Button
                                            variant="outlined"
                                            color="error"
                                            size="small"
                                            startIcon={<TrashIcon size={16} />}
                                            onClick={handleClearSelectedAnnotations}
                                            disabled={isAnnotating}
                                        >
                                            Clear annotations ({selectedFiles.size})
                                        </Button>
                                    )}
                                </Box>
                                {tier === 'rich' && (
                                    <ClapVocabAccordion disabled={isAnnotating} />
                                )}
                            </Stack>
                        }
                    />
                    <audio
                        ref={audioRef}
                        style={{ display: 'none' }}
                        onTimeUpdate={(e) => {
                            const a = e.currentTarget;
                            if (a.duration && isFinite(a.duration)) {
                                setPlayProgress(a.currentTime / a.duration);
                            }
                        }}
                        onEnded={() => { setPlayingFile(null); setPlayProgress(0); }}
                        onError={() => { setPlayingFile(null); setPlayProgress(0); }}
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
                onDeleteProject={handleDeleteProject}
            />

            <IngestDialog
                open={ingestOpen}
                projectName={project?.name}
                isDocker={isDocker}
                onClose={() => setIngestOpen(false)}
                onIngested={async () => {
                    setIngestOpen(false);
                    if (project) await refreshProject(project.name);
                    await refreshProjects();
                }}
            />

            <SliceDialog
                open={Boolean(sliceTarget)}
                projectName={project?.name}
                fileName={sliceTarget}
                onClose={() => setSliceTarget(null)}
                onSliced={async () => {
                    clearSelection();
                    if (project) await refreshProject(project.name);
                    await refreshProjects();
                }}
            />

            <Dialog
                open={Boolean(confirm)}
                onClose={confirmBusy ? undefined : () => setConfirm(null)}
                aria-labelledby="dataset-confirm-title"
            >
                <DialogTitle id="dataset-confirm-title">
                    {confirm?.title}
                </DialogTitle>
                <DialogContent>
                    <Typography sx={appStyles.dialogBodyText}>
                        {confirm?.body}
                    </Typography>
                    {confirm?.warning && (
                        <Typography variant="body2" color="warning.main" sx={appStyles.dialogErrorText}>
                            {confirm.warning}
                        </Typography>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setConfirm(null)} disabled={confirmBusy}>
                        Cancel
                    </Button>
                    <Button
                        onClick={async () => {
                            if (!confirm?.onConfirm) { setConfirm(null); return; }
                            setConfirmBusy(true);
                            try {
                                await confirm.onConfirm();
                            } finally {
                                setConfirmBusy(false);
                                setConfirm(null);
                            }
                        }}
                        color={confirm?.danger ? 'error' : 'primary'}
                        variant="contained"
                        disabled={confirmBusy}
                    >
                        {confirmBusy ? (confirm?.busyLabel || 'Working…') : (confirm?.confirmLabel || 'Confirm')}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Phase 6 — post-commit pre-encode dialog. Surfaces after a
                successful Create Dataset commit unless the user previously
                chose "Don't ask again". */}
            <Dialog
                open={preEncodeOffer}
                onClose={() => setPreEncodeOffer(false)}
                maxWidth="xs"
                fullWidth
            >
                <DialogTitle>Pre-encode latents?</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                        Encode your audio into SA3 latents now to speed up training. The
                        autoencoder runs once up-front instead of every training step.
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Takes a few minutes for ~50 clips. Latents live in
                        <code> {project?.name}/.latents/</code> and get wiped automatically
                        when you next commit or edit a clip.
                    </Typography>
                </DialogContent>
                <DialogActions sx={{ flexWrap: 'wrap' }}>
                    <Button
                        onClick={() => {
                            persistPreEncodeSuppression(true);
                            setPreEncodeOffer(false);
                        }}
                        sx={{ mr: 'auto' }}
                    >
                        Don't ask again
                    </Button>
                    <Button onClick={() => setPreEncodeOffer(false)}>Not now</Button>
                    <Button
                        variant="contained"
                        onClick={() => {
                            setPreEncodeOffer(false);
                            handleStartPreEncode();
                        }}
                    >
                        Pre-encode now
                    </Button>
                </DialogActions>
            </Dialog>

            <Portal>
                <Snackbar
                    open={Boolean(notice)}
                    autoHideDuration={4000}
                    onClose={() => setNotice(null)}
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                >
                    {notice ? (
                        <Alert
                            onClose={() => setNotice(null)}
                            severity={notice.severity}
                            variant="filled"
                            sx={{ width: '100%' }}
                        >
                            {notice.message}
                        </Alert>
                    ) : undefined}
                </Snackbar>
            </Portal>
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
        <Accordion
            disableGutters
            sx={{ '&, &.Mui-expanded': { mt: 0, mb: 0 } }}
        >
            <AccordionSummary
                expandIcon={<ChevronDownIcon size={18} />}
                sx={{
                    minHeight: 48,
                    '&.Mui-expanded': { minHeight: 48 },
                    '& .MuiAccordionSummary-content': {
                        margin: '12px 0',
                        '&.Mui-expanded': { margin: '12px 0' },
                    },
                }}
            >
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

function LoadProjectDialog({ open, projects, currentName, onClose, onLoad, onDeleteProject }) {
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
                            <Box
                                key={p.name}
                                sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 0.25 }}
                            >
                                <FormControlLabel
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
                                    sx={{ alignItems: 'flex-start', flex: 1, mr: 0 }}
                                />
                                <Tooltip title={TIPS.dataset.deleteProject}>
                                    <span>
                                        <IconButton
                                            size="small"
                                            sx={{ color: 'text.disabled', '&:hover': { color: 'error.main', bgcolor: 'action.hover' } }}
                                            onClick={() => onDeleteProject(p.name)}
                                        >
                                            <TrashIcon size={16} />
                                        </IconButton>
                                    </span>
                                </Tooltip>
                            </Box>
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

function ProjectHeader({ project, onSave, onCommit, onDiscard, onAddAudio, onClose, disabled }) {
    const stateLabel = (() => {
        if (project.dirty && project.has_unsaved_changes) return 'Unsaved changes';
        if (project.dirty && !project.has_unsaved_changes) return 'Draft saved · dataset not created';
        if (!project.dirty) return 'Dataset created';
        return '';
    })();
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ flex: 1, minWidth: 240 }}>
                <Box>
                    <Typography variant="h6">Project: &ldquo;{project.name}&rdquo;</Typography>
                    <Typography variant="body2" color="text.secondary">
                        {project.clip_count} clip{project.clip_count === 1 ? '' : 's'}
                        {' · '}{stateLabel}
                    </Typography>
                </Box>
                <Button
                    variant="outlined"
                    size="small"
                    startIcon={<MusicIcon size={16} />}
                    onClick={onAddAudio}
                    disabled={disabled}
                >
                    Add audio
                </Button>
            </Stack>
            <Stack direction="row" spacing={1}>
                <Tooltip title={TIPS.dataset.discardChanges}>
                    <span>
                        <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            startIcon={<TrashIcon size={16} />}
                            onClick={onDiscard}
                            disabled={disabled || !project.dirty}
                        >
                            Delete
                        </Button>
                    </span>
                </Tooltip>
                <Tooltip title={TIPS.dataset.saveDraft}>
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
                <Tooltip title={TIPS.dataset.createDataset}>
                    <span>
                        <Button
                            variant="contained"
                            size="small"
                            startIcon={<DatasetIcon size={16} />}
                            onClick={onCommit}
                            disabled={disabled || !project.dirty}
                        >
                            Create Dataset
                        </Button>
                    </span>
                </Tooltip>
                <Tooltip title={TIPS.dataset.closeProject}>
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<CloseIcon size={16} />}
                        onClick={onClose}
                    >
                        Close
                    </Button>
                </Tooltip>
            </Stack>
        </Box>
    );
}

function HealthStrip({ health, onSelectFiles }) {
    if (!health || health.total_clips === 0) return null;
    const empty = health.empty_prompts || { count: 0, files: [] };
    const tooShort = health.too_short || { count: 0, files: [] };
    const dups = health.duplicate_annotations || { count: 0, group_count: 0, files: [] };
    const unsupported = health.unsupported_format || { count: 0, accepted: [], files: [] };
    const issues = empty.count + tooShort.count
        + dups.count + unsupported.count;

    // Three-tier status driven by the share of unique clips touched by any
    // health check. A single file showing up in multiple categories only
    // counts once.
    const affected = new Set([
        ...empty.files,
        ...tooShort.files,
        ...dups.files,
        ...unsupported.files,
    ]);
    const affectedRatio = health.total_clips > 0 ? affected.size / health.total_clips : 0;
    let status;
    if (affected.size === 0) status = 'ok';
    else if (affectedRatio > 0.5) status = 'bad';
    else status = 'warn';

    const statusColor = (
        status === 'ok' ? 'success.main'
        : status === 'warn' ? 'warm.main'
        : 'error.main'
    );
    const statusText = (
        status === 'ok'
            ? `All clean · ${health.total_clips} clip${health.total_clips === 1 ? '' : 's'} ready`
            : `${affected.size} of ${health.total_clips} clip${health.total_clips === 1 ? '' : 's'} flagged`
    );

    return (
        <Paper variant="outlined" sx={{ borderRadius: 2.5 }}>
            <Box sx={{ px: 2, py: 1.25, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box component="span" sx={appStyles.sectionCardIcon}>
                    <HealthIcon size={18} />
                </Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, flex: 1 }}>
                    Dataset health
                </Typography>
                <Box
                    sx={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        bgcolor: statusColor,
                        // Soft halo so the dot reads as a status indicator,
                        // not stray decoration.
                        boxShadow: (theme) =>
                            `0 0 0 3px ${theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)'}`,
                    }}
                />
                <Typography variant="caption" sx={{ color: statusColor }}>
                    {statusText}
                </Typography>
            </Box>

            {issues > 0 && (
                <Box
                    sx={{
                        px: 2,
                        py: 1.25,
                        borderTop: 1,
                        borderColor: 'divider',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 0.75,
                        flexWrap: 'wrap',
                    }}
                >
                    {empty.count > 0 && (
                        <Tooltip title={TIPS.dataset.selectClips}>
                            <Chip
                                size="small"
                                variant="outlined"
                                color="warning"
                                label={`${empty.count} empty annotation${empty.count === 1 ? '' : 's'}`}
                                onClick={() => onSelectFiles(empty.files)}
                            />
                        </Tooltip>
                    )}
                    {tooShort.count > 0 && (
                        <Tooltip title={TIPS.dataset.tooShort(tooShort.threshold_sec)}>
                            <Chip
                                size="small"
                                variant="outlined"
                                color="error"
                                label={`${tooShort.count} too short (< ${tooShort.threshold_sec}s)`}
                                onClick={() => onSelectFiles(tooShort.files)}
                            />
                        </Tooltip>
                    )}
                    {dups.count > 0 && (
                        <Tooltip title={TIPS.dataset.duplicates(dups.group_count)}>
                            <Chip
                                size="small"
                                variant="outlined"
                                color="warning"
                                label={`${dups.count} duplicate annotation${dups.count === 1 ? '' : 's'}`}
                                onClick={() => onSelectFiles(dups.files)}
                            />
                        </Tooltip>
                    )}
                    {unsupported.count > 0 && (
                        <Tooltip title={TIPS.dataset.unsupported(unsupported.accepted)}>
                            <Chip
                                size="small"
                                variant="outlined"
                                color="error"
                                label={`${unsupported.count} unsupported format${unsupported.count === 1 ? '' : 's'}`}
                                onClick={() => onSelectFiles(unsupported.files)}
                            />
                        </Tooltip>
                    )}
                </Box>
            )}
        </Paper>
    );
}

function Waveform({ projectName, fileName, isActive, progress }) {
    const canvasRef = useRef(null);
    const theme = useTheme();
    const [peaks, setPeaks] = useState(null);
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        let cancelled = false;
        setPeaks(null);
        setFailed(false);
        if (!projectName || !fileName) return;
        const url = `/api/projects/${encodeURIComponent(projectName)}/clip/${encodeURIComponent(fileName)}/peaks?n=80`;
        api.get(url)
            .then(({ data }) => { if (!cancelled) setPeaks(data?.peaks || []); })
            .catch(() => { if (!cancelled) setFailed(true); });
        return () => { cancelled = true; };
    }, [projectName, fileName]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (canvas.width !== w * dpr) canvas.width = w * dpr;
        if (canvas.height !== h * dpr) canvas.height = h * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        if (!peaks || !peaks.length) {
            ctx.fillStyle = 'rgba(0,0,0,0.08)';
            const midY = h / 2;
            ctx.fillRect(0, midY - 0.5, w, 1);
            return;
        }

        const barCount = peaks.length;
        const barWidth = Math.max(1, w / barCount - 1);
        const playedIdx = isActive ? Math.floor(progress * barCount) : -1;
        // Match the Generated-Fragments waveforms: teal accent for the played
        // portion, dimmed (35% alpha) for the rest.
        const playedColor = '#279FBB';
        const restColor = '#279FBB59';

        for (let i = 0; i < barCount; i++) {
            const v = peaks[i];
            const barH = Math.max(1, v * (h - 2));
            const x = i * (w / barCount);
            const y = (h - barH) / 2;
            ctx.fillStyle = i <= playedIdx ? playedColor : restColor;
            ctx.fillRect(x, y, barWidth, barH);
        }
    }, [peaks, isActive, progress, theme]);

    return (
        <Box sx={{ width: 120, height: 28, flexShrink: 0, opacity: failed ? 0.3 : 1 }}>
            <canvas
                ref={canvasRef}
                style={{ width: '100%', height: '100%', display: 'block' }}
            />
        </Box>
    );
}

function ClipTable({ projectName, clips, playingFile, playProgress, onPlayToggle, onPromptChange, onAnnotate, onDelete, onSlice, selectedFiles, onToggleSelected, onToggleSelectAll, disabled, toolbar }) {
    const totalSelected = selectedFiles ? selectedFiles.size : 0;
    const allSelected = clips && clips.length > 0 && totalSelected === clips.length;
    const partiallySelected = totalSelected > 0 && !allSelected;
    if (!clips || clips.length === 0) {
        return (
            <Paper variant="outlined" sx={{ borderRadius: 2.5, overflow: 'hidden' }}>
                {toolbar && (
                    <Box sx={{ px: 1.5, py: 1, borderBottom: 1, borderColor: 'divider' }}>
                        {toolbar}
                    </Box>
                )}
                <Box sx={{ py: 4, textAlign: 'center', color: 'text.secondary' }}>
                    <Typography variant="body2">
                        No clips yet. Use “Add audio” to bring in a folder.
                    </Typography>
                </Box>
            </Paper>
        );
    }
    return (
        <Paper variant="outlined" sx={{ borderRadius: 2.5, overflow: 'hidden' }}>
            {toolbar && (
                <Box sx={{ px: 1.5, py: 1, borderBottom: 1, borderColor: 'divider' }}>
                    {toolbar}
                </Box>
            )}
            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell padding="checkbox">
                                <Checkbox
                                    size="small"
                                    checked={allSelected}
                                    indeterminate={partiallySelected}
                                    onChange={onToggleSelectAll}
                                    disabled={disabled || clips.length === 0}
                                />
                            </TableCell>
                            <TableCell sx={{ width: '36%' }}>File</TableCell>
                            <TableCell>Annotation</TableCell>
                            <TableCell sx={{ width: 132, textAlign: 'right' }}>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {clips.map((c) => (
                            <ClipRow
                                key={c.file_name}
                                projectName={projectName}
                                clip={c}
                                isPlaying={playingFile === c.file_name}
                                playProgress={playingFile === c.file_name ? playProgress : 0}
                                onPlayToggle={onPlayToggle}
                                onPromptChange={onPromptChange}
                                onAnnotate={onAnnotate}
                                onDelete={onDelete}
                                onSlice={onSlice}
                                selected={selectedFiles ? selectedFiles.has(c.file_name) : false}
                                onToggleSelected={onToggleSelected}
                                disabled={disabled}
                            />
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Paper>
    );
}

// React.memo so the 60Hz audio-playhead ticks don't reconcile every row in
// the table. Custom comparator: skip if visual props didn't change. Callback
// identity intentionally ignored — they're stable in behavior, just inline
// arrows from the parent, and re-creating a row only to re-bind a click
// handler isn't worth the work. playProgress only matters on the active row.
const ClipRow = React.memo(function ClipRow({ projectName, clip, isPlaying, playProgress, onPlayToggle, onPromptChange, onAnnotate, onDelete, onSlice, selected, onToggleSelected, disabled }) {
    const [draft, setDraft] = useState(clip.prompt);
    useEffect(() => { setDraft(clip.prompt); }, [clip.prompt]);

    const dirty = draft !== clip.prompt;

    return (
        <TableRow hover selected={selected}>
            <TableCell padding="checkbox">
                <Checkbox
                    size="small"
                    checked={!!selected}
                    onChange={() => onToggleSelected && onToggleSelected(clip.file_name)}
                />
            </TableCell>
            <TableCell sx={{ wordBreak: 'break-all' }}>
                <Stack direction="row" alignItems="center" spacing={1}>
                    <IconButton
                        size="small"
                        onClick={() => onPlayToggle(clip.file_name)}
                        sx={{ width: 28, height: 28 }}
                    >
                        {isPlaying ? <PauseIcon size={14} /> : <PlayIcon size={14} />}
                    </IconButton>
                    <Waveform
                        projectName={projectName}
                        fileName={clip.file_name}
                        isActive={isPlaying}
                        progress={playProgress}
                    />
                    <Typography variant="body2" sx={{ flex: 1, minWidth: 0, wordBreak: 'break-all' }}>
                        {clip.file_name}
                    </Typography>
                </Stack>
            </TableCell>
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
                <Tooltip title={TIPS.dataset.autoAnnotateClip}>
                    <span>
                        <IconButton
                            size="small"
                            onClick={() => onAnnotate(clip.file_name)}
                            disabled={disabled}
                            sx={{ color: 'warm.main', '&:hover': { color: 'warm.light', bgcolor: 'action.hover' } }}
                        >
                            <WandSparkles size={16} />
                        </IconButton>
                    </span>
                </Tooltip>
                <Tooltip title={TIPS.dataset.sliceClip}>
                    <span>
                        <IconButton
                            size="small"
                            onClick={() => onSlice(clip.file_name)}
                            disabled={disabled}
                        >
                            <ScissorsIcon size={16} />
                        </IconButton>
                    </span>
                </Tooltip>
                <Tooltip title={TIPS.dataset.removeClip}>
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
}, (prev, next) => {
    if (prev.clip !== next.clip) return false;
    if (prev.disabled !== next.disabled) return false;
    if (prev.projectName !== next.projectName) return false;
    if (prev.isPlaying !== next.isPlaying) return false;
    if (prev.selected !== next.selected) return false;
    // playProgress only matters when this row is the active one — inactive
    // rows always receive playProgress=0 from the parent, so they're skipped.
    if (next.isPlaying && prev.playProgress !== next.playProgress) return false;
    return true;
});

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

function IngestDialog({ open, projectName, onClose, onIngested, isDocker = false }) {
    const [folder, setFolder] = useState('');
    const [mode, setMode] = useState('copy');
    const [busy, setBusy] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadInfo, setUploadInfo] = useState('');
    const [dialogError, setDialogError] = useState('');
    const uploadInputRef = useRef(null);

    useEffect(() => {
        if (open) { setFolder(''); setMode('copy'); setDialogError(''); setUploadInfo(''); }
    }, [open]);

    async function pick() {
        try {
            const { data } = await api.post('/api/pick-folder', {});
            if (data?.path) setFolder(data.path);
        } catch (e) {
            setDialogError(extractError(e, 'Folder picker failed'));
        }
    }

    // Docker/web mode: the backend has no display server, so a native folder
    // dialog can't work. Upload the folder through the browser instead — the
    // audio is staged server-side under uploads/ and the returned staging path
    // feeds the same ingest flow as a locally picked folder.
    async function uploadFolder(fileList) {
        // Filter to audio up front (mirrors the backend's accepted set) so
        // artwork/project clutter doesn't count against the request size and
        // multipart part limits.
        const AUDIO_EXTS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'];
        const files = Array.from(fileList || []).filter((f) =>
            AUDIO_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext)));
        if (!files.length) {
            setDialogError('No audio files found in the selected folder.');
            return;
        }
        setUploading(true);
        setDialogError('');
        setUploadInfo('');
        try {
            const form = new FormData();
            for (const f of files) {
                form.append('files', f);
                form.append('rel_paths', f.webkitRelativePath || f.name);
            }
            const { data } = await api.post('/api/upload-folder', form);
            setFolder(data.path);
            setUploadInfo(`${data.file_count} audio file${data.file_count === 1 ? '' : 's'} uploaded`);
        } catch (e) {
            setDialogError(extractError(e, 'Folder upload failed'));
        } finally {
            setUploading(false);
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
                        {isDocker ? (
                            <>
                                <input
                                    ref={uploadInputRef}
                                    type="file"
                                    webkitdirectory=""
                                    multiple
                                    style={{ display: 'none' }}
                                    onChange={(e) => { uploadFolder(e.target.files); e.target.value = ''; }}
                                />
                                <Button
                                    variant="outlined"
                                    startIcon={<FolderOpenIcon size={18} />}
                                    disabled={uploading}
                                    onClick={() => uploadInputRef.current?.click()}
                                >
                                    {uploading ? 'Uploading…' : 'Upload folder'}
                                </Button>
                                <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-all' }}>
                                    {uploadInfo || 'No folder uploaded'}
                                </Typography>
                            </>
                        ) : (
                            <>
                                <Button variant="outlined" startIcon={<FolderOpenIcon size={18} />} onClick={pick}>
                                    Pick folder
                                </Button>
                                <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-all' }}>
                                    {folder || 'No folder selected'}
                                </Typography>
                            </>
                        )}
                    </Stack>

                    {/* Web uploads are staged copies already — symlinking into the
                        staging dir would break, so the mode choice is desktop-only
                        and Docker always ingests with the default 'copy'. */}
                    {!isDocker && (
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
                    )}

                    {dialogError && <Alert severity="error">{dialogError}</Alert>}
                </Stack>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} disabled={busy || uploading}>Cancel</Button>
                <Button variant="contained" onClick={submit} disabled={busy || uploading || !folder}>
                    {busy ? 'Adding…' : 'Add'}
                </Button>
            </DialogActions>
        </Dialog>
    );
}

function SliceDialog({ open, projectName, fileName, onClose, onSliced }) {
    const [target, setTarget] = useState(30);
    const [overlap, setOverlap] = useState(0);
    const [strategy, setStrategy] = useState('hard');
    const [duration, setDuration] = useState(null);
    const [busy, setBusy] = useState(false);
    const [dialogError, setDialogError] = useState('');

    useEffect(() => {
        if (!open) return;
        setTarget(30);
        setOverlap(0);
        setStrategy('hard');
        setDialogError('');
        setDuration(null);
        if (!projectName || !fileName) return;
        // Reuse the peaks endpoint to pull duration cheaply (cached server-side).
        api.get(`/api/projects/${encodeURIComponent(projectName)}/clip/${encodeURIComponent(fileName)}/peaks?n=20`)
            .then(({ data }) => setDuration(data?.duration || null))
            .catch(() => setDuration(null));
    }, [open, projectName, fileName]);

    const stepSec = Math.max(0.5, target - overlap);
    const estChildren = duration && target > 0 ? Math.max(1, Math.ceil(duration / stepSec)) : null;
    const tooShort = duration !== null && duration <= target;

    async function submit() {
        setBusy(true);
        setDialogError('');
        try {
            await api.post(
                `/api/projects/${encodeURIComponent(projectName)}/clip/${encodeURIComponent(fileName)}/slice`,
                { target_duration: target, overlap_sec: overlap, strategy },
            );
            await onSliced();
            onClose();
        } catch (e) {
            setDialogError(extractError(e, 'Slice failed'));
        } finally {
            setBusy(false);
        }
    }

    return (
        <Dialog open={open} onClose={busy ? undefined : onClose} maxWidth="sm" fullWidth>
            <DialogTitle>Slice {fileName || ''}</DialogTitle>
            <DialogContent>
                <Stack spacing={2.5} sx={{ pt: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                        The original file will be replaced by the children on disk. Children inherit this clip's annotation. They stay in the project until you Create Dataset (Delete reverts them).
                    </Typography>

                    <Stack direction="row" spacing={2}>
                        <TextField
                            label="Target duration (sec)"
                            type="number"
                            size="small"
                            value={target}
                            onChange={(e) => setTarget(Math.max(0.5, parseFloat(e.target.value) || 0))}
                            inputProps={{ step: 0.5, min: 0.5, max: 60 }}
                            fullWidth
                        />
                        <TextField
                            label="Overlap (sec)"
                            type="number"
                            size="small"
                            value={overlap}
                            onChange={(e) => setOverlap(Math.max(0, parseFloat(e.target.value) || 0))}
                            inputProps={{ step: 0.1, min: 0, max: Math.max(0, target - 0.5) }}
                            fullWidth
                            helperText="Head-overlap on every child after the first"
                        />
                    </Stack>

                    <FormControl>
                        <Typography variant="body2" gutterBottom>Where each cut should land:</Typography>
                        <RadioGroup value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                            <FormControlLabel
                                value="hard"
                                control={<Radio size="small" />}
                                label={<Typography variant="body2">Hard cut — exact intervals; fastest, can split mid-note</Typography>}
                            />
                            <FormControlLabel
                                value="transient"
                                control={<Radio size="small" />}
                                label={<Typography variant="body2">Transient-aware — snaps each cut to the nearest onset (good for drums / rhythmic)</Typography>}
                            />
                            <FormControlLabel
                                value="silence"
                                control={<Radio size="small" />}
                                label={<Typography variant="body2">Silence-aware — snaps to the quietest moment in each window (good for melodic / phrased)</Typography>}
                            />
                        </RadioGroup>
                    </FormControl>

                    {duration !== null && (
                        <Typography variant="caption" color="text.secondary">
                            Source: {duration.toFixed(1)}s
                            {estChildren !== null && !tooShort && ` · ~${estChildren} children at this setting`}
                            {tooShort && ' · already shorter than the target — nothing to slice'}
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
                    disabled={busy || tooShort || target <= 0 || overlap >= target}
                >
                    {busy ? 'Slicing…' : 'Slice'}
                </Button>
            </DialogActions>
        </Dialog>
    );
}

// ---------- utils ----------------------------------------------------------

