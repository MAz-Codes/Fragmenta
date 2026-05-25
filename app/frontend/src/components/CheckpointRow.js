import React, { useEffect, useRef, useState } from 'react';
import {
    Box,
    Typography,
    Button,
    Chip,
    LinearProgress,
    Stack,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    CloudDownload as DownloadIcon,
    Trash2 as DeleteIcon,
    X as CancelIcon,
} from 'lucide-react';
import api from '../api';

const fmtBytes = (n) => {
    if (!n && n !== 0) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let v = n;
    let u = 0;
    while (v >= 1000 && u < units.length - 1) { v /= 1000; u += 1; }
    return `${v.toFixed(v < 10 ? 2 : 1)} ${units[u]}`;
};

const hardwareLabel = (hw) => ({
    'cpu': 'CPU / GPU',
    'cuda': 'CUDA',
    'cuda+flash-attn': 'CUDA + Flash-Attn',
}[hw] || hw);

export default function CheckpointRow({ checkpoint, onAuthRequired, onChanged }) {
    const [jobId, setJobId] = useState(checkpoint.active_job?.job_id || null);
    const [job, setJob] = useState(checkpoint.active_job || null);
    const [error, setError] = useState(null);
    const [busy, setBusy] = useState(false);
    const pollTimer = useRef(null);

    // If the parent's refresh tells us about an in-flight job and we don't
    // already have one locally (typical case: dialog was closed mid-download
    // and just got reopened), adopt it. Don't stomp a freshly-started local
    // job_id with stale catalog data — only sync when the local state is empty
    // or a *different* job is now active for this checkpoint.
    useEffect(() => {
        const incoming = checkpoint.active_job?.job_id || null;
        if (incoming && incoming !== jobId) {
            setJobId(incoming);
            setJob(checkpoint.active_job);
        }
    }, [checkpoint.active_job, jobId]);

    useEffect(() => {
        if (!jobId) return undefined;
        const tick = async () => {
            try {
                const r = await api.get(`/api/checkpoints/jobs/${jobId}`);
                setJob(r.data);
                if (['complete', 'failed', 'cancelled'].includes(r.data.status)) {
                    if (r.data.status === 'failed' && (r.data.error || '').startsWith('hf_auth_required')) {
                        onAuthRequired?.();
                    } else if (r.data.status === 'failed') {
                        setError(r.data.error);
                    }
                    setJobId(null);
                    onChanged?.();
                }
            } catch (e) {
                setError(e.response?.data?.error || e.message);
                setJobId(null);
            }
        };
        tick();
        pollTimer.current = setInterval(tick, 1500);
        return () => clearInterval(pollTimer.current);
    }, [jobId, onAuthRequired, onChanged]);

    const startDownload = async () => {
        setBusy(true);
        setError(null);
        try {
            const r = await api.post(`/api/checkpoints/${checkpoint.id}/download`);
            setJobId(r.data.job_id);
        } catch (e) {
            setError(e.response?.data?.error || e.message);
        } finally {
            setBusy(false);
        }
    };

    const cancelDownload = async () => {
        try {
            await api.post(`/api/checkpoints/${checkpoint.id}/cancel-download`);
        } catch (e) {
            setError(e.response?.data?.error || e.message);
        }
    };

    const deleteCheckpoint = async () => {
        if (!window.confirm(`Delete ${checkpoint.name} (${fmtBytes(checkpoint.downloaded_bytes)})?`)) return;
        setBusy(true);
        try {
            await api.delete(`/api/checkpoints/${checkpoint.id}`);
            onChanged?.();
        } catch (e) {
            setError(e.response?.data?.error || e.message);
        } finally {
            setBusy(false);
        }
    };

    const downloading = !!jobId && job?.status === 'running';
    const queued = !!jobId && job?.status === 'queued';
    const pct = job?.total_bytes ? (job.downloaded_bytes / job.total_bytes) * 100 : 0;

    const renderAction = () => {
        if (downloading || queued) {
            return (
                <Tooltip title="Cancel download">
                    <IconButton size="small" onClick={cancelDownload}><CancelIcon size={16} /></IconButton>
                </Tooltip>
            );
        }
        if (checkpoint.downloaded) {
            return (
                <Tooltip title="Delete from disk">
                    <IconButton size="small" onClick={deleteCheckpoint} disabled={busy}>
                        <DeleteIcon size={16} />
                    </IconButton>
                </Tooltip>
            );
        }
        return (
            <Button
                size="small"
                variant="contained"
                startIcon={<DownloadIcon size={14} />}
                onClick={startDownload}
                disabled={busy}
            >
                Get
            </Button>
        );
    };

    return (
        <Box
            sx={{
                py: 1.25,
                px: 1.5,
                borderBottom: '1px solid',
                borderColor: 'divider',
                '&:last-child': { borderBottom: 'none' },
            }}
        >
            <Stack direction="row" alignItems="center" spacing={2}>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Stack direction="row" alignItems="center" spacing={1}>
                        <Tooltip title="Open on HuggingFace to accept the model's gated-access terms">
                            <Typography
                                component="a"
                                href={`https://huggingface.co/${checkpoint.repo}`}
                                target="_blank"
                                rel="noreferrer"
                                variant="body2"
                                sx={{
                                    fontWeight: 500,
                                    color: 'inherit',
                                    textDecoration: 'none',
                                    borderBottom: '1px dashed',
                                    borderColor: 'text.disabled',
                                    '&:hover': { color: 'primary.main', borderColor: 'primary.main' },
                                }}
                            >
                                {checkpoint.name}
                            </Typography>
                        </Tooltip>
                        <Chip
                            size="small"
                            label={hardwareLabel(checkpoint.hardware)}
                            variant="outlined"
                            sx={{ height: 18, fontSize: 10 }}
                        />
                        {checkpoint.downloaded && (
                            <Chip
                                size="small"
                                label="installed"
                                sx={{
                                    height: 18,
                                    fontSize: 10,
                                    fontWeight: 600,
                                    bgcolor: 'success.main',
                                    color: 'common.white',
                                }}
                            />
                        )}
                    </Stack>
                    <Typography variant="caption" color="text.secondary">
                        {fmtBytes(checkpoint.size_bytes)}
                        {checkpoint.max_duration_sec && ` · up to ${checkpoint.max_duration_sec}s`}
                    </Typography>
                </Box>
                <Box>{renderAction()}</Box>
            </Stack>

            {(downloading || queued) && (
                <Box sx={{ mt: 1 }}>
                    <LinearProgress
                        variant={queued ? 'indeterminate' : 'determinate'}
                        value={Math.min(100, pct)}
                        sx={{ height: 4, borderRadius: 2 }}
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                        {queued ? 'Queued…' : `${fmtBytes(job?.downloaded_bytes)} / ${fmtBytes(job?.total_bytes)}`}
                    </Typography>
                </Box>
            )}

            {error && (
                <Typography variant="caption" color="error" sx={{ mt: 0.5, display: 'block' }}>
                    {error}
                </Typography>
            )}
        </Box>
    );
}
