import React, { useEffect, useRef, useState } from 'react';
import {
    Box,
    Typography,
    Button,
    LinearProgress,
    Stack,
    IconButton,
} from '@mui/material';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import { Trash2 as DeleteIcon } from 'lucide-react';
import api from '../api';
import { LicenseAgreementDialog } from './StarterCard';

export const fmtBytes = (n) => {
    if (!n && n !== 0) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let v = n;
    let u = 0;
    while (v >= 1000 && u < units.length - 1) { v /= 1000; u += 1; }
    return `${v.toFixed(u > 0 && v < 100 ? 1 : 0)} ${units[u]}`;
};

// The look of a model card, shared with the Manager's Storage card.
export const MODEL_CARD_SX = (t) => ({
    backgroundColor: t.palette.fragmenta.surface,
    borderRadius: '20px',
    boxShadow: '0 2px 10px rgba(53, 57, 47, 0.07)',
    p: '13px 14px 13px 18px',
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px 10px',
    minWidth: 0,
});
export const MODEL_NAME_SX = { display: 'block', fontSize: 15.5, lineHeight: 1.3 };
export const MODEL_FACTS_SX = { fontSize: 12.5, color: 'text.secondary' };

// Size, then what the model is for or needs. MLX models share files, so one
// whose siblings are already here only costs what's left.
const modelFacts = (c) => {
    const left = c.size_bytes - (c.on_disk_bytes || 0);
    return [
        !c.downloaded && c.on_disk_bytes ? `${fmtBytes(Math.max(0, left))} more` : fmtBytes(c.size_bytes),
        c.max_duration_sec && `up to ${Math.floor(c.max_duration_sec / 60)} min`,
        c.kind === 'tagger' && 'tags your clips',
        c.hardware === 'cuda+flash-attn' && 'NVIDIA',
    ].filter(Boolean).join(' · ');
};

// Why this host can't run a given model, or null if it can. Mirrors the gate
// in audio_generator._ensure_model. `env` comes from GET /api/environment.
const hostIncompatReason = (hw, env) => {
    if (!env) return null;  // capabilities unknown — don't block
    if (hw === 'apple-silicon' && !env.mlx_available) {
        return 'Needs MLX, which isn\'t installed. Run python install.py again.';
    }
    if (hw === 'cuda+flash-attn') {
        if (!env.cuda_available) {
            return env.mlx_available
                ? 'Requires an NVIDIA CUDA GPU. On this Mac, use the Medium under Apple Silicon.'
                : 'Requires an NVIDIA CUDA GPU. Use a Small model — those run on CPU, Apple Silicon, or any GPU.';
        }
        // Gate on the real capability, not the platform: Windows works once a
        // matching flash-attn wheel is installed (Blackwell/Ampere + cu12x).
        // No wheel → guide the user to install one (or use Docker on WSL2).
        if (!env.flash_attn_available) {
            return env.platform === 'Windows'
                ? 'Requires Flash Attention 2 (flash-attn). No official Windows wheel — install a matching prebuilt/built wheel for your torch+CUDA, or run via Docker on WSL2.'
                : 'Requires Flash Attention 2 (flash-attn) — not installed. Install it, or use a Small model.';
        }
    }
    if (hw === 'cuda' && !env.cuda_available) {
        return 'Recommended on an NVIDIA CUDA GPU; this host has none.';
    }
    return null;
};

export default function CheckpointRow({ checkpoint, env, terms, onAuthRequired, onChanged }) {
    const [jobId, setJobId] = useState(checkpoint.active_job?.job_id || null);
    const [job, setJob] = useState(checkpoint.active_job || null);
    const [error, setError] = useState(null);
    const [busy, setBusy] = useState(false);
    const [asking, setAsking] = useState(false);
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
                if (['complete', 'failed', 'cancelled', 'paused'].includes(r.data.status)) {
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

    const startDownload = async (acceptTermsVersion) => {
        setAsking(false);
        setBusy(true);
        setError(null);
        try {
            const r = await api.post(`/api/checkpoints/${checkpoint.id}/download`,
                acceptTermsVersion ? { accept_terms_version: acceptTermsVersion } : undefined);
            setJobId(r.data.job_id);
            if (acceptTermsVersion) onChanged?.();
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
    const incompatReason = hostIncompatReason(checkpoint.hardware, env);

    const renderAction = () => {
        if (downloading || queued) return null;
        if (checkpoint.downloaded) {
            return (
                <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flex: 'none' }}>
                    <IconButton
                        className="delete-model"
                        size="small"
                        onClick={deleteCheckpoint}
                        disabled={busy}
                        aria-label={`Delete ${checkpoint.name} from disk`}
                    >
                        <DeleteIcon size={15} />
                    </IconButton>
                    <Box
                        component="span"
                        sx={{
                            height: 30,
                            px: 1.5,
                            display: 'inline-flex',
                            alignItems: 'center',
                            borderRadius: '999px',
                            backgroundColor: '#D5E6DB',
                            fontSize: 12,
                            fontWeight: 600,
                        }}
                    >
                        Installed
                    </Box>
                </Stack>
            );
        }
        if (incompatReason) {
            return (
                <Tooltip title={incompatReason}>
                    {/* span wrapper so the tooltip works on a disabled button */}
                    <Box component="span" sx={{ flex: 'none' }}>
                        <Button size="small" variant="contained" disabled>Get</Button>
                    </Box>
                </Tooltip>
            );
        }
        return (
            <Button
                size="small"
                variant="contained"
                onClick={() => {
                    // Gated models get their terms accepted on Hugging Face.
                    if (checkpoint.token_free && terms && !terms.terms_accepted) setAsking(true);
                    else startDownload();
                }}
                disabled={busy}
                sx={{ flex: 'none' }}
            >
                Get
            </Button>
        );
    };

    return (
        <Box
            sx={(t) => ({
                ...MODEL_CARD_SX(t),
                '& .delete-model': { opacity: 0 },
                '&:hover .delete-model, & .delete-model:focus-visible': { opacity: 1 },
            })}
        >
            <Box sx={{ flex: 1, minWidth: 0, opacity: (incompatReason && !checkpoint.downloaded) ? 0.55 : 1 }}>
                <Tooltip title={checkpoint.token_free ? TIPS.checkpoints.openOnHf : TIPS.checkpoints.gatedAccess}>
                    <Typography
                        component="a"
                        href={`https://huggingface.co/${checkpoint.repo}`}
                        target="_blank"
                        rel="noreferrer"
                        variant="subtitle1"
                        sx={{
                            ...MODEL_NAME_SX,
                            color: 'inherit',
                            textDecoration: 'none',
                            '&:hover': { textDecoration: 'underline' },
                        }}
                    >
                        {checkpoint.name}
                    </Typography>
                </Tooltip>
                <Typography sx={MODEL_FACTS_SX}>{modelFacts(checkpoint)}</Typography>
            </Box>
            {renderAction()}

            {(downloading || queued) && (
                <Box sx={{ flexBasis: '100%' }}>
                    <LinearProgress
                        variant={queued ? 'indeterminate' : 'determinate'}
                        value={Math.min(100, pct)}
                        sx={(t) => ({
                            height: 6,
                            borderRadius: '999px',
                            backgroundColor: t.palette.fragmenta.control,
                            '& .MuiLinearProgress-bar': {
                                backgroundColor: t.palette.fragmenta.accents.warm,
                                backgroundImage: 'none',
                            },
                        })}
                    />
                    <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mt: 0.5 }}>
                        <Typography sx={MODEL_FACTS_SX}>
                            {queued ? 'Queued…' : `${Math.floor(pct)}% · ${fmtBytes(job?.downloaded_bytes)} / ${fmtBytes(job?.total_bytes)}`}
                        </Typography>
                        <Button
                            size="small"
                            variant="text"
                            onClick={cancelDownload}
                            sx={{ minWidth: 0, minHeight: 0, p: 0, fontSize: 12, fontWeight: 600, color: 'text.primary' }}
                        >
                            Cancel
                        </Button>
                    </Stack>
                </Box>
            )}

            {error && (
                <Typography variant="caption" color="error" sx={{ flexBasis: '100%' }}>
                    {error}
                </Typography>
            )}

            <LicenseAgreementDialog
                checkpoint={asking ? checkpoint : null}
                terms={terms}
                onAgree={startDownload}
                onClose={() => setAsking(false)}
            />
        </Box>
    );
}
