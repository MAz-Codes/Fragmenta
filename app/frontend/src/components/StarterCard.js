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
    FormControlLabel,
    IconButton,
    LinearProgress,
    Stack,
    Tab,
    Tabs,
    Typography,
} from '@mui/material';
import { X as CloseIcon, ExternalLink as ExternalLinkIcon } from 'lucide-react';
import api from '../api';

const fmtGb = (bytes) => `${((bytes || 0) / 1e9).toFixed(1)}`;

const fmtLeft = (seconds) => {
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    if (seconds < 90) return 'under 2 min left';
    if (seconds < 3600) return `~${Math.round(seconds / 60)} min left`;
    return `~${(seconds / 3600).toFixed(1)} h left`;
};

const errorText = (e) => e.response?.data?.error || e.message;

// For pages opened from the card, so they don't just appear.
export const PAGE_FADE = {
    transitionDuration: { enter: 700, exit: 200 },
    TransitionProps: { easing: { enter: 'ease-in-out' } },
};

const TextButton = ({ onClick, children, sx }) => (
    <Typography
        component="button"
        type="button"
        variant="inherit"
        onClick={onClick}
        sx={{
            p: 0,
            border: 0,
            background: 'none',
            color: 'inherit',
            fontWeight: 600,
            textDecoration: 'underline',
            cursor: 'pointer',
            ...sx,
        }}
    >
        {children}
    </Typography>
);

/**
 * State of the first-run starter model (GET /api/starter). Polls while a
 * download runs, resumes one the app quit in the middle of, and calls onReady
 * once when the model lands.
 */
export function useStarterModel({ onReady }) {
    const [status, setStatus] = useState(null);
    const [resuming, setResuming] = useState(false);
    const wasDownloading = useRef(false);
    const resumedAfterQuit = useRef(false);
    const onReadyRef = useRef(onReady);
    onReadyRef.current = onReady;

    const refresh = useCallback(async () => {
        try {
            const r = await api.get('/api/starter');
            setStatus(r.data);
            return r.data;
        } catch (e) {
            return null;
        }
    }, []);

    useEffect(() => { refresh(); }, [refresh]);

    const phase = status?.phase;
    useEffect(() => {
        if (phase !== 'downloading') return undefined;
        const timer = setInterval(refresh, 1500);
        return () => clearInterval(timer);
    }, [phase, refresh]);

    useEffect(() => {
        if (!status) return;
        if (status.phase === 'downloading') {
            wasDownloading.current = true;
        } else if (wasDownloading.current) {
            wasDownloading.current = false;
            setResuming(false);
            if (status.installed) onReadyRef.current?.(status.model_id);
        }
    }, [status]);

    // Quit in the middle of a download: pick it up again on the next launch.
    useEffect(() => {
        if (phase !== 'interrupted' || resumedAfterQuit.current) return;
        resumedAfterQuit.current = true;
        setResuming(true);
        api.post('/api/starter/download', {}).catch(() => setResuming(false)).finally(refresh);
    }, [phase, refresh]);

    return { status, refresh, resuming, setResuming };
}

export function LicenseViewer({ open, onClose, licenses, fade }) {
    const [current, setCurrent] = useState(0);
    const [texts, setTexts] = useState({});
    const [error, setError] = useState(null);
    const license = licenses[current];

    useEffect(() => {
        if (!open || !license || texts[license.id]) return;
        setError(null);
        api.get(`/api/starter/licenses/${license.id}`)
            .then((r) => setTexts((prev) => ({ ...prev, [license.id]: r.data.text })))
            .catch((e) => setError(errorText(e)));
    }, [open, license, texts]);

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth {...fade}>
            <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ flex: 1 }}>Licenses</Box>
                <IconButton size="small" onClick={onClose} aria-label="Close licenses"><CloseIcon size={18} /></IconButton>
            </DialogTitle>
            <DialogContent>
                <Tabs value={current} onChange={(_e, v) => setCurrent(v)} variant="fullWidth" sx={{ mb: 1.5 }}>
                    {licenses.map((l) => (
                        <Tab key={l.id} label={l.tab} sx={{ textTransform: 'none', minWidth: 0, px: 1, fontSize: '0.85rem' }} />
                    ))}
                </Tabs>
                {license && (
                    <Typography variant="caption" color="text.secondary" component="div" sx={{ mb: 1 }}>
                        {license.title} ·{' '}
                        <a href={license.url} target="_blank" rel="noopener noreferrer" style={{ color: 'inherit' }}>
                            Official page <ExternalLinkIcon size={11} style={{ verticalAlign: '-1px' }} />
                        </a>
                    </Typography>
                )}
                {error && <Alert severity="error">{error}</Alert>}
                <Box sx={(t) => ({
                    height: '52vh',
                    overflow: 'auto',
                    p: 2,
                    borderRadius: '24px',
                    backgroundColor: t.palette.fragmenta.section,
                })}>
                    <Typography variant="body2" component="div" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                        {license ? texts[license.id] : ''}
                    </Typography>
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} variant="contained">Close</Button>
            </DialogActions>
        </Dialog>
    );
}

function AgreementTick({ checked, onChange, onRead }) {
    return (
        <FormControlLabel
            sx={{ m: 0 }}
            control={(
                <Checkbox
                    checked={checked}
                    onChange={(e) => onChange(e.target.checked)}
                    size="small"
                    sx={{ p: 0, pr: 1 }}
                />
            )}
            label={(
                <Typography variant="body2">
                    I have read and agree to the{' '}
                    <TextButton
                        onClick={(e) => {
                            // Inside the label: opening them mustn't also tick the box.
                            e.preventDefault();
                            onRead();
                        }}
                    >
                        licenses
                    </TextButton>
                </Typography>
            )}
        />
    );
}

/**
 * Asked once, before the first no-account model comes from the Checkpoint
 * Manager, unless the starter card already asked.
 *
 * Props:
 *   checkpoint:  the model about to download, or null when closed
 *   terms:       GET /api/starter (licenses, terms_version)
 *   onAgree:     (termsVersion) => void
 *   onClose:     () => void
 */
export function LicenseAgreementDialog({ checkpoint, terms, onAgree, onClose }) {
    const [agreed, setAgreed] = useState(false);
    const [licensesOpen, setLicensesOpen] = useState(false);

    useEffect(() => {
        if (!checkpoint) setAgreed(false);
    }, [checkpoint]);

    return (
        <>
            <Dialog open={Boolean(checkpoint)} onClose={onClose} maxWidth="sm" fullWidth>
                <DialogTitle>Agree to the licenses</DialogTitle>
                <DialogContent>
                    <Stack spacing={2}>
                        <Typography variant="body2" color="text.secondary">
                            This will download <strong>{checkpoint?.name}</strong>.
                        </Typography>
                        <AgreementTick checked={agreed} onChange={setAgreed} onRead={() => setLicensesOpen(true)} />
                    </Stack>
                </DialogContent>
                <DialogActions>
                    <Button onClick={onClose}>Cancel</Button>
                    <Button onClick={() => onAgree(terms.terms_version)} disabled={!agreed} variant="contained">
                        Download ({fmtGb(checkpoint?.size_bytes)} GB)
                    </Button>
                </DialogActions>
            </Dialog>
            <LicenseViewer open={licensesOpen} onClose={() => setLicensesOpen(false)} licenses={terms?.licenses || []} />
        </>
    );
}

/**
 * "Get your first model". Closing it stops nothing and decides nothing: a
 * running download carries on, and App opens the card again at the next launch
 * for as long as no model is on disk or downloading.
 *
 * Props:
 *   open, onClose:   the dialog
 *   starter:         what useStarterModel returns
 *   onOpenManager:   () => void, opens the Checkpoint Manager
 *   onImport:        () => void, opens Settings → Import
 */
export default function StarterCard({ open, onClose, starter, onOpenManager, onImport }) {
    const { status, refresh, resuming, setResuming } = starter;
    const [agreed, setAgreed] = useState(false);
    const [pausing, setPausing] = useState(false);
    const [requestError, setRequestError] = useState(null);
    const [licensesOpen, setLicensesOpen] = useState(false);
    const speed = useRef(null);

    const phase = status?.phase;
    const downloaded = status?.downloaded_bytes || 0;
    const size = status?.size_bytes || 1;

    useEffect(() => {
        if (phase !== 'downloading') {
            speed.current = null;
            setPausing(false);
        } else if (!speed.current) {
            speed.current = { at: Date.now(), bytes: downloaded };
        }
    }, [phase, downloaded]);

    if (!status) return null;

    const download = async () => {
        setRequestError(null);
        setResuming(phase === 'paused');
        try {
            await api.post('/api/starter/download',
                status.terms_accepted ? {} : { accept_terms_version: status.terms_version });
        } catch (e) {
            setRequestError(errorText(e));
            setResuming(false);
        }
        refresh();
    };

    const pause = async () => {
        setPausing(true);
        await api.post('/api/starter/pause').catch((e) => setRequestError(errorText(e)));
        refresh();
    };

    const cancel = async () => {
        await api.post('/api/starter/cancel').catch((e) => setRequestError(errorText(e)));
        setResuming(false);
        refresh();
    };

    const noSpace = status.free_bytes < status.needed_bytes;
    const canDownload = (status.terms_accepted || agreed) && !noSpace;
    const pct = Math.min(100, Math.floor((downloaded / size) * 100));
    let left = null;
    if (speed.current && phase === 'downloading') {
        const seconds = (Date.now() - speed.current.at) / 1000;
        const rate = (downloaded - speed.current.bytes) / seconds;
        if (seconds > 5 && rate > 0) left = fmtLeft((size - downloaded) / rate);
    }
    const started = phase === 'downloading' || phase === 'paused' || phase === 'interrupted';

    return (
        <>
            <Dialog
                open={open}
                onClose={onClose}
                aria-labelledby="starter-title"
                transitionDuration={{ enter: 900, exit: 200 }}
                PaperProps={{ sx: { position: 'relative', width: 440, maxWidth: 'calc(100% - 32px)', p: 4 } }}
            >
                <IconButton
                    size="small"
                    onClick={onClose}
                    aria-label="Close"
                    sx={{ position: 'absolute', top: 16, right: 16 }}
                >
                    <CloseIcon size={18} />
                </IconButton>

                <Stack spacing={2.75}>
                    <Box>
                        <Typography id="starter-title" variant="h5" component="h1" sx={{ pr: 3, mb: 1.75 }}>
                            {status.returning ? 'No models installed' : 'Get your first model'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            {status.model_name} · {fmtGb(size)} GB
                        </Typography>
                    </Box>

                    {!started && !status.terms_accepted && (
                        <AgreementTick checked={agreed} onChange={setAgreed} onRead={() => setLicensesOpen(true)} />
                    )}

                    {started && (
                        <Box>
                            <LinearProgress
                                variant={phase === 'interrupted' ? 'indeterminate' : 'determinate'}
                                value={pct}
                                sx={{ height: 8, borderRadius: 999 }}
                            />
                            <Stack direction="row" justifyContent="space-between" sx={{ mt: 1 }}>
                                <Typography variant="caption" color="text.secondary">
                                    {pct}% · {fmtGb(downloaded)} / {fmtGb(size)} GB
                                </Typography>
                                <Typography variant="caption" color="text.secondary">{left}</Typography>
                            </Stack>
                            {(phase === 'paused' || resuming) && (
                                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                                    {phase === 'paused'
                                        ? 'Paused. The current file restarts when you resume.'
                                        : 'Resuming. The current file restarts.'}
                                </Typography>
                            )}
                        </Box>
                    )}

                    {phase === 'failed' && status.error && <Alert severity="error">{status.error}</Alert>}
                    {requestError && <Alert severity="error">{requestError}</Alert>}
                    {!started && noSpace && (
                        <Alert severity="warning">
                            Not enough space (need {fmtGb(status.needed_bytes)} GB, have {fmtGb(status.free_bytes)} GB).
                        </Alert>
                    )}

                    {(phase === 'idle' || phase === 'failed') && (
                        <Button onClick={download} disabled={!canDownload} variant="contained" fullWidth sx={{ minHeight: 46 }}>
                            {phase === 'failed' ? 'Retry' : 'Download'}
                        </Button>
                    )}
                    {(phase === 'downloading' || phase === 'paused') && (
                        <Stack direction="row" spacing={1} justifyContent="center">
                            {phase === 'downloading' ? (
                                <Button onClick={pause} disabled={pausing} variant="contained">
                                    {pausing ? 'Pausing…' : 'Pause'}
                                </Button>
                            ) : (
                                <Button onClick={download} disabled={noSpace} variant="contained">Resume</Button>
                            )}
                            <Button onClick={cancel}>Cancel</Button>
                        </Stack>
                    )}

                    {!started && (
                        <Stack direction="row" justifyContent="center" spacing={2.25}>
                            <Typography variant="caption" color="text.secondary">
                                <TextButton onClick={onImport} sx={{ fontWeight: 400 }}>Import from an older Fragmenta</TextButton>
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                <TextButton onClick={onOpenManager} sx={{ fontWeight: 400 }}>More models</TextButton>
                            </Typography>
                        </Stack>
                    )}
                </Stack>
            </Dialog>

            <LicenseViewer
                open={licensesOpen}
                onClose={() => setLicensesOpen(false)}
                licenses={status.licenses}
                fade={PAGE_FADE}
            />
        </>
    );
}
