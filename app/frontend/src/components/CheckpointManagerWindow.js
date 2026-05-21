import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Box,
    Typography,
    Button,
    IconButton,
    Stack,
    Divider,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Chip,
    Alert,
    TextField,
    LinearProgress,
} from '@mui/material';
import {
    X as CloseIcon,
    ChevronDown as ExpandIcon,
    HardDrive as StorageIcon,
    LogIn as LoginIcon,
    LogOut as LogoutIcon,
} from 'lucide-react';
import api from '../api';
import CheckpointRow from './CheckpointRow';
import LicenseAcceptModal from './LicenseAcceptModal';
import StorageDrilldown from './StorageDrilldown';

const GROUP_LABELS = {
    'post-trained': 'Generation models',
    'base-for-lora': 'Base models (for LoRA training)',
    'autoencoder': 'Autoencoders',
};
const GROUP_DESCRIPTIONS = {
    'post-trained': 'Use these to generate audio. Smaller models run on CPU.',
    'base-for-lora': 'Pick one of these as the base when training a new LoRA.',
    'autoencoder': 'Required by the matching DiT. Pairing is automatic when downloading.',
};
const GROUP_ORDER = ['post-trained', 'base-for-lora', 'autoencoder'];

const fmtBytes = (n) => {
    if (!n && n !== 0) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let v = n;
    let u = 0;
    while (v >= 1024 && u < units.length - 1) { v /= 1024; u += 1; }
    return `${v.toFixed(v < 10 ? 2 : 1)} ${units[u]}`;
};

export default function CheckpointManagerWindow({ open, onClose }) {
    const [catalog, setCatalog] = useState([]);
    const [storage, setStorage] = useState(null);
    const [hfAuth, setHfAuth] = useState({ signed_in: false, username: null });
    const [tokenDraft, setTokenDraft] = useState('');
    const [showTokenInput, setShowTokenInput] = useState(false);
    const [authError, setAuthError] = useState(null);
    const [licenseModal, setLicenseModal] = useState({ open: false, checkpoint: null });
    const [showStorage, setShowStorage] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // --- data ---------------------------------------------------------------
    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [cat, store, auth] = await Promise.all([
                api.get('/api/checkpoints'),
                api.get('/api/checkpoints/storage'),
                api.get('/api/hf-auth/status'),
            ]);
            // Attach required_companions per row (the list endpoint doesn't
            // include it; the per-id endpoint does. Compute it client-side
            // from requires_ae + downloaded state.)
            const items = cat.data.checkpoints.map(c => ({
                ...c,
                required_companions:
                    c.requires_ae &&
                    !(cat.data.checkpoints.find(x => x.id === c.requires_ae)?.downloaded)
                        ? [c.requires_ae]
                        : [],
            }));
            setCatalog(items);
            setStorage(store.data);
            setHfAuth(auth.data);
        } catch (e) {
            setError(e.response?.data?.error || e.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (open) refresh();
    }, [open, refresh]);

    // --- auth ---------------------------------------------------------------
    const submitToken = async () => {
        setAuthError(null);
        try {
            await api.post('/api/hf-auth', { token: tokenDraft.trim() });
            setTokenDraft('');
            setShowTokenInput(false);
            refresh();
        } catch (e) {
            setAuthError(e.response?.data?.error || e.message);
        }
    };

    const logout = async () => {
        try {
            await api.delete('/api/hf-auth');
            refresh();
        } catch (e) {
            setAuthError(e.response?.data?.error || e.message);
        }
    };

    // --- license modal hook -------------------------------------------------
    const openLicense = (checkpoint) => setLicenseModal({ open: true, checkpoint });
    const closeLicense = () => setLicenseModal({ open: false, checkpoint: null });
    const onLicenseAccepted = () => refresh();

    // --- quick-start --------------------------------------------------------
    const quickStart = async () => {
        const sm = catalog.find(c => c.id === 'sa3-small-music');
        if (!sm) return;
        if (!sm.terms_accepted) {
            openLicense(sm);
            return;
        }
        try {
            await api.post('/api/checkpoints/sa3-small-music/download');
            refresh();
        } catch (e) {
            setError(e.response?.data?.error || e.message);
        }
    };

    // --- group view ---------------------------------------------------------
    const groups = useMemo(() => {
        const map = {};
        for (const c of catalog) {
            (map[c.group] = map[c.group] || []).push(c);
        }
        return GROUP_ORDER.map(g => ({ id: g, items: map[g] || [] })).filter(g => g.items.length);
    }, [catalog]);

    const anyInstalled = catalog.some(c => c.downloaded);

    return (
        <>
            <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth scroll="paper">
                <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>Checkpoint Manager</Box>
                    <IconButton size="small" onClick={onClose}><CloseIcon size={18} /></IconButton>
                </DialogTitle>

                <DialogContent dividers>
                    {/* Header strip: storage + HF auth */}
                    <Box sx={{ mb: 2 }}>
                        <Stack direction="row" alignItems="center" spacing={2} flexWrap="wrap">
                            <Button
                                size="small"
                                variant="text"
                                startIcon={<StorageIcon size={14} />}
                                onClick={() => setShowStorage(true)}
                                disabled={!storage}
                            >
                                {storage
                                    ? `${fmtBytes(storage.total_used_bytes)} used · ${fmtBytes(storage.total_free_bytes)} free`
                                    : '—'}
                            </Button>

                            <Box sx={{ flex: 1 }} />

                            {hfAuth.signed_in ? (
                                <Stack direction="row" alignItems="center" spacing={1}>
                                    <Typography variant="caption" color="text.secondary">
                                        HuggingFace: signed in as <strong>{hfAuth.username}</strong>
                                    </Typography>
                                    <Button
                                        size="small"
                                        variant="text"
                                        startIcon={<LogoutIcon size={14} />}
                                        onClick={logout}
                                    >
                                        Sign out
                                    </Button>
                                </Stack>
                            ) : showTokenInput ? (
                                <Stack direction="row" alignItems="center" spacing={1}>
                                    <TextField
                                        size="small"
                                        placeholder="hf_..."
                                        value={tokenDraft}
                                        onChange={(e) => setTokenDraft(e.target.value)}
                                        type="password"
                                        sx={{ width: 240 }}
                                    />
                                    <Button size="small" variant="contained" onClick={submitToken}>
                                        Sign in
                                    </Button>
                                    <Button size="small" onClick={() => { setShowTokenInput(false); setTokenDraft(''); }}>
                                        Cancel
                                    </Button>
                                </Stack>
                            ) : (
                                <Button
                                    size="small"
                                    variant="outlined"
                                    startIcon={<LoginIcon size={14} />}
                                    onClick={() => setShowTokenInput(true)}
                                >
                                    Sign in to HuggingFace
                                </Button>
                            )}
                        </Stack>
                        {authError && <Alert severity="error" sx={{ mt: 1 }}>{authError}</Alert>}
                    </Box>

                    {!hfAuth.signed_in && (
                        <Alert severity="info" sx={{ mb: 2 }}>
                            SA3 checkpoints are gated on HuggingFace. Sign in with a Read token before downloading,
                            and visit each repo to accept its gated-access terms — e.g.{' '}
                            <a href="https://huggingface.co/stabilityai/stable-audio-3-small-music" target="_blank" rel="noreferrer">
                                stable-audio-3-small-music
                            </a>.
                        </Alert>
                    )}

                    {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                    {loading && <LinearProgress sx={{ mb: 2 }} />}

                    {!loading && !anyInstalled && (
                        <Box sx={{
                            p: 2, mb: 2, borderRadius: 1, bgcolor: 'action.hover',
                            display: 'flex', alignItems: 'center', gap: 2,
                        }}>
                            <Box sx={{ flex: 1 }}>
                                <Typography variant="body2" fontWeight={500}>
                                    Pick a model to get started.
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Recommended: Small — Music (1.2 GB) + SAME-S (0.7 GB) ≈ 1.9 GB. Runs on CPU.
                                </Typography>
                            </Box>
                            <Button variant="contained" onClick={quickStart} disabled={!catalog.length}>
                                Quick Start
                            </Button>
                        </Box>
                    )}

                    {/* Groups */}
                    {groups.map((g) => (
                        <Accordion
                            key={g.id}
                            defaultExpanded={g.id === 'post-trained'}
                            disableGutters
                            sx={{ '&:before': { display: 'none' }, mb: 0.5 }}
                        >
                            <AccordionSummary expandIcon={<ExpandIcon size={18} />}>
                                <Stack direction="row" alignItems="center" spacing={1}>
                                    <Typography variant="subtitle2">{GROUP_LABELS[g.id]}</Typography>
                                    <Chip size="small" label={g.items.length} sx={{ height: 18, fontSize: 10 }} />
                                </Stack>
                            </AccordionSummary>
                            <AccordionDetails sx={{ p: 0 }}>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ display: 'block', px: 1.5, pt: 0.5, pb: 1 }}
                                >
                                    {GROUP_DESCRIPTIONS[g.id]}
                                </Typography>
                                <Divider />
                                {g.items.map(c => (
                                    <CheckpointRow
                                        key={c.id}
                                        checkpoint={c}
                                        catalog={catalog}
                                        onRequestLicense={openLicense}
                                        onAuthRequired={() => setShowTokenInput(true)}
                                        onChanged={refresh}
                                    />
                                ))}
                            </AccordionDetails>
                        </Accordion>
                    ))}
                </DialogContent>

                <DialogActions>
                    <Button onClick={refresh} disabled={loading}>Refresh</Button>
                    <Button onClick={onClose} variant="contained">Close</Button>
                </DialogActions>
            </Dialog>

            <LicenseAcceptModal
                open={licenseModal.open}
                checkpoint={licenseModal.checkpoint}
                onClose={closeLicense}
                onAccepted={onLicenseAccepted}
            />

            <StorageDrilldown
                open={showStorage}
                onClose={() => setShowStorage(false)}
                storage={storage}
                catalog={catalog}
            />
        </>
    );
}
