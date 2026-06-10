import React, { useCallback, useEffect, useState } from 'react';
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
    Alert,
    TextField,
    LinearProgress,
} from '@mui/material';
import {
    X as CloseIcon,
    HardDrive as StorageIcon,
    LogIn as LoginIcon,
    LogOut as LogoutIcon,
} from 'lucide-react';
import api from '../api';
import CheckpointRow from './CheckpointRow';
import StorageDrilldown from './StorageDrilldown';
import Tooltip from './Tooltip';
import { TIPS } from '../tooltips';

const fmtBytes = (n) => {
    if (!n && n !== 0) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let v = n;
    let u = 0;
    while (v >= 1000 && u < units.length - 1) { v /= 1000; u += 1; }
    return `${v.toFixed(v < 10 ? 2 : 1)} ${units[u]}`;
};

export default function CheckpointManagerWindow({ open, onClose }) {
    const [catalog, setCatalog] = useState([]);
    const [storage, setStorage] = useState(null);
    const [env, setEnv] = useState(null);
    const [hfAuth, setHfAuth] = useState({ signed_in: false, username: null });
    const [tokenDraft, setTokenDraft] = useState('');
    const [showTokenInput, setShowTokenInput] = useState(false);
    const [authError, setAuthError] = useState(null);
    const [showStorage, setShowStorage] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [cat, store, auth, environment] = await Promise.all([
                api.get('/api/checkpoints'),
                api.get('/api/checkpoints/storage'),
                api.get('/api/hf-auth/status'),
                api.get('/api/environment'),
            ]);
            setCatalog(cat.data.checkpoints);
            setStorage(store.data);
            setHfAuth(auth.data);
            setEnv(environment.data);
        } catch (e) {
            setError(e.response?.data?.error || e.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (open) refresh();
    }, [open, refresh]);

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

    const anyInstalled = catalog.some(c => c.downloaded);

    return (
        <>
            <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth scroll="paper">
                <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>Checkpoint Manager</Box>
                    <IconButton size="small" onClick={onClose}><CloseIcon size={18} /></IconButton>
                </DialogTitle>

                <DialogContent dividers>
                    <Box sx={{ mb: 2 }}>
                        <Stack direction="row" alignItems="center" spacing={2} flexWrap="wrap">
                            <Tooltip title={TIPS.manager.storage}>
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
                            </Tooltip>

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
                                    <Tooltip title={TIPS.manager.hfToken}>
                                    <TextField
                                        size="small"
                                        placeholder="hf_..."
                                        value={tokenDraft}
                                        onChange={(e) => setTokenDraft(e.target.value)}
                                        type="password"
                                        sx={{ width: 240 }}
                                    />
                                    </Tooltip>
                                    <Button size="small" variant="contained" onClick={submitToken}>
                                        Sign in
                                    </Button>
                                    <Button size="small" onClick={() => { setShowTokenInput(false); setTokenDraft(''); }}>
                                        Cancel
                                    </Button>
                                </Stack>
                            ) : (
                                <Tooltip title={TIPS.manager.hfLogin}>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    startIcon={<LoginIcon size={14} />}
                                    onClick={() => setShowTokenInput(true)}
                                >
                                    Sign in to HuggingFace
                                </Button>
                                </Tooltip>
                            )}
                        </Stack>
                        {authError && <Alert severity="error" sx={{ mt: 1 }}>{authError}</Alert>}
                    </Box>

                    {!hfAuth.signed_in ? (
                        <Alert severity="info" sx={{ mb: 2 }}>
                            SA3 checkpoints are gated on HuggingFace. You need a{' '}
                            <a href="https://huggingface.co/join" target="_blank" rel="noreferrer">
                                HuggingFace account
                            </a>
                            {' '}to continue. Then{' '}
                            <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                                create a Read access token
                            </a>
                            {' '}and sign in above.
                        </Alert>
                    ) : (
                        <Alert severity="info" sx={{ mb: 2 }}>
                            You're signed in. Each model is gated — click its name below to open the
                            HuggingFace page and accept the model's terms before downloading.
                        </Alert>
                    )}

                    {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                    {loading && <LinearProgress sx={{ mb: 2 }} />}

                    {!loading && !anyInstalled && catalog.length > 0 && (
                        <Box sx={{
                            p: 2, mb: 2, borderRadius: 1, bgcolor: 'action.hover',
                        }}>
                            <Typography variant="body2" fontWeight={500}>
                                Pick a model to get started.
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                Small - Music (1.2 GB) is a good first choice on a laptop or any GPU.
                            </Typography>
                        </Box>
                    )}

                    {[
                        { kind: 'post-trained', label: 'Distilled (fast)', hint: '8 steps, cfg locked at 1.0. Prompt, duration and seed only.' },
                        { kind: 'base', label: 'Base (full control)', hint: 'CFG-aware. ~50 steps, cfg ~7. Cfg-scale and steps are live controls.' },
                        { kind: 'tagger', label: 'Auto-annotation tools', hint: 'Optional helpers for dataset prep. CLAP scores audio against your vocabulary.' },
                    ].map(group => {
                        const rows = catalog.filter(c => c.kind === group.kind);
                        if (!rows.length) return null;
                        return (
                            <Box key={group.kind} sx={{ mb: 2 }}>
                                <Typography variant="subtitle2" sx={{ mb: 0.25 }}>{group.label}</Typography>
                                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
                                    {group.hint}
                                </Typography>
                                <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                                    {rows.map(c => (
                                        <CheckpointRow
                                            key={c.id}
                                            checkpoint={c}
                                            env={env}
                                            onAuthRequired={() => setShowTokenInput(true)}
                                            onChanged={refresh}
                                        />
                                    ))}
                                </Box>
                            </Box>
                        );
                    })}
                </DialogContent>

                <DialogActions>
                    <Tooltip title={TIPS.manager.refresh}>
                        <Button onClick={refresh} disabled={loading}>Refresh</Button>
                    </Tooltip>
                    <Button onClick={onClose} variant="contained">Close</Button>
                </DialogActions>
            </Dialog>

            <StorageDrilldown
                open={showStorage}
                onClose={() => setShowStorage(false)}
                storage={storage}
                catalog={catalog}
            />
        </>
    );
}
