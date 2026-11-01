import React, { useCallback, useEffect, useState } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    Box,
    ButtonBase,
    Typography,
    Button,
    IconButton,
    Stack,
    Alert,
    TextField,
    LinearProgress,
} from '@mui/material';
import { X as CloseIcon } from 'lucide-react';
import api from '../api';
import CheckpointRow, { fmtBytes, MODEL_CARD_SX, MODEL_NAME_SX, MODEL_FACTS_SX } from './CheckpointRow';
import StorageDrilldown from './StorageDrilldown';
import Tooltip from './Tooltip';
import { TIPS } from '../tooltips';
import { hiddenOnMac } from '../utils/macModels';

const ROW_COLUMNS = '190px repeat(3, minmax(0, 1fr))';

// One row per purpose. On a Mac the MLX models come first, and the PyTorch
// fast ones only show if they're already here, so they can still be deleted.
const purposeRows = (mac, signedIn) => [
    {
        key: 'mlx',
        title: 'Optimized for Mac (MLX)',
        about: 'Quick results, built for Apple Silicon. No account needed.',
    },
    {
        key: 'post-trained',
        title: 'Fast generation',
        about: mac
            ? 'The PyTorch versions you already have.'
            : signedIn
                ? 'Quick results. Click a name to accept its terms on Hugging Face.'
                : 'Quick results. Free Hugging Face account needed.',
    },
    {
        key: 'base',
        title: mac ? 'Training' : 'Full control and training',
        about: mac
            ? 'LoRAs train on these. Slower to generate with, every setting open. No account needed.'
            : 'Every setting open. Train LoRAs on these. No account needed.',
    },
    {
        key: 'tagger',
        title: 'Dataset tools',
        about: 'For preparing audio to train on.',
    },
];

export default function CheckpointManagerWindow({ open, onClose, fade }) {
    const [catalog, setCatalog] = useState([]);
    const [storage, setStorage] = useState(null);
    const [env, setEnv] = useState(null);
    const [terms, setTerms] = useState(null);
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
            const [cat, store, auth, environment, starter] = await Promise.all([
                api.get('/api/checkpoints'),
                api.get('/api/checkpoints/storage'),
                api.get('/api/hf-auth/status'),
                api.get('/api/environment'),
                api.get('/api/starter'),
            ]);
            setCatalog(cat.data.checkpoints);
            setStorage(store.data);
            setHfAuth(auth.data);
            setEnv(environment.data);
            setTerms(starter.data);
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
    const appleSilicon = catalog.some(c => c.kind === 'mlx');
    const shown = catalog.filter(c => !(appleSilicon && hiddenOnMac(c)));
    const hasGatedRow = shown.some(c => c.kind === 'post-trained' && !c.downloaded);
    // Public HF Spaces demo: checkpoints are pre-provisioned and visitors
    // must not hand a personal HF token to a shared Space, so sign-in is
    // greyed out there. Every other platform is unaffected.
    const isHfDemo = Boolean(env?.hf_space);
    const usedShare = storage
        ? storage.total_used_bytes / Math.max(1, storage.total_used_bytes + storage.total_free_bytes)
        : 0;

    return (
        <>
            <Dialog
                open={open}
                onClose={onClose}
                maxWidth={false}
                fullWidth
                scroll="paper"
                PaperProps={{ sx: (t) => ({ width: 1180, backgroundColor: t.palette.fragmenta.bg, borderRadius: '28px' }) }}
                {...fade}
            >
                <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 2.5, p: '28px 32px 0' }}>
                    <Box sx={{ flex: 1, fontSize: 30, lineHeight: 1.2 }}>Checkpoint Manager</Box>

                    {hfAuth.signed_in ? (
                        <Stack direction="row" alignItems="center" spacing={1}>
                            <Typography variant="caption" color="text.secondary">
                                Hugging Face: signed in as <strong>{hfAuth.username}</strong>
                            </Typography>
                            <Tooltip title={TIPS.manager.hfLogout}>
                                <Button size="small" variant="text" onClick={logout}>Sign out</Button>
                            </Tooltip>
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
                    ) : hasGatedRow && (
                        <Tooltip title={isHfDemo ? TIPS.manager.hfLoginDemo : TIPS.manager.hfLogin}>
                            {/* span wrapper so the tooltip works on a disabled button */}
                            <span>
                                <Button
                                    size="small"
                                    variant="text"
                                    onClick={() => setShowTokenInput(true)}
                                    disabled={isHfDemo}
                                    sx={(t) => ({ height: 38, px: 2.25, backgroundColor: t.palette.fragmenta.surface })}
                                >
                                    Sign in to Hugging Face
                                </Button>
                            </span>
                        </Tooltip>
                    )}

                    <IconButton onClick={onClose} aria-label="Close"><CloseIcon size={20} /></IconButton>
                </DialogTitle>

                <DialogContent sx={{ p: '22px 32px 32px !important' }}>
                    {showTokenInput && !hfAuth.signed_in && (
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'right', mb: 2, '& a': { color: 'inherit' } }}>
                            Needs a free{' '}
                            <a href="https://huggingface.co/join" target="_blank" rel="noreferrer">Hugging Face account</a>
                            {' '}and a{' '}
                            <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">Read access token</a>.
                        </Typography>
                    )}
                    {authError && <Alert severity="error" sx={{ mb: 2 }}>{authError}</Alert>}

                    {isHfDemo && (
                        <Alert severity="info" sx={{ mb: 2 }}>
                            Hugging Face sign-in is disabled in this demonstration Space.{' '}
                            <a href="https://www.misaghazimi.com/fragmenta" target="_blank" rel="noreferrer">
                                Install Fragmenta locally
                            </a>
                            {' '}to sign in with your own account and download checkpoints.
                        </Alert>
                    )}

                    {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                    {loading && !catalog.length && <LinearProgress sx={{ mb: 2 }} />}

                    {!anyInstalled && catalog.length > 0 && (
                        <Typography variant="body2" sx={{ mb: 2.5 }}>
                            <strong>Pick a model to get started.</strong>{' '}
                            {appleSilicon
                                ? 'Small - Music under Optimized for Mac (MLX) downloads without an account (1.9 GB).'
                                : 'Small - Music (Base) downloads without an account (3.5 GB).'}
                        </Typography>
                    )}

                    <Stack spacing={2.5}>
                        {purposeRows(appleSilicon, hfAuth.signed_in).filter(row => shown.some(c => c.kind === row.key)).map((row, index) => {
                            const models = shown.filter(c => c.kind === row.key);
                            const withStorage = row.key === 'tagger';
                            return (
                                <React.Fragment key={row.key}>
                                    {index > 0 && <Box sx={{ height: '1px', backgroundColor: 'divider' }} />}
                                    <Box sx={{ display: 'grid', gridTemplateColumns: ROW_COLUMNS, gap: 1.5 }}>
                                        <Box sx={{ alignSelf: 'center', pr: 1.5 }}>
                                            <Typography variant="subtitle1" component="h3" sx={{ fontSize: 20, lineHeight: 1.15 }}>
                                                {row.title}
                                            </Typography>
                                            <Typography sx={{ fontSize: 13, lineHeight: 1.4, mt: 0.75, color: 'text.secondary' }}>
                                                {row.about}
                                            </Typography>
                                        </Box>
                                        {models.map(c => (
                                            <CheckpointRow
                                                key={c.id}
                                                checkpoint={c}
                                                env={env}
                                                terms={terms}
                                                onAuthRequired={() => setShowTokenInput(true)}
                                                onChanged={refresh}
                                            />
                                        ))}
                                        {withStorage && (
                                            <Tooltip title={TIPS.manager.storage}>
                                                {/* span wrapper so the tooltip works before storage has loaded */}
                                                <Box component="span" sx={{ display: 'grid' }}>
                                                    <ButtonBase
                                                        onClick={() => setShowStorage(true)}
                                                        disabled={!storage}
                                                        sx={(t) => ({ ...MODEL_CARD_SX(t), textAlign: 'left' })}
                                                    >
                                                        <Box sx={{ flexBasis: '100%' }}>
                                                            <Typography variant="subtitle1" sx={MODEL_NAME_SX}>Storage</Typography>
                                                            <Typography sx={MODEL_FACTS_SX}>
                                                                {storage
                                                                    ? `${fmtBytes(storage.total_used_bytes)} used · ${fmtBytes(storage.total_free_bytes)} free`
                                                                    : '—'}
                                                            </Typography>
                                                        </Box>
                                                        <Box sx={(t) => ({ flexBasis: '100%', height: 5, borderRadius: '999px', overflow: 'hidden', backgroundColor: t.palette.fragmenta.control })}>
                                                            <Box sx={(t) => ({ width: `${usedShare * 100}%`, height: '100%', borderRadius: '999px', backgroundColor: t.palette.fragmenta.accents.fresh })} />
                                                        </Box>
                                                    </ButtonBase>
                                                </Box>
                                            </Tooltip>
                                        )}
                                    </Box>
                                </React.Fragment>
                            );
                        })}
                    </Stack>
                </DialogContent>
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
