import React, { useEffect, useState } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Typography,
    Box,
    CircularProgress,
    Alert,
} from '@mui/material';
import api from '../api';

/**
 * Stability AI Community License accept modal.
 *
 * Opens for a single catalog entry. Fetches the license text from the backend
 * (`/api/checkpoints/<id>/license`) and presents Accept/Decline. On accept,
 * persists via `/api/checkpoints/<id>/accept-terms` and calls `onAccepted()`.
 */
export default function LicenseAcceptModal({ checkpoint, open, onClose, onAccepted }) {
    const [text, setText] = useState('');
    const [loading, setLoading] = useState(false);
    const [accepting, setAccepting] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!open || !checkpoint) return;
        setLoading(true);
        setError(null);
        api.get(`/api/checkpoints/${checkpoint.id}/license`)
            .then(r => setText(r.data.license || ''))
            .catch(e => setError(e.response?.data?.error || e.message))
            .finally(() => setLoading(false));
    }, [open, checkpoint]);

    const handleAccept = async () => {
        if (!checkpoint) return;
        setAccepting(true);
        setError(null);
        try {
            await api.post(`/api/checkpoints/${checkpoint.id}/accept-terms`);
            onAccepted?.(checkpoint);
            onClose();
        } catch (e) {
            setError(e.response?.data?.error || e.message);
        } finally {
            setAccepting(false);
        }
    };

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle>
                License — {checkpoint?.name}
                <Typography variant="caption" display="block" color="text.secondary">
                    Stability AI Community License ({checkpoint?.license})
                </Typography>
            </DialogTitle>
            <DialogContent dividers>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {loading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                        <CircularProgress />
                    </Box>
                ) : (
                    <Box
                        component="pre"
                        sx={{
                            whiteSpace: 'pre-wrap',
                            fontFamily: 'monospace',
                            fontSize: 13,
                            maxHeight: 400,
                            overflow: 'auto',
                            m: 0,
                        }}
                    >
                        {text}
                    </Box>
                )}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} disabled={accepting}>Decline</Button>
                <Button
                    onClick={handleAccept}
                    variant="contained"
                    disabled={loading || accepting || !text}
                >
                    {accepting ? <CircularProgress size={20} /> : 'Accept'}
                </Button>
            </DialogActions>
        </Dialog>
    );
}
