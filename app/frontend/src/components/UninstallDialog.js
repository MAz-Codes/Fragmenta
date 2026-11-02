import React, { useEffect, useState } from 'react';
import {
    Alert,
    Button,
    Checkbox,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    FormControlLabel,
    Typography,
} from '@mui/material';
import api from '../api';

const fmtGb = (bytes) => {
    const gb = (bytes || 0) / 1e9;
    if (gb >= 0.1) return `${gb.toFixed(1)} GB`;
    return `${Math.ceil((bytes || 0) / 1e6)} MB`;
};

const errorText = (e) => e.response?.data?.error || e.message;

/**
 * Settings → About → Uninstall Fragmenta…: the installed app removes itself and
 * the packages it downloaded, and the user's work too unless they keep it.
 * The app closes once the backend has handed the work to its script.
 */
export default function UninstallDialog({ open, onClose }) {
    const [plan, setPlan] = useState(null);
    const [keep, setKeep] = useState(true);
    const [started, setStarted] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!open) return undefined;
        let cancelled = false;
        setPlan(null);
        setKeep(true);
        setError(null);
        api.get('/api/uninstall', { params: { sizes: 1 } })
            .then(({ data }) => { if (!cancelled) setPlan(data); })
            .catch((e) => { if (!cancelled) setError(errorText(e)); });
        return () => { cancelled = true; };
    }, [open]);

    const uninstall = async () => {
        setError(null);
        try {
            await api.post('/api/uninstall', { keep_data: keep });
            setStarted(true);
        } catch (e) {
            setError(errorText(e));
        }
    };

    const sized = plan && plan.packages_bytes !== undefined;
    return (
        <Dialog open={open} onClose={started ? undefined : onClose} maxWidth="xs" fullWidth>
            <DialogTitle>Uninstall Fragmenta</DialogTitle>
            <DialogContent>
                {started ? (
                    <DialogContentText>Fragmenta is closing and removing itself.</DialogContentText>
                ) : (
                    <>
                        <DialogContentText>
                            Fragmenta closes and removes itself, along with the Python packages it
                            downloaded{sized ? ` (${fmtGb(plan.packages_bytes)})` : ''}.
                        </DialogContentText>
                        <FormControlLabel
                            sx={{ mt: 1.5 }}
                            control={<Checkbox size="small" checked={keep} onChange={(e) => setKeep(e.target.checked)} />}
                            label={`Keep my models, projects and outputs${sized ? ` (${fmtGb(plan.yours_bytes)})` : ''}`}
                        />
                        {plan && keep && (
                            <Typography variant="caption" color="text.secondary" component="div"
                                sx={{ ml: 4, wordBreak: 'break-all' }}>
                                They stay in {plan.data_dir}
                            </Typography>
                        )}
                    </>
                )}
                {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
            </DialogContent>
            {!started && (
                <DialogActions>
                    <Button onClick={onClose}>Cancel</Button>
                    <Button onClick={uninstall} color="error" variant="contained" disabled={!plan?.available}>
                        Uninstall
                    </Button>
                </DialogActions>
            )}
        </Dialog>
    );
}
