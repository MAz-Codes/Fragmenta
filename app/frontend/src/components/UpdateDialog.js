import React from 'react';
import {
    Alert,
    Box,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Typography,
} from '@mui/material';

/**
 * "A new version is available" — highlights, how to update for this kind of
 * install, and Skip / Later. Text and buttons come from the backend's status
 * (channel + upgrade_kind); URLs are resolved server-side via `onOpen`.
 *
 * Props: open, onClose, update (from useUpdateStatus)
 */

// Backticks in the backend's instructions mark commands — render them as code.
function withCode(text) {
    return String(text || '').split('`').map((part, i) => (i % 2
        ? <Box key={i} component="code" sx={{ fontFamily: 'monospace', px: 0.5, borderRadius: 0.5, bgcolor: 'action.hover' }}>{part}</Box>
        : part));
}

export default function UpdateDialog({ open, onClose, update }) {
    const s = update.status;
    if (!s || !s.available) return null;
    const major = s.upgrade_kind === 'major';
    const go = (target) => { update.open(target); };
    const act = async (fn) => { await fn(); onClose(); };

    let actions;
    if (s.channel === 'packaged') {
        actions = <Button variant="contained" onClick={() => go('download')}>Download</Button>;
    } else if (major) {
        actions = (
            <>
                <Button variant="contained" onClick={() => go('guide')}>Upgrade guide</Button>
                <Button variant="outlined" onClick={() => go('download')}>Download page</Button>
            </>
        );
    } else if (s.channel === 'source-zip') {
        actions = <Button variant="contained" onClick={() => go('download')}>Download page</Button>;
    } else {
        actions = <Button variant="contained" onClick={() => go('notes')}>Release notes</Button>;
    }

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth aria-labelledby="update-dialog-title">
            <DialogTitle id="update-dialog-title">
                {major ? `Fragmenta ${s.latest} — major upgrade` : `Fragmenta ${s.latest} is available`}
            </DialogTitle>
            <DialogContent>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                    You have {s.current}.
                </Typography>
                {(s.critical || s.unsupported) && (
                    <Alert severity="warning" sx={{ mb: 1.5 }}>
                        {s.critical
                            ? 'This update fixes a security or data-loss problem.'
                            : 'Your version is no longer supported.'}
                    </Alert>
                )}
                {s.highlights && s.highlights.length > 0 && (
                    <Box component="ul" sx={{ mt: 0, mb: 1.5, pl: 2.5 }}>
                        {s.highlights.map((h, i) => (
                            <Typography key={i} component="li" variant="body2" sx={{ mb: 0.5 }}>{h}</Typography>
                        ))}
                    </Box>
                )}
                <Typography variant="body2">{withCode(s.instructions)}</Typography>
            </DialogContent>
            <DialogActions sx={{ px: 3, pb: 2, justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button size="small" onClick={() => act(update.skip)}>Skip this version</Button>
                    <Button size="small" onClick={() => act(update.snooze)}>Later</Button>
                </Box>
                <Box sx={{ display: 'flex', gap: 1 }}>{actions}</Box>
            </DialogActions>
        </Dialog>
    );
}
