import React, { useRef } from 'react';
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
} from '@mui/material';

/**
 * A Fragmenta card for "are you sure?", instead of window.confirm: in the
 * desktop window that's a native alert with Python's icon on it.
 *
 * prompt: { title, text, confirmLabel, danger, onConfirm } while open, else null.
 */
export default function ConfirmDialog({ prompt, onClose }) {
    // Keeps its words while it fades out, after the prompt has gone.
    const shown = useRef(prompt);
    if (prompt) shown.current = prompt;
    const { title, text, confirmLabel = 'OK', danger = false, onConfirm } = shown.current || {};

    return (
        <Dialog open={Boolean(prompt)} onClose={onClose} maxWidth="xs" fullWidth>
            <DialogTitle>{title}</DialogTitle>
            <DialogContent>
                <DialogContentText>{text}</DialogContentText>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Button
                    onClick={() => { onClose(); onConfirm?.(); }}
                    color={danger ? 'error' : 'primary'}
                    variant="contained"
                >
                    {confirmLabel}
                </Button>
            </DialogActions>
        </Dialog>
    );
}
