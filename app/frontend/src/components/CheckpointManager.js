import React, { useState } from 'react';
import {
    Box,
    Typography,
    Card,
    Chip,
    Button,
    Alert,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Snackbar,
} from '@mui/material';
import { CloudDownload as CloudDownloadIcon, Trash2 as DeleteIcon } from 'lucide-react';
import api from '../api';
import { checkpointManagerStyles } from '../theme';

export default function CheckpointManager({ model, onRefresh }) {
    const [loadingStates, setLoadingStates] = useState({});
    const [error, setError] = useState(null);
    const [deleteTarget, setDeleteTarget] = useState(null);
    const [toast, setToast] = useState({
        open: false,
        message: '',
        severity: 'success'
    });

    const handleUnwrapCheckpoint = async (checkpoint) => {
        const checkpointId = checkpoint.path;
        setLoadingStates(prev => ({ ...prev, [checkpointId]: { unwrapping: true } }));
        setError(null);
        try {
            await api.post('/api/unwrap-model', {
                model_config: model.config_path,
                ckpt_path: checkpoint.path,
                name: `${checkpoint.name}_unwrapped`
            });
            setError(null);
            setToast({
                open: true,
                message: `Checkpoint "${checkpoint.name}" unwrapped successfully.`,
                severity: 'success'
            });
            onRefresh();
        } catch (err) {
            setError(`Failed to unwrap ${checkpoint.name}: ${err.response?.data?.error || err.message}`);
        } finally {
            setLoadingStates(prev => ({ ...prev, [checkpointId]: { unwrapping: false } }));
        }
    };

    const handleDeleteCheckpoint = async () => {
        if (!deleteTarget) {
            return;
        }

        const checkpointId = deleteTarget.path;
        setLoadingStates(prev => ({ ...prev, [checkpointId]: { deleting: true } }));
        setError(null);

        try {
            await api.post('/api/delete-checkpoint', {
                checkpoint_path: deleteTarget.path
            });
            setToast({
                open: true,
                message: `Checkpoint "${deleteTarget.name}" deleted successfully.`,
                severity: 'success'
            });
            onRefresh();
        } catch (err) {
            setError(`Failed to delete ${deleteTarget.name}: ${err.response?.data?.error || err.message}`);
        } finally {
            setDeleteTarget(null);
            setLoadingStates(prev => ({ ...prev, [checkpointId]: { deleting: false } }));
        }
    };

    const closeToast = (_, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setToast(prev => ({ ...prev, open: false }));
    };

    const checkpoints = model.checkpoints || [];

    return (
        <>
            <Box sx={checkpointManagerStyles.root}>
                <Typography variant="subtitle2" color="textSecondary">
                    Checkpoints ({checkpoints.length})
                </Typography>

                {checkpoints.length === 0 ? (
                    <Typography variant="caption" color="textSecondary" sx={checkpointManagerStyles.emptyText}>
                        No checkpoints yet.
                    </Typography>
                ) : (
                    <Box sx={checkpointManagerStyles.checkpointsList}>
                        {checkpoints.map((checkpoint, index) => {
                            const checkpointId = checkpoint.path;
                            const isUnwrapping = loadingStates[checkpointId]?.unwrapping;
                            const isDeleting = loadingStates[checkpointId]?.deleting;

                            const hasUnwrappedVersion = model.unwrapped_models?.some(unwrapped =>
                                unwrapped.name.includes(checkpoint.name) ||
                                checkpoint.name.includes(unwrapped.name.replace('_unwrapped', ''))
                            );

                            return (
                                <Card key={index} sx={checkpointManagerStyles.checkpointCard}>
                                    <Box sx={checkpointManagerStyles.checkpointRow}>
                                        <Box sx={checkpointManagerStyles.checkpointInfo}>
                                            <Typography variant="body2" sx={checkpointManagerStyles.checkpointName}>
                                                {checkpoint.name}
                                                {hasUnwrappedVersion && (
                                                    <Chip
                                                        label="Unwrapped"
                                                        size="small"
                                                        color="success"
                                                        sx={checkpointManagerStyles.unwrappedChip}
                                                    />
                                                )}
                                            </Typography>
                                            <Typography variant="caption" color="textSecondary">
                                                {checkpoint.size_mb} MB
                                            </Typography>
                                        </Box>
                                        <Box sx={checkpointManagerStyles.actions}>
                                            {!hasUnwrappedVersion && (
                                                <Button
                                                    variant="outlined"
                                                    color="primary"
                                                    size="small"
                                                    startIcon={<CloudDownloadIcon />}
                                                    onClick={() => handleUnwrapCheckpoint(checkpoint)}
                                                    disabled={isUnwrapping || isDeleting}
                                                >
                                                    {isUnwrapping ? 'Unwrapping...' : 'Unwrap'}
                                                </Button>
                                            )}

                                            {hasUnwrappedVersion && (
                                                <Button
                                                    variant="outlined"
                                                    color="error"
                                                    size="small"
                                                    startIcon={<DeleteIcon />}
                                                    onClick={() => setDeleteTarget(checkpoint)}
                                                    disabled={isDeleting || isUnwrapping}
                                                >
                                                    {isDeleting ? 'Deleting...' : 'Delete'}
                                                </Button>
                                            )}
                                        </Box>
                                    </Box>
                                </Card>
                            );
                        })}
                    </Box>
                )}

                {error && (
                    <Alert severity="error" sx={checkpointManagerStyles.errorAlert}>{error}</Alert>
                )}
            </Box>

            <Dialog
                open={Boolean(deleteTarget)}
                onClose={() => setDeleteTarget(null)}
                aria-labelledby="delete-checkpoint-dialog-title"
            >
                <DialogTitle id="delete-checkpoint-dialog-title">
                    Delete Wrapped Checkpoint
                </DialogTitle>
                <DialogContent>
                    <Typography sx={checkpointManagerStyles.deleteDialogText}>
                        {deleteTarget
                            ? `Delete "${deleteTarget.name}"? This action cannot be undone.`
                            : 'Delete this checkpoint?'}
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDeleteTarget(null)}>
                        Cancel
                    </Button>
                    <Button
                        variant="contained"
                        color="error"
                        onClick={handleDeleteCheckpoint}
                    >
                        Delete
                    </Button>
                </DialogActions>
            </Dialog>

            <Snackbar
                open={toast.open}
                autoHideDuration={3200}
                onClose={closeToast}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            >
                <Alert onClose={closeToast} severity={toast.severity} variant="filled" sx={checkpointManagerStyles.snackbarAlert}>
                    {toast.message}
                </Alert>
            </Snackbar>
        </>
    );
}
