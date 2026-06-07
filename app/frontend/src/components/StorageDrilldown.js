import React from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Typography,
    Box,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
} from '@mui/material';

const fmtBytes = (n) => {
    if (!n) return '—';
    // Decimal (SI) units — matches what HuggingFace shows next to safetensors files.
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let v = n;
    let u = 0;
    while (v >= 1000 && u < units.length - 1) { v /= 1000; u += 1; }
    return `${v.toFixed(v < 10 ? 2 : 1)} ${units[u]}`;
};

export default function StorageDrilldown({ open, onClose, storage, catalog }) {
    if (!storage) return null;

    const usedPct = storage.total_used_bytes && (storage.total_used_bytes + storage.total_free_bytes)
        ? (storage.total_used_bytes / (storage.total_used_bytes + storage.total_free_bytes)) * 100
        : 0;

    const nameFor = (id) => catalog?.find(c => c.id === id)?.name || id;

    const rows = (storage.per_model || [])
        .filter(m => m.downloaded)
        .sort((a, b) => b.bytes - a.bytes);

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle>Storage</DialogTitle>
            <DialogContent dividers>
                <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                        {fmtBytes(storage.total_used_bytes)} used · {fmtBytes(storage.total_free_bytes)} free
                    </Typography>
                    <LinearProgress
                        variant="determinate"
                        value={Math.min(100, usedPct)}
                        sx={{ mt: 1, height: 6, borderRadius: 3 }}
                    />
                </Box>

                {rows.length === 0 ? (
                    <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                        Nothing downloaded yet.
                    </Typography>
                ) : (
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Checkpoint</TableCell>
                                <TableCell align="right">Size</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {rows.map(m => (
                                <TableRow key={m.id}>
                                    <TableCell>{nameFor(m.id)}</TableCell>
                                    <TableCell align="right">{fmtBytes(m.bytes)}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                )}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
}
