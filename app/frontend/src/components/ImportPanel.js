import React, { useCallback, useEffect, useState } from 'react';
import {
    Alert,
    Box,
    Button,
    Checkbox,
    FormControlLabel,
    LinearProgress,
    Stack,
    TextField,
    Typography,
} from '@mui/material';
import api from '../api';

const fmtGb = (bytes) => {
    const gb = (bytes || 0) / 1e9;
    if (gb >= 0.1) return `${gb.toFixed(1)} GB`;
    return `${Math.ceil((bytes || 0) / 1e6)} MB`;
};

const errorText = (e) => e.response?.data?.error || e.message;

/** Settings → Import: bring models, projects and audio over from an older Fragmenta. */
export default function ImportPanel() {
    const [path, setPath] = useState('');
    const [found, setFound] = useState(null);
    const [chosen, setChosen] = useState({});
    const [keepOld, setKeepOld] = useState(false);
    const [job, setJob] = useState(null);
    const [error, setError] = useState(null);
    const [busy, setBusy] = useState(false);

    const refreshJob = useCallback(async () => {
        try {
            const r = await api.get('/api/import/status');
            setJob(r.data);
        } catch (e) {
            setError(errorText(e));
        }
    }, []);

    useEffect(() => { refreshJob(); }, [refreshJob]);

    const running = job?.state === 'running';
    useEffect(() => {
        if (!running) return undefined;
        const timer = setInterval(refreshJob, 700);
        return () => clearInterval(timer);
    }, [running, refreshJob]);

    const scan = async (folder) => {
        setError(null);
        setFound(null);
        setJob(null);
        setBusy(true);
        try {
            const r = await api.post('/api/import/scan', { path: folder });
            setFound(r.data);
            setChosen(Object.fromEntries(r.data.items.map((i) => [i.id, i.count > 0])));
            setKeepOld(false);
        } catch (e) {
            setError(errorText(e));
        } finally {
            setBusy(false);
        }
    };

    const chooseFolder = async () => {
        setError(null);
        try {
            const r = await api.post('/api/pick-folder', { purpose: 'import', start_dir: path || undefined });
            if (r.data.error) setError(r.data.error);
            if (r.data.path) {
                setPath(r.data.path);
                scan(r.data.path);
            }
        } catch (e) {
            setError(errorText(e));
        }
    };

    const start = async () => {
        setError(null);
        try {
            const r = await api.post('/api/import/start', {
                path: found.source,
                items: Object.keys(chosen).filter((id) => chosen[id]),
                keep_old: keepOld,
            });
            setJob(r.data);
        } catch (e) {
            setError(errorText(e));
        }
    };

    const finish = () => {
        // Everything reads these folders when it opens, so start the page again.
        try { window.sessionStorage.setItem('fragmenta-skip-welcome-once', '1'); } catch (_) { /* optional */ }
        window.location.reload();
    };

    const selected = found ? found.items.filter((i) => chosen[i.id] && i.count > 0) : [];
    const totalBytes = selected.reduce((sum, i) => sum + i.bytes, 0);
    const copying = found && (!found.same_drive || keepOld);
    const noSpace = copying && found.free_bytes < totalBytes + found.margin_bytes;
    const pct = job?.total_bytes ? Math.min(100, Math.floor((job.done_bytes / job.total_bytes) * 100)) : 0;
    const finished = job && ['done', 'cancelled', 'failed'].includes(job.state);

    return (
        <Stack spacing={2}>
            <Typography variant="h6">Import from an older Fragmenta</Typography>

            <Stack direction="row" spacing={1} alignItems="center">
                <TextField
                    size="small"
                    fullWidth
                    placeholder="Fragmenta folder"
                    value={path}
                    disabled={running}
                    onChange={(e) => setPath(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && path.trim()) scan(path.trim()); }}
                    inputProps={{ 'aria-label': 'Fragmenta folder' }}
                />
                <Button onClick={chooseFolder} disabled={running} sx={{ flexShrink: 0 }}>Choose…</Button>
                <Button variant="contained" onClick={() => scan(path.trim())} disabled={running || busy || !path.trim()} sx={{ flexShrink: 0 }}>
                    Look inside
                </Button>
            </Stack>

            {busy && <LinearProgress />}
            {error && <Alert severity="error">{error}</Alert>}

            {found && !running && !finished && (
                <>
                    <Box sx={(t) => ({ borderRadius: '24px', backgroundColor: t.palette.fragmenta.section, py: 1, px: 2 })}>
                        {found.items.map((item) => (
                            <Stack key={item.id} direction="row" alignItems="center" justifyContent="space-between">
                                <FormControlLabel
                                    control={(
                                        <Checkbox
                                            size="small"
                                            checked={Boolean(chosen[item.id])}
                                            disabled={item.count === 0}
                                            onChange={(e) => setChosen((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                                        />
                                    )}
                                    label={<Typography variant="body2">{item.label}</Typography>}
                                />
                                <Typography variant="caption" color="text.secondary">
                                    {item.count === 0 ? 'Already here' : fmtGb(item.bytes)}
                                    {item.count > 0 && item.already > 0 ? ` · ${item.already} already here` : ''}
                                </Typography>
                            </Stack>
                        ))}
                    </Box>

                    {found.same_drive && (
                        <FormControlLabel
                            control={<Checkbox size="small" checked={keepOld} onChange={(e) => setKeepOld(e.target.checked)} />}
                            label={<Typography variant="body2">Keep the old folder as it is (uses {fmtGb(totalBytes)} more)</Typography>}
                        />
                    )}
                    {noSpace && (
                        <Alert severity="warning">
                            Not enough space (need {fmtGb(totalBytes + found.margin_bytes)}, have {fmtGb(found.free_bytes)}).
                        </Alert>
                    )}
                    <Box>
                        <Button variant="contained" onClick={start} disabled={selected.length === 0 || noSpace}>
                            {copying ? 'Copy' : 'Move'} {fmtGb(totalBytes)} here
                        </Button>
                    </Box>
                </>
            )}

            {running && (
                <Box>
                    <LinearProgress variant="determinate" value={pct} sx={{ height: 8, borderRadius: 999 }} />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
                        {pct}% · {fmtGb(job.done_bytes)} / {fmtGb(job.total_bytes)}
                    </Typography>
                    <Button sx={{ mt: 1 }} onClick={() => api.post('/api/import/cancel').then(refreshJob)}>Cancel</Button>
                </Box>
            )}

            {finished && (
                <>
                    {job.state === 'done' && !job.failed?.length && (
                        <Alert severity="success">Imported.</Alert>
                    )}
                    {job.state === 'cancelled' && <Alert severity="info">Stopped. What came over so far stays here.</Alert>}
                    {job.state === 'failed' && <Alert severity="error">{job.error}</Alert>}
                    {job.failed?.length > 0 && (
                        <Alert severity="warning">
                            Couldn't bring over: {job.failed.map((f) => f.name).join(', ')}
                        </Alert>
                    )}
                    <Box><Button variant="contained" onClick={finish}>Done</Button></Box>
                </>
            )}
        </Stack>
    );
}
