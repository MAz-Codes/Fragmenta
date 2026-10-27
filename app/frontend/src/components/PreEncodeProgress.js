import React, { useEffect, useRef, useState } from 'react';
import { Box, Button, LinearProgress, Typography } from '@mui/material';
import { Square as StopIcon } from 'lucide-react';

const ENCODER_NAMES = { 'same-s': 'small-model', 'same-l': 'medium-model' };

function fmtClock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/**
 * Progress for a running pre-encode job (Training and Dataset pages).
 *
 * The encoder reports only when each clip starts, and loading it plus a
 * long clip can take minutes on a CPU, so a bare "2 / 4" looks stuck. This
 * shows the phase, a bar that keeps moving (the buffer segment is the clip in
 * progress), elapsed time, and — once a clip has finished — the time left.
 *
 * Props:
 *   job:    pre-encode job from /pre-encode/status (state, phase, current,
 *           total, started_at, encoding_started_at, clip_started_at, now)
 *   onStop: cancel handler
 */
export default function PreEncodeProgress({ job, onStop }) {
    // Server time at the moment of the last poll vs. our clock, so elapsed
    // time works off the server's timestamps even if the clocks differ.
    const offsetRef = useRef(0);
    useEffect(() => {
        if (job?.now) offsetRef.current = job.now - Date.now() / 1000;
    }, [job]);
    // Re-render every second so the clock and estimate tick between polls.
    const [, setTick] = useState(0);
    useEffect(() => {
        const id = window.setInterval(() => setTick((t) => t + 1), 1000);
        return () => window.clearInterval(id);
    }, []);

    if (!job) return null;
    const now = Date.now() / 1000 + offsetRef.current;
    const total = job.total || 0;
    const encoding = job.phase === 'encoding' && total > 0;
    const done = encoding ? Math.max(0, job.current - 1) : 0;

    const encoderName = ENCODER_NAMES[job.autoencoder];
    let label = encoderName ? `Loading the ${encoderName} encoder…` : 'Loading the encoder…';
    if (encoding) label = `Encoding clip ${job.current} of ${total}`;

    let detail = job.started_at ? `${fmtClock(now - job.started_at)} elapsed` : '';
    if (encoding && done > 0 && job.encoding_started_at && job.clip_started_at) {
        const perClip = (job.clip_started_at - job.encoding_started_at) / done;
        const left = perClip * (total - done) - (now - job.clip_started_at);
        detail += left > 60
            ? ` · about ${Math.round(left / 60)} min left`
            : ' · less than a minute left';
    }

    return (
        <Box>
            <LinearProgress
                variant={encoding ? 'buffer' : 'indeterminate'}
                value={encoding ? (done / total) * 100 : undefined}
                valueBuffer={encoding ? (job.current / total) * 100 : undefined}
            />
            <Box sx={{ mt: 0.75, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="caption" sx={{ display: 'block' }}>
                        {label}
                    </Typography>
                    {detail && (
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                            {detail}
                        </Typography>
                    )}
                </Box>
                <Button
                    size="small"
                    variant="outlined"
                    color="error"
                    startIcon={<StopIcon size={14} />}
                    onClick={onStop}
                >
                    Stop
                </Button>
            </Box>
        </Box>
    );
}
