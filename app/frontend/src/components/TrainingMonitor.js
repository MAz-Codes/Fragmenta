import React, { useMemo } from 'react';
import { Paper, Box, Typography, LinearProgress, Alert } from '@mui/material';
import { Activity as ActivityIcon } from 'lucide-react';
import LossChart from './LossChart';
import Tooltip from './Tooltip';
import { TIPS } from '../tooltips';
import { trainingMonitorStyles } from '../theme';

/**
 * TrainingMonitor — right-pane status card for the Training tab.
 *
 * Reads SA3-shaped status from the backend:
 *   { is_training, status, step, total_steps, current_step, progress,
 *     loss, checkpoints, checkpoints_saved, error, ... }
 *
 * SA3 trains by step count, not epochs — the panel surfaces step / total
 * directly. Loss curve is built frontend-side from successive poll snapshots
 * (trainingHistory) so we don't depend on the backend emitting a history
 * array.
 */
export default function TrainingMonitor({
    trainingProgress,
    trainingStatus,
    trainingHistory,
    trainingError,
    indicatorState,
}) {
    const getProgressColor = () => {
        if (trainingError) return 'error';
        if (trainingProgress === 100) return 'success';
        return 'primary';
    };

    const status = indicatorState?.status || 'idle';
    const label = indicatorState?.label || 'Idle';
    const animate = indicatorState?.animate || false;

    // Loss points for the chart. We prefer the backend's loss_history
    // (built from Lightning's metrics.csv, which records per-step loss
    // from step 0) so the chart shows the full curve even before PL's
    // tqdm postfix surfaces train/loss (which only appears after the
    // first metrics flush, typically end of epoch 0). Falls back to the
    // frontend-built trainingHistory if the backend hasn't populated
    // loss_history yet (very early in the run, before PL writes CSV).
    const lossPoints = useMemo(() => {
        const fromBackend = trainingStatus?.loss_history;
        if (Array.isArray(fromBackend) && fromBackend.length > 0) {
            return fromBackend
                .filter(p => Number.isFinite(p?.step) && Number.isFinite(p?.loss))
                .sort((a, b) => a.step - b.step);
        }
        if (!trainingHistory || trainingHistory.length === 0) return [];
        const byStep = new Map();
        for (const h of trainingHistory) {
            const step = h.current_step ?? h.step;
            const loss = typeof h.loss === 'number' ? h.loss : parseFloat(h.loss);
            if (Number.isFinite(step) && Number.isFinite(loss)) {
                byStep.set(step, { step, loss });
            }
        }
        return Array.from(byStep.values()).sort((a, b) => a.step - b.step);
    }, [trainingHistory, trainingStatus?.loss_history]);

    const step = trainingStatus?.current_step ?? trainingStatus?.step ?? 0;
    const totalSteps = trainingStatus?.total_steps ?? 0;
    const checkpointsSaved = trainingStatus?.checkpoints_saved
        ?? trainingStatus?.checkpoints?.length
        ?? 0;
    const currentLoss = trainingStatus?.loss;
    // Checkpoint steps, read from file names like "epoch=4-step=500.ckpt",
    // for the markers on the loss chart.
    const checkpointSteps = useMemo(() => (trainingStatus?.checkpoints || [])
        .map((c) => {
            const m = String(c?.name ?? c?.path ?? c).match(/step=(\d+)/);
            return m ? Number(m[1]) : null;
        })
        .filter((v) => Number.isFinite(v)), [trainingStatus?.checkpoints]);

    return (
        <Paper sx={trainingMonitorStyles.rootPaper}>
            <Box sx={trainingMonitorStyles.headerRow}>
                <Box sx={trainingMonitorStyles.headerTitleWrap}>
                    <Box component="span" sx={trainingMonitorStyles.headerIcon}>
                        <ActivityIcon size={20} />
                    </Box>
                    <Typography variant="h6" sx={trainingMonitorStyles.headerTitle}>
                        Monitor
                    </Typography>
                </Box>
                <Box sx={trainingMonitorStyles.statusInline(status)}>
                    <Box sx={trainingMonitorStyles.statusDot(status, animate)} />
                    <Typography variant="caption" sx={trainingMonitorStyles.statusText(status)}>
                        {label}
                    </Typography>
                </Box>
            </Box>

            <Box sx={trainingMonitorStyles.progressSection}>
                <Box sx={trainingMonitorStyles.progressHeader}>
                    <Typography component="span" sx={trainingMonitorStyles.bigPercent}>{trainingProgress}%</Typography>
                    <Tooltip title={TIPS.monitor.steps}>
                        <Typography component="span" sx={trainingMonitorStyles.stepText}>
                            {totalSteps > 0 ? `Step ${step.toLocaleString()} of ${totalSteps.toLocaleString()}` : `Step ${step}`}
                        </Typography>
                    </Tooltip>
                </Box>
                <LinearProgress
                    variant="determinate"
                    value={trainingProgress}
                    aria-label="Training progress"
                    sx={trainingMonitorStyles.progressBar}
                />
            </Box>

            <Box sx={trainingMonitorStyles.metricsGrid}>
                <Box sx={trainingMonitorStyles.metricTile}>
                    <Typography component="span" sx={trainingMonitorStyles.metricLabel}>Loss</Typography>
                    <Typography component="span" sx={trainingMonitorStyles.metricValue}>
                        {Number.isFinite(currentLoss) ? parseFloat(currentLoss).toFixed(3) : '—'}
                    </Typography>
                </Box>
                <Tooltip title={TIPS.monitor.checkpoints}>
                    <Box sx={trainingMonitorStyles.metricTile}>
                        <Typography component="span" sx={trainingMonitorStyles.metricLabel}>Checkpoints</Typography>
                        <Typography component="span" sx={trainingMonitorStyles.metricValue}>{checkpointsSaved}</Typography>
                    </Box>
                </Tooltip>
                <Box sx={trainingMonitorStyles.metricTile}>
                    <Typography component="span" sx={trainingMonitorStyles.metricLabel}>Phase</Typography>
                    <Typography component="span" sx={[trainingMonitorStyles.metricValue, { textTransform: 'capitalize' }]}>
                        {trainingStatus?.status || 'idle'}
                    </Typography>
                </Box>
                <Box sx={trainingMonitorStyles.metricTile}>
                    <Typography component="span" sx={trainingMonitorStyles.metricLabel}>Seed</Typography>
                    <Typography component="span" sx={trainingMonitorStyles.metricValue}>
                        {Number.isFinite(trainingStatus?.seed) ? trainingStatus.seed : '—'}
                    </Typography>
                </Box>
            </Box>

            {lossPoints.length > 0 && (
                <Box sx={trainingMonitorStyles.lossSection}>
                    <Tooltip title={TIPS.monitor.lossChart}>
                        <Typography component="span" sx={[trainingMonitorStyles.metricLabel, { width: 'fit-content' }]}>
                            Loss
                        </Typography>
                    </Tooltip>
                    <Box sx={trainingMonitorStyles.lossChartBox}>
                        <LossChart data={lossPoints} checkpoints={checkpointSteps} />
                    </Box>
                    <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={trainingMonitorStyles.lossDisclaimer}
                    >
                        LoRA diffusion loss is noisy by design. Judge the result
                        with your ears, not only with this chart.
                    </Typography>
                </Box>
            )}

            {lossPoints.length === 0 && (trainingStatus?.is_training || status === 'training') && (
                <Box sx={trainingMonitorStyles.lossSection}>
                    <Tooltip title={TIPS.monitor.lossChart}>
                        <Typography component="span" sx={[trainingMonitorStyles.metricLabel, { width: 'fit-content' }]}>
                            Loss
                        </Typography>
                    </Tooltip>
                    <Typography variant="caption" color="textSecondary">
                        Warming up — loss is logged periodically, so the first point
                        appears a little into the run (around step 50). The curve fills
                        in as training proceeds.
                    </Typography>
                </Box>
            )}

            {trainingError && (
                <Alert severity="error" sx={trainingMonitorStyles.errorAlert}>
                    <Typography variant="body2">
                        <strong>Training Error:</strong> {trainingError}
                    </Typography>
                </Alert>
            )}
        </Paper>
    );
}
