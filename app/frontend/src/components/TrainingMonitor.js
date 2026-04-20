import React from 'react';
import { Paper, Box, Typography, LinearProgress, Grid, Alert } from '@mui/material';
import { Activity as ActivityIcon } from 'lucide-react';
import LossChart from './LossChart';
import { trainingMonitorStyles } from '../theme';

export default function TrainingMonitor({
    trainingProgress,
    trainingStatus,
    trainingError,
    trainingConfig,
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

    return (
        <Paper sx={trainingMonitorStyles.rootPaper}>
            <Box sx={trainingMonitorStyles.headerRow}>
                <Box sx={trainingMonitorStyles.headerTitleWrap}>
                    <Box component="span" sx={trainingMonitorStyles.headerIcon}>
                        <ActivityIcon size={20} />
                    </Box>
                    <Typography variant="h6" sx={trainingMonitorStyles.headerTitle}>
                        Training Monitor
                    </Typography>
                </Box>
                <Box sx={trainingMonitorStyles.statusInline}>
                    <Box sx={trainingMonitorStyles.statusDot(status, animate)} />
                    <Typography variant="caption" sx={trainingMonitorStyles.statusText(status)}>
                        {label}
                    </Typography>
                </Box>
            </Box>

            <Box sx={trainingMonitorStyles.progressSection}>
                <Box sx={trainingMonitorStyles.progressHeader}>
                    <Typography variant="body2">Progress</Typography>
                    <Typography variant="body2">{trainingProgress}%</Typography>
                </Box>
                <LinearProgress
                    variant="determinate"
                    value={trainingProgress}
                    color={getProgressColor()}
                    sx={trainingMonitorStyles.progressBar}
                />
            </Box>

            {trainingStatus?.device_info && (
                <Box sx={trainingMonitorStyles.deviceSection}>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                        <strong>Device Used for Training</strong>
                    </Typography>
                    <Typography variant="body2">
                        Device: {trainingStatus.device_info.device} ({trainingStatus.device_info.memory_gb?.toFixed(2)}GB VRAM)
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={trainingMonitorStyles.deviceInfo}>
                        Info: {trainingStatus.device_info.type === 'cuda' ? 'CUDA GPU available and selected for training' :
                            trainingStatus.device_info.type === 'cpu' ? 'Using CPU (no CUDA GPU available or compatible)' :
                                'Using MPS (Apple Silicon GPU)'}
                    </Typography>
                </Box>
            )}

            <Grid container spacing={2} sx={trainingMonitorStyles.metricsGrid}>
                <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">Current Epoch</Typography>
                    <Typography variant="body1">
                        {trainingStatus?.current_epoch !== undefined ?
                            `${trainingStatus.current_epoch + 1} / ${trainingConfig.epochs}` :
                            '0 / ' + trainingConfig.epochs}
                    </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">Global Step / Total Steps</Typography>
                    <Typography variant="body1" color="primary">
                        {trainingStatus?.global_step !== undefined && trainingStatus?.total_steps !== undefined ?
                            `${trainingStatus.global_step} / ${trainingStatus.total_steps}` :
                            'N/A'}
                    </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">Checkpoints Saved</Typography>
                    <Typography variant="body1">
                        {trainingStatus?.checkpoints_saved || 0}
                    </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="textSecondary">Current Loss</Typography>
                    <Typography variant="body1">
                        {trainingStatus?.loss ? parseFloat(trainingStatus.loss).toFixed(4) : 'N/A'}
                    </Typography>
                </Grid>
            </Grid>

            {trainingStatus?.loss_history && trainingStatus.loss_history.length > 0 && (
                <Box sx={trainingMonitorStyles.lossSection}>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                        <strong>Loss History</strong>
                    </Typography>
                    <Box sx={trainingMonitorStyles.lossChartBox}>
                        <LossChart data={trainingStatus.loss_history} />
                    </Box>
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
