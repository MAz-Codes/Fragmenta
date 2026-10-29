import React from 'react';
import { Box, CircularProgress, IconButton, Typography } from '@mui/material';
import {
    ArrowUpCircle as UpdateIcon,
    Download as DownloadIcon,
    FolderOpen as FolderOpenIcon,
    HelpCircle as InfoViewIcon,
    RefreshCw as RefreshIcon,
    Settings as SettingsIcon,
} from 'lucide-react';
import Tooltip from './Tooltip';
import { appToolbarStyles as styles } from '../theme';

/**
 * Top-right app toolbar, on every page: GPU memory, then the app-wide
 * actions (get models, outputs folder, free GPU, Info View, update, settings).
 */
export default function AppToolbar({
    gpu,
    isDocker,
    isFreeingGPU,
    onGetModels,
    onOpenOutputs,
    onFreeGPU,
    infoViewEnabled,
    onToggleInfoView,
    updateVisible,
    onShowUpdate,
    onSettings,
}) {
    const cuda = gpu && gpu.cuda;
    // Apple Silicon: the GPU works out of unified memory, so that's the
    // meter (same meaning as free VRAM: whether a model fits).
    const unified = !cuda && gpu && gpu.unified;
    const mem = cuda || unified;
    const usedPct = mem
        ? Math.min(Math.max(((mem.total - mem.free) / mem.total) * 100, 0), 100)
        : 0;

    return (
        <Box role="toolbar" aria-label="App tools" sx={styles.root}>
            <Tooltip title={
                unified ? `${unified.device} · unified memory, shared by the GPU and apps`
                    : cuda ? `${cuda.device} · memory in use by every process`
                        : 'No GPU found; models run on the CPU'
            }>
                <Box sx={styles.gpu}>
                    <Typography component="span" sx={styles.gpuLabel}>{unified ? 'Memory' : 'GPU'}</Typography>
                    {mem ? (
                        <>
                            <Box sx={styles.meter} aria-hidden>
                                <Box sx={styles.meterFill(usedPct)} />
                            </Box>
                            <Typography component="span" sx={styles.gpuText}>
                                {mem.free.toFixed(1)} / {Math.ceil(mem.total)} GB free
                            </Typography>
                        </>
                    ) : (
                        <Typography component="span" sx={styles.gpuText}>Not detected · CPU mode</Typography>
                    )}
                </Box>
            </Tooltip>
            <Box sx={styles.actions}>
                <Tooltip title="Get models">
                    <IconButton aria-label="Get models" onClick={onGetModels} sx={styles.button}>
                        <DownloadIcon size={16} strokeWidth={1.8} />
                    </IconButton>
                </Tooltip>
                {!isDocker && (
                    <Tooltip title="Open the outputs folder">
                        <IconButton aria-label="Open outputs folder" onClick={onOpenOutputs} sx={styles.button}>
                            <FolderOpenIcon size={16} strokeWidth={1.8} />
                        </IconButton>
                    </Tooltip>
                )}
                <Tooltip title="Free GPU memory">
                    <span>
                        <IconButton
                            aria-label="Free GPU memory"
                            onClick={onFreeGPU}
                            disabled={isFreeingGPU || !mem}
                            sx={styles.button}
                        >
                            {isFreeingGPU
                                ? <CircularProgress size={14} color="inherit" />
                                : <RefreshIcon size={16} strokeWidth={1.8} />}
                        </IconButton>
                    </span>
                </Tooltip>
                <Tooltip title={infoViewEnabled ? 'Info View on: hover anything for help' : 'Info View off'}>
                    <IconButton
                        aria-label={infoViewEnabled ? 'Turn off Info View' : 'Turn on Info View'}
                        aria-pressed={infoViewEnabled}
                        onClick={onToggleInfoView}
                        sx={styles.toggle(infoViewEnabled)}
                    >
                        <InfoViewIcon size={16} strokeWidth={1.8} />
                    </IconButton>
                </Tooltip>
                {updateVisible && (
                    <Tooltip title="Update available">
                        <IconButton aria-label="Update available" onClick={onShowUpdate} sx={styles.update}>
                            <UpdateIcon size={16} strokeWidth={1.8} />
                        </IconButton>
                    </Tooltip>
                )}
                <Tooltip title="Settings: audio, MIDI, about">
                    <IconButton aria-label="Settings" onClick={onSettings} sx={styles.button}>
                        <SettingsIcon size={16} strokeWidth={1.8} />
                    </IconButton>
                </Tooltip>
            </Box>
        </Box>
    );
}
