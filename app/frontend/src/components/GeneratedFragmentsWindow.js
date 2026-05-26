import React, { useState, useRef, useCallback } from 'react';
import {
    Paper, Box, Typography, List, ListItem, IconButton, Tooltip,
    Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, Button,
} from '@mui/material';
import {
    Square as StopIcon,
    Play as PlayIcon,
    CloudDownload as DownloadIcon,
    AudioLines as TitleIcon,
    Info as InfoIcon,
    Trash2 as DeleteIcon,
    Eraser as ClearAllIcon,
} from 'lucide-react';
import { generatedFragmentsWindowStyles } from '../theme';

// Compact human-readable "X ago" with absolute fallback for stale items.
function relativeTime(createdAt) {
    if (!createdAt) return '';
    const sec = Math.max(0, (Date.now() - createdAt) / 1000);
    if (sec < 10) return 'just now';
    if (sec < 60) return `${Math.floor(sec)}s ago`;
    const min = sec / 60;
    if (min < 60) return `${Math.floor(min)}m ago`;
    const hr = min / 60;
    if (hr < 24) return `${Math.floor(hr)}h ago`;
    const day = hr / 24;
    if (day < 7) return `${Math.floor(day)}d ago`;
    // Older than a week — show absolute date, no time
    return new Date(createdAt).toLocaleDateString();
}

export default function GeneratedFragmentsWindow({ fragments, onDownload, onDelete, onClearAll }) {
    const [playingFragment, setPlayingFragment] = useState(null);
    const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
    const audioRefs = useRef({});

    const handlePlayPause = (fragment) => {
        const audio = audioRefs.current[fragment.id];
        if (!audio) return;

        if (playingFragment === fragment.id) {
            audio.pause();
            setPlayingFragment(null);
        } else {
            if (playingFragment && audioRefs.current[playingFragment]) {
                audioRefs.current[playingFragment].pause();
            }
            audio.play();
            setPlayingFragment(fragment.id);
        }
    };

    const setAudioRef = useCallback((fragmentId, audioElement) => {
        if (audioElement) {
            audioRefs.current[fragmentId] = audioElement;
        }
    }, []);

    return (
        <Paper variant="outlined" sx={generatedFragmentsWindowStyles.rootPaper}>
            <Box sx={generatedFragmentsWindowStyles.headerRow}>
                <Box sx={generatedFragmentsWindowStyles.titleRow}>
                    <Box component="span" sx={generatedFragmentsWindowStyles.titleIcon}>
                        <TitleIcon size={20} />
                    </Box>
                    <Typography variant="h6" sx={generatedFragmentsWindowStyles.titleText}>
                        Generated Fragments
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Typography variant="caption" color="textSecondary" sx={generatedFragmentsWindowStyles.countText}>
                        {fragments.length}
                    </Typography>
                    {fragments.length > 0 && onClearAll && (
                        <Tooltip title="Clear all (delete every fragment from disk)" placement="top" arrow>
                            <IconButton
                                size="small"
                                onClick={() => setClearConfirmOpen(true)}
                                sx={{ color: 'text.disabled', '&:hover': { color: 'error.main' } }}
                            >
                                <ClearAllIcon size={14} />
                            </IconButton>
                        </Tooltip>
                    )}
                </Box>
            </Box>

            <Dialog open={clearConfirmOpen} onClose={() => setClearConfirmOpen(false)}>
                <DialogTitle>Clear all generated fragments?</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Permanently delete all {fragments.length} fragment{fragments.length === 1 ? '' : 's'} from disk.
                        Uploaded source clips (used by Edit mode) are not affected.
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setClearConfirmOpen(false)}>Cancel</Button>
                    <Button
                        onClick={() => { setClearConfirmOpen(false); onClearAll?.(); }}
                        color="error"
                        variant="contained"
                    >
                        Delete all
                    </Button>
                </DialogActions>
            </Dialog>

            {fragments.length === 0 ? (
                <Box sx={generatedFragmentsWindowStyles.emptyState}>
                    <Typography variant="body2">No fragments generated yet</Typography>
                </Box>
            ) : (
                <List sx={generatedFragmentsWindowStyles.listRoot}>
                    {fragments.slice().reverse().map((fragment) => {
                        const isPlaying = playingFragment === fragment.id;
                        const ago = relativeTime(fragment.createdAt);
                        // The compact line keeps duration + relative time inline.
                        // CFG, seed, full timestamp, and model go in the info
                        // tooltip — accessible but not pushing the row out.
                        const tooltipLines = [
                            `Seed: ${fragment.seed ?? '—'}`,
                            `CFG: ${fragment.cfgScale ?? '—'}`,
                            fragment.steps != null ? `Steps: ${fragment.steps}` : null,
                            fragment.modelId ? `Model: ${fragment.modelId}` : null,
                            fragment.editMode ? `Mode: ${fragment.editMode}` : null,
                            fragment.timestamp ? `Generated: ${fragment.timestamp}` : null,
                        ].filter(Boolean).join('\n');

                        return (
                            <ListItem
                                key={fragment.id}
                                sx={generatedFragmentsWindowStyles.listItem}
                            >
                                <Tooltip
                                    title={isPlaying ? 'Stop' : 'Play'}
                                    placement="top"
                                    arrow
                                >
                                    <IconButton
                                        size="small"
                                        onClick={() => handlePlayPause(fragment)}
                                        sx={generatedFragmentsWindowStyles.playPauseButton(isPlaying)}
                                    >
                                        {isPlaying ? <StopIcon size={16} /> : <PlayIcon size={16} />}
                                    </IconButton>
                                </Tooltip>

                                <Box sx={generatedFragmentsWindowStyles.fragmentMeta}>
                                    <Typography
                                        variant="body2"
                                        sx={generatedFragmentsWindowStyles.fragmentPrompt}
                                        title={fragment.prompt}
                                    >
                                        {fragment.batchTotal > 1 && (
                                            <Box component="span" sx={generatedFragmentsWindowStyles.batchTag}>
                                                {fragment.batchIndex}/{fragment.batchTotal}
                                            </Box>
                                        )}
                                        {fragment.prompt}
                                    </Typography>
                                </Box>

                                <Typography
                                    variant="caption"
                                    color="textSecondary"
                                    sx={generatedFragmentsWindowStyles.fragmentMetaInline}
                                >
                                    {fragment.duration}s
                                    {ago && ` · ${ago}`}
                                </Typography>

                                <Tooltip
                                    title={
                                        <Box component="span" sx={{ whiteSpace: 'pre-line' }}>
                                            {tooltipLines}
                                        </Box>
                                    }
                                    arrow
                                    placement="top"
                                >
                                    <Box
                                        component="span"
                                        sx={generatedFragmentsWindowStyles.fragmentInfoIcon}
                                    >
                                        <InfoIcon size={14} />
                                    </Box>
                                </Tooltip>

                                <Tooltip title="Download" placement="top" arrow>
                                    <IconButton
                                        size="small"
                                        onClick={() => onDownload(fragment)}
                                        sx={generatedFragmentsWindowStyles.downloadButton}
                                    >
                                        <DownloadIcon size={16} />
                                    </IconButton>
                                </Tooltip>

                                {onDelete && (
                                    <Tooltip title="Delete from disk" placement="top" arrow>
                                        <IconButton
                                            size="small"
                                            onClick={() => onDelete(fragment)}
                                            sx={{ color: 'text.disabled', '&:hover': { color: 'error.main', bgcolor: 'action.hover' } }}
                                        >
                                            <DeleteIcon size={16} />
                                        </IconButton>
                                    </Tooltip>
                                )}

                                <audio
                                    ref={el => setAudioRef(fragment.id, el)}
                                    src={fragment.audioUrl}
                                    onEnded={() => setPlayingFragment(null)}
                                    onPause={() => setPlayingFragment(null)}
                                    style={generatedFragmentsWindowStyles.hiddenAudio}
                                />
                            </ListItem>
                        );
                    })}
                </List>
            )}
        </Paper>
    );
}
