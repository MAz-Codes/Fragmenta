import React, { useState, useRef, useCallback } from 'react';
import { Paper, Box, Typography, Button, List, ListItem, IconButton } from '@mui/material';
import { Square as StopIcon, Play as PlayIcon, Download as DownloadIcon } from 'lucide-react';
import api from '../api';
import { generatedFragmentsWindowStyles } from '../theme';

export default function GeneratedFragmentsWindow({ fragments, onDownload }) {
    const [playingFragment, setPlayingFragment] = useState(null);
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
        <Paper
            variant="outlined"
            sx={generatedFragmentsWindowStyles.rootPaper}
        >
            <Box sx={generatedFragmentsWindowStyles.headerRow}>
                <Box sx={generatedFragmentsWindowStyles.titleRow}>
                    <Box component="span" sx={generatedFragmentsWindowStyles.titleIcon}>
                        <DownloadIcon size={20} />
                    </Box>
                    <Typography variant="h6" sx={generatedFragmentsWindowStyles.titleText}>
                        Generated Fragments
                    </Typography>
                </Box>
                <Typography variant="caption" color="textSecondary" sx={generatedFragmentsWindowStyles.countText}>
                    {fragments.length}
                </Typography>
            </Box>

            {fragments.length === 0 ? (
                <Box
                    sx={generatedFragmentsWindowStyles.emptyState}
                >
                    <Typography variant="body2">
                        No fragments generated yet
                    </Typography>
                </Box>
            ) : (
                <List
                    sx={generatedFragmentsWindowStyles.listRoot}
                >
                    {fragments.slice().reverse().map((fragment) => (
                        <ListItem
                            key={fragment.id}
                            sx={generatedFragmentsWindowStyles.listItem}
                        >
                            <Box sx={generatedFragmentsWindowStyles.fragmentRow}>
                                <Box sx={generatedFragmentsWindowStyles.fragmentMeta}>
                                    <Typography
                                        variant="subtitle2"
                                        sx={generatedFragmentsWindowStyles.fragmentPrompt}
                                    >
                                        {fragment.batchTotal > 1 && (
                                            <Box component="span" sx={{ fontWeight: 700, mr: 0.75 }}>
                                                [{fragment.batchIndex}/{fragment.batchTotal}]
                                            </Box>
                                        )}
                                        {fragment.prompt}
                                    </Typography>
                                    <Typography variant="caption" color="textSecondary">
                                        {fragment.duration}s
                                        {fragment.cfgScale !== undefined && ` • CFG ${fragment.cfgScale}`}
                                        {fragment.seed !== undefined && ` • Seed ${fragment.seed}`}
                                        {' • '}{fragment.timestamp}
                                    </Typography>
                                </Box>
                                <Box sx={generatedFragmentsWindowStyles.fragmentActions}>
                                    <IconButton
                                        size="small"
                                        onClick={() => handlePlayPause(fragment)}
                                        color={playingFragment === fragment.id ? "primary" : "default"}
                                        sx={generatedFragmentsWindowStyles.playPauseButton(playingFragment === fragment.id)}
                                    >
                                        {playingFragment === fragment.id ? <StopIcon /> : <PlayIcon />}
                                    </IconButton>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        startIcon={<DownloadIcon />}
                                        onClick={() => onDownload(fragment)}
                                    >
                                        Download
                                    </Button>
                                </Box>
                            </Box>

                            <audio
                                ref={el => setAudioRef(fragment.id, el)}
                                src={fragment.audioUrl}
                                onEnded={() => setPlayingFragment(null)}
                                onPause={() => setPlayingFragment(null)}
                                style={generatedFragmentsWindowStyles.hiddenAudio}
                            />
                        </ListItem>
                    ))}
                </List>
            )}
        </Paper>
    );
}
