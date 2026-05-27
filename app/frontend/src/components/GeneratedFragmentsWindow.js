import React, { useState, useRef, useCallback } from 'react';
import {
    Paper, Box, Typography, List, ListItem, IconButton, Tooltip,
    Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, Button,
} from '@mui/material';
import {
    Square as StopIcon,
    Play as PlayIcon,
    AudioLines as TitleIcon,
    Info as InfoIcon,
    Trash2 as DeleteIcon,
    Eraser as ClearAllIcon,
} from 'lucide-react';
import { generatedFragmentsWindowStyles } from '../theme';
import GenerationWaveform from './GenerationWaveform';

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

export default function GeneratedFragmentsWindow({ fragments, onDelete, onClearAll }) {
    const [playingFragment, setPlayingFragment] = useState(null);
    const [playingTime, setPlayingTime] = useState(0);
    const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
    const audioRefs = useRef({});
    // Tracks a play request that's between "user clicked Play" and "audio
    // actually started". If the user clicks again during this window we
    // need to either no-op (same fragment) or cleanly cancel (different
    // fragment) — re-entering load() would abort the first play() and
    // both attempts would fail with AbortError.
    const playInFlightRef = useRef(null);

    // Strict single-play with first-click readiness gate.
    //
    // Race-fixes the old version had:
    //   1. Iterate audioRefs.current and pause everything that isn't the
    //      new target — avoids losing the race when two play clicks land
    //      before React state settles.
    //   2. For blob URLs, Chromium often doesn't actually pull bytes until
    //      the first play() call, and play() rejects/hangs if readyState
    //      is too low. If we're not ready, call load() and wait for
    //      `canplay` (with a 1500 ms safety timeout) before play().
    //   3. Guard against the user clicking Play twice during loading. A
    //      second load() while the first play() is still pending aborts
    //      the first with AbortError. playInFlightRef tracks the active
    //      request: same-fragment second click is a no-op; different
    //      fragment cleanly cancels the prior load timer/listener.
    const handlePlayPause = (fragment) => {
        const audio = audioRefs.current[fragment.id];
        if (!audio) return;

        // Stop case: this fragment is currently playing → pause it.
        if (!audio.paused) {
            playInFlightRef.current?.cleanup?.();
            playInFlightRef.current = null;
            audio.pause();
            audio.currentTime = 0;
            setPlayingFragment(null);
            setPlayingTime(0);
            return;
        }

        // Click during loading of the SAME fragment → ignore.
        if (playInFlightRef.current?.fragmentId === fragment.id) {
            return;
        }
        // Click during loading of a DIFFERENT fragment → cancel that.
        if (playInFlightRef.current) {
            playInFlightRef.current.cleanup?.();
            playInFlightRef.current = null;
        }

        Object.values(audioRefs.current).forEach((el) => {
            if (el && el !== audio) {
                el.pause();
                el.currentTime = 0;
            }
        });

        const startedFor = fragment.id;
        setPlayingFragment(startedFor);
        setPlayingTime(0);

        const startPlayback = () => {
            audio.currentTime = 0;
            Promise.resolve(audio.play())
                .then(() => {
                    // Successfully playing — clear the in-flight marker so
                    // the next Play click can fire a fresh request.
                    if (playInFlightRef.current?.fragmentId === startedFor) {
                        playInFlightRef.current = null;
                    }
                })
                .catch((err) => {
                    // AbortError is expected when the user cancels (clicks
                    // Stop or switches fragments) — don't noise the log.
                    if (err && err.name !== 'AbortError') {
                        console.warn(`Fragment play failed (${fragment.filename || fragment.id}):`, err);
                    }
                    setPlayingFragment((prev) => (prev === startedFor ? null : prev));
                    setPlayingTime(0);
                    if (playInFlightRef.current?.fragmentId === startedFor) {
                        playInFlightRef.current = null;
                    }
                });
        };

        if (audio.readyState >= 2) {
            playInFlightRef.current = { fragmentId: startedFor, cleanup: null };
            startPlayback();
            return;
        }

        // Not ready yet — load and wait for canplay (or 1.5 s timeout).
        try { audio.load(); } catch { /* ignore */ }
        let cancelled = false;
        const onReady = () => {
            audio.removeEventListener('canplay', onReady);
            clearTimeout(timer);
            if (cancelled) return;
            startPlayback();
        };
        audio.addEventListener('canplay', onReady, { once: true });
        // 5 s — disk-hydrated fragments fetch from /api/fragments/...
        // over HTTP, which can take a couple of seconds on first request.
        // Blob-URL fragments (in-memory) hit canplay almost instantly.
        const timer = setTimeout(() => {
            audio.removeEventListener('canplay', onReady);
            if (!cancelled) startPlayback();
        }, 5000);
        playInFlightRef.current = {
            fragmentId: startedFor,
            cleanup: () => {
                cancelled = true;
                audio.removeEventListener('canplay', onReady);
                clearTimeout(timer);
            },
        };
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
                        // CFG, seed, full timestamp, and model go in the info
                        // tooltip — accessible but not pushing the row out.
                        const tooltipLines = [
                            `Seed: ${fragment.seed ?? '—'}`,
                            `CFG: ${fragment.cfgScale ?? '—'}`,
                            fragment.steps != null ? `Steps: ${fragment.steps}` : null,
                            fragment.modelId ? `Model: ${fragment.modelId}` : null,
                            fragment.editMode ? `Mode: ${fragment.editMode}` : null,
                            `Duration: ${fragment.duration}s`,
                            ago ? `Generated: ${ago}` : null,
                            fragment.timestamp ? fragment.timestamp : null,
                        ].filter(Boolean).join('\n');

                        return (
                            <ListItem
                                key={fragment.id}
                                sx={generatedFragmentsWindowStyles.listItem}
                            >
                                <IconButton
                                    size="small"
                                    onClick={() => handlePlayPause(fragment)}
                                    aria-label={isPlaying ? 'Stop' : 'Play'}
                                    sx={generatedFragmentsWindowStyles.playPauseButton(isPlaying)}
                                >
                                    {isPlaying ? <StopIcon size={16} /> : <PlayIcon size={16} />}
                                </IconButton>

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

                                <GenerationWaveform
                                    blob={fragment.audioBlob}
                                    audioUrl={fragment.audioUrl}
                                    filename={fragment.filename || 'fragment.wav'}
                                    currentTime={isPlaying ? playingTime : 0}
                                    duration={fragment.duration || 0}
                                />

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
                                    preload="auto"
                                    onTimeUpdate={(e) => {
                                        if (playingFragment === fragment.id) {
                                            setPlayingTime(e.target.currentTime);
                                        }
                                    }}
                                    onEnded={() => {
                                        if (playingFragment === fragment.id) {
                                            setPlayingFragment(null);
                                            setPlayingTime(0);
                                        }
                                    }}
                                    onPause={() => {
                                        if (playingFragment === fragment.id) {
                                            setPlayingFragment(null);
                                        }
                                    }}
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
