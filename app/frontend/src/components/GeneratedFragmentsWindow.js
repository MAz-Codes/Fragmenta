import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
    Paper, Box, Typography, List, ListItem, IconButton,
    Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, Button,
    CircularProgress, Alert,
} from '@mui/material';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import {
    Square as StopIcon,
    Play as PlayIcon,
    AudioLines as TitleIcon,
    Info as InfoIcon,
    Trash2 as DeleteIcon,
    Eraser as ClearAllIcon,
    FolderOpen as RevealIcon,
    Download as DownloadIcon,
} from 'lucide-react';
import { useTheme } from '@mui/material/styles';
import { generatedFragmentsWindowStyles, waveColor } from '../theme';
import GenerationWaveform from './GenerationWaveform';
import api from '../api';
import { setFragmentDragPayload, clearFragmentDragPayload } from '../utils/fragmentDrag';
import { GENERATION_AUDIO_EDIT } from '../features';
import { extractError } from '../utils/errors';

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

// Module-level preload cache, keyed by filename (stable across reloads; id
// is a per-mount fallback). The window unmounts on every tab switch, so a
// per-mount cache meant returning to this tab re-downloaded and re-decoded
// every fragment — and the 5 s grace gate then hid the list again each time.
// Capped with insertion-order eviction; evicted entries get their object URL
// revoked.
const preloadCache = new Map();      // key -> { blob, blobUrl }
const preloadInFlight = new Set();   // keys with a fetch outstanding
// Preloads that failed (file gone, server error): they count as settled, so
// one missing file can't hold the whole list behind the loading screen.
const preloadFailed = new Set();
const PRELOAD_CACHE_MAX = 150;

const preloadKeyOf = (frag) => frag.filename || String(frag.id);

function prunePreloadCache() {
    while (preloadCache.size > PRELOAD_CACHE_MAX) {
        const [oldestKey, entry] = preloadCache.entries().next().value;
        try { URL.revokeObjectURL(entry.blobUrl); } catch { /* ignore */ }
        preloadCache.delete(oldestKey);
    }
}

function evictPreloadEntry(frag) {
    const key = preloadKeyOf(frag);
    const entry = preloadCache.get(key);
    if (entry) {
        try { URL.revokeObjectURL(entry.blobUrl); } catch { /* ignore */ }
        preloadCache.delete(key);
    }
}

export default function GeneratedFragmentsWindow({ fragments, onDelete, onClearAll, isDocker = false }) {
    const theme = useTheme();
    const [playingFragment, setPlayingFragment] = useState(null);
    const [playingTime, setPlayingTime] = useState(0);
    const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
    // Transient feedback for row actions (reveal failures, the web-mode
    // "use download instead" message) — these used to die in the console.
    const [actionMessage, setActionMessage] = useState(null);
    const audioRefs = useRef({});
    // Tracks a play request that's between "user clicked Play" and "audio
    // actually started". If the user clicks again during this window we
    // need to either no-op (same fragment) or cleanly cancel (different
    // fragment) — re-entering load() would abort the first play() and
    // both attempts would fail with AbortError.
    const playInFlightRef = useRef(null);
    // Where the next start lands, 0..1 (set by Play = 0 or a waveform click).
    const startFracRef = useRef(0);

    // `timeupdate` only fires ~4x a second, which makes the playhead jump.
    // Follow the playing element every frame instead.
    useEffect(() => {
        if (!playingFragment) return undefined;
        let raf = 0;
        const tick = () => {
            const el = audioRefs.current[playingFragment];
            if (el && !el.paused) setPlayingTime(el.currentTime);
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [playingFragment]);

    // Background-preload of disk-hydrated fragments. On app reload the parent
    // gives us fragment metadata + the backend URL (/api/fragments/...) but
    // no in-memory Blob. The first Play click on those would HTTP-fetch the
    // file synchronously through the <audio> element and freeze briefly. We
    // pre-fetch them in parallel on mount and gate the UI behind a single
    // loading screen — once everything is ready, plays + waveform decodes
    // are instant because they work off blob: URLs.
    const [loadedTick, setLoadedTick] = useState(0);

    useEffect(() => {
        let cancelled = false;
        fragments.forEach((frag) => {
            if (frag.audioBlob) return;             // already in memory
            const key = preloadKeyOf(frag);
            if (preloadCache.has(key)) return;      // already preloaded
            if (preloadInFlight.has(key)) return;
            if (preloadFailed.has(key)) return;
            if (!frag.audioUrl) return;
            preloadInFlight.add(key);
            fetch(frag.audioUrl)
                .then((r) => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    return r.blob();
                })
                .then((blob) => {
                    // Cache even when unmounted mid-fetch — the next mount
                    // gets it for free. Only the re-render is gated.
                    const blobUrl = URL.createObjectURL(blob);
                    preloadCache.set(key, { blob, blobUrl });
                    prunePreloadCache();
                    if (!cancelled) setLoadedTick((t) => t + 1);
                })
                .catch((err) => {
                    console.warn(`Fragment preload failed (${frag.filename || frag.id}):`, err);
                    preloadFailed.add(key);
                    if (!cancelled) setLoadedTick((t) => t + 1);
                })
                .finally(() => {
                    preloadInFlight.delete(key);
                });
        });
        return () => { cancelled = true; };
    }, [fragments]);

    // Per-fragment helpers that prefer the in-memory blob (immediate) over
    // the HTTP URL. Defined after loadedTick is read so React knows to
    // re-render when a new fragment finishes preloading.
    void loadedTick;
    const effectiveBlob = (frag) => frag.audioBlob || preloadCache.get(preloadKeyOf(frag))?.blob || null;
    const effectiveUrl = (frag) => preloadCache.get(preloadKeyOf(frag))?.blobUrl || frag.audioUrl;
    const isFragmentReady = (frag) => !!frag.audioBlob || preloadCache.has(preloadKeyOf(frag));

    const isSettled = (frag) => isFragmentReady(frag) || preloadFailed.has(preloadKeyOf(frag)) || !frag.audioUrl;
    const readyCount = fragments.filter(isSettled).length;
    const allReady = fragments.length === 0 || readyCount === fragments.length;

    // Safety buffer: once everything reports ready, keep the loading overlay
    // up for an extra 5s before revealing the list. Audio decodes that are
    // still settling in the background can't be poked (and can't crash the
    // list) while the user is gated behind the spinner.
    const GRACE_MS = 5000;
    const [graceDone, setGraceDone] = useState(false);
    // Only worth waiting out when something actually had to download this
    // mount. With everything served from the module cache (every tab switch
    // after the first), the gate would just hide an already-ready list for
    // five pointless seconds.
    const hadToLoadRef = useRef(false);
    useEffect(() => {
        if (fragments.length === 0) { setGraceDone(true); return undefined; }
        if (!allReady) {
            hadToLoadRef.current = true;
            setGraceDone(false);
            return undefined;
        }
        if (!hadToLoadRef.current) { setGraceDone(true); return undefined; }
        const t = setTimeout(() => setGraceDone(true), GRACE_MS);
        return () => clearTimeout(t);
    }, [allReady, fragments.length]);
    const showLoading = fragments.length > 0 && (!allReady || !graceDone);

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
    const handlePlayPause = (fragment, startFrac = 0) => {
        const audio = audioRefs.current[fragment.id];
        if (!audio) return;
        startFracRef.current = startFrac;

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
            // Start where the waveform was clicked (0 for the Play button).
            // Read at start time, so a scrub during loading still lands.
            const d = audio.duration;
            const frac = startFracRef.current;
            audio.currentTime = (frac > 0 && d && isFinite(d))
                ? Math.min(frac * d, Math.max(0, d - 0.05))
                : 0;
            setPlayingTime(audio.currentTime);
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

    // Click / drag on a fragment's waveform. On the playing fragment it jumps
    // there; on any other it starts playback from that spot (same gesture as
    // the dataset page). A scrub during loading just moves the start point.
    const handleSeek = (fragment, frac) => {
        const audio = audioRefs.current[fragment.id];
        if (!audio) return;
        if (playInFlightRef.current?.fragmentId === fragment.id) {
            startFracRef.current = frac;
            return;
        }
        if (audio.paused) {
            handlePlayPause(fragment, frac);
            return;
        }
        const d = audio.duration;
        if (!d || !isFinite(d)) return;
        // Nudge off the very end so the last pixel doesn't fire `ended`.
        audio.currentTime = Math.min(frac * d, Math.max(0, d - 0.05));
        setPlayingTime(audio.currentTime);
    };

    // Reveal a fragment in the OS file manager (folder opens with the file
    // highlighted where the platform supports it). Disk-hydrated fragments
    // always have a filename; in-memory-only ones (not yet flushed) won't.
    const revealInFolder = (fragment) => {
        if (!fragment.filename) return;
        api.post('/api/reveal-fragment', { filename: fragment.filename })
            .then((res) => {
                // Headless (web/Docker) replies success:false with a hint to
                // use the download button — show it instead of swallowing it.
                if (res.data?.success === false && res.data?.message) {
                    setActionMessage(res.data.message);
                }
            })
            .catch((err) => {
                console.warn(`Reveal failed (${fragment.filename}):`, err);
                setActionMessage(extractError(err, `Could not reveal ${fragment.filename}.`));
            });
    };

    // Web/Docker has no host file manager to reveal in, so we offer a direct
    // browser download instead. effectiveUrl is a blob: URL once preloaded, or
    // the same-origin /api/fragments/... URL — both honour the download attr.
    const downloadFragment = (fragment) => {
        if (!fragment.filename) return;
        const url = effectiveUrl(fragment);
        if (!url) return;
        const a = document.createElement('a');
        a.href = url;
        a.download = fragment.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
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
                        Generated fragments
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Typography variant="caption" color="textSecondary" sx={generatedFragmentsWindowStyles.countText}>
                        {fragments.length}
                    </Typography>
                    {fragments.length > 0 && onClearAll && (
                        <Tooltip title={TIPS.fragments.clearAll} placement="top" arrow>
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

            {actionMessage && (
                <Alert
                    severity="info"
                    onClose={() => setActionMessage(null)}
                    sx={{ mx: 1.5, mb: 1, py: 0, fontSize: 12 }}
                >
                    {actionMessage}
                </Alert>
            )}

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
                        onClick={() => {
                            setClearConfirmOpen(false);
                            fragments.forEach(evictPreloadEntry);
                            audioRefs.current = {};
                            onClearAll?.();
                        }}
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
            ) : showLoading ? (
                <Box sx={{
                    ...generatedFragmentsWindowStyles.emptyState,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 1.5,
                }}>
                    <CircularProgress size={28} />
                    <Typography variant="body2">
                        {allReady
                            ? 'Finishing up…'
                            : `Loading fragments… ${readyCount} / ${fragments.length}`}
                    </Typography>
                </Box>
            ) : (
                <List sx={generatedFragmentsWindowStyles.listRoot}>
                    {fragments.slice().reverse().map((fragment, fragmentIndex) => {
                        const isPlaying = playingFragment === fragment.id;
                        const ago = relativeTime(fragment.createdAt);
                        // CFG, seed, full timestamp, and model go in the info
                        // tooltip — accessible but not pushing the row out.
                        const tooltipLines = [
                            // Pre-fix fragments stored -1 for a random seed;
                            // show that as "random" rather than a bare -1.
                            `Seed: ${(fragment.seed != null && fragment.seed >= 0) ? fragment.seed : 'random'}`,
                            // Distilled SA3 models have CFG distilled away — it's
                            // genuinely not applicable, not missing.
                            `CFG: ${fragment.cfgScale ?? 'n/a'}`,
                            fragment.steps != null ? `Steps: ${fragment.steps}` : null,
                            fragment.modelId ? `Model: ${fragment.modelId}` : null,
                            fragment.editMode ? `Mode: ${fragment.editMode}` : null,
                            fragment.duration ? `Duration: ${fragment.duration}s` : null,
                            ago ? `Generated: ${ago}` : null,
                            fragment.timestamp ? fragment.timestamp : null,
                        ].filter(Boolean).join('\n');

                        return (
                            <ListItem
                                key={fragment.id}
                                sx={generatedFragmentsWindowStyles.listItem(isPlaying)}
                            >
                                <Box sx={generatedFragmentsWindowStyles.rowBody}>
                                <Box sx={generatedFragmentsWindowStyles.rowTop}>

                                {/* The prompt text is the row's drag handle (the waveform
                                    next to it scrubs). One drag carries both payloads:
                                    the in-app one for Edit existing's drop zone (gated
                                    with it, see features.js) and a DownloadURL so
                                    engines that support it (Chrome-based, Windows/macOS)
                                    can drop the WAV on the desktop or into a DAW. */}
                                <Box
                                    sx={{
                                        ...generatedFragmentsWindowStyles.fragmentMeta,
                                        ...(effectiveUrl(fragment) && { cursor: 'grab' }),
                                    }}
                                    {...(effectiveUrl(fragment) ? {
                                        draggable: true,
                                        onDragStart: (e) => {
                                            e.dataTransfer.effectAllowed = 'copy';
                                            // The OS needs an absolute URL: blob: as is, the
                                            // backend's /api/fragments/... path resolved
                                            // against this origin.
                                            const raw = effectiveUrl(fragment);
                                            const absolute = (raw.startsWith('http') || raw.startsWith('blob:'))
                                                ? raw
                                                : `${window.location.origin}${raw.startsWith('/') ? '' : '/'}${raw}`;
                                            e.dataTransfer.setData(
                                                'DownloadURL',
                                                `audio/wav:${fragment.filename || 'fragment.wav'}:${absolute}`,
                                            );
                                            if (!GENERATION_AUDIO_EDIT) return;
                                            // In-app payload consumed by EditPanel's drop zone
                                            // ("drag a clip into Edit existing").
                                            e.dataTransfer.setData(
                                                'application/x-fragmenta-fragment',
                                                fragment.filename || '',
                                            );
                                            // Hand off the in-memory blob too so the drop can
                                            // use it directly — no disk fetch, immune to any
                                            // name mismatch. Falls back to the filename when
                                            // the blob isn't preloaded yet.
                                            const blob = effectiveBlob(fragment);
                                            if (blob) {
                                                setFragmentDragPayload({
                                                    filename: fragment.filename || '',
                                                    blob,
                                                });
                                            }
                                        },
                                        onDragEnd: () => clearFragmentDragPayload(),
                                        title: GENERATION_AUDIO_EDIT
                                            ? 'Drag into Edit existing to use as a source clip'
                                            : 'Drag to save the file',
                                    } : {})}
                                >
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
                                <Typography component="span" sx={generatedFragmentsWindowStyles.metaText}>
                                    {[fragment.duration ? `${Math.round(Number(fragment.duration))} s` : null, ago].filter(Boolean).join(' · ')}
                                </Typography>
                                </Box>

                                {/* Play, waveform and actions on one line under the title. */}
                                <Box sx={generatedFragmentsWindowStyles.rowWave}>
                                <IconButton
                                    size="small"
                                    onClick={() => handlePlayPause(fragment)}
                                    aria-label={isPlaying ? 'Stop' : 'Play'}
                                    sx={generatedFragmentsWindowStyles.playPauseButton(isPlaying)}
                                >
                                    {isPlaying ? <StopIcon size={14} fill="currentColor" /> : <PlayIcon size={14} fill="currentColor" />}
                                </IconButton>
                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                <GenerationWaveform
                                    blob={effectiveBlob(fragment)}
                                    audioUrl={effectiveUrl(fragment)}
                                    height={40}
                                    color={waveColor(theme, fragmentIndex).main}
                                    pastel={waveColor(theme, fragmentIndex).faint}
                                    currentTime={isPlaying ? playingTime : 0}
                                    duration={audioRefs.current[fragment.id]?.duration || fragment.duration || 0}
                                    onSeek={(frac) => handleSeek(fragment, frac)}
                                />
                                </Box>

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

                                {fragment.filename && (
                                    isDocker ? (
                                        <Tooltip title={TIPS.fragments.download} placement="top" arrow>
                                            <IconButton
                                                size="small"
                                                onClick={() => downloadFragment(fragment)}
                                                aria-label="Download fragment"
                                                sx={{ color: 'text.disabled', '&:hover': { color: 'text.primary', bgcolor: 'action.hover' } }}
                                            >
                                                <DownloadIcon size={16} />
                                            </IconButton>
                                        </Tooltip>
                                    ) : (
                                        <Tooltip title={TIPS.fragments.revealInFolder} placement="top" arrow>
                                            <IconButton
                                                size="small"
                                                onClick={() => revealInFolder(fragment)}
                                                aria-label="Show in folder"
                                                sx={{ color: 'text.disabled', '&:hover': { color: 'text.primary', bgcolor: 'action.hover' } }}
                                            >
                                                <RevealIcon size={16} />
                                            </IconButton>
                                        </Tooltip>
                                    )
                                )}

                                {onDelete && (
                                    <Tooltip title={TIPS.fragments.deleteFromDisk} placement="top" arrow>
                                        <IconButton
                                            size="small"
                                            onClick={() => {
                                                evictPreloadEntry(fragment);
                                                delete audioRefs.current[fragment.id];
                                                onDelete(fragment);
                                            }}
                                            sx={{ color: 'text.disabled', '&:hover': { color: 'error.main', bgcolor: 'action.hover' } }}
                                        >
                                            <DeleteIcon size={16} />
                                        </IconButton>
                                    </Tooltip>
                                )}
                                </Box>
                                </Box>

                                <audio
                                    ref={el => setAudioRef(fragment.id, el)}
                                    src={effectiveUrl(fragment)}
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
