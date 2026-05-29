import React, { useState } from 'react';
import {
    Box,
    IconButton,
    Tooltip,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogContentText,
    DialogActions,
    Button,
} from '@mui/material';
import {
    Play as PlayIcon,
    Square as StopIcon,
    Star as StarIcon,
    Trash2 as DeleteIcon,
    Check as CommitIcon,
    Eraser as ClearAllIcon,
} from 'lucide-react';
import { performanceChannelStyles as styles } from '../theme';
import { MidiMappable } from './MidiContext';

/**
 * Per-channel rolling fragment history. Always visible (empty-state included)
 * so the user knows the strip exists. Chronological order — oldest at
 * the top, newest at the bottom; scrolls vertically when the list grows
 * past ~4 visible rows.
 *
 * Each row exposes four actions, all visible by default (no hover-reveal —
 * Performance use is fast, can't afford the discoverability tax):
 *   • Cue ▶/■   — audition through the cue output (separate from main mix)
 *   • Star ★/☆ — mark as a keeper. Starred fragments survive the cap
 *                  eviction; unstarred get dropped FIFO when over cap.
 *   • Delete ⌫  — remove this fragment from history (cancellable confirm not
 *                  shown for single deletes — the entry can be regenerated
 *                  or audition can be retriggered after a quick re-tap).
 *   • Load ✓   — commit this fragment to the channel strip (becomes the
 *                  audio the channel plays). Disabled while already loaded.
 *
 * Props:
 *   fragments:      [{ id, audioUrl, blob, prompt, duration, createdAt,
 *                     starred, number }]
 *   color:          channel accent color
 *   auditioningId:  the id currently playing through cue, or null
 *   committedId:    the id currently loaded into the channel strip, or null
 *   maxFragments:   cap, default 50 (informational; eviction lives in parent)
 *   on{Audition,Commit,ToggleStar,Delete}:  (fragmentId) => void
 *   onClearAll:     () => void  (parent confirms separately — we still show
 *                   a confirm dialog here for the trash-everything action)
 */
export default function ChannelFragmentHistory({
    fragments,
    color,
    channelIndex,
    auditioningId,
    committedId,
    maxFragments = 50,
    onAudition,
    onCommit,
    onToggleStar,
    onDelete,
    onClearAll,
}) {
    const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
    // Channel-scoped MIME type for drag-and-drop. The waveform drop target on
    // this same channel listens for this exact type — cross-channel drags
    // won't highlight or accept because the mime won't match.
    const dragMime = `application/x-fragmenta-fragment-ch${channelIndex}`;

    return (
        <Box sx={styles.fragmentHistoryPanel}>
            <Box sx={styles.fragmentHistoryHeader}>
                <Box component="span" sx={styles.fragmentHistoryHeaderText}>
                    Fragments
                </Box>
                {fragments.length > 0 && (
                    <Tooltip title="Clear fragment history" placement="top" arrow>
                        <IconButton
                            size="small"
                            onClick={() => setClearConfirmOpen(true)}
                            sx={styles.fragmentHistoryHeaderBtn}
                            aria-label="Clear all fragments"
                        >
                            <ClearAllIcon size={12} />
                        </IconButton>
                    </Tooltip>
                )}
            </Box>

            {fragments.length === 0 ? (
                <Box sx={styles.fragmentHistoryEmpty}>Empty</Box>
            ) : (
                <Box sx={styles.fragmentHistoryList}>
                    {fragments.map((fragment) => {
                        const isAuditioning = auditioningId === fragment.id;
                        const isCommitted = committedId === fragment.id;
                        return (
                            <Box
                                key={fragment.id}
                                draggable
                                onDragStart={(e) => {
                                    e.dataTransfer.setData(dragMime, fragment.id);
                                    e.dataTransfer.effectAllowed = 'copy';
                                }}
                                sx={{
                                    ...styles.fragmentRow(color, isCommitted, isAuditioning),
                                    cursor: 'grab',
                                    '&:active': { cursor: 'grabbing' },
                                }}
                            >
                                <MidiMappable
                                    id={`channel.${channelIndex}.fragment.${fragment.id}.audition`}
                                    label={`Ch ${channelIndex + 1} · Fragment ${fragment.number} audition`}
                                    kind="trigger"
                                    onChange={() => onAudition(fragment.id)}
                                >
                                    <Tooltip
                                        title={isAuditioning ? 'Stop cue' : 'Audition through cue output'}
                                        placement="top"
                                        arrow
                                        enterDelay={300}
                                    >
                                        <IconButton
                                            size="small"
                                            onClick={() => onAudition(fragment.id)}
                                            sx={styles.fragmentIconBtn(color, isAuditioning, true)}
                                            aria-label={isAuditioning ? 'Stop cue' : 'Audition'}
                                        >
                                            {isAuditioning
                                                ? <StopIcon size={12} />
                                                : <PlayIcon size={12} />}
                                        </IconButton>
                                    </Tooltip>
                                </MidiMappable>

                                <Box sx={styles.fragmentMeta}>
                                    <Box component="span" sx={styles.fragmentOrdinal}>
                                        F{fragment.number}
                                    </Box>
                                </Box>

                                <Tooltip
                                    title={fragment.starred ? 'Unstar' : 'Star (keep through eviction)'}
                                    placement="top"
                                    arrow
                                    enterDelay={300}
                                >
                                    <IconButton
                                        size="small"
                                        onClick={() => onToggleStar(fragment.id)}
                                        sx={styles.fragmentIconBtn(color, fragment.starred)}
                                        aria-label={fragment.starred ? 'Unstar fragment' : 'Star fragment'}
                                    >
                                        <StarIcon
                                            size={12}
                                            fill={fragment.starred ? color : 'none'}
                                            strokeWidth={2}
                                        />
                                    </IconButton>
                                </Tooltip>

                                <Tooltip title="Delete fragment" placement="top" arrow enterDelay={300}>
                                    <IconButton
                                        size="small"
                                        onClick={() => onDelete(fragment.id)}
                                        sx={styles.fragmentDeleteBtn}
                                        aria-label="Delete fragment"
                                    >
                                        <DeleteIcon size={12} />
                                    </IconButton>
                                </Tooltip>

                                <Tooltip
                                    title={isCommitted ? 'Currently loaded' : 'Load into channel'}
                                    placement="top"
                                    arrow
                                    enterDelay={300}
                                >
                                    <span>
                                        <IconButton
                                            size="small"
                                            onClick={() => onCommit(fragment.id)}
                                            disabled={isCommitted}
                                            sx={styles.fragmentIconBtn(color, isCommitted, true)}
                                            aria-label="Load fragment into channel"
                                        >
                                            <CommitIcon size={12} strokeWidth={isCommitted ? 3 : 2} />
                                        </IconButton>
                                    </span>
                                </Tooltip>
                            </Box>
                        );
                    })}
                </Box>
            )}

            <Dialog open={clearConfirmOpen} onClose={() => setClearConfirmOpen(false)}>
                <DialogTitle>Clear fragment history?</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Removes all {fragments.length} fragments from this channel's history,
                        including starred ones. The currently loaded clip stays loaded
                        — only the history entries are dropped.
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setClearConfirmOpen(false)}>Cancel</Button>
                    <Button
                        onClick={() => { setClearConfirmOpen(false); onClearAll?.(); }}
                        color="error"
                        variant="contained"
                    >
                        Clear all
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}
