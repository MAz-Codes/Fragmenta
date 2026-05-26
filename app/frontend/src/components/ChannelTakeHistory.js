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

/**
 * Per-channel rolling take history. Always visible (empty-state included)
 * so the user knows the strip exists. Chronological order — oldest at
 * the top, newest at the bottom; scrolls vertically when the list grows
 * past ~4 visible rows.
 *
 * Each row exposes four actions, all visible by default (no hover-reveal —
 * Performance use is fast, can't afford the discoverability tax):
 *   • Cue ▶/■   — audition through the cue output (separate from main mix)
 *   • Star ★/☆ — mark as a keeper. Starred takes survive the 50-cap
 *                  eviction; unstarred get dropped FIFO when over cap.
 *   • Delete ⌫  — remove this take from history (cancellable confirm not
 *                  shown for single deletes — the entry can be regenerated
 *                  or audition can be retriggered after a quick re-tap).
 *   • Load ✓   — commit this take to the channel strip (becomes the
 *                  audio the channel plays). Disabled while already loaded.
 *
 * Props:
 *   takes:          [{ id, audioUrl, blob, prompt, duration, createdAt,
 *                     starred, number }]
 *   color:          channel accent color
 *   auditioningId:  the id currently playing through cue, or null
 *   committedId:    the id currently loaded into the channel strip, or null
 *   maxTakes:       cap, default 50 (informational; eviction lives in parent)
 *   on{Audition,Commit,ToggleStar,Delete}:  (takeId) => void
 *   onClearAll:     () => void  (parent confirms separately — we still show
 *                   a confirm dialog here for the trash-everything action)
 */
export default function ChannelTakeHistory({
    takes,
    color,
    channelIndex,
    auditioningId,
    committedId,
    maxTakes = 50,
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
    const dragMime = `application/x-fragmenta-take-ch${channelIndex}`;

    return (
        <Box sx={styles.takeHistoryPanel}>
            <Box sx={styles.takeHistoryHeader}>
                <Box component="span" sx={styles.takeHistoryHeaderText}>
                    Takes
                </Box>
                {takes.length > 0 && (
                    <Tooltip title="Clear take history" placement="top" arrow>
                        <IconButton
                            size="small"
                            onClick={() => setClearConfirmOpen(true)}
                            sx={styles.takeHistoryHeaderBtn}
                            aria-label="Clear all takes"
                        >
                            <ClearAllIcon size={12} />
                        </IconButton>
                    </Tooltip>
                )}
            </Box>

            {takes.length === 0 ? (
                <Box sx={styles.takeHistoryEmpty}>Empty</Box>
            ) : (
                <Box sx={styles.takeHistoryList}>
                    {takes.map((take) => {
                        const isAuditioning = auditioningId === take.id;
                        const isCommitted = committedId === take.id;
                        return (
                            <Box
                                key={take.id}
                                draggable
                                onDragStart={(e) => {
                                    e.dataTransfer.setData(dragMime, take.id);
                                    e.dataTransfer.effectAllowed = 'copy';
                                }}
                                sx={{
                                    ...styles.takeRow(color, isCommitted, isAuditioning),
                                    cursor: 'grab',
                                    '&:active': { cursor: 'grabbing' },
                                }}
                            >
                                <Tooltip
                                    title={isAuditioning ? 'Stop cue' : 'Audition through cue output'}
                                    placement="top"
                                    arrow
                                    enterDelay={300}
                                >
                                    <IconButton
                                        size="small"
                                        onClick={() => onAudition(take.id)}
                                        sx={styles.takeIconBtn(color, isAuditioning)}
                                        aria-label={isAuditioning ? 'Stop cue' : 'Audition'}
                                    >
                                        {isAuditioning
                                            ? <StopIcon size={12} />
                                            : <PlayIcon size={12} />}
                                    </IconButton>
                                </Tooltip>

                                <Box sx={styles.takeMeta}>
                                    <Box component="span" sx={styles.takeOrdinal}>
                                        T{take.number}
                                    </Box>
                                </Box>

                                <Tooltip
                                    title={take.starred ? 'Unstar' : 'Star (keep through eviction)'}
                                    placement="top"
                                    arrow
                                    enterDelay={300}
                                >
                                    <IconButton
                                        size="small"
                                        onClick={() => onToggleStar(take.id)}
                                        sx={styles.takeIconBtn(color, take.starred)}
                                        aria-label={take.starred ? 'Unstar take' : 'Star take'}
                                    >
                                        <StarIcon
                                            size={12}
                                            fill={take.starred ? color : 'none'}
                                            strokeWidth={2}
                                        />
                                    </IconButton>
                                </Tooltip>

                                <Tooltip title="Delete take" placement="top" arrow enterDelay={300}>
                                    <IconButton
                                        size="small"
                                        onClick={() => onDelete(take.id)}
                                        sx={styles.takeDeleteBtn}
                                        aria-label="Delete take"
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
                                            onClick={() => onCommit(take.id)}
                                            disabled={isCommitted}
                                            sx={styles.takeIconBtn(color, isCommitted, true)}
                                            aria-label="Load take into channel"
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
                <DialogTitle>Clear take history?</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Removes all {takes.length} takes from this channel's history,
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
