import React, { useState } from 'react';
import {
    Box,
    IconButton,
    InputBase,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogContentText,
    DialogActions,
    Button,
} from '@mui/material';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import {
    Play as PlayIcon,
    Square as StopIcon,
    Star as StarIcon,
    Trash2 as DeleteIcon,
    Check as CommitIcon,
    Eraser as ClearAllIcon,
    ChevronUp as MoveUpIcon,
    ChevronDown as MoveDownIcon,
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
 * Two further affordances are hover/focus-revealed rather than always on —
 * they're housekeeping, not performance actions, and the row can't spare the
 * width for six permanent buttons across four channels:
 *   • ▴/▾       — move the row up or down. Ordering is purely presentational.
 *   • rename    — double-click the F# / label to type a name. Empty clears
 *                  back to the F# ordinal.
 *
 * Props:
 *   fragments:      [{ id, audioUrl, blob, prompt, duration, createdAt,
 *                     starred, number, label? }]
 *   color:          channel accent color
 *   auditioningId:  the id currently playing through cue, or null
 *   committedId:    the id currently loaded into the channel strip, or null
 *   maxFragments:   cap, default 50 (informational; eviction lives in parent)
 *   on{Audition,Commit,ToggleStar,Delete}:  (fragmentId) => void
 *   onRename:       (fragmentId, label) => void  ('' clears the label)
 *   onMove:         (fragmentId, delta) => void  (-1 up, +1 down)
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
    onRename,
    onMove,
    onClearAll,
}) {
    const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
    // Id of the row whose name is being edited, plus the in-progress text.
    // Kept here rather than in the parent so a rename never re-renders the
    // channel strip on every keystroke.
    const [editingId, setEditingId] = useState(null);
    const [draftLabel, setDraftLabel] = useState('');

    const startRename = (fragment) => {
        setEditingId(fragment.id);
        setDraftLabel(fragment.label || '');
    };
    const commitRename = () => {
        if (editingId != null) onRename?.(editingId, draftLabel);
        setEditingId(null);
        setDraftLabel('');
    };
    const cancelRename = () => {
        setEditingId(null);
        setDraftLabel('');
    };
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
                    <IconButton
                        size="small"
                        onClick={() => setClearConfirmOpen(true)}
                        sx={styles.fragmentHistoryHeaderBtn}
                        aria-label="Clear all fragments"
                    >
                        <ClearAllIcon size={12} />
                    </IconButton>
                )}
            </Box>

            {fragments.length === 0 ? (
                <Box sx={styles.fragmentHistoryEmpty}>Empty</Box>
            ) : (
                <Box sx={styles.fragmentHistoryList}>
                    {fragments.map((fragment, rowIndex) => {
                        const isAuditioning = auditioningId === fragment.id;
                        const isCommitted = committedId === fragment.id;
                        const isEditing = editingId === fragment.id;
                        const displayName = fragment.label || `F${fragment.number}`;
                        return (
                            <Box
                                key={fragment.id}
                                // Dragging is suspended mid-rename so selecting
                                // text in the input doesn't start a row drag.
                                draggable={!isEditing}
                                onDragStart={(e) => {
                                    e.dataTransfer.setData(dragMime, fragment.id);
                                    e.dataTransfer.effectAllowed = 'copy';
                                }}
                                sx={{
                                    ...styles.fragmentRow(color, isCommitted, isAuditioning),
                                    cursor: isEditing ? 'default' : 'grab',
                                    '&:active': { cursor: isEditing ? 'default' : 'grabbing' },
                                }}
                            >
                                <Box className="frag-reorder" sx={styles.fragmentReorderCol}>
                                    <Tooltip title={TIPS.fragments.moveUp} placement="top" arrow enterDelay={400}>
                                        <span>
                                            <IconButton
                                                size="small"
                                                onClick={() => onMove?.(fragment.id, -1)}
                                                disabled={rowIndex === 0}
                                                sx={styles.fragmentReorderBtn}
                                                aria-label="Move fragment up"
                                            >
                                                <MoveUpIcon size={10} />
                                            </IconButton>
                                        </span>
                                    </Tooltip>
                                    <Tooltip title={TIPS.fragments.moveDown} placement="top" arrow enterDelay={400}>
                                        <span>
                                            <IconButton
                                                size="small"
                                                onClick={() => onMove?.(fragment.id, 1)}
                                                disabled={rowIndex === fragments.length - 1}
                                                sx={styles.fragmentReorderBtn}
                                                aria-label="Move fragment down"
                                            >
                                                <MoveDownIcon size={10} />
                                            </IconButton>
                                        </span>
                                    </Tooltip>
                                </Box>

                                <MidiMappable
                                    id={`channel.${channelIndex}.fragment.${fragment.id}.audition`}
                                    label={`Ch ${channelIndex + 1} · ${displayName} audition`}
                                    kind="trigger"
                                    onChange={() => onAudition(fragment.id)}
                                >
                                    <Tooltip
                                        title={TIPS.fragments.audition(isAuditioning)}
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
                                    {isEditing ? (
                                        <InputBase
                                            value={draftLabel}
                                            autoFocus
                                            fullWidth
                                            placeholder={`F${fragment.number}`}
                                            inputProps={{ maxLength: 40, 'aria-label': 'Fragment name' }}
                                            onChange={(e) => setDraftLabel(e.target.value)}
                                            onBlur={commitRename}
                                            onKeyDown={(e) => {
                                                // Enter commits, Escape reverts. Both stop
                                                // propagation so the panel's transport
                                                // keyboard shortcuts don't fire mid-typing.
                                                e.stopPropagation();
                                                if (e.key === 'Enter') { e.preventDefault(); commitRename(); }
                                                else if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
                                            }}
                                            sx={styles.fragmentLabelInput}
                                        />
                                    ) : (
                                        <Tooltip
                                            title={TIPS.fragments.rename}
                                            placement="top"
                                            arrow
                                            enterDelay={600}
                                        >
                                            <Box
                                                component="span"
                                                onDoubleClick={() => startRename(fragment)}
                                                sx={fragment.label
                                                    ? styles.fragmentLabelText
                                                    : { ...styles.fragmentOrdinal, cursor: 'text' }}
                                            >
                                                {displayName}
                                            </Box>
                                        </Tooltip>
                                    )}
                                </Box>

                                <Tooltip
                                    title={TIPS.fragments.star(fragment.starred)}
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

                                <IconButton
                                    size="small"
                                    onClick={() => onDelete(fragment.id)}
                                    sx={styles.fragmentDeleteBtn}
                                    aria-label="Delete fragment"
                                >
                                    <DeleteIcon size={12} />
                                </IconButton>

                                <Tooltip
                                    title={TIPS.fragments.commit(isCommitted)}
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
