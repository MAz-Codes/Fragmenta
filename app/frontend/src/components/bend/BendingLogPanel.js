import React, { useState } from 'react';
import {
    Box, Typography, IconButton, TextField, Chip,
} from '@mui/material';
import {
    RotateCcw as RecallIcon, Trash2 as TrashIcon, Check as CheckIcon,
    Pencil as PencilIcon,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import api from '../../api';
import { describeModule } from './bendUtils';

/**
 * The Bending Log (Kotowski & Font §3.3.1): every bent generation lands
 * here automatically; the user adds the one-line sonic-result note —
 * "smears the transients", "adds metallic ring" — turning exploration
 * into a reusable vocabulary. Any row recalls its full patch into the
 * rack.
 */
export default function BendingLogPanel({ entries, registry, onRecall, onChanged }) {
    const [editing, setEditing] = useState(null);      // entry id
    const [draft, setDraft] = useState('');

    const saveNote = async (id) => {
        try {
            await api.patch(`/api/bend/log/${encodeURIComponent(id)}`, { note: draft });
            onChanged();
        } catch { /* non-fatal */ }
        setEditing(null);
    };

    const remove = async (id) => {
        try {
            await api.delete(`/api/bend/log/${encodeURIComponent(id)}`);
            onChanged();
        } catch { /* non-fatal */ }
    };

    if (!entries.length) {
        return (
            <Typography variant="body2" color="textSecondary" sx={{ py: 2, textAlign: 'center' }}>
                No bends logged yet for this model. Every bent generation lands
                here — note down what it did to the sound, and recall any row
                back into the rack.
            </Typography>
        );
    }

    return (
        <Box sx={{ maxHeight: 340, overflowY: 'auto', pr: 0.5 }}>
            {entries.map((e) => (
                <Box key={e.id} sx={{
                    display: 'flex', alignItems: 'flex-start', gap: 1,
                    py: 1, px: 0.5, borderBottom: '1px solid', borderColor: 'divider',
                    '&:last-of-type': { borderBottom: 'none' },
                }}>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 0.25 }}>
                            {(e.patch?.modules || []).map((m, i) => (
                                <Chip key={i} size="small" variant="outlined"
                                      label={describeModule(m, registry)}
                                      sx={{ fontSize: '0.62rem', height: 20 }} />
                            ))}
                            <Typography variant="caption" sx={{ color: 'text.disabled', ml: 0.5 }}>
                                seed {e.seed}{e.fragment ? ` · ${e.fragment}` : ''}
                            </Typography>
                        </Box>
                        {editing === e.id ? (
                            <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
                                <TextField
                                    size="small" fullWidth autoFocus value={draft}
                                    placeholder="What did it do to the sound?"
                                    onChange={(ev) => setDraft(ev.target.value)}
                                    onKeyDown={(ev) => { if (ev.key === 'Enter') saveNote(e.id); }}
                                    inputProps={{ maxLength: 500 }}
                                />
                                <IconButton size="small" onClick={() => saveNote(e.id)}>
                                    <CheckIcon size={15} />
                                </IconButton>
                            </Box>
                        ) : (
                            <Typography
                                variant="body2"
                                onClick={() => { setEditing(e.id); setDraft(e.note || ''); }}
                                sx={{
                                    cursor: 'text',
                                    color: e.note ? 'text.primary' : 'text.disabled',
                                    fontStyle: e.note ? 'normal' : 'italic',
                                    display: 'flex', alignItems: 'center', gap: 0.5,
                                }}
                            >
                                {e.note || 'add a sonic-result note…'}
                                <PencilIcon size={11} style={{ opacity: 0.4 }} />
                            </Typography>
                        )}
                    </Box>
                    <Tooltip title="Recall this bend into the rack">
                        <IconButton size="small" onClick={() => onRecall(e)}>
                            <RecallIcon size={15} />
                        </IconButton>
                    </Tooltip>
                    <IconButton size="small" onClick={() => remove(e.id)}
                                sx={{ '&:hover': { color: 'error.main' } }}>
                        <TrashIcon size={15} />
                    </IconButton>
                </Box>
            ))}
        </Box>
    );
}
