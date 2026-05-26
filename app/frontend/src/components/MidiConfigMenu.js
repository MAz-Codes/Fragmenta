import React from 'react';
import {
    Popover,
    Box,
    Typography,
    FormControl,
    Select,
    MenuItem,
    Button,
    IconButton,
    Tooltip,
    Divider,
    ToggleButton,
    ToggleButtonGroup,
    Alert,
} from '@mui/material';
import { Trash2 as DeleteIcon, X as CloseIcon } from 'lucide-react';
import { useMidi, formatMidi } from './MidiContext';
import { perfTokens, performancePanelStyles as panelStyles } from '../theme';

const CHANNEL_OPTIONS = [
    { value: 0, label: 'Any' },
    ...Array.from({ length: 16 }, (_, i) => ({ value: i + 1, label: `Ch ${i + 1}` })),
];

export default function MidiConfigMenu({ anchorEl, open, onClose }) {
    const ctx = useMidi();
    if (!ctx) return null;

    const {
        config,
        inputs,
        supported,
        permissionError,
        setDevice,
        setChannelFilter,
        setTakeover,
        clearMapping,
        clearAll,
    } = ctx;

    const sortedMappings = [...config.mappings].sort((a, b) => a.label.localeCompare(b.label));

    return (
        <Popover
            anchorEl={anchorEl}
            open={open}
            onClose={onClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
            slotProps={{
                paper: {
                    sx: {
                        width: 380,
                        maxHeight: '70vh',
                        p: 2,
                        borderRadius: 2,
                        border: '1px solid',
                        borderColor: 'divider',
                    },
                },
            }}
        >
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography sx={{ ...perfTokens.caps, color: 'text.secondary' }}>
                    MIDI Settings
                </Typography>
                <IconButton size="small" onClick={onClose} sx={panelStyles.compactIconBtn('lg')}>
                    <CloseIcon size={perfTokens.icon.md} />
                </IconButton>
            </Box>

            {!supported && (
                <Alert severity="warning" sx={{ mb: 1.5 }}>
                    {permissionError || 'Web MIDI is not available in this browser. Try Chrome / Edge / Electron.'}
                </Alert>
            )}

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Box>
                    <Typography sx={{ ...perfTokens.labelMuted, color: 'text.secondary', display: 'block', mb: 0.5 }}>
                        Input device
                    </Typography>
                    <FormControl size="small" fullWidth>
                        <Select
                            value={config.deviceId && inputs.some(i => i.id === config.deviceId) ? config.deviceId : ''}
                            onChange={(e) => setDevice(e.target.value || null)}
                            displayEmpty
                            disabled={!supported}
                            renderValue={(value) => {
                                if (!value) return <em style={{ opacity: 0.6 }}>None</em>;
                                const found = inputs.find(i => i.id === value);
                                return found ? found.name : 'Disconnected';
                            }}
                            sx={{ fontSize: perfTokens.fontSize.sm }}
                        >
                            <MenuItem value="" sx={{ fontSize: perfTokens.fontSize.sm }}>
                                <em>None</em>
                            </MenuItem>
                            {inputs.map((input) => (
                                <MenuItem key={input.id} value={input.id} sx={{ fontSize: perfTokens.fontSize.sm }}>
                                    {input.name}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    {config.deviceName && !inputs.some(i => i.name === config.deviceName) && (
                        <Typography sx={{ fontSize: perfTokens.fontSize.xs, color: 'warning.main', display: 'block', mt: 0.5 }}>
                            Saved device "{config.deviceName}" not connected
                        </Typography>
                    )}
                </Box>

                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>
                        <Typography sx={{ ...perfTokens.labelMuted, color: 'text.secondary', display: 'block', mb: 0.5 }}>
                            Channel filter
                        </Typography>
                        <FormControl size="small" fullWidth>
                            <Select
                                value={config.channelFilter}
                                onChange={(e) => setChannelFilter(Number(e.target.value))}
                                disabled={!supported}
                                sx={{ fontSize: perfTokens.fontSize.sm }}
                            >
                                {CHANNEL_OPTIONS.map(opt => (
                                    <MenuItem key={opt.value} value={opt.value} sx={{ fontSize: perfTokens.fontSize.sm }}>{opt.label}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>

                    <Box sx={{ flex: 1 }}>
                        <Typography sx={{ ...perfTokens.labelMuted, color: 'text.secondary', display: 'block', mb: 0.5 }}>
                            Takeover
                        </Typography>
                        <ToggleButtonGroup
                            size="small"
                            value={config.takeover}
                            exclusive
                            onChange={(_, v) => { if (v) setTakeover(v); }}
                            fullWidth
                            sx={{ height: perfTokens.height.compact }}
                        >
                            <ToggleButton value="jump" sx={{ fontSize: perfTokens.fontSize.sm, textTransform: 'none' }}>Jump</ToggleButton>
                            <ToggleButton value="pickup" sx={{ fontSize: perfTokens.fontSize.sm, textTransform: 'none' }}>Pickup</ToggleButton>
                        </ToggleButtonGroup>
                    </Box>
                </Box>

                <Divider sx={{ my: 0.5 }} />

                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography sx={{ ...perfTokens.caps, color: 'text.secondary' }}>
                        Mappings ({config.mappings.length})
                    </Typography>
                    <Button
                        size="small"
                        onClick={clearAll}
                        disabled={config.mappings.length === 0}
                        sx={{ fontSize: perfTokens.fontSize.sm, textTransform: 'none' }}
                    >
                        Clear all
                    </Button>
                </Box>

                <Box
                    sx={{
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        maxHeight: 280,
                        overflowY: 'auto',
                        bgcolor: 'background.default',
                    }}
                >
                    {sortedMappings.length === 0 ? (
                        <Box sx={{ p: 2, textAlign: 'center' }}>
                            <Typography sx={{ fontSize: perfTokens.fontSize.xs, color: 'text.disabled', fontStyle: 'italic' }}>
                                No mappings yet. Enable MIDI mode (the MIDI button), click a control, then move a hardware knob, fader, or button.
                            </Typography>
                        </Box>
                    ) : (
                        sortedMappings.map((m) => (
                            <Box
                                key={m.controlId}
                                sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 1,
                                    px: 1,
                                    py: 0.6,
                                    borderBottom: '1px solid',
                                    borderColor: 'divider',
                                    '&:last-child': { borderBottom: 'none' },
                                }}
                            >
                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                    <Typography sx={{ fontSize: perfTokens.fontSize.sm, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {m.label}
                                    </Typography>
                                    <Typography sx={{ color: 'text.secondary', fontSize: perfTokens.fontSize.xs, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' }}>
                                        {formatMidi(m.midi)}
                                    </Typography>
                                </Box>
                                <Tooltip title="Remove mapping">
                                    <IconButton
                                        size="small"
                                        onClick={() => clearMapping(m.controlId)}
                                        sx={panelStyles.compactIconBtn('sm', 'danger')}
                                    >
                                        <DeleteIcon size={perfTokens.icon.sm} />
                                    </IconButton>
                                </Tooltip>
                            </Box>
                        ))
                    )}
                </Box>

                <Typography sx={{ color: 'text.disabled', fontSize: perfTokens.fontSize.xs, lineHeight: 1.4 }}>
                    Pickup = ignore the hardware until its position matches the on-screen value (no jumps).
                    Right-click a control while in MIDI mode to clear its mapping.
                </Typography>
            </Box>
        </Popover>
    );
}
