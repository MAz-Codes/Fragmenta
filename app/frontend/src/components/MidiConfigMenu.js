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
            anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
            transformOrigin={{ vertical: 'top', horizontal: 'left' }}
            slotProps={{
                paper: {
                    sx: {
                        width: 360,
                        maxHeight: '70vh',
                        p: 0,
                        borderRadius: 2,
                        border: '1px solid',
                        borderColor: 'divider',
                        overflow: 'hidden',
                    },
                },
            }}
        >
            {/* Title bar — same pattern as Presets / Audio menus. */}
            <Box sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                px: 1.5,
                pt: 1.25,
                pb: 1,
            }}>
                <Typography sx={{ ...perfTokens.caps, color: 'text.secondary' }}>
                    MIDI Settings
                </Typography>
                <IconButton onClick={onClose} sx={panelStyles.compactIconBtn('md')}>
                    <CloseIcon size={perfTokens.icon.sm} />
                </IconButton>
            </Box>

            <Divider />

            {!supported && (
                <Box sx={{ px: 1.5, pt: 1.25 }}>
                    <Alert severity="warning" sx={{ py: 0.5 }}>
                        {permissionError || 'Web MIDI is not available in this browser. Try Chrome / Edge / Electron.'}
                    </Alert>
                </Box>
            )}

            {/* SETTINGS — input device + channel filter + takeover. */}
            <Box sx={{
                px: 1.5,
                pt: 1.25,
                pb: 1.25,
                display: 'flex',
                flexDirection: 'column',
                gap: 1.25,
            }}>
                <Box>
                    <Typography sx={{ ...perfTokens.labelMuted, display: 'block', mb: 0.5 }}>
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
                        <Typography sx={{
                            fontSize: perfTokens.fontSize.xs,
                            color: 'warning.main',
                            fontStyle: 'italic',
                            display: 'block',
                            mt: 0.5,
                        }}>
                            Saved device "{config.deviceName}" not connected
                        </Typography>
                    )}
                </Box>

                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>
                        <Typography sx={{ ...perfTokens.labelMuted, display: 'block', mb: 0.5 }}>
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
                        <Typography sx={{ ...perfTokens.labelMuted, display: 'block', mb: 0.5 }}>
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
            </Box>

            <Divider />

            {/* MAPPINGS — header row + bordered scrollable list. */}
            <Box sx={{ px: 1.5, pt: 1.25, pb: 1.25 }}>
                <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    mb: 0.75,
                }}>
                    <Typography sx={{ ...perfTokens.labelMuted, display: 'block' }}>
                        Mappings ({config.mappings.length})
                    </Typography>
                    <Button
                        size="small"
                        onClick={clearAll}
                        disabled={config.mappings.length === 0}
                        sx={{
                            fontSize: perfTokens.fontSize.xs,
                            color: 'error.main',
                            textTransform: 'none',
                            py: 0,
                            px: 0.75,
                            minWidth: 0,
                            '&:hover': { bgcolor: 'action.hover' },
                            '&.Mui-disabled': { color: 'text.disabled' },
                        }}
                    >
                        Clear all
                    </Button>
                </Box>

                <Box
                    sx={{
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        maxHeight: 240,
                        overflowY: 'auto',
                        bgcolor: 'background.default',
                    }}
                >
                    {sortedMappings.length === 0 ? (
                        <Box sx={{ px: 1.5, py: 1.5, textAlign: 'center' }}>
                            <Typography sx={{
                                fontSize: perfTokens.fontSize.xs,
                                color: 'text.disabled',
                                fontStyle: 'italic',
                                lineHeight: 1.4,
                            }}>
                                No mappings yet. Enable MIDI mode, click a control,
                                then move a hardware knob, fader, or button.
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
                                    '&:hover': { bgcolor: 'action.hover' },
                                    transition: 'background-color 120ms',
                                }}
                            >
                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                    <Typography sx={{
                                        fontSize: perfTokens.fontSize.sm,
                                        fontWeight: 500,
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap',
                                    }}>
                                        {m.label}
                                    </Typography>
                                    <Typography sx={{
                                        color: 'text.secondary',
                                        fontSize: perfTokens.fontSize.xs,
                                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                                    }}>
                                        {formatMidi(m.midi)}
                                    </Typography>
                                </Box>
                                <IconButton
                                    size="small"
                                    onClick={() => clearMapping(m.controlId)}
                                    sx={panelStyles.compactIconBtn('sm', 'danger')}
                                    aria-label="Remove mapping"
                                >
                                    <DeleteIcon size={perfTokens.icon.sm} />
                                </IconButton>
                            </Box>
                        ))
                    )}
                </Box>
            </Box>

            <Divider />

            {/* Footer help text. */}
            <Box sx={{ px: 1.5, pt: 1, pb: 1.25 }}>
                <Typography sx={{
                    color: 'text.disabled',
                    fontSize: perfTokens.fontSize.xs,
                    fontStyle: 'italic',
                    lineHeight: 1.4,
                }}>
                    Pickup ignores the hardware until its position matches the on-screen
                    value. Right-click a control while in MIDI mode to clear its mapping.
                </Typography>
            </Box>
        </Popover>
    );
}
