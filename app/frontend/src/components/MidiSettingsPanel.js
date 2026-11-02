import React, { useEffect } from 'react';
import {
    Box,
    Typography,
    FormControl,
    Select,
    MenuItem,
    Button,
    Switch,
    IconButton,
    Divider,
    ToggleButton,
    ToggleButtonGroup,
    Alert,
} from '@mui/material';
import { Trash2 as DeleteIcon } from 'lucide-react';
import { useMidi, formatMidi } from './MidiContext';
import Tooltip from './Tooltip';
import { TIPS } from '../tooltips';
import { appStyles, perfTokens, performancePanelStyles as panelStyles } from '../theme';

// How often the open tab looks for controllers plugged in or removed.
const DEVICE_POLL_MS = 3000;

const CHANNEL_OPTIONS = [
    { value: 0, label: 'Any' },
    ...Array.from({ length: 16 }, (_, i) => ({ value: i + 1, label: `Ch ${i + 1}` })),
];

// The MIDI settings (devices, channel filter, takeover, mappings), shown as
// the MIDI tab of the app's Settings dialog.
export default function MidiSettingsPanel({ open = true }) {
    const ctx = useMidi();
    const refreshInputs = ctx?.refreshInputs;
    // Devices are otherwise only enumerated on provider mount, so a
    // controller plugged in mid-session would never show up until a page
    // reload. Re-enumerate when the menu opens, and keep looking while it is.
    useEffect(() => {
        if (!open || !refreshInputs) return undefined;
        refreshInputs();
        const timer = setInterval(refreshInputs, DEVICE_POLL_MS);
        return () => clearInterval(timer);
    }, [open, refreshInputs]);
    if (!ctx) return null;

    const {
        config,
        inputs,
        failedInputs,
        supported,
        permissionError,
        setDeviceEnabled,
        setChannelFilter,
        setTakeover,
        clearMapping,
        clearAll,
    } = ctx;

    const sortedMappings = [...config.mappings].sort((a, b) => a.label.localeCompare(b.label));
    // Every input plugged in, plus switched-on ones that aren't, so they can
    // be switched off.
    const devices = [
        ...inputs.map((i) => i.id),
        ...config.deviceIds.filter((id) => !inputs.some((i) => i.id === id)),
    ];

    return (
        <Box>
            {!supported && (
                <Box sx={{ px: 0, pt: 1.25 }}>
                    <Alert severity="warning" sx={{ py: 0.5 }}>
                        {permissionError || 'Web MIDI is not available in this browser. Try Chrome / Edge / Electron.'}
                    </Alert>
                </Box>
            )}

            {/* SETTINGS — input device + channel filter + takeover. */}
            <Box sx={{
                px: 0,
                pt: 1.25,
                pb: 1.25,
                display: 'flex',
                flexDirection: 'column',
                gap: 1.25,
            }}>
                <Box>
                    <Tooltip title={TIPS.midi.device}>
                        <Typography sx={{ ...appStyles.fieldCaps, display: 'inline-block', mb: 0.5 }}>
                            Input devices
                        </Typography>
                    </Tooltip>
                    <Box sx={{
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        bgcolor: 'background.default',
                    }}>
                        {devices.length === 0 ? (
                            <Typography sx={{
                                px: 1,
                                py: 1.5,
                                textAlign: 'center',
                                fontSize: perfTokens.fontSize.xs,
                                color: 'text.disabled',
                                fontStyle: 'italic',
                            }}>
                                No MIDI devices found. Plug one in and it shows up here.
                            </Typography>
                        ) : devices.map((id) => {
                            const on = config.deviceIds.includes(id);
                            const present = inputs.some((i) => i.id === id);
                            const status = !present
                                ? 'Not connected'
                                : on && failedInputs.includes(id)
                                    ? "Couldn't open it. Is another app using it?"
                                    : null;
                            return (
                                <Box
                                    key={id}
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 1,
                                        pl: 1,
                                        pr: 0.5,
                                        py: 0.4,
                                        borderBottom: '1px solid',
                                        borderColor: 'divider',
                                        '&:last-child': { borderBottom: 'none' },
                                    }}
                                >
                                    <Box sx={{ flex: 1, minWidth: 0 }}>
                                        <Typography sx={{
                                            fontSize: perfTokens.fontSize.sm,
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis',
                                            whiteSpace: 'nowrap',
                                        }}>
                                            {id}
                                        </Typography>
                                        {status && (
                                            <Typography sx={{
                                                fontSize: perfTokens.fontSize.xs,
                                                color: 'warning.main',
                                                fontStyle: 'italic',
                                            }}>
                                                {status}
                                            </Typography>
                                        )}
                                    </Box>
                                    <Switch
                                        size="small"
                                        checked={on}
                                        onChange={(e) => setDeviceEnabled(id, e.target.checked)}
                                        disabled={!supported}
                                        inputProps={{ 'aria-label': `Listen to ${id}` }}
                                    />
                                </Box>
                            );
                        })}
                    </Box>
                    <Button size="small" variant="outlined" onClick={() => refreshInputs?.()} sx={{ mt: 1 }}>Refresh devices</Button>
                </Box>

                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>
                        <Tooltip title={TIPS.midi.channelFilter}>
                            <Typography sx={{ ...appStyles.fieldCaps, display: 'inline-block', mb: 0.5 }}>
                                Channel filter
                            </Typography>
                        </Tooltip>
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
                        <Tooltip title={TIPS.midi.takeover}>
                            <Typography sx={{ ...appStyles.fieldCaps, display: 'inline-block', mb: 0.5 }}>
                                Takeover
                            </Typography>
                        </Tooltip>
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
            <Box sx={{ px: 0, pt: 1.25, pb: 1.25 }}>
                <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    mb: 0.75,
                }}>
                    <Tooltip title={TIPS.midi.mappings}>
                        <Typography sx={{ ...appStyles.fieldCaps, display: 'inline-block' }}>
                            Mappings ({config.mappings.length})
                        </Typography>
                    </Tooltip>
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
                        <Box sx={{ px: 0, py: 1.5, textAlign: 'center' }}>
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
            <Box sx={{ px: 0, pt: 1, pb: 1.25 }}>
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
        </Box>
    );
}
