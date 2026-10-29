import React, { useEffect, useState } from 'react';
import {
    Box,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    MenuItem,
    Select,
    Tab,
    Tabs,
    Typography,
} from '@mui/material';
import { Info as InfoIcon, Volume2 as AudioIcon, Piano as MidiIcon } from 'lucide-react';
import { AboutPanel } from './AboutDialog';
import MidiSettingsPanel from './MidiSettingsPanel';
import { useAudioSettings } from './AudioSettingsContext';
import { appStyles } from '../theme';

function AudioSettingsPanel({ open }) {
    const { api } = useAudioSettings();
    const refresh = api?.refresh;
    // Devices plugged in mid-session only show up after a fresh enumeration.
    useEffect(() => { if (open) refresh?.(); }, [open, refresh]);

    if (!api) {
        return (
            <Typography variant="body2" color="text.secondary">
                Audio output is set up once the Performance engine has started.
            </Typography>
        );
    }
    if (!api.supported) {
        return (
            <Typography variant="body2" color="text.secondary">
                Choosing an output device needs a Chromium-based engine; audio plays on the system device.
            </Typography>
        );
    }
    // Without a media permission the browser lists the system device with an
    // empty id; that's the "System device (default)" entry already.
    const devices = api.devices.filter((d) => d.deviceId);
    const known = api.deviceId === '' || devices.some((d) => d.deviceId === api.deviceId);
    return (
        <Box>
            <Typography component="div" sx={[appStyles.fieldCaps, { mb: 0.75 }]}>
                Output device
            </Typography>
            <Select
                fullWidth
                size="small"
                displayEmpty
                value={known ? api.deviceId : ''}
                onChange={(e) => api.pick(e.target.value)}
            >
                <MenuItem value="">System device (default)</MenuItem>
                {devices.map((d, i) => (
                    <MenuItem key={d.deviceId} value={d.deviceId}>
                        {d.label || `Output ${i + 1}`}
                    </MenuItem>
                ))}
            </Select>
            {devices.length === 0 && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                    No additional output devices detected.
                </Typography>
            )}
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                Main and cue channel pairs are chosen in the Performance toolbar.
            </Typography>
        </Box>
    );
}

/**
 * Settings (the gear in the top toolbar): About first, exactly as the old
 * About dialog, then Audio and MIDI, as tabs down the left side.
 */
export default function SettingsDialog({ open, onClose, aboutProps }) {
    const [tab, setTab] = useState(0);
    useEffect(() => { if (open) setTab(0); }, [open]);
    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth aria-label="Settings">
            <DialogContent sx={{ display: 'flex', gap: 3, pt: 3, minHeight: 520 }}>
                <Tabs
                    orientation="vertical"
                    value={tab}
                    onChange={(_, v) => setTab(v)}
                    sx={{
                        flexShrink: 0,
                        width: 150,
                        '& .MuiTabs-indicator': { display: 'none' },
                        '& .MuiTabs-flexContainer': { gap: 0.5 },
                        '& .MuiTab-root': {
                            minHeight: 40,
                            justifyContent: 'flex-start',
                            gap: 1,
                            px: 1.5,
                            borderRadius: '999px',
                            textTransform: 'none',
                            fontSize: '0.9rem',
                            color: 'text.secondary',
                        },
                        '& .MuiTab-root.Mui-selected': {
                            color: 'text.primary',
                            backgroundColor: (t) => t.palette.fragmenta.navSelected,
                        },
                    }}
                >
                    <Tab icon={<InfoIcon size={16} />} iconPosition="start" label="About" />
                    <Tab icon={<AudioIcon size={16} />} iconPosition="start" label="Audio" />
                    <Tab icon={<MidiIcon size={16} />} iconPosition="start" label="MIDI" />
                </Tabs>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                    {tab === 0 && <AboutPanel {...aboutProps} />}
                    {tab === 1 && <AudioSettingsPanel open={open} />}
                    {tab === 2 && <MidiSettingsPanel open={open} />}
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
}
