import React, { useEffect, useState } from 'react';
import {
    Box,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    MenuItem,
    Select,
    Tab,
    Tabs,
    Typography,
} from '@mui/material';
import { Info as InfoIcon } from 'lucide-react';
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
                {devices.map((d) => (
                    <MenuItem key={d.deviceId} value={d.deviceId}>
                        {d.label || `Output (${d.deviceId.slice(0, 6)}…)`}
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
 * App settings (the gear in the top toolbar): audio output and MIDI, with
 * About one click away.
 */
export default function SettingsDialog({ open, onClose, onAbout }) {
    const [tab, setTab] = useState(0);
    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle>Settings</DialogTitle>
            <DialogContent>
                <Tabs
                    value={tab}
                    onChange={(_, v) => setTab(v)}
                    sx={{ mb: 2.5, minHeight: 0 }}
                >
                    <Tab label="Audio" />
                    <Tab label="MIDI" />
                </Tabs>
                {tab === 0 && <AudioSettingsPanel open={open} />}
                {tab === 1 && <MidiSettingsPanel open={open} />}
            </DialogContent>
            <DialogActions>
                <Button variant="text" startIcon={<InfoIcon size={16} />} onClick={onAbout}>
                    About Fragmenta
                </Button>
                <Box sx={{ flex: 1 }} />
                <Button variant="outlined" onClick={onClose}>Done</Button>
            </DialogActions>
        </Dialog>
    );
}
