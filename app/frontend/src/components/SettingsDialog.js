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
import { Info as InfoIcon, Volume2 as AudioIcon, Piano as MidiIcon, FolderInput as ImportIcon } from 'lucide-react';
import { AboutPanel } from './AboutDialog';
import ImportPanel from './ImportPanel';
import MidiSettingsPanel from './MidiSettingsPanel';
import UninstallDialog from './UninstallDialog';
import { useAudioSettings } from './AudioSettingsContext';
import api from '../api';
import { appStyles } from '../theme';

// How often an open Settings tab looks for devices plugged in or removed.
const DEVICE_POLL_MS = 3000;

function AudioSettingsPanel({ open }) {
    const { api } = useAudioSettings();
    const refresh = api?.refresh;
    const rescan = api?.rescan;
    // Devices plugged in mid-session only show up after a fresh enumeration.
    useEffect(() => { if (open) refresh?.(); }, [open, refresh]);
    useEffect(() => {
        if (!open || !rescan) return undefined;
        const timer = setInterval(rescan, DEVICE_POLL_MS);
        const media = navigator.mediaDevices;
        media?.addEventListener?.('devicechange', rescan);
        return () => {
            clearInterval(timer);
            media?.removeEventListener?.('devicechange', rescan);
        };
    }, [open, rescan]);

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
    // empty id; that's the "System device (default)" entry already. Chromium's
    // 'default' and 'communications' are aliases of devices listed anyway.
    const devices = api.devices.filter((d) => d.deviceId
        && d.deviceId !== 'default' && d.deviceId !== 'communications');
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
                {!api.systemOutput && <MenuItem value="">System device (default)</MenuItem>}
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
            <Button size="small" onClick={() => refresh?.()} sx={{ mt: 1 }}>Refresh devices</Button>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                {api.systemOutput
                    ? "The Mac's output until Fragmenta closes. Main and cue pairs: master strip."
                    : 'Main and cue channel pairs are chosen on the Performance master strip.'}
            </Typography>
        </Box>
    );
}

/**
 * Settings (the gear in the top toolbar): About first, exactly as the old
 * About dialog, then Audio and MIDI, as tabs down the left side.
 */
export default function SettingsDialog({ open, onClose, aboutProps, showImport = false, initialTab = 0, fade }) {
    const [tab, setTab] = useState(0);
    useEffect(() => { if (open) setTab(initialTab); }, [open, initialTab]);
    // Only the installed app can uninstall itself; source installs never see the button.
    const [canUninstall, setCanUninstall] = useState(false);
    const [uninstalling, setUninstalling] = useState(false);
    useEffect(() => {
        if (!open) return undefined;
        let cancelled = false;
        api.get('/api/uninstall')
            .then(({ data }) => { if (!cancelled) setCanUninstall(Boolean(data.available)); })
            .catch(() => {});
        return () => { cancelled = true; };
    }, [open]);
    return (
        // Fixed at the About page's size: every tab gets the same window, and
        // longer pages (the MIDI mappings) scroll inside it.
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="md"
            fullWidth
            aria-label="Settings"
            {...fade}
            PaperProps={{ sx: { height: 'min(660px, calc(100vh - 64px))' } }}
        >
            <DialogContent sx={{ display: 'flex', gap: 3, pt: 3, overflow: 'hidden' }}>
                <Box sx={{ flexShrink: 0, width: 200, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                    {/* The same card and tabs as the app's left sidebar. */}
                    <Box sx={(t) => ({
                        borderRadius: '24px',
                        backgroundColor: t.palette.fragmenta.surface,
                        boxShadow: t.palette.fragmenta.shadowSoft,
                    })}>
                        <Tabs
                            orientation="vertical"
                            value={tab}
                            onChange={(_, v) => setTab(v)}
                            sx={appStyles.navigationTabs(false, false)}
                        >
                            <Tab icon={<InfoIcon size={20} />} iconPosition="start" label="About" />
                            <Tab icon={<AudioIcon size={20} />} iconPosition="start" label="Audio" />
                            <Tab icon={<MidiIcon size={20} />} iconPosition="start" label="MIDI" />
                            {showImport && <Tab icon={<ImportIcon size={20} />} iconPosition="start" label="Import" />}
                        </Tabs>
                    </Box>
                    {canUninstall && (
                        <Button variant="contained" color="error"
                            onClick={() => setUninstalling(true)}
                            sx={(t) => ({
                                alignSelf: 'center',
                                minHeight: 44,
                                px: 2.5,
                                fontWeight: 600,
                                backgroundColor: t.palette.error.main,
                                color: '#FFFFFF',
                                '&:hover': { backgroundColor: t.palette.fragmenta.errorLo },
                            })}>
                            Uninstall Fragmenta
                        </Button>
                    )}
                </Box>
                <UninstallDialog open={uninstalling} onClose={() => setUninstalling(false)} />
                <Box sx={{ flex: 1, minWidth: 0, height: '100%', overflowY: 'auto', pr: 0.5 }}>
                    {tab === 0 && <AboutPanel {...aboutProps} />}
                    {tab === 1 && <AudioSettingsPanel open={open} />}
                    {tab === 2 && <MidiSettingsPanel open={open} />}
                    {tab === 3 && showImport && <ImportPanel />}
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
}
