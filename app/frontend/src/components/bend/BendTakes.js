import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, IconButton, List, ListItem } from '@mui/material';
import {
    Play as PlayIcon, Square as StopIcon, Info as InfoIcon,
    FolderOpen as RevealIcon, Download as DownloadIcon,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import GenerationWaveform from '../GenerationWaveform';
import api from '../../api';
import { TIPS } from '../../tooltips';
import { generatedFragmentsWindowStyles as fragStyles } from '../../theme';

/**
 * The bent / clean takes, as rows of the Generated Fragments list: round
 * play/stop, title, bar waveform with playhead, info tooltip, and
 * show-in-folder (download on Docker/web).
 *
 * One take plays at a time, and switching takes mid-playback carries the
 * position over — that is what makes it an A/B: the ear compares the same
 * moment, bent against clean.
 *
 * takes: [{ key, title, titleColor, color, url, blob, filename, seed, body }]
 */
export default function BendTakes({ takes, isDocker = false, onMessage }) {
    const audioRefs = useRef({});
    const [playing, setPlaying] = useState(null);      // take key
    const [time, setTime] = useState(0);
    const [durations, setDurations] = useState({});

    // A replaced take (new generation) stops if it was the one playing.
    const urls = takes.map(t => `${t.key}:${t.url}`).join('|');
    useEffect(() => {
        setPlaying(p => (p && takes.some(t => t.key === p) ? p : null));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [urls]);

    const toggle = (key) => {
        const audio = audioRefs.current[key];
        if (!audio) return;
        if (!audio.paused) {
            audio.pause();
            audio.currentTime = 0;
            setPlaying(null);
            setTime(0);
            return;
        }
        let from = 0;
        Object.entries(audioRefs.current).forEach(([k, el]) => {
            if (!el || k === key) return;
            if (!el.paused) from = el.currentTime;
            el.pause();
            el.currentTime = 0;
        });
        audio.currentTime = Number.isFinite(audio.duration)
            ? Math.min(from, Math.max(0, audio.duration - 0.05)) : from;
        setPlaying(key);
        setTime(audio.currentTime);
        Promise.resolve(audio.play()).catch((err) => {
            if (err?.name !== 'AbortError') console.warn('Bend take play failed:', err);
            setPlaying(p => (p === key ? null : p));
        });
    };

    const reveal = (t) => {
        api.post('/api/reveal-fragment', { filename: t.filename })
            .then((res) => {
                if (res.data?.success === false && res.data?.message) onMessage?.(res.data.message);
            })
            .catch(() => onMessage?.(`Could not reveal ${t.filename}.`));
    };
    const download = (t) => {
        const a = document.createElement('a');
        a.href = t.url;
        a.download = t.filename || 'fragment.wav';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    return (
        <List sx={{ ...fragStyles.listRoot, flex: 'none', overflow: 'visible', mb: 1.5 }}>
            {takes.map((t) => {
                const isPlaying = playing === t.key;
                const b = t.body || {};
                const info = [
                    `Seed: ${t.seed}`,
                    b.model_id ? `Model: ${b.model_id}` : null,
                    b.steps ? `Steps: ${b.steps}` : null,
                    b.duration ? `Duration: ${b.duration}s` : null,
                    b.prompt ? `Prompt: ${b.prompt}` : null,
                ].filter(Boolean).join('\n');
                return (
                    <ListItem key={t.key} sx={fragStyles.listItem}>
                        <IconButton
                            size="small"
                            onClick={() => toggle(t.key)}
                            aria-label={isPlaying ? `Stop ${t.title}` : `Play ${t.title}`}
                            sx={fragStyles.playPauseButton(isPlaying)}
                        >
                            {isPlaying ? <StopIcon size={16} /> : <PlayIcon size={16} />}
                        </IconButton>
                        <Box sx={fragStyles.fragmentMeta}>
                            <Typography variant="body2" sx={fragStyles.fragmentPrompt}>
                                <Box component="span" sx={{ color: t.titleColor || 'text.primary' }}>
                                    {t.title}
                                </Box>
                                <Box component="span" sx={{ color: 'text.secondary', fontWeight: 400 }}>
                                    {` · seed ${t.seed}`}
                                </Box>
                            </Typography>
                        </Box>
                        {/* Same slot width as a Generated Fragments row, so
                            the waveform reads the same (thin bars). */}
                        <Box sx={{ display: 'flex', width: { xs: 140, sm: 200 }, flexShrink: 0 }}>
                            <GenerationWaveform
                                blob={t.blob}
                                audioUrl={t.url}
                                filename={t.filename || 'fragment.wav'}
                                currentTime={isPlaying ? time : 0}
                                duration={durations[t.key] || 0}
                                color={t.color}
                            />
                        </Box>
                        <Tooltip
                            title={<Box component="span" sx={{ whiteSpace: 'pre-line' }}>{info}</Box>}
                            arrow placement="top"
                        >
                            <Box component="span" sx={fragStyles.fragmentInfoIcon}>
                                <InfoIcon size={14} />
                            </Box>
                        </Tooltip>
                        {t.filename && (
                            <Tooltip title={isDocker ? TIPS.fragments.download : TIPS.fragments.revealInFolder}
                                     placement="top" arrow>
                                <IconButton
                                    size="small"
                                    onClick={() => (isDocker ? download(t) : reveal(t))}
                                    aria-label={isDocker ? 'Download fragment' : 'Show in folder'}
                                    sx={{ color: 'text.disabled', '&:hover': { color: 'primary.main', bgcolor: 'action.hover' } }}
                                >
                                    {isDocker ? <DownloadIcon size={16} /> : <RevealIcon size={16} />}
                                </IconButton>
                            </Tooltip>
                        )}
                        <audio
                            key={t.url}
                            ref={(el) => {
                                if (el) audioRefs.current[t.key] = el;
                                else delete audioRefs.current[t.key];
                            }}
                            src={t.url}
                            preload="auto"
                            onLoadedMetadata={(e) => {
                                const d = e.target.duration;
                                setDurations(prev => ({ ...prev, [t.key]: Number.isFinite(d) ? d : 0 }));
                            }}
                            onTimeUpdate={(e) => { if (isPlaying) setTime(e.target.currentTime); }}
                            // Functional: the take being switched away from
                            // fires its pause after the new one took over.
                            onPause={() => setPlaying(p => (p === t.key ? null : p))}
                            onEnded={() => {
                                setPlaying(p => (p === t.key ? null : p));
                                setTime(0);
                            }}
                            style={fragStyles.hiddenAudio}
                        />
                    </ListItem>
                );
            })}
        </List>
    );
}
