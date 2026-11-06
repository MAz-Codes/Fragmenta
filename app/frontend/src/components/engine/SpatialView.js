import React from 'react';
import { Box, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';

/**
 * Top-down view of the listening space: the listener in the middle, beds as
 * rings, events as dots at their azimuth (0 = front, positive = left) and
 * distance. Fed by the engine's stream; nothing here is interactive.
 */
const MAX_DIST = 12;

export default function SpatialView({ beds = {}, events = [], size = 260 }) {
    const theme = useTheme();
    const ink = theme.palette.text.primary;
    const faint = theme.palette.divider;
    const accent = theme.palette.primary.main;
    const warm = theme.palette.warm?.main || '#FFB123';
    const bedList = Object.entries(beds);

    const place = (az, dist) => {
        const r = 8 + Math.min(1, (dist || 1) / MAX_DIST) * 38;
        const a = ((az || 0) + 90) * Math.PI / 180;   // 0° front = up; positive = left
        return [50 - Math.cos(a) * r, 50 - Math.sin(a) * r];
    };

    return (
        <Box sx={{ width: '100%', maxWidth: size }}>
            <Box sx={{
                position: 'relative', width: '100%', aspectRatio: '1 / 1', borderRadius: '24px',
                background: theme.palette.background.paper, border: `1px solid ${faint}`,
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.55)',
            }}>
                <svg viewBox="0 0 100 100" width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
                    {[46, 31, 16].map((r) => <circle key={r} cx="50" cy="50" r={r} fill="none" stroke={faint} strokeWidth="0.5" />)}
                    <line x1="50" y1="4" x2="50" y2="96" stroke={faint} strokeWidth="0.3" />
                    <line x1="4" y1="50" x2="96" y2="50" stroke={faint} strokeWidth="0.3" />
                    {bedList.map(([key, b], i) => {
                        const g = Math.max(0, Math.min(1, b.gain || 0));
                        const [bx, by] = place(b.azimuth || 0, 1 + i * 0.6);
                        return (
                            <g key={key}>
                                <circle cx="50" cy="50" r={44 - i * 3} fill="none" stroke={accent} strokeWidth="1.4"
                                        strokeOpacity={0.15 + g * 0.5} strokeDasharray={b.state === 'receding' ? '2 2' : undefined} />
                                {Math.abs(b.azimuth || 0) > 1 && <circle cx={bx} cy={by} r="2" fill={accent} fillOpacity="0.8" />}
                            </g>
                        );
                    })}
                    {events.map((e) => {
                        const [x, y] = place(e.azimuth, e.distance);
                        const r = 2.2 + 2.5 / Math.max(1, e.distance || 1);
                        return (
                            <g key={`${e.clip_id}-${e.t}`}>
                                <circle cx={x} cy={y} r={r} fill={warm} fillOpacity="0.85" stroke={ink} strokeOpacity="0.4" strokeWidth="0.4" />
                                <text x={x} y={y - r - 1.5} textAnchor="middle" fontSize="4" fill={ink} fillOpacity="0.8">{e.type}</text>
                            </g>
                        );
                    })}
                    <circle cx="50" cy="50" r="2.4" fill={ink} />
                    <path d="M50 44 L47.5 48 L52.5 48 Z" fill={ink} />
                </svg>
                <Typography variant="caption" sx={{ position: 'absolute', left: 10, bottom: 6, color: 'text.secondary', fontSize: 10 }}>
                    front ↑ · left is left
                </Typography>
            </Box>
        </Box>
    );
}
