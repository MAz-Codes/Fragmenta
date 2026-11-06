import React, { useCallback, useMemo, useRef, useState } from 'react';
import { Box, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';

/**
 * The blend pad: anchors sit on a plane, one point is dragged between them.
 * Weights are inverse-distance-squared, normalised, so sitting on an anchor
 * gives it everything and the middle gives a true mixture. The pad emits
 * {anchor: weight}; the engine does the embedding maths.
 */
function weightsAt(anchors, x, y) {
    const raw = anchors.map((a) => {
        const dx = x - a.position[0];
        const dy = y - a.position[1];
        const d2 = dx * dx + dy * dy;
        return d2 < 1e-5 ? Infinity : 1 / (d2 * d2);
    });
    if (raw.some((r) => r === Infinity)) {
        return Object.fromEntries(anchors.map((a, i) => [a.id, raw[i] === Infinity ? 1 : 0]));
    }
    const sum = raw.reduce((s, r) => s + r, 0) || 1;
    return Object.fromEntries(anchors.map((a, i) => [a.id, raw[i] / sum]));
}

function pointFor(anchors, blend) {
    let x = 0; let y = 0; let w = 0;
    anchors.forEach((a) => {
        const v = blend?.[a.id] || 0;
        x += a.position[0] * v;
        y += a.position[1] * v;
        w += v;
    });
    return w > 0 ? [x / w, y / w] : [0.5, 0.5];
}

export default function BlendPad({ anchors = [], blend, onChange, disabled = false, size = 260 }) {
    const theme = useTheme();
    const ref = useRef(null);
    const [drag, setDrag] = useState(null);     // [x, y] while dragging
    const point = drag || pointFor(anchors, blend);
    const live = useMemo(() => (drag ? weightsAt(anchors, drag[0], drag[1]) : (blend || {})), [drag, anchors, blend]);
    const ink = theme.palette.text.primary;
    const faint = theme.palette.divider;

    const toLocal = useCallback((e) => {
        const r = ref.current.getBoundingClientRect();
        return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
    }, []);
    const onDown = (e) => {
        if (disabled || e.button !== 0) return;
        e.preventDefault();
        try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* noop */ }
        const p = toLocal(e);
        setDrag(p);
        onChange?.(weightsAt(anchors, p[0], p[1]), false);
    };
    const onMove = (e) => {
        if (!drag) return;
        const p = toLocal(e);
        setDrag(p);
        onChange?.(weightsAt(anchors, p[0], p[1]), false);
    };
    const onUp = (e) => {
        if (!drag) return;
        const p = toLocal(e);
        setDrag(null);
        onChange?.(weightsAt(anchors, p[0], p[1]), true);
    };

    return (
        <Box sx={{ userSelect: 'none', width: '100%', maxWidth: size }}>
            <Box
                ref={ref}
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={onUp}
                onPointerCancel={onUp}
                sx={{
                    position: 'relative', width: '100%', aspectRatio: '1 / 1', borderRadius: '24px',
                    background: theme.palette.background.paper, border: `1px solid ${faint}`,
                    cursor: disabled ? 'default' : 'crosshair', touchAction: 'none', opacity: disabled ? 0.6 : 1,
                    boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.55)',
                }}
            >
                <svg viewBox="0 0 100 100" width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
                    <circle cx="50" cy="50" r="42" fill="none" stroke={faint} strokeWidth="0.6" />
                    <circle cx="50" cy="50" r="21" fill="none" stroke={faint} strokeWidth="0.4" />
                    {anchors.map((a) => {
                        const w = live[a.id] || 0;
                        const [ax, ay] = a.position;
                        const col = a.color || theme.palette.primary.main;
                        return (
                            <g key={a.id}>
                                <line x1={ax * 100} y1={ay * 100} x2={point[0] * 100} y2={point[1] * 100}
                                      stroke={col} strokeWidth={0.3 + w * 1.6} strokeOpacity={0.15 + w * 0.6} />
                                <circle cx={ax * 100} cy={ay * 100} r={3 + w * 4} fill={col} fillOpacity={0.55 + w * 0.45}
                                        stroke={ink} strokeOpacity={a.encoded ? 0.35 : 0.1} strokeWidth="0.5" />
                            </g>
                        );
                    })}
                    <circle cx={point[0] * 100} cy={point[1] * 100} r="3.2" fill={ink} />
                    <circle cx={point[0] * 100} cy={point[1] * 100} r="5.5" fill="none" stroke={ink} strokeWidth="0.6" strokeOpacity="0.5" />
                </svg>
                {anchors.map((a) => {
                    const [ax, ay] = a.position;
                    const w = live[a.id] || 0;
                    return (
                        <Typography key={a.id} variant="caption" sx={{
                            position: 'absolute', left: `${ax * 100}%`, top: `${ay * 100}%`,
                            transform: `translate(-50%, ${ay < 0.5 ? '-170%' : '70%'})`,
                            color: 'text.secondary', fontWeight: w > 0.3 ? 600 : 400, pointerEvents: 'none', whiteSpace: 'nowrap',
                            fontSize: 11,
                        }}>
                            {a.label}{w > 0.02 ? ` ${Math.round(w * 100)}%` : ''}
                        </Typography>
                    );
                })}
            </Box>
        </Box>
    );
}
