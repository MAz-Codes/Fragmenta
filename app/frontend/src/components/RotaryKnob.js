import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import { useTheme } from '@mui/material/styles';

// 270° of travel, centred on 12 o'clock.
const SWEEP = 135;
// Vertical drag distance for the whole range (detent included).
const DRAG_PX = 160;

const toRad = (deg) => ((deg - 90) * Math.PI) / 180;
const polar = (c, r, deg) => [c + r * Math.cos(toRad(deg)), c + r * Math.sin(toRad(deg))];
function arc(c, r, from, to) {
    if (Math.abs(to - from) < 0.01) return '';
    const [x1, y1] = polar(c, r, from);
    const [x2, y2] = polar(c, r, to);
    const large = Math.abs(to - from) > 180 ? 1 : 0;
    const sweep = to > from ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} ${sweep} ${x2} ${y2}`;
}

/**
 * Rotary knob: a softly sculpted body with an indicator line, ringed by a
 * fine arc that follows the value. Drag vertically, scroll, or use the arrow
 * keys; double-click returns to `center`. The detent is extra drag travel
 * around `center` (in value units), so a sweep sticks there briefly instead
 * of skipping through (0 = no detent).
 *
 * Props: value, min, max, step (1), center (0: reset/detent point),
 * origin (where the arc starts; defaults to `center`), detent (1), size (28),
 * color (the arc), busy (dims while a change is applied), format(v),
 * label (shown under the knob, with the value; without it the value sits
 * in the middle), ariaLabel, onChange(v).
 */
export default function RotaryKnob({
    value, min, max, step = 1, center = 0, origin, detent = 1, size = 28,
    color, busy = false, format = (v) => String(v), label, ariaLabel, onChange,
}) {
    const theme = useTheme();
    const arcColor = color || theme.palette.primary.main;
    const valueBelow = label !== undefined;
    const ref = useRef(null);
    const drag = useRef(null);
    const latest = useRef({ value, onChange });
    latest.current = { value, onChange };

    const clamp = (v) => Math.min(max, Math.max(min, v));
    const toRaw = (v) => (v === center ? center : center + Math.sign(v - center) * (Math.abs(v - center) + detent));
    const fromRaw = (raw) => {
        const off = raw - center;
        if (Math.abs(off) <= detent + step / 2) return center;
        const steps = Math.round((Math.abs(off) - detent) / step) * step;
        return clamp(center + Math.sign(off) * steps);
    };
    const emit = (v) => { if (v !== latest.current.value) latest.current.onChange?.(v); };

    // Wheel needs a non-passive listener to keep the page from scrolling.
    useEffect(() => {
        const el = ref.current;
        if (!el) return undefined;
        const onWheel = (e) => {
            e.preventDefault();
            emit(clamp(latest.current.value + (e.deltaY < 0 ? step : -step)));
        };
        el.addEventListener('wheel', onWheel, { passive: false });
        return () => el.removeEventListener('wheel', onWheel);
    });

    const onPointerDown = (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* noop */ }
        drag.current = { y: e.clientY, raw: toRaw(value) };
    };
    const onPointerMove = (e) => {
        if (!drag.current) return;
        const span = (max - min) + 2 * detent;
        const raw = Math.min(max + detent, Math.max(min - detent,
            drag.current.raw + ((drag.current.y - e.clientY) * span) / DRAG_PX));
        emit(fromRaw(raw));
    };
    const endDrag = (e) => {
        drag.current = null;
        try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* noop */ }
    };
    const onKeyDown = (e) => {
        const keys = { ArrowUp: step, ArrowRight: step, ArrowDown: -step, ArrowLeft: -step };
        if (e.key in keys) emit(clamp(value + keys[e.key]));
        else if (e.key === 'Home') emit(min);
        else if (e.key === 'End') emit(max);
        else if (e.key === '0' || e.key === 'Delete' || e.key === 'Backspace') emit(center);
        else return;
        e.preventDefault();
    };

    const c = size / 2;
    const r = c - 2;
    const angle = (v) => -SWEEP + ((v - min) / (max - min)) * 2 * SWEEP;
    const a0 = angle(origin ?? center);
    const a = angle(value);
    const body = r - 4.5;
    // Indicator line from near the centre out toward the rim.
    const [ix0, iy0] = polar(c, body * (valueBelow ? 0.25 : 0.62), a);
    const [ix1, iy1] = polar(c, body - 1.5, a);
    const gradId = `knob-body-${size}`;

    const knob = (
        <Box
            ref={ref}
            role="slider"
            tabIndex={0}
            aria-label={ariaLabel}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={value}
            aria-valuetext={format(value)}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onDoubleClick={() => emit(center)}
            onKeyDown={onKeyDown}
            sx={{
                width: size,
                height: size,
                flexShrink: 0,
                cursor: 'ns-resize',
                touchAction: 'none',
                borderRadius: '50%',
                opacity: busy ? 0.5 : 1,
                transition: 'opacity 120ms ease',
                outline: 'none',
                color: 'text.primary',
                '&:focus-visible': { boxShadow: `0 0 0 2px ${arcColor}66` },
            }}
        >
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: 'block', overflow: 'visible' }}>
                <defs>
                    {/* Soft top-left light on the body. */}
                    <radialGradient id={gradId} cx="35%" cy="30%" r="75%">
                        <stop offset="0%" stopColor="#FFFFFF" stopOpacity={0.9} />
                        <stop offset="100%" stopColor="#FFFFFF" stopOpacity={0.25} />
                    </radialGradient>
                </defs>
                <path d={arc(c, r, -SWEEP, SWEEP)} fill="none" stroke="currentColor" strokeOpacity={0.12}
                    strokeWidth={2} strokeLinecap="round" />
                <path d={arc(c, r, Math.min(a0, a), Math.max(a0, a))} fill="none" stroke={arcColor}
                    strokeWidth={2} strokeLinecap="round" style={{ transition: 'd 90ms linear' }} />
                {/* Sculpted body: a faint drop shadow, then the lit face. */}
                <circle cx={c} cy={c + 0.8} r={body} fill="currentColor" fillOpacity={0.10} />
                <circle cx={c} cy={c} r={body} fill={`url(#${gradId})`} stroke="currentColor" strokeOpacity={0.10} strokeWidth={0.75} />
                <line x1={ix0} y1={iy0} x2={ix1} y2={iy1} stroke="currentColor" strokeOpacity={0.8}
                    strokeWidth={1.6} strokeLinecap="round" />
                {!valueBelow && (
                    <text x={c} y={c + 3} textAnchor="middle" fontSize={8} fontWeight={600}
                        fill="currentColor" fillOpacity={0.85} style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {format(value)}
                    </text>
                )}
            </svg>
        </Box>
    );

    if (!valueBelow) return knob;
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.25, minWidth: 0 }}>
            {knob}
            <Box component="span" sx={{
                fontSize: '0.58rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
                color: 'text.secondary', lineHeight: 1.1, mt: 0.25,
            }}>
                {label}
            </Box>
            <Box component="span" sx={{
                fontSize: '0.62rem', fontVariantNumeric: 'tabular-nums', color: 'text.primary',
                lineHeight: 1.1, whiteSpace: 'nowrap',
            }}>
                {format(value)}
            </Box>
        </Box>
    );
}
