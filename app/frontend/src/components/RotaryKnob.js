import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';

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
 * Bipolar rotary knob with a centre detent. Drag vertically, scroll, or use
 * the arrow keys; double-click returns to `center`. The detent is extra drag
 * travel around `center` (in value units), so a sweep sticks there briefly
 * instead of skipping through.
 *
 * Props: value, min, max, step (1), center (0), detent (1), size (28),
 * color, busy (dims while the change is being applied), format(v) for the
 * centre readout, ariaLabel, onChange(v).
 */
export default function RotaryKnob({
    value, min, max, step = 1, center = 0, detent = 1, size = 28,
    color = '#279FBB', busy = false, format = (v) => String(v), ariaLabel, onChange,
}) {
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
    const a0 = angle(center);
    const a = angle(value);
    const [px, py] = polar(c, r, a);

    return (
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
                '&:focus-visible': { boxShadow: `0 0 0 2px ${color}66` },
            }}
        >
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: 'block' }}>
                <path d={arc(c, r, -SWEEP, SWEEP)} fill="none" stroke="currentColor" strokeOpacity={0.15}
                    strokeWidth={2.5} strokeLinecap="round" />
                <path d={arc(c, r, Math.min(a0, a), Math.max(a0, a))} fill="none" stroke={color}
                    strokeWidth={2.5} strokeLinecap="round" />
                {/* Centre tick, so the detent position is visible. */}
                <line x1={c} y1={0.5} x2={c} y2={3.5} stroke="currentColor" strokeOpacity={0.45} strokeWidth={1} />
                <circle cx={c} cy={c} r={r - 4} fill="currentColor" fillOpacity={0.06} />
                {/* Position marker on the ring; the value sits in the middle. */}
                <circle cx={px} cy={py} r={2.4} fill={value === center ? 'currentColor' : color}
                    fillOpacity={value === center ? 0.7 : 1} />
                <text x={c} y={c + 3} textAnchor="middle" fontSize={8} fontWeight={600}
                    fill="currentColor" fillOpacity={0.85} style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {format(value)}
                </text>
            </svg>
        </Box>
    );
}
