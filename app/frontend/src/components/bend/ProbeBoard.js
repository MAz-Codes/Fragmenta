import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Box, Typography, Chip, CircularProgress, Fade, useTheme } from '@mui/material';
import { keyframes } from '@emotion/react';
import {
    STAGE_SHORT, PART_LABEL, boardColumns, padModule, parseJumperId, wireEndPad, wireIsForward,
    isNoopWire,
} from './bendUtils';

/**
 * ProbeBoard — the model as a circuit board of unlabelled contacts.
 *
 * Touch a pad and the fragment renders through it; shift-click latches a pad
 * so touches combine; drag between two DiT pads to solder a jumper wire,
 * which stays in the circuit for every render until it is disconnected:
 * alt-click (option-click on a Mac) it, or its chip's × — holding
 * alt/option turns the hovered wire red first.
 * Each wire keeps its own colour while connected, matched by its chip. A DiT column's rows are the block itself (top), its feed-forward
 * (middle) and its attention (bottom): touching the lower two bends their
 * weights, wiring them taps or feeds their signal. What a pad does is a property of the
 * contact (bendUtils.padModule), never chosen and never named by the app:
 * the only names on the board are the ones the user gives after listening.
 * "Show wiring" reveals the architecture for anyone who wants the theory.
 *
 * Columns run in signal order, so position still carries the path; DiT
 * columns hold a signal pad on top and two weight pads below. Only DiT
 * contacts take wires, so only they are drawn as solder holes; every other
 * contact is a push-button (press, or shift-press to latch).
 */

const PAD = 20;
const COL_GAP = 8;
const GROUP_GAP = 18;
const ROW = PAD + COL_GAP;
// Space between the zone labels and the pads: forward wires arc through it,
// so it must stay clear of both.
const WIRE_LANE = 30;
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
// Reference designators, like the silkscreen on a real board: they mark the
// parts without saying what they do. "Show wiring" adds the names.
const DESIGNATOR = { cond: 'U1', timestep: 'U2', dit: 'U3', latent: 'U4', decoder: 'U5' };

// The last contact touched keeps breathing until the next touch.
const pulse = keyframes`
  0%   { box-shadow: 0 0 0 0 var(--pad-glow), 0 0 10px var(--pad-glow); }
  70%  { box-shadow: 0 0 0 9px transparent, 0 0 14px var(--pad-glow); }
  100% { box-shadow: 0 0 0 0 transparent, 0 0 10px var(--pad-glow); }
`;
const DRAG_THRESHOLD = 6;

const wireLabel = (j) => {
    if (!j) return null;
    const end = (b, part) => `DiT ${b}${part === 'block' ? '' : ` ${PART_LABEL[part]}`}`;
    const kind = wireIsForward(j.from, j.fromPart, j.to, j.toPart) ? 'forward' : 'feedback';
    return `wire ${end(j.from, j.fromPart)} → ${end(j.to, j.toPart)} (${kind})`;
};

export default function ProbeBoard({
    registry, names, held, lastTouched, wires, showWiring, disabled, loading = false,
    onTouch, onWire, onForget, onDisconnect,
}) {
    const theme = useTheme();
    const accent = theme.palette.bend?.main || '#AEB9C4';

    const columns = useMemo(() => boardColumns(registry), [registry]);
    const groups = useMemo(() => {
        const out = [];
        columns.forEach((c) => {
            const last = out[out.length - 1];
            if (last && last.stage === c.stage) last.cols.push(c);
            else out.push({ stage: c.stage, cols: [c] });
        });
        return out;
    }, [columns]);

    // What each pad does — only ever shown with "show wiring".
    const wiring = useMemo(() => {
        if (!showWiring) return {};
        const opLabel = Object.fromEntries((registry?.operators || []).map(o => [o.name, o.label]));
        const out = {};
        columns.forEach(c => c.pads.forEach((p) => {
            const m = padModule(p, registry);
            const where = p.block !== undefined ? `${STAGE_SHORT[p.stage]} ${p.block}`
                : p.stage === 'latent' ? `Latent ${['early', 'middle', 'late'][p.window]}`
                    : STAGE_SHORT[p.stage];
            const what = p.domain === 'weight'
                ? (p.slot === 'self_attn' ? 'attention weights' : p.slot === 'ff' ? 'feed-forward weights' : 'weights')
                : 'signal';
            out[p.id] = `${where} · ${what} · ${opLabel[m.operator] || m.operator}`;
        }));
        return out;
    }, [showWiring, columns, registry]);

    const heldSet = useMemo(() => new Set(held), [held]);
    const stageOf = useMemo(() => {
        const out = {};
        columns.forEach(c => c.pads.forEach((p) => { out[p.id] = p.stage; }));
        return out;
    }, [columns]);

    // Each wire keeps its colour for as long as it is connected: a new wire
    // takes the first colour no connected wire is using.
    const wirePalette = theme.palette.board?.wires || [accent];
    const colorSlots = useRef({});
    const wireColorOf = useMemo(() => {
        const slots = colorSlots.current;
        Object.keys(slots).forEach((id) => { if (!wires.includes(id)) delete slots[id]; });
        wires.forEach((id) => {
            if (slots[id] !== undefined) return;
            const used = new Set(Object.values(slots));
            let k = 0;
            while (used.has(k % wirePalette.length) && k < wirePalette.length) k++;
            slots[id] = k % wirePalette.length;
        });
        return (id) => wirePalette[slots[id] ?? 0];
    }, [wires, wirePalette]);

    // Alt/option held: the hovered wire shows red — a click will remove it.
    const [altDown, setAltDown] = useState(false);
    useEffect(() => {
        const down = (e) => { if (e.key === 'Alt') setAltDown(true); };
        const up = (e) => { if (e.key === 'Alt') setAltDown(false); };
        const reset = () => setAltDown(false);
        window.addEventListener('keydown', down);
        window.addEventListener('keyup', up);
        window.addEventListener('blur', reset);
        return () => {
            window.removeEventListener('keydown', down);
            window.removeEventListener('keyup', up);
            window.removeEventListener('blur', reset);
        };
    }, []);

    // --- pointer: tap = touch, drag DiT→DiT = jumper -------------------------
    const innerRef = useRef(null);
    const padEls = useRef({});
    const drag = useRef(null);            // {id, block, part, x, y, moving}
    const [dragLine, setDragLine] = useState(null);
    const [hover, setHover] = useState(null);

    const relPoint = (clientX, clientY) => {
        const r = innerRef.current?.getBoundingClientRect();
        return r ? { x: clientX - r.left, y: clientY - r.top } : { x: 0, y: 0 };
    };
    // An element's centre in the board's drawing space. The overlays are
    // positioned inside the board's border, so measure from there too —
    // from the outer edge, every wire end lands a border-width off its hole.
    const centerIn = useCallback((el) => {
        const box = innerRef.current;
        if (!el || !box) return null;
        const a = el.getBoundingClientRect();
        const b = box.getBoundingClientRect();
        return {
            x: a.left - b.left - box.clientLeft + a.width / 2,
            y: a.top - b.top - box.clientTop + a.height / 2,
        };
    }, []);
    const padCenter = useCallback((id) => centerIn(padEls.current[id]), [centerIn]);

    const onPadDown = (e, pad) => {
        if (disabled || e.button > 0) return;
        e.preventDefault();
        drag.current = { id: pad.id, block: pad.stage === 'dit' ? pad.block : null,
                         part: pad.part, x: e.clientX, y: e.clientY, moving: false };
    };

    useEffect(() => {
        const move = (e) => {
            const d = drag.current;
            if (!d) return;
            if (!d.moving && Math.hypot(e.clientX - d.x, e.clientY - d.y) > DRAG_THRESHOLD) {
                d.moving = d.block !== null;      // only DiT pads take a wire
            }
            if (d.moving) {
                const from = padCenter(d.id);
                if (from) setDragLine({ from, to: relPoint(e.clientX, e.clientY) });
            }
        };
        const up = (e) => {
            const d = drag.current;
            drag.current = null;
            setDragLine(null);
            if (!d) return;
            const el = document.elementFromPoint(e.clientX, e.clientY)?.closest?.('[data-pad]');
            const targetId = el?.getAttribute('data-pad');
            if (d.moving) {
                const target = el && el.getAttribute('data-block');
                const toPart = el && el.getAttribute('data-part');
                // Points that are already joined take no wire: it snaps back.
                if (targetId && targetId !== d.id && target !== null && target !== ''
                    && !isNoopWire(d.block, d.part, Number(target), toPart)) {
                    onWire(d.block, d.part, Number(target), toPart);
                }
            } else if (targetId === d.id) {
                onTouch(d.id, { hold: e.shiftKey || e.metaKey || e.ctrlKey });
            }
        };
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', up);
        return () => {
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
        };
    }, [onTouch, onWire, padCenter]);

    // --- wires: measured from the pads they join -----------------------------
    const [wirePaths, setWirePaths] = useState([]);
    const [bus, setBus] = useState(null);
    const inPin = useRef(null);
    const outPin = useRef(null);
    const measure = useCallback(() => {
        // The board's main trace: IN → through every zone → OUT.
        const i = centerIn(inPin.current);
        const o = centerIn(outPin.current);
        setBus(i && o ? { x1: i.x, x2: o.x, y: i.y } : null);
        const paths = [];
        wires.forEach((id) => {
            const j = parseJumperId(id);
            if (!j) return;
            const a = padCenter(wireEndPad(j.from, j.fromPart));
            const b = padCenter(wireEndPad(j.to, j.toPart));
            if (!a || !b) return;
            let d;
            if (j.from === j.to && j.fromPart === j.toPart) {
                // Feedback into itself: a small loop over the pad.
                const y0 = a.y - PAD / 2 + 2;
                d = `M ${a.x - 5} ${y0} C ${a.x - 18} ${a.y - 38}, ${a.x + 18} ${a.y - 38}, ${a.x + 5} ${y0}`;
            } else {
                // Forward wires arc over the board, feedback wires under it,
                // whichever rows they start and end on. The curve's apex sits
                // just past the DiT rows (a quadratic peaks halfway between
                // its endpoints' mean and its control point).
                const top = padCenter('dit.0')?.y ?? Math.min(a.y, b.y);
                const bottom = padCenter(wireEndPad(0, 'attn'))?.y ?? Math.max(a.y, b.y);
                const reach = Math.min(Math.abs(b.x - a.x) * 0.06, 14);
                const forward = wireIsForward(j.from, j.fromPart, j.to, j.toPart);
                const apex = forward ? top - PAD / 2 - 6 - reach : bottom + PAD / 2 + 12 + reach;
                const cy = 2 * apex - (a.y + b.y) / 2;
                d = `M ${a.x} ${a.y} Q ${(a.x + b.x) / 2} ${cy} ${b.x} ${b.y}`;
            }
            paths.push({ id, d, a, b });
        });
        setWirePaths(paths);
    }, [wires, padCenter, centerIn]);
    useLayoutEffect(() => { measure(); }, [measure, columns, showWiring]);
    useEffect(() => {
        // Web fonts landing after the first paint can nudge the zones.
        let live = true;
        document.fonts?.ready?.then(() => { if (live) measure(); });
        if (!innerRef.current || typeof ResizeObserver === 'undefined') return () => { live = false; };
        const ro = new ResizeObserver(() => measure());
        ro.observe(innerRef.current);
        return () => { live = false; ro.disconnect(); };
    }, [measure]);

    // Named contacts, plus named wires that aren't connected right now
    // (touching one solders it back); connected wires have their own row.
    const connected = useMemo(() => new Set(wires), [wires]);
    const finds = useMemo(() => Object.entries(names || {})
        .filter(([id]) => !connected.has(id)), [names, connected]);
    const holdKey = (e) => e.shiftKey || e.metaKey || e.ctrlKey;
    const cutting = altDown && hover && wires.includes(hover) ? hover : null;
    // How to use what's under the pointer, when there's nothing else to say.
    const affordance = (id) => (parseJumperId(id) ? 'click to hear · alt/option-click to disconnect'
        : stageOf[id] === 'dit' ? 'press to hear · shift-press to hold · drag to another hole to wire'
            : stageOf[id] ? 'press to hear · shift-press to hold' : null);
    const readout = cutting
        ? 'Click to disconnect this wire.'
        : hover
            ? ([names?.[hover]?.name, showWiring && (wiring[hover] || wireLabel(parseJumperId(hover)))]
                .filter(Boolean).join('  ·  ') || affordance(hover))
            : '';

    if (!columns.length) {
        return (
            <Typography variant="body2" color="textSecondary" sx={{ py: 3, textAlign: 'center' }}>
                Loading the board…
            </Typography>
        );
    }

    const isDark = theme.palette.mode === 'dark';
    const board = theme.palette.board || {};
    const cutColor = theme.palette.error.main;
    const dragColor = isDark ? (theme.palette.bend?.light || accent) : (theme.palette.bend?.dark || accent);
    const metal = isDark ? '#C9D2CD' : '#8C9590';

    const padSx = (pad) => (pad.stage === 'dit' ? holeSx(pad) : buttonSx(pad));

    // A push-button: raised keycap, sinks in and lights up when latched.
    const buttonSx = (pad) => {
        const color = board[pad.stage] || accent;
        const isHeld = heldSet.has(pad.id);
        const named = !!names?.[pad.id];
        const isLast = lastTouched === pad.id;
        const cap = `linear-gradient(180deg, rgba(255,255,255,${isDark ? 0.32 : 0.55}) 0%, rgba(255,255,255,0.06) 48%, rgba(0,0,0,0.16) 100%)`;
        const mark = `radial-gradient(circle, ${isHeld ? board.surface : color} 0 16%, transparent 19%)`;
        return {
            '--pad-glow': `${color}99`,
            position: 'relative', zIndex: 1,
            width: PAD, height: PAD, borderRadius: '6px', flexShrink: 0,
            cursor: disabled ? 'progress' : 'pointer', touchAction: 'none',
            backgroundColor: isHeld ? color : `${color}${isDark ? '8C' : 'A6'}`,
            backgroundImage: [named && mark, cap].filter(Boolean).join(', '),
            transform: isHeld ? 'translateY(1px)' : 'none',
            boxShadow: isHeld
                ? `inset 0 2px 3px rgba(0,0,0,0.35), 0 0 0 2px ${color}40, 0 0 14px ${color}AA`
                : `inset 0 1px 0 rgba(255,255,255,${isDark ? 0.35 : 0.6}), 0 2px 0 rgba(0,0,0,${isDark ? 0.55 : 0.22}), 0 3px 5px rgba(0,0,0,${isDark ? 0.35 : 0.12})`,
            animation: isLast ? `${pulse} 1.8s ease-out infinite` : 'none',
            // A wire can't land here: step back while one is being dragged.
            opacity: dragLine ? 0.25 : disabled && !isLast ? 0.55 : 1,
            transition: 'transform 120ms ease, box-shadow 200ms ease, background-color 160ms ease, opacity 160ms ease',
            '&:hover': dragLine ? {} : {
                backgroundColor: color,
                transform: isHeld ? 'translateY(1px)' : 'translateY(-1px)',
            },
            '&:active': { transform: 'translateY(1px)' },
        };
    };

    const holeSx = (pad) => {
        const color = board[pad.stage] || accent;
        const isHeld = heldSet.has(pad.id);
        const named = !!names?.[pad.id];
        const isLast = lastTouched === pad.id;
        // A plated through-hole: coloured ring, drilled centre, a sheen.
        const sheen = `radial-gradient(circle at 34% 28%, ${isDark ? 'rgba(255,255,255,0.45)' : 'rgba(255,255,255,0.75)'} 0 10%, transparent 42%)`;
        const hole = `radial-gradient(circle, ${board.surface} 0 30%, transparent 33%)`;
        const mark = `radial-gradient(circle, ${color} 0 15%, transparent 17%)`;
        return {
            '--pad-glow': `${color}99`,
            position: 'relative', zIndex: 1,
            width: PAD, height: PAD, borderRadius: '50%', flexShrink: 0,
            cursor: disabled ? 'progress' : 'pointer', touchAction: 'none',
            backgroundColor: isHeld ? color : `${color}${isDark ? 'B8' : 'D0'}`,
            backgroundImage: isHeld ? sheen : [named && mark, hole, sheen].filter(Boolean).join(', '),
            boxShadow: isHeld
                ? `0 0 0 2px ${color}40, 0 0 14px ${color}AA`
                : `inset 0 -1px 1px rgba(0,0,0,0.28), 0 1px 2px rgba(0,0,0,${isDark ? 0.5 : 0.18})`,
            animation: isLast ? `${pulse} 1.8s ease-out infinite` : 'none',
            opacity: disabled && !isLast ? 0.55 : 1,
            transition: 'transform 140ms ease, box-shadow 200ms ease, background-color 160ms ease',
            '&:hover': {
                transform: 'scale(1.18)',
                backgroundColor: color,
                boxShadow: `0 0 0 3px ${color}33, 0 0 14px ${color}99`,
            },
        };
    };

    const zoneSx = (stage) => {
        const color = board[stage] || accent;
        return {
            position: 'relative', zIndex: 1,
            display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
            px: 1.5, pt: 0.75, pb: 1.25,
            borderRadius: 1.5,
            border: '1px solid', borderColor: `${color}70`,
            backgroundColor: board.surface,
            backgroundImage: `linear-gradient(180deg, ${color}1C, ${color}0A)`,
            boxShadow: `inset 0 1px 0 ${color}1F, 0 2px 8px rgba(0,0,0,${isDark ? 0.35 : 0.08})`,
        };
    };

    // A pin header at each end of the board, its three pins on the pad rows.
    const header = (label, pinRef) => (
        <Box sx={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography sx={{
                fontFamily: MONO, fontSize: '0.62rem', letterSpacing: '0.12em', lineHeight: '14px',
                height: 14, mb: `${WIRE_LANE}px`, color: 'text.secondary', mt: 0.75,
            }}>
                {label}
            </Typography>
            <Box sx={{
                display: 'flex', flexDirection: 'column', gap: `${COL_GAP}px`, p: '3px', mb: 1.25,
                border: '1px solid', borderColor: board.trace, borderRadius: 1,
                backgroundColor: board.surface,
            }}>
                {[0, 1, 2].map(r => (
                    <Box key={r} sx={{ width: PAD - 6, height: PAD - 6, m: '3px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Box ref={r === 1 ? pinRef : undefined} sx={{
                            width: 9, height: 9, borderRadius: '2px',
                            background: `linear-gradient(135deg, ${metal}, ${metal}88)`,
                            boxShadow: '0 1px 1px rgba(0,0,0,0.35)',
                        }} />
                    </Box>
                ))}
            </Box>
        </Box>
    );

    return (
        <Box sx={{ mb: 1.5 }}>
            <Box sx={{ overflowX: 'auto', overflowY: 'hidden', pb: 0.5 }}>
                <Box ref={innerRef} sx={{
                    position: 'relative', display: 'inline-flex', alignItems: 'flex-end',
                    justifyContent: 'center',
                    gap: `${GROUP_GAP}px`, px: 3, pt: 3.5, pb: 7, minWidth: '100%',
                    borderRadius: 3,
                    backgroundColor: board.surface,
                    backgroundImage: [
                        `radial-gradient(ellipse at 50% 0%, ${isDark ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.6)'}, transparent 70%)`,
                        `linear-gradient(${board.grid} 1px, transparent 1px)`,
                        `linear-gradient(90deg, ${board.grid} 1px, transparent 1px)`,
                    ].join(', '),
                    backgroundSize: '100% 100%, 16px 16px, 16px 16px',
                    border: '1px solid',
                    borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(30,45,38,0.16)',
                    boxShadow: isDark
                        ? 'inset 0 1px 0 rgba(255,255,255,0.05), inset 0 0 40px rgba(0,0,0,0.35)'
                        : 'inset 0 1px 0 rgba(255,255,255,0.8), inset 0 0 30px rgba(30,45,38,0.06)',
                    userSelect: 'none',
                }}>
                    {/* Model weights loading: the board waits under a veil. */}
                    <Fade in={loading} unmountOnExit>
                        <Box sx={{
                            position: 'absolute', inset: 0, zIndex: 3, borderRadius: 'inherit',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1.5,
                            backgroundColor: `${board.surface}C7`, backdropFilter: 'blur(2px)',
                        }}>
                            <CircularProgress size={20} thickness={5} sx={{ color: 'bend.main' }} />
                            <Typography sx={{ fontFamily: MONO, fontSize: '0.75rem', letterSpacing: '0.08em', color: 'text.secondary' }}>
                                Loading model weights…
                            </Typography>
                        </Box>
                    </Fade>

                    {/* mounting holes */}
                    {[{ top: 10, left: 10 }, { top: 10, right: 10 }, { bottom: 10, left: 10 }, { bottom: 10, right: 10 }].map((pos, i) => (
                        <Box key={i} sx={{
                            position: 'absolute', ...pos, width: 10, height: 10, borderRadius: '50%',
                            border: '2px solid', borderColor: board.trace, backgroundColor: 'transparent',
                        }} />
                    ))}

                    {/* the board's main trace, under the zones */}
                    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 0 }}>
                        {bus && (
                            <line x1={bus.x1} y1={bus.y} x2={bus.x2} y2={bus.y}
                                  stroke={board.trace} strokeWidth="3" strokeLinecap="round" />
                        )}
                    </svg>

                    {header('IN', inPin)}
                    {groups.map(g => {
                        const color = board[g.stage] || accent;
                        return (
                            <Box key={g.stage} sx={zoneSx(g.stage)}>
                                <Typography sx={{
                                    fontFamily: MONO, fontSize: '0.64rem', fontWeight: 600, letterSpacing: '0.08em',
                                    lineHeight: '14px', height: 14, mb: `${WIRE_LANE}px`, color, whiteSpace: 'nowrap',
                                }}>
                                    {DESIGNATOR[g.stage]}
                                </Typography>
                                {/* The stage's name sits above the zone, out of the
                                    layout, so showing the wiring never resizes it. */}
                                <Typography sx={{
                                    position: 'absolute', bottom: '100%', left: '50%', mb: '4px',
                                    transform: 'translateX(-50%)', whiteSpace: 'nowrap',
                                    fontFamily: MONO, fontSize: '0.6rem', letterSpacing: '0.06em', lineHeight: '12px',
                                    color, opacity: showWiring ? 0.85 : 0, transition: 'opacity 160ms ease',
                                    pointerEvents: 'none',
                                }}>
                                    {STAGE_SHORT[g.stage]}
                                </Typography>
                                <Box sx={{
                                    position: 'relative', display: 'flex', gap: `${COL_GAP}px`,
                                    // a trace along the middle row of the zone
                                    '&::before': g.cols.length > 1 ? {
                                        content: '""', position: 'absolute', zIndex: 0,
                                        left: PAD / 2, right: PAD / 2, top: ROW + PAD / 2 - 1, height: 2,
                                        backgroundColor: `${color}45`,
                                    } : undefined,
                                }}>
                                    {g.cols.map(c => (
                                        // Every column is three pads tall (shorter ones sit at
                                        // the bottom), so all labels and rows line up.
                                        <Box key={c.key} sx={{
                                            position: 'relative',
                                            display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
                                            gap: `${COL_GAP}px`, minHeight: 3 * PAD + 2 * COL_GAP,
                                            '&::before': c.pads.length > 1 ? {
                                                content: '""', position: 'absolute', zIndex: 0,
                                                left: PAD / 2 - 1, width: 2, bottom: PAD / 2,
                                                top: (3 - c.pads.length) * ROW + PAD / 2,
                                                backgroundColor: `${color}45`,
                                            } : undefined,
                                        }}>
                                            {c.pads.map(p => (
                                                <Box
                                                    key={p.id}
                                                    ref={(el) => { if (el) padEls.current[p.id] = el; else delete padEls.current[p.id]; }}
                                                    data-pad={p.id}
                                                    data-block={p.stage === 'dit' ? p.block : ''}
                                                    data-part={p.part || ''}
                                                    role="button"
                                                    aria-label={names?.[p.id]?.name || 'contact'}
                                                    onPointerDown={(e) => onPadDown(e, p)}
                                                    onPointerEnter={() => setHover(p.id)}
                                                    onPointerLeave={() => setHover(h => (h === p.id ? null : h))}
                                                    sx={padSx(p)}
                                                />
                                            ))}
                                        </Box>
                                    ))}
                                </Box>
                            </Box>
                        );
                    })}
                    {header('OUT', outPin)}

                    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none', zIndex: 2 }}>
                        {wirePaths.map(w => {
                            const cut = cutting === w.id;
                            const on = cut || lastTouched === w.id || hover === w.id;
                            const named = !!names?.[w.id];
                            const wireColor = cut ? cutColor : wireColorOf(w.id);
                            return (
                                <g key={w.id}
                                   style={{
                                       pointerEvents: disabled ? 'none' : 'stroke', cursor: 'pointer',
                                       filter: on ? `drop-shadow(0 0 4px ${wireColor})` : `drop-shadow(0 1px 1px rgba(0,0,0,${isDark ? 0.6 : 0.2}))`,
                                   }}
                                   onPointerEnter={() => setHover(w.id)}
                                   onPointerLeave={() => setHover(h => (h === w.id ? null : h))}
                                   onClick={(e) => (e.altKey ? onDisconnect(w.id) : onTouch(w.id))}>
                                    <path d={w.d} fill="none" stroke="transparent" strokeWidth="12" />
                                    <path d={w.d} fill="none" stroke={wireColor} strokeLinecap="round"
                                          strokeWidth={cut ? 3.25 : on ? 2.75 : 2.25}
                                          strokeOpacity={on ? 1 : 0.9}
                                          strokeDasharray={cut ? '6 4' : 'none'} />
                                    {/* solder blobs where the wire meets the pads */}
                                    {[w.a, w.b].map((pt, k) => (
                                        <circle key={k} cx={pt.x} cy={pt.y} r={on ? 3.5 : 3} fill={wireColor}
                                                fillOpacity={on ? 1 : 0.8} />
                                    ))}
                                </g>
                            );
                        })}
                        {dragLine && (
                            <line x1={dragLine.from.x} y1={dragLine.from.y} x2={dragLine.to.x} y2={dragLine.to.y}
                                  stroke={dragColor} strokeWidth="2" strokeDasharray="5 4" strokeLinecap="round" />
                        )}
                    </svg>
                </Box>
            </Box>

            <Typography variant="caption" sx={{ display: 'block', minHeight: 18, color: 'text.secondary', mt: 0.5 }}>
                {readout || ' '}
            </Typography>

            {wires.length > 0 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5, alignItems: 'center' }}>
                    <Typography variant="caption" color="textSecondary" sx={{ mr: 0.5 }}>Wires</Typography>
                    {/* Every connected wire is live, so the chips have no held
                        state: click re-plays the board, × disconnects. */}
                    {wires.map((id, i) => {
                        const c = wireColorOf(id);
                        return (
                            <Chip
                                key={id} size="small" disabled={disabled} variant="outlined"
                                label={names?.[id]?.name || (showWiring ? wireLabel(parseJumperId(id)) : `wire ${i + 1}`)}
                                icon={<Box component="span" sx={{
                                    width: 14, height: 3, borderRadius: 2, ml: '8px !important', backgroundColor: c,
                                }} />}
                                onClick={() => onTouch(id)}
                                onDelete={disabled ? undefined : () => onDisconnect(id)}
                                onMouseEnter={() => setHover(id)}
                                onMouseLeave={() => setHover(h => (h === id ? null : h))}
                                sx={{
                                    fontSize: '0.68rem', height: 22, borderColor: c, color: c,
                                    '& .MuiChip-deleteIcon': { color: `${c}AA`, '&:hover': { color: cutColor } },
                                }}
                            />
                        );
                    })}
                </Box>
            )}

            {finds.length > 0 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5, alignItems: 'center' }}>
                    <Typography variant="caption" color="textSecondary" sx={{ mr: 0.5 }}>Your finds</Typography>
                    {finds.map(([id, f]) => (
                        <Chip
                            key={id} size="small" label={f.name} disabled={disabled}
                            variant={heldSet.has(id) ? 'filled' : 'outlined'}
                            icon={<Box component="span" sx={{
                                width: 8, height: 8, borderRadius: '50%', ml: '6px !important',
                                backgroundColor: theme.palette.board?.[stageOf[id]] || accent,
                            }} />}
                            onClick={(e) => onTouch(id, { hold: holdKey(e) })}
                            onDelete={disabled ? undefined : () => onForget(id)}
                            onMouseEnter={() => setHover(id)}
                            onMouseLeave={() => setHover(h => (h === id ? null : h))}
                            sx={{
                                fontSize: '0.68rem', height: 22, borderColor: `${accent}88`,
                                ...(heldSet.has(id) ? { backgroundColor: accent, color: theme.palette.bend?.contrastText } : {}),
                            }}
                        />
                    ))}
                </Box>
            )}
        </Box>
    );
}
