import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
    Box,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Button,
    Typography,
    Stack,
    MenuItem,
    MenuList,
    Paper,
    Select,
    Slider,
    IconButton,
    Chip,
    Alert,
} from '@mui/material';
import { createPortal } from 'react-dom';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import {
    Plus as AddIcon,
    Trash2 as RemoveIcon,
    GripVertical as DragIcon,
    Power as BypassIcon,
    ChevronDown as ChevronDownIcon,
    ChevronRight as ChevronRightIcon,
} from 'lucide-react';
import api from '../api';
import { isLoraCompatible } from '../utils/loraMatch';

const MAX_SLOTS = 4;

// "…/epoch=57-step=1500.safetensors" → "Epoch 57 · step 1500".
// Matches the label format used on the Generation page elsewhere.
const parseCheckpointLabel = (filepath) => {
    const name = (filepath || '').split('/').pop() || filepath || '';
    const m = name.match(/epoch=(\d+)-step=(\d+)/);
    if (m) return `Epoch ${m[1]} · step ${m[2]}`;
    return name.replace(/\.(safetensors|ckpt)$/i, '');
};

// Every selectable checkpoint for a run, oldest→latest. The API already
// returns all of them in `all_checkpoints`; `path` is just the latest, so
// falling back to it keeps single-checkpoint runs working.
const checkpointsOf = (lora) => (
    (lora.all_checkpoints && lora.all_checkpoints.length)
        ? lora.all_checkpoints
        : [lora.path]
);

/**
 * Multi-LoRA stack for the Generation panel.
 *
 * Props:
 *   selectedModel: the currently-selected base model id (e.g. "sa3-medium-base")
 *   value:         array of { path, strength, bypassed } slots
 *   onChange:      (newSlots) => void
 *
 * The picker filters available LoRAs by base-model compatibility (a `*-base`
 * LoRA also runs on its distilled sibling — see utils/loraMatch). Slot order
 * is the load order (slot 0 first); drag the handle to reorder. Bypass keeps
 * a slot in the stack but sends strength 0.
 */
const SUBMENU_SWITCH_MS = 180;

// Where a fixed panel goes beside a row, on the row's right if it fits. The app
// zooms the page on small windows, and browsers differ in whether measured sizes
// include that zoom, so the scale between the two is measured rather than assumed.
const besideRow = (row, panel) => {
    const probe = document.createElement('div');
    probe.style.cssText = 'position:fixed;left:0;top:0;width:100px;height:100px;visibility:hidden';
    document.body.append(probe);
    const scale = probe.getBoundingClientRect().width / 100;
    probe.style.cssText = 'position:fixed;inset:0;visibility:hidden';
    const view = probe.getBoundingClientRect();
    probe.remove();

    const gap = 4 * scale;
    const anchor = row.getBoundingClientRect();
    const { width, height } = panel.getBoundingClientRect();
    const fitsRight = anchor.right + gap + width <= view.width;
    const x = fitsRight ? anchor.right + gap : Math.max(0, anchor.left - gap - width);
    const y = Math.max(0, Math.min(anchor.top, view.height - height));
    return { left: x / scale, top: y / scale };
};

export default function LoraStack({ selectedModel, value, onChange, bare = false }) {
    const [available, setAvailable] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [dragIndex, setDragIndex] = useState(null);
    // Which slot's Select is open (controlled so a submenu pick can close it),
    // and the run whose checkpoint submenu is currently showing.
    const [openSlot, setOpenSlot] = useState(null);
    const [submenu, setSubmenu] = useState(null);  // { anchorEl, lora } | null
    // The run the pointer is aiming at, the timer that shows it, and where the
    // pointer is (a scroll moves rows under it without any mouse event).
    const aimed = useRef(null);
    const aimTimer = useRef(null);
    const pointer = useRef(null);
    const panelRef = useRef(null);
    const [panelPosition, setPanelPosition] = useState({ left: 0, top: 0, visibility: 'hidden' });
    const [listScroll, setListScroll] = useState(0);

    // Switching waits a moment, so crossing a row on the way into the submenu
    // doesn't swap it out from under the pointer.
    const aimAt = (next) => {
        const id = next ? next.lora.id : null;
        if (aimed.current === id) return;
        aimed.current = id;
        clearTimeout(aimTimer.current);
        aimTimer.current = setTimeout(() => setSubmenu(next), next && !submenu ? 0 : SUBMENU_SWITCH_MS);
    };

    const stayOnSubmenu = () => {
        clearTimeout(aimTimer.current);
        aimed.current = submenu ? submenu.lora.id : null;
    };

    const closeSubmenu = () => {
        clearTimeout(aimTimer.current);
        aimed.current = null;
        setSubmenu(null);
    };

    useEffect(() => () => clearTimeout(aimTimer.current), []);

    useLayoutEffect(() => {
        if (submenu && panelRef.current) {
            setPanelPosition(besideRow(submenu.anchorEl, panelRef.current));
        }
    }, [submenu, listScroll]);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        api.get('/api/loras')
            .then(r => { if (!cancelled) setAvailable(r.data.loras || []); })
            .catch(e => { if (!cancelled) setError(e.response?.data?.error || e.message); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, []);

    // LoRAs compatible with the current generation model. A LoRA trained
    // against `*-base` is compatible with both that base and its distilled
    // sibling (same backbone, differ only in CFG state) — loraMatch strips
    // the trailing `-base` before comparing.
    const compatible = available.filter(l =>
        isLoraCompatible(l.base_model, selectedModel)
    );

    // The single-LoRA case stays one click: when no slots are populated AND
    // there's a compatible LoRA, surface one empty slot so the user sees a
    // "Pick a LoRA" dropdown immediately.
    const slots = (value && value.length > 0)
        ? value
        : (compatible.length ? [{ path: '', strength: 1.0, bypassed: false }] : []);

    const addSlot = () => {
        if (slots.length >= MAX_SLOTS) return;
        onChange([...slots, { path: '', strength: 1.0, bypassed: false }]);
    };

    const removeSlot = (idx) => onChange(slots.filter((_, i) => i !== idx));

    const setSlot = (idx, patch) => {
        onChange(slots.map((s, i) => i === idx ? { ...s, ...patch } : s));
    };

    // --- drag-to-reorder (slot 0 is loaded first) ---------------------------
    const onDrop = (target) => {
        if (dragIndex === null || dragIndex === target) { setDragIndex(null); return; }
        const next = [...slots];
        const [moved] = next.splice(dragIndex, 1);
        next.splice(target, 0, moved);
        setDragIndex(null);
        onChange(next);
    };

    // Which run's row is under the pointer, for hovering and for scrolling.
    const aimAtRowAt = (x, y) => {
        const row = document.elementFromPoint(x, y)?.closest('[data-lora-row]');
        const lora = row && compatible.find((l) => String(l.id) === row.dataset.loraRow);
        aimAt(lora && checkpointsOf(lora).length > 1 ? { anchorEl: row, lora } : null);
    };

    const listPaperProps = {
        // The theme's menus don't scroll, and on a zoomed page a tall one is pushed off the top.
        sx: { overflowY: 'auto', maxHeight: '60vh' },
        onMouseMove: (e) => {
            pointer.current = { x: e.clientX, y: e.clientY };
            aimAtRowAt(e.clientX, e.clientY);
        },
        onMouseLeave: () => aimAt(null),
        // Capture: the list inside this paper is what scrolls, and scrolls don't bubble.
        onScrollCapture: (e) => {
            setListScroll(e.target.scrollTop);
            if (pointer.current) aimAtRowAt(pointer.current.x, pointer.current.y);
        },
    };

    const hint = (() => {
        if (!selectedModel) return 'Pick a model first.';
        if (loading) return 'Loading LoRAs…';
        if (!compatible.length) {
            return 'No LoRAs for this model yet. Train one in the Training tab.';
        }
        return null;
    })();

    const body = (
        <>
            {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
            {hint && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    {hint}
                </Typography>
            )}

            {slots.length > 0 && (
                <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                    {slots.map((slot, idx) => {
                        // Match on any checkpoint of the run — a slot can now hold an
                        // intermediate one, which `l.path === slot.path` would miss.
                        const choice = available.find(
                            l => l.path === slot.path || checkpointsOf(l).includes(slot.path)
                        );
                        const bypassed = !!slot.bypassed;
                        return (
                            <Box
                                key={idx}
                                onDragOver={(e) => { if (dragIndex !== null) e.preventDefault(); }}
                                onDrop={() => onDrop(idx)}
                                sx={{
                                    p: 1.5,
                                    borderBottom: '1px solid',
                                    borderColor: 'divider',
                                    '&:last-child': { borderBottom: 'none' },
                                    bgcolor: dragIndex === idx ? 'action.hover' : 'transparent',
                                    opacity: bypassed ? 0.5 : 1,
                                }}
                            >
                                <Stack direction="row" alignItems="center" spacing={1}>
                                    <Tooltip title={TIPS.lora.dragReorder}>
                                        <Box
                                            draggable={slots.length > 1}
                                            onDragStart={() => setDragIndex(idx)}
                                            onDragEnd={() => setDragIndex(null)}
                                            sx={{
                                                display: 'flex',
                                                cursor: slots.length > 1 ? 'grab' : 'default',
                                                color: 'text.disabled',
                                            }}
                                        >
                                            <DragIcon size={16} />
                                        </Box>
                                    </Tooltip>
                                    <Typography variant="caption" color="text.disabled" sx={{ width: 14 }}>
                                        {idx}
                                    </Typography>
                                    <Select
                                        size="small"
                                        value={slot.path}
                                        displayEmpty
                                        open={openSlot === idx}
                                        onOpen={() => setOpenSlot(idx)}
                                        onClose={() => { setOpenSlot(null); closeSubmenu(); }}
                                        MenuProps={{ PaperProps: listPaperProps }}
                                        onChange={(e) => setSlot(idx, { path: String(e.target.value) })}
                                        renderValue={(v) => {
                                            if (!v) return <em style={{ opacity: 0.6 }}>Pick a LoRA</em>;
                                            const run = available.find(
                                                l => l.path === v || checkpointsOf(l).includes(v)
                                            );
                                            if (!run) return v;
                                            return checkpointsOf(run).length > 1
                                                ? `${run.name} · ${parseCheckpointLabel(v)}`
                                                : run.name;
                                        }}
                                        sx={{ flex: 1, minWidth: 0 }}
                                    >
                                        <MenuItem value="" disabled>
                                            <em>Pick a LoRA</em>
                                        </MenuItem>
                                        {/* One row per RUN. Clicking it takes the latest checkpoint
                                            (the long-standing one-click behaviour); hovering opens a
                                            submenu to pick an earlier one, since LoRAs often peak
                                            before the final step. Runs with a single checkpoint have
                                            nothing to expand and just select. */}
                                        {compatible.flatMap(l => {
                                            const ckpts = checkpointsOf(l);
                                            const multi = ckpts.length > 1;
                                            // Hidden rows for the earlier checkpoints. The visible list
                                            // stays one row per run, but Select still finds a matching
                                            // child for whatever path the slot holds — without these it
                                            // warns "out-of-range value" whenever a slot holds an
                                            // intermediate checkpoint picked from the submenu.
                                            const hidden = ckpts.slice(0, -1).map(ckpt => (
                                                <MenuItem key={`${l.id}::${ckpt}`} value={ckpt} sx={{ display: 'none' }} />
                                            ));
                                            return [(
                                                <MenuItem
                                                    key={l.id}
                                                    value={ckpts[ckpts.length - 1]}
                                                    data-lora-row={l.id}
                                                >
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                                                        <Box sx={{ flex: 1, minWidth: 0 }}>
                                                            <Typography variant="body2" noWrap>{l.name}</Typography>
                                                            <Stack direction="row" spacing={0.5} sx={{ mt: 0.25 }}>
                                                                <Chip size="small" label={l.adapter_type || 'lora'} sx={{ height: 16, fontSize: 9 }} />
                                                                {l.rank && <Chip size="small" label={`r=${l.rank}`} sx={{ height: 16, fontSize: 9 }} />}
                                                                {multi && (
                                                                    <Chip size="small" label={`${ckpts.length} checkpoints`} variant="outlined" sx={{ height: 16, fontSize: 9 }} />
                                                                )}
                                                                {/* Trained in Break mode or made in the Adapter Lab. */}
                                                                {l.bent && (
                                                                    <Chip size="small" label="bent" variant="outlined"
                                                                          sx={{ height: 16, fontSize: 9, color: 'bend.main', borderColor: 'bend.main' }} />
                                                                )}
                                                            </Stack>
                                                        </Box>
                                                        {multi && <ChevronRightIcon size={14} style={{ opacity: 0.5, flexShrink: 0 }} />}
                                                    </Box>
                                                </MenuItem>
                                            ), ...hidden];
                                        })}
                                    </Select>
                                    <Tooltip title={TIPS.lora.bypass(bypassed)}>
                                        <IconButton
                                            size="small"
                                            color={bypassed ? 'default' : 'primary'}
                                            onClick={() => setSlot(idx, { bypassed: !bypassed })}
                                        >
                                            <BypassIcon size={14} />
                                        </IconButton>
                                    </Tooltip>
                                    <IconButton size="small" onClick={() => removeSlot(idx)} aria-label="Remove slot">
                                        <RemoveIcon size={14} />
                                    </IconButton>
                                </Stack>

                                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mt: 1, mb: 2 }}>
                                    <Typography variant="caption" color="text.secondary" sx={{ width: 60 }}>
                                        Strength
                                    </Typography>
                                    <Slider
                                        size="small"
                                        value={slot.strength}
                                        disabled={bypassed}
                                        onChange={(e, v) => setSlot(idx, { strength: v })}
                                        min={-2}
                                        max={2}
                                        step={0.05}
                                        valueLabelDisplay="auto"
                                        marks={[
                                            { value: 0, label: '0' },
                                            { value: 1, label: '1' },
                                        ]}
                                        sx={{ flex: 1 }}
                                    />
                                    <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                                        {bypassed ? '—' : slot.strength.toFixed(2)}
                                    </Typography>
                                </Stack>

                                {choice && choice.base_model && (
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
                                        Trained on {choice.base_model}
                                    </Typography>
                                )}
                            </Box>
                        );
                    })}
                </Box>
            )}

            <Stack direction="row" sx={{ mt: 1 }}>
                <Button
                    size="small"
                    variant="outlined"
                    startIcon={<AddIcon size={14} />}
                    disabled={slots.length >= MAX_SLOTS || !compatible.length}
                    onClick={addSlot}
                >
                    Add LoRA
                </Button>
            </Stack>

            {/* Checkpoint submenu. Not a Menu: a Menu is a modal, whose full-screen
                layer would keep the pointer and the wheel from the list underneath,
                so nothing but a click could change or close it. */}
            {submenu && createPortal(
            <Box
                ref={panelRef}
                onMouseEnter={stayOnSubmenu}
                onMouseLeave={() => aimAt(null)}
                sx={(t) => ({ position: 'fixed', zIndex: t.zIndex.modal + 1, ...panelPosition })}
            >
                <Paper sx={{ maxHeight: 320, overflowY: 'auto' }}>
                    <MenuList dense sx={{ py: 0.5 }}>
                        {checkpointsOf(submenu.lora).map((ckpt, ci, arr) => (
                            <MenuItem
                                key={ckpt}
                                selected={slots[openSlot]?.path === ckpt}
                                onClick={() => {
                                    if (openSlot !== null) setSlot(openSlot, { path: ckpt });
                                    closeSubmenu();
                                    setOpenSlot(null);
                                }}
                                sx={{ gap: 1 }}
                            >
                                <Typography variant="body2">{parseCheckpointLabel(ckpt)}</Typography>
                                {ci === arr.length - 1 && (
                                    <Chip size="small" label="latest" color="primary" variant="outlined" sx={{ height: 16, fontSize: 9 }} />
                                )}
                            </MenuItem>
                        ))}
                    </MenuList>
                </Paper>
            </Box>,
            document.body,
            )}
        </>
    );
    if (bare) {
        // Inside a rail band (the Bend tab): no accordion of its own, just a
        // caption and the slots.
        return (
            <Box>
                <Tooltip title={TIPS.lora.stackInfo(MAX_SLOTS)}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
                        LoRA stack
                    </Typography>
                </Tooltip>
                {body}
            </Box>
        );
    }
    return (
        <Accordion
            disableGutters
            defaultExpanded={Boolean(value && value.some((s) => s.path))}
        >
            <AccordionSummary expandIcon={<ChevronDownIcon size={18} />}>
                {/* Hover the title to surface the help in the Info View pill
                    (when it's on) — no inline "i", matching the rest of the app. */}
                <Tooltip title={TIPS.lora.stackInfo(MAX_SLOTS)}>
                    <Typography variant="subtitle1">LoRA Stack</Typography>
                </Tooltip>
            </AccordionSummary>
            <AccordionDetails>{body}</AccordionDetails>
        </Accordion>
    );
}
