import React, { useEffect, useState } from 'react';
import {
    Box,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Button,
    Typography,
    Stack,
    Menu,
    MenuItem,
    Select,
    Slider,
    IconButton,
    Chip,
    Alert,
} from '@mui/material';
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
export default function LoraStack({ selectedModel, value, onChange }) {
    const [available, setAvailable] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [dragIndex, setDragIndex] = useState(null);
    // Which slot's Select is open (controlled so a submenu pick can close it),
    // and the run whose checkpoint submenu is currently showing.
    const [openSlot, setOpenSlot] = useState(null);
    const [submenu, setSubmenu] = useState(null);  // { anchorEl, lora } | null

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

    const hint = (() => {
        if (!selectedModel) return 'Pick a model first.';
        if (!selectedModel.endsWith('-base')) {
            return 'LoRAs need a Base model. Switch to a *-base checkpoint to use LoRAs.';
        }
        if (loading) return 'Loading LoRAs…';
        if (!compatible.length) {
            return `No LoRAs trained against ${selectedModel} yet. Train one in the Training tab.`;
        }
        return null;
    })();

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
            <AccordionDetails>
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
                                        onClose={() => { setOpenSlot(null); setSubmenu(null); }}
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
                                        <MenuItem value="" disabled onMouseEnter={() => setSubmenu(null)}>
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
                                                    // Entering any row retargets the submenu, so moving
                                                    // down the list never leaves a stale one open. We
                                                    // deliberately don't close on mouseleave — that would
                                                    // fight the diagonal travel into the submenu itself.
                                                    onMouseEnter={(e) => setSubmenu(
                                                        multi ? { anchorEl: e.currentTarget, lora: l } : null
                                                    )}
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

            {/* Checkpoint submenu. Rendered outside the Select (both are
                portaled, and this mounts later so it paints above) and anchored
                to whichever run row the pointer is over. autoFocus is off so the
                Select keeps keyboard focus and the two menus don't fight. */}
            <Menu
                open={Boolean(submenu)}
                anchorEl={submenu?.anchorEl || null}
                onClose={() => setSubmenu(null)}
                anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
                transformOrigin={{ vertical: 'top', horizontal: 'left' }}
                autoFocus={false}
                disableAutoFocusItem
                disableAutoFocus
                disableEnforceFocus
                disableRestoreFocus
                // The submenu opens on top of the Select's own popover, so its
                // backdrop would stack with that one — the theme's global
                // MuiBackdrop override (rgba(5,4,3,.6) + blur) beats MUI's
                // normally-invisible popover backdrop, and two of them read as
                // a heavy double dim. Keep this one fully transparent so the
                // page fades exactly once.
                BackdropProps={{
                    invisible: true,
                    sx: { backgroundColor: 'transparent', backdropFilter: 'none' },
                }}
                // Instant: the submenu re-targets on every row hover, and a
                // grow/fade on each one makes scanning the list feel sticky.
                transitionDuration={0}
                MenuListProps={{ dense: true, sx: { py: 0.5 } }}
                PaperProps={{ sx: { maxHeight: 320, ml: 0.5 } }}
            >
                {(submenu ? checkpointsOf(submenu.lora) : []).map((ckpt, ci, arr) => (
                    <MenuItem
                        key={ckpt}
                        selected={slots[openSlot]?.path === ckpt}
                        onClick={() => {
                            if (openSlot !== null) setSlot(openSlot, { path: ckpt });
                            setSubmenu(null);
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
            </Menu>
            </AccordionDetails>
        </Accordion>
    );
}
