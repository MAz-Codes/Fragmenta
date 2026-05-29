import React, { useEffect, useState } from 'react';
import {
    Box,
    Button,
    Typography,
    Stack,
    MenuItem,
    Select,
    Slider,
    IconButton,
    Tooltip,
    Chip,
    Alert,
} from '@mui/material';
import {
    Plus as AddIcon,
    Trash2 as RemoveIcon,
    GripVertical as DragIcon,
    Power as BypassIcon,
    Save as SaveIcon,
} from 'lucide-react';
import api from '../api';
import { isLoraCompatible } from '../utils/loraMatch';

const MAX_SLOTS = 4;
const PRESETS_KEY = 'fragmenta.lora.presets';

const loadPresets = () => {
    try { return JSON.parse(window.localStorage.getItem(PRESETS_KEY) || '{}'); }
    catch { return {}; }
};
const savePresets = (obj) => {
    try { window.localStorage.setItem(PRESETS_KEY, JSON.stringify(obj)); } catch { /* ignore */ }
};

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
 * a slot in the stack but sends strength 0. Presets persist to localStorage.
 */
export default function LoraStack({ selectedModel, value, onChange }) {
    const [available, setAvailable] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [dragIndex, setDragIndex] = useState(null);
    const [presets, setPresets] = useState(loadPresets);

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

    // --- presets ------------------------------------------------------------
    const saveCurrentPreset = () => {
        const active = slots.filter(s => s.path);
        if (!active.length) return;
        const name = window.prompt('Save LoRA stack preset as:');
        if (!name) return;
        const next = { ...presets, [name]: active };
        setPresets(next);
        savePresets(next);
    };
    const loadPreset = (name) => {
        const p = presets[name];
        if (p) onChange(p.map(s => ({ bypassed: false, ...s })));
    };
    const deletePreset = (name) => {
        const next = { ...presets };
        delete next[name];
        setPresets(next);
        savePresets(next);
    };
    const presetNames = Object.keys(presets);

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
        <Box>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                <Typography variant="subtitle2">LoRA Stack</Typography>
                <Typography variant="caption" color="text.secondary">
                    Blend up to {MAX_SLOTS} LoRAs at any strength
                </Typography>
                <Box sx={{ flex: 1 }} />
                {presetNames.length > 0 && (
                    <Select
                        size="small"
                        displayEmpty
                        value=""
                        onChange={(e) => loadPreset(e.target.value)}
                        sx={{ height: 30, fontSize: 12, minWidth: 110 }}
                        renderValue={() => 'Presets'}
                    >
                        {presetNames.map(n => (
                            <MenuItem key={n} value={n} sx={{ fontSize: 12 }}>
                                <Box sx={{ flex: 1 }}>{n}</Box>
                                <Tooltip title="Delete preset">
                                    <IconButton
                                        size="small"
                                        onClick={(e) => { e.stopPropagation(); deletePreset(n); }}
                                    >
                                        <RemoveIcon size={12} />
                                    </IconButton>
                                </Tooltip>
                            </MenuItem>
                        ))}
                    </Select>
                )}
                <Tooltip title="Save current stack as a preset">
                    <span>
                        <IconButton
                            size="small"
                            onClick={saveCurrentPreset}
                            disabled={!slots.some(s => s.path)}
                        >
                            <SaveIcon size={15} />
                        </IconButton>
                    </span>
                </Tooltip>
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

            {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
            {hint && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    {hint}
                </Typography>
            )}

            {slots.length > 0 && (
                <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                    {slots.map((slot, idx) => {
                        const choice = available.find(l => l.path === slot.path);
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
                                    <Tooltip title="Drag to reorder (slot 0 loads first)">
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
                                        onChange={(e) => setSlot(idx, { path: String(e.target.value) })}
                                        sx={{ flex: 1, minWidth: 0 }}
                                    >
                                        <MenuItem value="" disabled>
                                            <em>Pick a LoRA</em>
                                        </MenuItem>
                                        {compatible.map(l => (
                                            <MenuItem key={l.id} value={l.path}>
                                                <Box>
                                                    <Typography variant="body2">
                                                        {l.name} · {l.checkpoint}
                                                    </Typography>
                                                    <Stack direction="row" spacing={0.5} sx={{ mt: 0.25 }}>
                                                        <Chip size="small" label={l.adapter_type || 'lora'} sx={{ height: 16, fontSize: 9 }} />
                                                        {l.rank && <Chip size="small" label={`r=${l.rank}`} sx={{ height: 16, fontSize: 9 }} />}
                                                    </Stack>
                                                </Box>
                                            </MenuItem>
                                        ))}
                                    </Select>
                                    <Tooltip title={bypassed ? 'Bypassed (strength 0) — click to enable' : 'Bypass this slot'}>
                                        <IconButton
                                            size="small"
                                            color={bypassed ? 'default' : 'primary'}
                                            onClick={() => setSlot(idx, { bypassed: !bypassed })}
                                        >
                                            <BypassIcon size={14} />
                                        </IconButton>
                                    </Tooltip>
                                    <Tooltip title="Remove slot">
                                        <IconButton size="small" onClick={() => removeSlot(idx)}>
                                            <RemoveIcon size={14} />
                                        </IconButton>
                                    </Tooltip>
                                </Stack>

                                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mt: 1 }}>
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
        </Box>
    );
}
