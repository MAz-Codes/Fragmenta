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
import { Plus as AddIcon, Trash2 as RemoveIcon } from 'lucide-react';
import api from '../api';

const MAX_SLOTS = 4;

/**
 * Multi-LoRA stack for the Generation panel.
 *
 * Props:
 *   selectedModel: the currently-selected base model id (e.g. "sa3-medium-base")
 *   value:         array of { path, strength } slots
 *   onChange:      (newSlots) => void
 *
 * The picker filters available LoRAs by base_model match. SA3 LoRAs trained
 * against `sa3-medium-base` can only be stacked on `sa3-medium-base` — we
 * surface that as a compat note rather than letting the backend 400.
 */
export default function LoraStack({ selectedModel, value, onChange }) {
    const [available, setAvailable] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        api.get('/api/loras')
            .then(r => { if (!cancelled) setAvailable(r.data.loras || []); })
            .catch(e => { if (!cancelled) setError(e.response?.data?.error || e.message); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, []);

    // LoRAs compatible with the current generation model. Strict id match —
    // the LoRA's base_model must equal selectedModel (typically a *-base).
    const compatible = available.filter(l =>
        selectedModel && l.base_model === selectedModel
    );

    // The single-LoRA case stays one click: when no slots are populated AND
    // there's a compatible LoRA, surface one empty slot so the user sees a
    // "Pick a LoRA" dropdown immediately. They never need to hit "Add LoRA"
    // unless they want a second slot. Empty path slots are filtered out in
    // the caller's request body, so this has no effect on the wire.
    const slots = (value && value.length > 0)
        ? value
        : (compatible.length ? [{ path: '', strength: 1.0 }] : []);

    const addSlot = () => {
        if (slots.length >= MAX_SLOTS) return;
        onChange([...slots, { path: '', strength: 1.0 }]);
    };

    const removeSlot = (idx) => {
        const next = slots.filter((_, i) => i !== idx);
        onChange(next);
    };

    const setSlot = (idx, patch) => {
        const next = slots.map((s, i) => i === idx ? { ...s, ...patch } : s);
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
        <Box>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                <Typography variant="subtitle2">LoRA Stack</Typography>
                <Typography variant="caption" color="text.secondary">
                    Blend up to {MAX_SLOTS} LoRAs at any strength
                </Typography>
                <Box sx={{ flex: 1 }} />
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
                        return (
                            <Box
                                key={idx}
                                sx={{
                                    p: 1.5,
                                    borderBottom: '1px solid',
                                    borderColor: 'divider',
                                    '&:last-child': { borderBottom: 'none' },
                                }}
                            >
                                <Stack direction="row" alignItems="center" spacing={1.5}>
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
                                        {slot.strength.toFixed(2)}
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
