import React, { useMemo, useRef } from 'react';
import {
    Box, Typography, IconButton, Slider, Select, MenuItem, Switch,
    ToggleButton, ToggleButtonGroup, TextField, useTheme,
} from '@mui/material';
import {
    Dices as DicesIcon, X as XIcon, ChevronUp, ChevronDown,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import { defaultParams, describeModule } from './bendUtils';

/**
 * One module in the bend rack. Declarative: the card edits the patch
 * entry; nothing is applied until Generate. Sliders are continuous
 * (Brave: continuous modification, dry/wet per module).
 */

/** Compact block-grid selector for DiT / decoder stages. */
function BlockGrid({ count, value, onChange, warm }) {
    const selected = useMemo(
        () => new Set(value === null || value === undefined ? [] : value),
        [value]);
    const allSelected = value === null || value === undefined;
    const toggle = (i) => {
        const next = new Set(allSelected ? Array.from({ length: count }, (_, k) => k) : selected);
        if (next.has(i)) next.delete(i); else next.add(i);
        onChange(next.size === 0 ? null : [...next].sort((a, b) => a - b));
    };
    return (
        <Box>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.4 }}>
                {Array.from({ length: count }, (_, i) => {
                    const on = allSelected || selected.has(i);
                    return (
                        <Box
                            key={i}
                            onClick={() => toggle(i)}
                            sx={{
                                width: 20, height: 20, borderRadius: 0.75,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '0.55rem', cursor: 'pointer', userSelect: 'none',
                                fontFamily: 'inherit',
                                border: '1px solid',
                                borderColor: on ? warm : 'divider',
                                backgroundColor: on ? `${warm}33` : 'transparent',
                                color: on ? warm : 'text.disabled',
                                transition: 'all 120ms ease',
                                '&:hover': { borderColor: warm },
                            }}
                        >
                            {i}
                        </Box>
                    );
                })}
            </Box>
            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                {[
                    ['all', () => onChange(null)],
                    ['early', () => onChange(Array.from({ length: Math.ceil(count / 3) }, (_, i) => i))],
                    ['late', () => onChange(Array.from({ length: Math.ceil(count / 3) }, (_, i) => count - Math.ceil(count / 3) + i))],
                    ['odd', () => onChange(Array.from({ length: count }, (_, i) => i).filter(i => i % 2))],
                ].map(([label, fn]) => (
                    <Typography
                        key={label} variant="caption" onClick={fn}
                        sx={{ cursor: 'pointer', color: 'text.disabled',
                              '&:hover': { color: 'primary.main' }, fontSize: '0.65rem' }}
                    >
                        {label}
                    </Typography>
                ))}
            </Box>
        </Box>
    );
}

/** Minimal freehand curve editor for the `draw` operator (K&F). Rendered
 *  as an abstract bend curve over labelled index axes — deliberately NOT
 *  waveform-styled (Brave: weight arrays that look like waveforms confuse
 *  musicians). Y range 0..2 (1 = unity for scale mode). */
function CurveEditor({ points, onChange, warm }) {
    const N = 24;
    const H = 64;
    const ref = useRef(null);
    const pts = points && points.length >= 2
        ? points
        : Array.from({ length: N }, () => 1.0);

    const setFromEvent = (e) => {
        const rect = ref.current.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const x = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
        const y = Math.min(1, Math.max(0, (clientY - rect.top) / rect.height));
        const idx = Math.min(N - 1, Math.round(x * (N - 1)));
        const next = [...(pts.length === N ? pts : Array.from({ length: N }, (_, i) =>
            pts[Math.min(pts.length - 1, Math.round(i * (pts.length - 1) / (N - 1)))]))];
        next[idx] = (1 - y) * 2;
        onChange(next);
    };
    const dragging = useRef(false);

    const path = pts.map((v, i) =>
        `${i === 0 ? 'M' : 'L'} ${(i / (pts.length - 1)) * 100} ${H - (v / 2) * H}`).join(' ');

    return (
        <Box>
            <Box
                ref={ref}
                onMouseDown={(e) => { dragging.current = true; setFromEvent(e); }}
                onMouseMove={(e) => dragging.current && setFromEvent(e)}
                onMouseUp={() => { dragging.current = false; }}
                onMouseLeave={() => { dragging.current = false; }}
                onTouchStart={(e) => { dragging.current = true; setFromEvent(e); }}
                onTouchMove={(e) => dragging.current && setFromEvent(e)}
                onTouchEnd={() => { dragging.current = false; }}
                sx={{
                    position: 'relative', height: H, borderRadius: 1.5,
                    border: '1px dashed', borderColor: 'divider',
                    cursor: 'crosshair', touchAction: 'none', overflow: 'hidden',
                }}
            >
                <svg width="100%" height={H} viewBox={`0 0 100 ${H}`} preserveAspectRatio="none"
                     style={{ display: 'block' }}>
                    <line x1="0" y1={H / 2} x2="100" y2={H / 2}
                          stroke="currentColor" strokeOpacity="0.15" strokeDasharray="2 3" />
                    <path d={path} fill="none" stroke={warm} strokeWidth="1.6"
                          vectorEffect="non-scaling-stroke" />
                </svg>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '0.6rem' }}>
                    feature 0 → N · drag to draw · centre = unity
                </Typography>
                <Typography
                    variant="caption" onClick={() => onChange([])}
                    sx={{ cursor: 'pointer', color: 'text.disabled', fontSize: '0.6rem',
                          '&:hover': { color: 'primary.main' } }}
                >
                    reset
                </Typography>
            </Box>
        </Box>
    );
}

function ParamControls({ opSpec, params, onChange, warm }) {
    if (!opSpec) return null;
    return (
        <>
            {Object.entries(opSpec.params || {}).map(([name, desc]) => {
                const value = params?.[name] ?? desc.default;
                if (desc.type === 'bool') {
                    return (
                        <Box key={name} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="caption" color="textSecondary">{name}</Typography>
                            <Switch size="small" checked={!!value}
                                    onChange={(e) => onChange({ ...params, [name]: e.target.checked })} />
                        </Box>
                    );
                }
                if (desc.type === 'enum') {
                    return (
                        <Box key={name} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="caption" color="textSecondary">{name}</Typography>
                            <Select
                                size="small" value={value ?? desc.options?.[0]}
                                onChange={(e) => onChange({ ...params, [name]: e.target.value })}
                                sx={{ minWidth: 92, '& .MuiSelect-select': { py: 0.4 } }}
                            >
                                {(desc.options || []).map(o =>
                                    <MenuItem key={o} value={o}>{o}</MenuItem>)}
                            </Select>
                        </Box>
                    );
                }
                if (desc.type === 'curve') {
                    return <CurveEditor key={name} points={value}
                                        onChange={(pts) => onChange({ ...params, [name]: pts })}
                                        warm={warm} />;
                }
                return (
                    <Box key={name} sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <Typography variant="caption" color="textSecondary"
                                    sx={{ minWidth: 52 }}>{name}</Typography>
                        <Slider
                            size="small" value={Number(value ?? 0)}
                            min={desc.min} max={desc.max}
                            step={desc.step || (desc.type === 'int' ? 1 : 0.01)}
                            onChange={(_, v) => onChange({
                                ...params,
                                [name]: desc.type === 'int' ? Math.round(v) : v,
                            })}
                            valueLabelDisplay="auto"
                        />
                        <Typography variant="caption" sx={{ minWidth: 34, textAlign: 'right' }}>
                            {desc.type === 'int' ? Math.round(value) : Number(value ?? 0).toFixed(2)}
                        </Typography>
                    </Box>
                );
            })}
        </>
    );
}

export default function BendModuleCard({
    module: mod, registry, onChange, onRemove, onRandomize,
    onMoveUp, onMoveDown, isFirst, isLast,
}) {
    const theme = useTheme();
    const warm = theme.palette.warm?.main || '#FDA22B';
    const stageInfo = (registry?.stages || []).find(s => s.stage === mod.target?.stage);
    const domain = mod.target?.domain || 'activation';
    const isStructure = domain === 'structure';
    const opSpec = (registry?.operators || []).find(o => o.name === mod.operator);
    const availableOps = (registry?.operators || []).filter(o =>
        o.domains.includes(domain === 'latent' ? 'latent' : domain));

    const set = (patch) => onChange({ ...mod, ...patch });
    const setTarget = (patch) => onChange({ ...mod, target: { ...mod.target, ...patch } });

    const enabled = mod.enabled !== false;
    const features = mod.target?.features || { mode: 'all' };

    return (
        <Box sx={{
            borderRadius: 2.5, p: { xs: 1.25, sm: 1.75 }, mb: 1.25,
            border: '1px solid',
            borderColor: enabled ? `${warm}66` : 'divider',
            backgroundColor: enabled ? `${warm}0A` : 'transparent',
            opacity: enabled ? 1 : 0.55,
            transition: 'all 220ms ease',
        }}>
            {/* header */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                <Switch size="small" checked={enabled}
                        onChange={(e) => set({ enabled: e.target.checked })} />
                <Typography variant="subtitle2" sx={{
                    flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap', color: enabled ? warm : 'text.secondary',
                    textTransform: 'none', fontSize: '0.8rem',
                }}>
                    {describeModule(mod, registry)}
                </Typography>
                <Tooltip title="Re-roll this module's settings (chance, scoped small)">
                    <IconButton size="small" onClick={onRandomize}><DicesIcon size={15} /></IconButton>
                </Tooltip>
                <IconButton size="small" disabled={isFirst} onClick={onMoveUp}><ChevronUp size={15} /></IconButton>
                <IconButton size="small" disabled={isLast} onClick={onMoveDown}><ChevronDown size={15} /></IconButton>
                <IconButton size="small" onClick={onRemove}
                            sx={{ '&:hover': { color: 'error.main' } }}>
                    <XIcon size={15} />
                </IconButton>
            </Box>

            <Box sx={{ display: 'grid', gap: 1.25,
                       gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
                {/* left column: where */}
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {(stageInfo?.domains || []).length > 1 && (
                        <ToggleButtonGroup
                            size="small" exclusive value={domain}
                            onChange={(_, v) => {
                                if (!v || v === domain) return;
                                const next = { ...mod, target: { ...mod.target, domain: v } };
                                if (v === 'structure') {
                                    delete next.operator; delete next.params; delete next.mix;
                                    next.structure = { type: 'bypass' };
                                } else {
                                    delete next.structure;
                                    if (!next.operator) {
                                        next.operator = 'scale';
                                        next.params = defaultParams(
                                            (registry?.operators || []).find(o => o.name === 'scale'));
                                        next.mix = 1.0;
                                    }
                                    if (v === 'weight') {
                                        next.target.param = next.target.param || 'weight';
                                        if (mod.target.stage === 'dit') {
                                            next.target.weight_slot = next.target.weight_slot || 'ff';
                                        }
                                    }
                                }
                                onChange(next);
                            }}
                            sx={{ '& .MuiToggleButton-root': { py: 0.25, px: 1.25, fontSize: '0.7rem' } }}
                        >
                            {(stageInfo.domains).map(d => (
                                <ToggleButton key={d} value={d}>
                                    {d === 'activation' ? 'signal' : d === 'weight' ? 'weights' : d}
                                </ToggleButton>
                            ))}
                        </ToggleButtonGroup>
                    )}

                    {stageInfo?.kind === 'blocks' && (
                        <BlockGrid
                            count={stageInfo.count || 4}
                            value={mod.target?.blocks}
                            onChange={(blocks) => setTarget({ blocks })}
                            warm={warm}
                        />
                    )}

                    {domain === 'weight' && (
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            {mod.target?.stage === 'dit' && (
                                <Select size="small" value={mod.target?.weight_slot || 'ff'}
                                        onChange={(e) => setTarget({ weight_slot: e.target.value })}
                                        sx={{ '& .MuiSelect-select': { py: 0.4 } }}>
                                    {(stageInfo?.weight_slots || []).map(s => (
                                        <MenuItem key={s.name} value={s.name}>{s.label}</MenuItem>
                                    ))}
                                </Select>
                            )}
                            <Select size="small" value={mod.target?.param || 'weight'}
                                    onChange={(e) => setTarget({ param: e.target.value })}
                                    sx={{ '& .MuiSelect-select': { py: 0.4 } }}>
                                <MenuItem value="weight">weights</MenuItem>
                                <MenuItem value="bias">biases</MenuItem>
                                <MenuItem value="both">both</MenuItem>
                            </Select>
                        </Box>
                    )}

                    {!isStructure && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                            <Typography variant="caption" color="textSecondary">features</Typography>
                            <Select size="small" value={features.mode || 'all'}
                                    onChange={(e) => {
                                        const mode = e.target.value;
                                        setTarget({
                                            features: mode === 'all' ? { mode: 'all' }
                                                : mode === 'cluster'
                                                    ? { mode: 'cluster', k: features.k ?? 4,
                                                        index: features.index ?? 0, seed: features.seed ?? 1 }
                                                    : { mode: 'random',
                                                        fraction: features.fraction ?? 0.5,
                                                        seed: features.seed ?? 1 },
                                        });
                                    }}
                                    sx={{ '& .MuiSelect-select': { py: 0.4 } }}>
                                <MenuItem value="all">all</MenuItem>
                                <MenuItem value="random">random subset</MenuItem>
                                <MenuItem value="cluster">cluster (features that act together)</MenuItem>
                            </Select>
                            {features.mode === 'cluster' && (
                                <>
                                    <Select size="small" value={features.k ?? 4}
                                            onChange={(e) => setTarget({ features: {
                                                ...features, k: e.target.value,
                                                index: Math.min(features.index ?? 0, e.target.value - 1) } })}
                                            renderValue={(v) => `${v} groups`}
                                            sx={{ '& .MuiSelect-select': { py: 0.4 } }}>
                                        {[2, 3, 4, 6, 8].map(k => <MenuItem key={k} value={k}>{k} groups</MenuItem>)}
                                    </Select>
                                    <Select size="small" value={features.index ?? 0}
                                            onChange={(e) => setTarget({ features: { ...features, index: e.target.value } })}
                                            renderValue={(v) => `group ${v + 1}`}
                                            sx={{ '& .MuiSelect-select': { py: 0.4 } }}>
                                        {Array.from({ length: features.k ?? 4 }, (_, i) => (
                                            <MenuItem key={i} value={i}>group {i + 1}</MenuItem>
                                        ))}
                                    </Select>
                                </>
                            )}
                            {features.mode === 'random' && (
                                <>
                                    <Slider size="small" value={features.fraction ?? 0.5}
                                            min={0.05} max={1} step={0.05}
                                            onChange={(_, v) => setTarget({
                                                features: { ...features, fraction: v } })}
                                            sx={{ width: 90 }} valueLabelDisplay="auto" />
                                    <Typography variant="caption" color="textSecondary">
                                        {Math.round((features.fraction ?? 0.5) * 100)}%
                                    </Typography>
                                </>
                            )}
                        </Box>
                    )}

                    {mod.steps && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <Tooltip title="Which part of the denoising the bend is active in — early steps shape structure, late steps shape texture.">
                                <Typography variant="caption" color="textSecondary"
                                            sx={{ minWidth: 52 }}>steps</Typography>
                            </Tooltip>
                            <Slider
                                size="small"
                                value={[mod.steps.from ?? 0, mod.steps.to ?? 1]}
                                min={0} max={1} step={0.05}
                                onChange={(_, v) => set({ steps: { from: v[0], to: v[1] } })}
                                valueLabelDisplay="auto"
                                valueLabelFormat={(v) => `${Math.round(v * 100)}%`}
                            />
                        </Box>
                    )}
                </Box>

                {/* right column: what */}
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {isStructure ? (
                        <>
                            <Select size="small" value={mod.structure?.type || 'bypass'}
                                    onChange={(e) => {
                                        const type = e.target.value;
                                        const structure = { type };
                                        if (type === 'repeat') structure.times = 2;
                                        if (type === 'swap_nonlinearity') structure.fn = 'sin';
                                        if (type === 'reorder') {
                                            const n = stageInfo?.count || 4;
                                            structure.order = Array.from({ length: n }, (_, i) => n - 1 - i);
                                        }
                                        set({ structure });
                                    }}
                                    sx={{ '& .MuiSelect-select': { py: 0.5 } }}>
                                <MenuItem value="bypass">Bypass — skip the blocks</MenuItem>
                                <MenuItem value="repeat">Repeat — run them again</MenuItem>
                                <MenuItem value="reorder">Reorder — reversed order</MenuItem>
                                <MenuItem value="swap_nonlinearity">Swap nonlinearity</MenuItem>
                            </Select>
                            {mod.structure?.type === 'repeat' && (
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                    <Typography variant="caption" color="textSecondary">times</Typography>
                                    <Slider size="small" value={mod.structure.times || 2}
                                            min={2} max={4} step={1} marks
                                            onChange={(_, v) => set({ structure: { ...mod.structure, times: v } })}
                                            valueLabelDisplay="auto" sx={{ maxWidth: 120 }} />
                                </Box>
                            )}
                            {mod.structure?.type === 'swap_nonlinearity' && (
                                <Select size="small" value={mod.structure.fn || 'sin'}
                                        onChange={(e) => set({ structure: { ...mod.structure, fn: e.target.value } })}
                                        sx={{ '& .MuiSelect-select': { py: 0.4 }, maxWidth: 140 }}>
                                    {(registry?.swap_fns || ['sin']).map(fn => (
                                        <MenuItem key={fn} value={fn}>{fn}</MenuItem>
                                    ))}
                                </Select>
                            )}
                        </>
                    ) : (
                        <>
                            <Select
                                size="small" value={mod.operator || 'scale'}
                                onChange={(e) => {
                                    const spec = availableOps.find(o => o.name === e.target.value);
                                    set({ operator: e.target.value, params: defaultParams(spec) });
                                }}
                                sx={{ '& .MuiSelect-select': { py: 0.5 } }}
                            >
                                {availableOps.map(o => (
                                    <MenuItem key={o.name} value={o.name}>
                                        <Box>
                                            <Typography variant="body2">{o.label}</Typography>
                                            <Typography variant="caption" color="textSecondary"
                                                        sx={{ display: 'block', whiteSpace: 'normal' }}>
                                                {o.description}
                                            </Typography>
                                        </Box>
                                    </MenuItem>
                                ))}
                            </Select>
                            <ParamControls opSpec={opSpec} params={mod.params}
                                           onChange={(params) => set({ params })} warm={warm} />
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <Tooltip title="Dry/wet: crossfade between the original signal and the bent one.">
                                    <Typography variant="caption" color="textSecondary"
                                                sx={{ minWidth: 52 }}>mix</Typography>
                                </Tooltip>
                                <Slider size="small" value={mod.mix ?? 1}
                                        min={0} max={1} step={0.01}
                                        onChange={(_, v) => set({ mix: v })}
                                        valueLabelDisplay="auto" color="warm" />
                                <Typography variant="caption" sx={{ minWidth: 34, textAlign: 'right' }}>
                                    {Math.round((mod.mix ?? 1) * 100)}%
                                </Typography>
                            </Box>
                        </>
                    )}
                </Box>
            </Box>
        </Box>
    );
}
