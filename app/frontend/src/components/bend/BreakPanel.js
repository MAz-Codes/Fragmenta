import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    Box, Typography, Button, Select, MenuItem, TextField, Slider, Switch,
    Accordion, AccordionSummary, AccordionDetails, FormControl, InputLabel,
    Alert, ToggleButton, ToggleButtonGroup, useTheme,
} from '@mui/material';
import {
    ChevronDown as ExpandMoreIcon, Hammer as HammerIcon,
    GitMerge as MergeIcon,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import TrainingMonitor from '../TrainingMonitor';
import api from '../../api';
import { TIPS } from '../../tooltips';

/**
 * Break — model bending: deliberate mis-training (Gillespie & Schachter's
 * practice moved to the training loop). Recipes over the existing trainer
 * plus the train_bend.py interventions, and the Adapter Lab (LoRA bending
 * + Model Blending) that fixes bends into persistent instruments.
 *
 * Training runs through the exact same /api/start-training pipeline as
 * the Training tab — same run dirs, same LoRA picker, same monitor.
 */

// VRAM figures match the README's LoRA table (rank-16 defaults).
const BASES = [
    { id: 'sa3-small-music-base', label: 'Small Music (base)', vram: '~2.5 GB' },
    { id: 'sa3-small-sfx-base', label: 'Small SFX (base)', vram: '~2.5 GB' },
    { id: 'sa3-medium-base', label: 'Medium (base) — NVIDIA GPU', vram: '~6.5 GB' },
];

// Honest recipes: each is just trainer knobs + interventions, all visible
// after selection so the "expert panel" is the same panel.
const RECIPES = {
    underfit: {
        label: 'Underfit sketch',
        hint: 'Barely learn: few steps, dense checkpoints from the very start. Every half-formed checkpoint is an instrument — pick them from the LoRA list afterwards.',
        train: { steps: 300, checkpointSteps: 25, learningRate: 5e-5, loraDropout: 0.0 },
        bend: {},
    },
    overfit: {
        label: 'Overfit mantra',
        hint: 'Memorize a tiny dataset until it warps: many steps over few clips, no dropout. The model recites your material — then mutates it.',
        train: { steps: 8000, checkpointSteps: 500, learningRate: 2e-4, loraDropout: 0.0 },
        bend: {},
    },
    amnesia: {
        label: 'Amnesia',
        hint: 'Learn, then audibly unlearn: the learning rate dips negative in periodic windows.',
        train: { steps: 2000, checkpointSteps: 100, learningRate: 1e-4, loraDropout: 0.0 },
        bend: { amnesia: { period: 200, duty: 0.25, lr_scale: -1.0 } },
    },
    wrongLessons: {
        label: 'Wrong lessons',
        hint: 'A third of the clips train against another clip\'s caption — words and sounds mis-associate.',
        train: { steps: 3000, checkpointSteps: 250, learningRate: 1e-4, loraDropout: 0.0 },
        bend: { caption_shuffle: 0.33 },
    },
    gradientStorm: {
        label: 'Gradient storm',
        hint: 'Noise injected into every gradient — the adapter never settles, textures smear.',
        train: { steps: 2000, checkpointSteps: 200, learningRate: 1e-4, loraDropout: 0.0 },
        bend: { grad_noise: 1.0 },
    },
    splitBrain: {
        label: 'Split brain',
        hint: 'Every 100 steps a different random half of the adapter is frozen — two half-trained minds in one file.',
        train: { steps: 2000, checkpointSteps: 200, learningRate: 1e-4, loraDropout: 0.0 },
        bend: { freeze_rotation: { period: 100, fraction: 0.5, seed: 0 } },
    },
    custom: { label: 'Custom', hint: 'Start from the defaults and shape every intervention yourself.', train: {}, bend: {} },
};

function InterventionSlider({ label, hint, value, onChange, min, max, step, format, accent }) {
    const active = value !== null && value !== undefined;
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Switch size="small" checked={active}
                    onChange={(e) => onChange(e.target.checked ? ((min + max) / 2) : null)} />
            <Tooltip title={hint}>
                <Typography variant="caption" sx={{
                    minWidth: 110, color: active ? accent : 'text.secondary',
                }}>{label}</Typography>
            </Tooltip>
            <Slider size="small" disabled={!active}
                    value={active ? value : min} min={min} max={max} step={step}
                    onChange={(_, v) => onChange(v)} valueLabelDisplay="auto"
                    color={active ? 'bend' : 'primary'} sx={active ? { color: 'bend.main' } : { opacity: 0.3 }} />
            <Typography variant="caption" sx={{ minWidth: 42, textAlign: 'right' }}>
                {active ? (format ? format(value) : value) : '—'}
            </Typography>
        </Box>
    );
}

export default function BreakPanel({ defaultBase }) {
    const theme = useTheme();
    const accent = theme.palette.bend?.main || '#AEB9C4';

    const [projects, setProjects] = useState([]);
    const [project, setProject] = useState('');
    const [base, setBase] = useState(defaultBase || 'sa3-small-music-base');
    const [runName, setRunName] = useState('');
    const [recipe, setRecipe] = useState('underfit');

    const [train, setTrain] = useState({ steps: 300, checkpointSteps: 25, learningRate: 5e-5, loraDropout: 0.0 });
    // Interventions: null = off.
    const [gradNoise, setGradNoise] = useState(null);
    const [gradFlip, setGradFlip] = useState(null);
    const [amnesiaDuty, setAmnesiaDuty] = useState(null);
    const [captionShuffle, setCaptionShuffle] = useState(null);
    const [timestepSkew, setTimestepSkew] = useState('none');
    const [lossScale, setLossScale] = useState(null);
    const [freezeFraction, setFreezeFraction] = useState(null);

    const [status, setStatus] = useState(null);
    const [history, setHistory] = useState([]);
    const [message, setMessage] = useState('');
    const [starting, setStarting] = useState(false);
    const pollRef = useRef(null);

    // Adapter Lab state.
    const [loras, setLoras] = useState([]);
    const [blendA, setBlendA] = useState('');
    const [blendB, setBlendB] = useState('');
    const [blendK, setBlendK] = useState(0.5);
    const [labName, setLabName] = useState('');
    const [labBusy, setLabBusy] = useState(false);
    const [labMsg, setLabMsg] = useState('');
    const [bendSource, setBendSource] = useState('');
    const [bendScale, setBendScale] = useState(1.0);
    const [bendNoise, setBendNoise] = useState(0.0);

    useEffect(() => {
        api.get('/api/projects')
            .then(({ data }) => setProjects(data.projects || []))
            .catch(() => {});
        api.get('/api/loras')
            .then(({ data }) => setLoras(data.loras || []))
            .catch(() => {});
    }, []);

    const applyRecipe = (key) => {
        setRecipe(key);
        const r = RECIPES[key];
        if (!r) return;
        setTrain(t => ({ ...t, ...r.train }));
        setGradNoise(r.bend.grad_noise ?? null);
        setGradFlip(r.bend.grad_flip?.fraction ?? null);
        setAmnesiaDuty(r.bend.amnesia?.duty ?? null);
        setCaptionShuffle(r.bend.caption_shuffle ?? null);
        setTimestepSkew(r.bend.timestep_skew ?? 'none');
        setLossScale(r.bend.loss_scale ?? null);
        setFreezeFraction(r.bend.freeze_rotation?.fraction ?? null);
    };

    const buildBend = useCallback(() => {
        const bend = {};
        if (gradNoise != null) bend.grad_noise = gradNoise;
        if (gradFlip != null) bend.grad_flip = { fraction: gradFlip, seed: 0 };
        if (amnesiaDuty != null) bend.amnesia = { period: 200, duty: amnesiaDuty, lr_scale: -1.0 };
        if (captionShuffle != null) bend.caption_shuffle = captionShuffle;
        if (timestepSkew !== 'none') bend.timestep_skew = timestepSkew;
        if (lossScale != null) bend.loss_scale = lossScale;
        if (freezeFraction != null) bend.freeze_rotation = { period: 100, fraction: freezeFraction, seed: 0 };
        return bend;
    }, [gradNoise, gradFlip, amnesiaDuty, captionShuffle, timestepSkew, lossScale, freezeFraction]);

    const stopPolling = () => {
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
    useEffect(() => stopPolling, []);

    const startPolling = () => {
        stopPolling();
        pollRef.current = setInterval(async () => {
            try {
                const { data } = await api.get('/api/training-status');
                setStatus(data);
                setHistory(h => [...h.slice(-400), data]);
                if (!data.is_training) stopPolling();
            } catch { /* keep polling */ }
        }, 2000);
    };

    const start = async (overwrite = false) => {
        if (!project) { setMessage('Pick a Dataset Workbench project first.'); return; }
        if (!runName.trim()) { setMessage('Name the run.'); return; }
        setStarting(true);
        setMessage('');
        try {
            await api.post('/api/start-training', {
                modelName: runName.trim(),
                baseModel: base,
                projectName: project,
                steps: train.steps,
                checkpointSteps: train.checkpointSteps,
                learningRate: train.learningRate,
                loraDropout: train.loraDropout,
                overwrite,
                bend: buildBend(),
            });
            setMessage('Bent training started — watch it below. Checkpoints appear in the LoRA picker as they are written.');
            setHistory([]);
            startPolling();
        } catch (err) {
            const data = err.response?.data;
            if (data?.code === 'run_exists') {
                if (window.confirm(`${data.message}\n\nOverwrite?`)) { setStarting(false); return start(true); }
            } else {
                setMessage(data?.error || data?.message || String(err.message));
            }
        } finally {
            setStarting(false);
        }
    };

    const stop = async () => {
        try { await api.post('/api/stop-training'); } catch { /* ignore */ }
    };

    // /api/loras returns all_checkpoints as plain relative path strings.
    const allCheckpoints = loras.flatMap(l =>
        (l.all_checkpoints || []).map(p => ({
            label: `${l.name} / ${String(p).split('/').pop().replace('.safetensors', '')}${l.bent ? ' ⟡bent' : ''}`,
            path: p,
        })));

    const runLab = async (payload) => {
        setLabBusy(true);
        setLabMsg('');
        try {
            const { data } = await api.post('/api/bend/lora', payload);
            setLabMsg(`Saved: ${data.output?.split('/').slice(-3).join('/')}${
                data.warnings?.length ? ` (${data.warnings.length} keys kept from A)` : ''}`);
            const r = await api.get('/api/loras');
            setLoras(r.data.loras || []);
        } catch (err) {
            setLabMsg(err.response?.data?.error || String(err.message));
        } finally {
            setLabBusy(false);
        }
    };

    const interventionsOn = Object.keys(buildBend()).length > 0;

    return (
        <Box>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Break trains adapters wrong on purpose — underfit them, overfit
                them, corrupt the lessons. Runs use the same pipeline and
                appear in the same LoRA picker as normal training, tagged
                “bent”.
            </Typography>

            {/* recipes */}
            <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mb: 1 }}>
                {Object.entries(RECIPES).map(([key, r]) => (
                    <Tooltip key={key} title={r.hint}>
                        <ToggleButton
                            size="small" value={key} selected={recipe === key}
                            onChange={() => applyRecipe(key)}
                            sx={{ px: 1.5, py: 0.4, fontSize: '0.72rem' }}
                        >
                            {r.label}
                        </ToggleButton>
                    </Tooltip>
                ))}
            </Box>
            <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 2 }}>
                {RECIPES[recipe]?.hint}
            </Typography>

            {/* run setup */}
            <Box sx={{ display: 'grid', gap: 1.5, mb: 2,
                       gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' } }}>
                <FormControl size="small">
                    <InputLabel>Dataset project</InputLabel>
                    <Select value={project} label="Dataset project"
                            onChange={(e) => setProject(e.target.value)}>
                        {projects.map(p => (
                            <MenuItem key={p.name || p} value={p.name || p}>{p.name || p}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <FormControl size="small">
                    <InputLabel>Base model</InputLabel>
                    <Select value={base} label="Base model"
                            onChange={(e) => setBase(e.target.value)}>
                        {BASES.map(b => <MenuItem key={b.id} value={b.id}>{b.label}</MenuItem>)}
                    </Select>
                </FormControl>
                <TextField size="small" label="Run name" value={runName}
                           placeholder="e.g. broken-choir-v1"
                           onChange={(e) => setRunName(e.target.value)} />
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <Tooltip title="Training steps. Underfitting lives at the low end.">
                        <TextField size="small" label="Steps" type="number" value={train.steps}
                                   onChange={(e) => setTrain(t => ({ ...t, steps: Math.max(1, parseInt(e.target.value || 1, 10)) }))}
                                   sx={{ width: 110 }} />
                    </Tooltip>
                    <Tooltip title="Checkpoint cadence — dense early checkpoints are the underfit archaeology: every half-learned state is kept as a playable adapter.">
                        <TextField size="small" label="Ckpt every" type="number" value={train.checkpointSteps}
                                   onChange={(e) => setTrain(t => ({ ...t, checkpointSteps: Math.max(1, parseInt(e.target.value || 1, 10)) }))}
                                   sx={{ width: 110 }} />
                    </Tooltip>
                    <Tooltip title="Learning rate.">
                        <TextField size="small" label="LR" type="number" value={train.learningRate}
                                   inputProps={{ step: 1e-5 }}
                                   onChange={(e) => setTrain(t => ({ ...t, learningRate: parseFloat(e.target.value) || 1e-4 }))}
                                   sx={{ width: 130 }} />
                    </Tooltip>
                </Box>
            </Box>

            <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: -1, mb: 2 }}>
                Breaking is real training: {BASES.find(b => b.id === base)?.vram || '~2.5 GB'} of
                VRAM, and {train.steps} steps take as long as a normal run of the same length —
                {' '}~{Math.max(1, Math.round(train.steps / train.checkpointSteps))} checkpoints will be written.
            </Typography>

            {/* interventions */}
            <Typography variant="subtitle2" sx={{ mb: 1, color: interventionsOn ? accent : 'text.secondary' }}>
                Bent backprop
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75, mb: 2 }}>
                <InterventionSlider label="Gradient noise" accent={accent}
                    hint="Gaussian noise added to every gradient, scaled to its own level — learning that never settles."
                    value={gradNoise} onChange={setGradNoise} min={0.1} max={3} step={0.05} />
                <InterventionSlider label="Gradient flip" accent={accent}
                    hint="This fraction of the adapter's parameters gets its gradients negated — those layers learn away from the data."
                    value={gradFlip} onChange={setGradFlip} min={0.05} max={1} step={0.05}
                    format={(v) => `${Math.round(v * 100)}%`} />
                <InterventionSlider label="Amnesia" accent={accent}
                    hint="Periodic windows of negative learning rate — learn, then audibly unlearn. Value = fraction of each 200-step period spent unlearning."
                    value={amnesiaDuty} onChange={setAmnesiaDuty} min={0.05} max={0.9} step={0.05}
                    format={(v) => `${Math.round(v * 100)}%`} />
                <InterventionSlider label="Caption shuffle" accent={accent}
                    hint="Probability a clip trains against a random other clip's caption — wrong word–sound associations."
                    value={captionShuffle} onChange={setCaptionShuffle} min={0.05} max={1} step={0.05}
                    format={(v) => `${Math.round(v * 100)}%`} />
                <InterventionSlider label="Loss scale" accent={accent}
                    hint="Global loss multiplier. Negative = pure anti-learning."
                    value={lossScale} onChange={setLossScale} min={-2} max={2} step={0.1}
                    format={(v) => v.toFixed(1)} />
                <InterventionSlider label="Split brain" accent={accent}
                    hint="Every 100 steps a different random fraction of the adapter is frozen."
                    value={freezeFraction} onChange={setFreezeFraction} min={0.1} max={0.9} step={0.05}
                    format={(v) => `${Math.round(v * 100)}%`} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, pl: 5 }}>
                    <Tooltip title="Train only one denoising regime: texture = the fine-detail steps, structure = the composition-defining steps.">
                        <Typography variant="caption" sx={{
                            minWidth: 110, color: timestepSkew !== 'none' ? accent : 'text.secondary',
                        }}>Timestep skew</Typography>
                    </Tooltip>
                    <ToggleButtonGroup size="small" exclusive value={timestepSkew}
                                       onChange={(_, v) => v && setTimestepSkew(v)}
                                       sx={{ '& .MuiToggleButton-root': { py: 0.25, px: 1.25, fontSize: '0.7rem' } }}>
                        <ToggleButton value="none">off</ToggleButton>
                        <ToggleButton value="structure">structure</ToggleButton>
                        <ToggleButton value="texture">texture</ToggleButton>
                    </ToggleButtonGroup>
                </Box>
            </Box>

            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', mb: 2 }}>
                <Button variant="contained" color="bend" disabled={starting || status?.is_training}
                        onClick={() => start(false)} startIcon={<HammerIcon size={16} />}>
                    {status?.is_training ? 'Training…' : 'Start bent training'}
                </Button>
                {status?.is_training && (
                    <Button variant="outlined" color="error" onClick={stop}>Stop</Button>
                )}
                {message && (
                    <Typography variant="caption" color="textSecondary">{message}</Typography>
                )}
            </Box>

            {(status?.is_training || status?.status === 'completed' || history.length > 0) && (
                <Box sx={{ mb: 2 }}>
                    <TrainingMonitor
                        trainingProgress={status?.progress || 0}
                        trainingStatus={status}
                        trainingHistory={history}
                        trainingError={status?.error}
                        indicatorState={{
                            status: status?.is_training ? 'running' : (status?.status || 'idle'),
                            label: status?.is_training ? 'Breaking' : (status?.status || 'Idle'),
                            animate: !!status?.is_training,
                        }}
                    />
                </Box>
            )}

            {/* Adapter Lab */}
            <Accordion disableGutters>
                <AccordionSummary expandIcon={<ExpandMoreIcon size={18} />}>
                    <Typography variant="subtitle1">Adapter Lab — bend & blend trained LoRAs</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                        Fix a bend into a persistent instrument: results are
                        written as new adapters in the LoRA picker, sources
                        stay untouched. Blending two checkpoints of the same
                        run plays the training trajectory itself.
                    </Typography>

                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Model Blending</Typography>
                    <Box sx={{ display: 'grid', gap: 1.5, mb: 1,
                               gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' } }}>
                        <FormControl size="small">
                            <InputLabel>Adapter A</InputLabel>
                            <Select value={blendA} label="Adapter A" onChange={(e) => setBlendA(e.target.value)}>
                                {allCheckpoints.map(c => (
                                    <MenuItem key={c.path} value={c.path}>{c.label}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                        <FormControl size="small">
                            <InputLabel>Adapter B</InputLabel>
                            <Select value={blendB} label="Adapter B" onChange={(e) => setBlendB(e.target.value)}>
                                {allCheckpoints.map(c => (
                                    <MenuItem key={c.path} value={c.path}>{c.label}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                        <Typography variant="caption" color="textSecondary">A</Typography>
                        <Slider size="small" value={blendK} min={0} max={1} step={0.01}
                                onChange={(_, v) => setBlendK(v)} valueLabelDisplay="auto"
                                color="bend" sx={{ maxWidth: 220, color: 'bend.main' }} />
                        <Typography variant="caption" color="textSecondary">B</Typography>
                        <TextField size="small" label="New adapter name" value={labName}
                                   onChange={(e) => setLabName(e.target.value)} sx={{ flex: 1, minWidth: 140 }} />
                        <Button variant="outlined" size="small" startIcon={<MergeIcon size={14} />}
                                disabled={labBusy || !blendA || !blendB || !labName.trim()}
                                onClick={() => runLab({ mode: 'blend', source: blendA, source_b: blendB, k: blendK, output_name: labName.trim() })}>
                            Blend
                        </Button>
                    </Box>

                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Bend one adapter</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                        <FormControl size="small" sx={{ minWidth: 200 }}>
                            <InputLabel>Adapter</InputLabel>
                            <Select value={bendSource} label="Adapter" onChange={(e) => setBendSource(e.target.value)}>
                                {allCheckpoints.map(c => (
                                    <MenuItem key={c.path} value={c.path}>{c.label}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="caption" color="textSecondary">scale</Typography>
                            <Slider size="small" value={bendScale} min={-2} max={3} step={0.05}
                                    onChange={(_, v) => setBendScale(v)} valueLabelDisplay="auto"
                                    sx={{ width: 110 }} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="caption" color="textSecondary">noise</Typography>
                            <Slider size="small" value={bendNoise} min={0} max={2} step={0.05}
                                    onChange={(_, v) => setBendNoise(v)} valueLabelDisplay="auto"
                                    sx={{ width: 110 }} />
                        </Box>
                        <Button variant="outlined" size="small"
                                disabled={labBusy || !bendSource || !labName.trim()
                                          || (bendScale === 1.0 && bendNoise === 0)}
                                onClick={() => runLab({
                                    mode: 'bend', source: bendSource, output_name: labName.trim(),
                                    ops: [
                                        ...(bendScale !== 1.0 ? [{ op: 'scale', factor: bendScale }] : []),
                                        ...(bendNoise > 0 ? [{ op: 'noise', amount: bendNoise, seed: Math.floor(Math.random() * 9999) }] : []),
                                    ],
                                })}>
                            Bend
                        </Button>
                    </Box>
                    {labMsg && <Alert severity="info" sx={{ mt: 2 }}>{labMsg}</Alert>}
                </AccordionDetails>
            </Accordion>
        </Box>
    );
}
