import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Box, Typography, Paper, Button, Select, MenuItem, TextField, FormControl, InputLabel,
    Chip, Switch, FormControlLabel, LinearProgress, Alert, Divider, ToggleButton, ToggleButtonGroup,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
    Infinity as EngineIcon, Play as PlayIcon, Square as StopIcon, Circle as RecordIcon,
    Headphones as ListenIcon, Radio as PowerIcon, RefreshCw as RefreshIcon, Waves as RenderIcon,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import RotaryKnob from '../RotaryKnob';
import RailSection from '../RailSection';
import BetaBadge from '../bend/BetaBadge';
import { MidiMappable } from '../MidiContext';
import { TIPS } from '../../tooltips';
import { appStyles } from '../../theme';
import useEngine, { v1 } from './useEngine';
import BlendPad from './BlendPad';
import SpatialView from './SpatialView';

/**
 * Engine — the perpetual, parametric generator as a tab.
 *
 * A client of the engine's API and nothing more: the backend only starts and
 * stops the process (/api/engine/*); everything else here is a call any
 * script could make against /v1 (redesign/PERPETUAL_ENGINE_PLAN.md, §5 of the
 * strategy). Blend pad over the anchors, the conductor's knobs, scenes with
 * transitions, a top-down view of the events, buffers and GPU, Listen and
 * Record.
 */

const STORE_KEY = 'fragmenta.engine.v1';
const readStore = () => {
    try { return JSON.parse(window.localStorage.getItem(STORE_KEY) || '{}'); } catch { return {}; }
};
const KNOBS = [
    { path: '/events/density', label: 'Density', fmt: (v) => `${Math.round(v * 100)}%`, step: 0.01, midi: 'engine.density' },
    { path: '/events/sparseness', label: 'Silence', fmt: (v) => `${Math.round(v * 100)}%`, step: 0.01, midi: 'engine.silence' },
    { path: '/events/max_concurrent', label: 'Max events', fmt: (v) => `${v}`, step: 1 },
    { path: '/events/min_interval', label: 'Interval', fmt: (v) => `${Number(v).toFixed(1)} s`, step: 0.1 },
    { path: '/bed/takes', label: 'Takes', fmt: (v) => `${v}`, step: 1 },
    { path: '/bed/clip_length', label: 'Bed clip', fmt: (v) => `${Math.round(v)} s`, step: 1 },
    { path: '/bed/crossfade_s', label: 'Crossfade', fmt: (v) => `${Math.round(v)} s`, step: 0.5 },
    { path: '/spatial/events/distance_max', label: 'Farthest', fmt: (v) => `${Math.round(v)} m`, step: 0.5 },
    { path: '/render/lora_strength', label: 'LoRA', fmt: (v) => Number(v).toFixed(2), step: 0.05 },
    { path: '/master/gain', label: 'Master', fmt: (v) => `${Math.round(v * 100)}%`, step: 0.01, midi: 'engine.master' },
];

const errorText = (e, fallback) => (e && e.message) || fallback;

export default function EnginePanel({ models = [], loras = [], active = true, isDocker = false }) {
    const theme = useTheme();
    const stored = useMemo(readStore, []);
    const eng = useEngine({ active });
    const { proc, base, status, meters, liveEvents, log, connected, call } = eng;
    const running = !!(proc && proc.running);

    // launch options (used by Start; also sent to /v1/model while running)
    const baseModels = useMemo(() => models.filter((m) => m.kind === 'base'), [models]);
    const [modelId, setModelId] = useState(stored.modelId || 'sa3-small-sfx-base');
    const [loraPath, setLoraPath] = useState(stored.loraPath || '');
    const [layout, setLayout] = useState(stored.layout || 'stereo');
    const [radioUrl, setRadioUrl] = useState(stored.radioUrl || '');
    const [clock, setClock] = useState(!!stored.clock);
    const [busy, setBusy] = useState('');
    const [error, setError] = useState('');

    // the engine's parameters, anchors, scenes
    const [params, setParams] = useState({});
    const [anchors, setAnchors] = useState([]);
    const [blend, setBlend] = useState({});
    const [scenes, setScenes] = useState([]);
    const [transition, setTransition] = useState(stored.transition ?? 20);
    const [fromAz, setFromAz] = useState(stored.fromAz ?? 0);
    const [listening, setListening] = useState(false);
    const [oneshot, setOneshot] = useState(null);     // {id, state, url}
    const audioRef = useRef(null);
    const throttles = useRef({});

    useEffect(() => {
        try {
            window.localStorage.setItem(STORE_KEY, JSON.stringify({ modelId, loraPath, layout, transition, fromAz, radioUrl, clock }));
        } catch { /* noop */ }
    }, [modelId, loraPath, layout, transition, fromAz, radioUrl, clock]);

    const loadAll = useCallback(async () => {
        if (!base) return;
        try {
            const [p, a, s] = await Promise.all([
                v1(base, 'GET', '/v1/params?flat=1'), v1(base, 'GET', '/v1/anchors'), v1(base, 'GET', '/v1/scenes'),
            ]);
            setParams(p);
            setAnchors(a.anchors || []);
            setBlend(a.blend || {});
            setScenes(s.scenes || []);
        } catch (e) {
            setError(errorText(e, 'Could not read the engine'));
        }
    }, [base]);

    useEffect(() => { if (connected) loadAll(); }, [connected, loadAll]);
    // scene changes rewrite parameters and the blend
    const sceneId = status?.conductor?.scene;
    useEffect(() => { if (connected) loadAll(); }, [sceneId, connected, loadAll]);
    useEffect(() => { if (status?.conductor?.blend && !throttles.current.blendDrag) setBlend(status.conductor.blend); }, [status]);

    // --- actions -------------------------------------------------------------
    const startEngine = async () => {
        setBusy('start');
        setError('');
        try {
            const loraList = loraPath ? [{ path: loraPath, strength: Number(params['/render/lora_strength']?.value ?? 0.85) }] : [];
            const radio = layout === 'radio';
            await eng.start({
                model_id: modelId, loras: loraList, layout: radio ? 'stereo' : layout, scene: radio || clock ? 'tehran_radio' : 'places',
                sink: radio ? 'radio' : 'auto', radio: radio && radioUrl ? [radioUrl] : [], clock,
            });
        } catch (e) {
            setError(errorText(e, 'The engine did not start'));
        } finally {
            setBusy('');
        }
    };
    const stopEngine = async () => {
        setBusy('stop');
        try { setListening(false); await eng.stop(); } catch (e) { setError(errorText(e, 'Stop failed')); } finally { setBusy(''); }
    };
    const loadModel = async () => {
        setBusy('model');
        try {
            const loraList = loraPath ? [{ path: loraPath, strength: Number(params['/render/lora_strength']?.value ?? 0.85) }] : [];
            await call('POST', '/v1/model', { model_id: modelId, loras: loraList });
        } catch (e) { setError(errorText(e, 'Could not load the model')); } finally { setBusy(''); }
    };
    const transport = async (on) => {
        try { await call('POST', `/v1/transport/${on ? 'start' : 'stop'}`); eng.refreshStatus(); } catch (e) { setError(errorText(e, 'Transport failed')); }
    };
    const record = async (on) => {
        try { await call('POST', `/v1/record/${on ? 'start' : 'stop'}`); eng.refreshStatus(); } catch (e) { setError(errorText(e, 'Record failed')); }
    };
    const activateScene = async (id) => {
        try {
            await call('POST', `/v1/scenes/${encodeURIComponent(id)}/activate`, { transition_s: Number(transition), from_azimuth: Number(fromAz) });
            eng.refreshStatus();
        } catch (e) { setError(errorText(e, 'Scene change failed')); }
    };

    // throttled parameter writes: at most one in flight per 80 ms, the last value always lands
    const setParam = useCallback((path, value) => {
        setParams((p) => ({ ...p, [path]: { ...(p[path] || {}), value } }));
        const t = throttles.current[path] || (throttles.current[path] = { timer: null, pending: null });
        t.pending = value;
        if (t.timer) return;
        const send = () => {
            const v = t.pending; t.pending = null; t.timer = null;
            if (v === null || v === undefined) return;
            call('PATCH', `/v1/params${path}`, { value: v }).catch(() => {});
        };
        t.timer = window.setTimeout(send, 80);
    }, [call]);

    const setBlendWeights = useCallback((w, final) => {
        throttles.current.blendDrag = !final;
        setBlend(w);
        const t = throttles.current.blend || (throttles.current.blend = { timer: null, pending: null });
        t.pending = w;
        if (t.timer && !final) return;
        const send = () => {
            const v = t.pending; t.pending = null; t.timer = null;
            if (v) call('PUT', '/v1/blend', { blend: v }).catch(() => {});
        };
        if (final) { if (t.timer) window.clearTimeout(t.timer); send(); } else t.timer = window.setTimeout(send, 120);
    }, [call]);

    const renderOneshot = async () => {
        try {
            const r = await call('POST', '/v1/render', { blend, duration_s: 10 });
            setOneshot({ id: r.job_id, state: 'queued', url: null });
        } catch (e) { setError(errorText(e, 'Render failed')); }
    };
    useEffect(() => {
        if (!oneshot || oneshot.state === 'done' || oneshot.state === 'error' || !base) return undefined;
        const t = window.setInterval(async () => {
            try {
                const r = await v1(base, 'GET', `/v1/render/${oneshot.id}`);
                if (r.state === 'done') setOneshot({ id: oneshot.id, state: 'done', url: `${base}/v1/clips/${r.clips[0].id}`, render_s: r.render_s });
                else if (r.state === 'error') setOneshot({ id: oneshot.id, state: 'error', error: r.error });
                else setOneshot((o) => ({ ...o, state: r.state, progress: r.progress }));
            } catch { /* keep polling */ }
        }, 700);
        return () => window.clearInterval(t);
    }, [oneshot, base]);

    // the monitor stream
    useEffect(() => {
        const el = audioRef.current;
        if (!el) return;
        if (listening && base) {
            el.src = `${base}/v1/monitor.wav?t=${Date.now()}`;
            el.play().catch(() => setListening(false));
        } else {
            el.pause();
            el.removeAttribute('src');
            el.load();
        }
    }, [listening, base]);

    // --- derived ---------------------------------------------------------------
    const modelState = status?.model?.state || (running ? '…' : 'off');
    const playing = !!status?.conductor?.playing;
    const recording = !!status?.audio?.recording;
    const rend = status?.renderer || {};
    const beds = meters?.beds || status?.audio?.beds || {};
    const pools = status?.conductor?.pools || {};
    const warmColor = theme.palette.warm?.main || '#FFB123';
    const procLabel = !proc ? '…' : !proc.available ? 'not available here' : running ? (connected ? 'running' : 'starting…') : 'off';
    const knobVal = (k) => Number(params[k.path]?.value ?? 0);
    const knobRange = (k) => params[k.path]?.range || [0, 1];

    return (
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', lg: 'row' }, alignItems: 'flex-start', gap: 2 }}>
            <audio ref={audioRef} style={{ display: 'none' }} />
            <Box sx={{ flex: 1, minWidth: 0, width: '100%' }}>
                <Paper sx={appStyles.elevatedInfoCard}>
                    <Box sx={appStyles.sectionCardHeader}>
                        <Box component="span" sx={appStyles.sectionCardIcon}><EngineIcon size={20} /></Box>
                        <Typography variant="h6" sx={appStyles.sectionCardTitle}>Engine</Typography>
                        <BetaBadge />
                        <Chip size="small" label={procLabel} variant="outlined"
                              sx={{ ml: 1, borderColor: running && connected ? warmColor : undefined, color: running && connected ? warmColor : undefined }} />
                        {running && <Chip size="small" label={`model ${modelState}`} variant="outlined" color={modelState === 'error' ? 'error' : 'default'} />}
                        <Box sx={{ flex: 1 }} />
                        {!running ? (
                            <Tooltip title={TIPS.engine.start}>
                                <span>
                                    <Button size="small" variant="contained" startIcon={<PowerIcon size={14} />} onClick={startEngine}
                                            disabled={busy === 'start' || isDocker || (proc && proc.available === false)}>
                                        Start engine
                                    </Button>
                                </span>
                            </Tooltip>
                        ) : (
                            <>
                                <Tooltip title={playing ? TIPS.engine.stopTransport : TIPS.engine.play}>
                                    <span>
                                        <Button size="small" variant="contained" color={playing ? 'inherit' : 'primary'}
                                                startIcon={playing ? <StopIcon size={14} /> : <PlayIcon size={14} />}
                                                onClick={() => transport(!playing)} disabled={!connected || modelState !== 'ready'}>
                                            {playing ? 'Stop' : 'Play'}
                                        </Button>
                                    </span>
                                </Tooltip>
                                <Tooltip title={TIPS.engine.record}>
                                    <span>
                                        <Button size="small" variant={recording ? 'contained' : 'outlined'} color="error"
                                                startIcon={<RecordIcon size={12} fill={recording ? 'currentColor' : 'none'} />}
                                                onClick={() => record(!recording)} disabled={!connected}>
                                            {recording ? `Rec ${Math.round(status?.audio?.record_s || 0)} s` : 'Record'}
                                        </Button>
                                    </span>
                                </Tooltip>
                                <Tooltip title={TIPS.engine.listen}>
                                    <span>
                                        <Button size="small" variant={listening ? 'contained' : 'outlined'} startIcon={<ListenIcon size={14} />}
                                                onClick={() => setListening((l) => !l)} disabled={!connected}>
                                            Listen
                                        </Button>
                                    </span>
                                </Tooltip>
                                <Tooltip title={TIPS.engine.stopEngine}>
                                    <span>
                                        <Button size="small" variant="outlined" onClick={stopEngine} disabled={busy === 'stop'}>Stop engine</Button>
                                    </span>
                                </Tooltip>
                            </>
                        )}
                    </Box>

                    {error && <Alert severity="error" onClose={() => setError('')} sx={{ mb: 1.5 }}>{error}</Alert>}
                    {proc && proc.available === false && (
                        <Alert severity="info" sx={{ mb: 1.5 }}>The engine runs on a desktop install, not on the Space or in Docker.</Alert>
                    )}
                    {running && status?.notes?.length > 0 && (
                        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 1 }}>{status.notes.join(' · ')}</Typography>
                    )}

                    {/* model and output */}
                    <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', mb: 2 }}>
                        <FormControl size="small" sx={{ minWidth: 200 }}>
                            <InputLabel>Model</InputLabel>
                            <Tooltip title={TIPS.engine.model}>
                                <Select value={modelId} label="Model" onChange={(e) => setModelId(e.target.value)}>
                                    {baseModels.map((m) => (
                                        <MenuItem key={m.name} value={m.name} disabled={!m.downloaded}>
                                            {m.displayName}{m.downloaded ? '' : ' — not downloaded'}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </Tooltip>
                        </FormControl>
                        <FormControl size="small" sx={{ minWidth: 200 }}>
                            <InputLabel>LoRA</InputLabel>
                            <Tooltip title={TIPS.engine.lora}>
                                <Select value={loraPath} label="LoRA" onChange={(e) => setLoraPath(e.target.value)}>
                                    <MenuItem value="">None (plain base)</MenuItem>
                                    {loras.map((l) => <MenuItem key={l.path || l.name} value={l.path || l.name}>{l.name}</MenuItem>)}
                                </Select>
                            </Tooltip>
                        </FormControl>
                        <FormControl size="small" sx={{ minWidth: 130 }}>
                            <InputLabel>Output</InputLabel>
                            <Tooltip title={TIPS.engine.layout}>
                                <Select value={layout} label="Output" onChange={(e) => setLayout(e.target.value)}>
                                    <MenuItem value="stereo">Stereo</MenuItem>
                                    <MenuItem value="binaural">Binaural (soon)</MenuItem>
                                    <MenuItem value="8ch">8 speakers (soon)</MenuItem>
                                    <MenuItem value="radio">Radio stream</MenuItem>
                                </Select>
                            </Tooltip>
                        </FormControl>
                        {layout === 'radio' && (
                            <Tooltip title={TIPS.engine.radioUrl}>
                                <TextField size="small" label="Stream target" value={radioUrl} onChange={(e) => setRadioUrl(e.target.value)}
                                           placeholder="icecast://source:password@host:8000/tehran.mp3" sx={{ minWidth: 320 }} disabled={running} />
                            </Tooltip>
                        )}
                        <Tooltip title={TIPS.engine.clock}>
                            <FormControlLabel control={
                                <Switch size="small" checked={running ? !!params['/scene/clock/on']?.value : clock}
                                        onChange={(e) => { setClock(e.target.checked); if (running) setParam('/scene/clock/on', e.target.checked); }} />
                            } label={<Typography variant="caption">Tehran's clock{status?.conductor?.clock ? ` · ${status.conductor.clock.local_time}` : ''}</Typography>} />
                        </Tooltip>
                        {running && (
                            <Tooltip title={TIPS.engine.loadModel}>
                                <span>
                                    <Button size="small" variant="outlined" startIcon={<RefreshIcon size={14} />} onClick={loadModel}
                                            disabled={!connected || busy === 'model' || modelState === 'loading'}>
                                        Load
                                    </Button>
                                </span>
                            </Tooltip>
                        )}
                    </Box>

                    {/* scenes */}
                    <Typography variant="subtitle2" sx={{ mb: 0.75 }}>Scenes</Typography>
                    <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', alignItems: 'center', mb: 1.5 }}>
                        {scenes.length === 0 && <Typography variant="caption" color="textSecondary">{running ? 'No scenes loaded.' : 'Start the engine to see its scenes.'}</Typography>}
                        {scenes.map((s) => (
                            <Chip key={s.id} label={s.label} size="small" clickable onClick={() => activateScene(s.id)}
                                  color={sceneId === s.id ? 'primary' : 'default'} variant={sceneId === s.id ? 'filled' : 'outlined'}
                                  disabled={!connected} />
                        ))}
                    </Box>
                    <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center', justifyContent: 'flex-end', mb: 1.5 }}>
                        <Typography variant="caption" color="textSecondary" sx={{ mr: 0.5 }}>Next scene arrives over</Typography>
                        <Tooltip title={TIPS.engine.transition}>
                            <TextField size="small" type="number" label="Transition s" value={transition} sx={{ width: 110 }}
                                       inputProps={{ min: 0, max: 120, step: 1 }} onChange={(e) => setTransition(e.target.value)} />
                        </Tooltip>
                        <Tooltip title={TIPS.engine.fromAzimuth}>
                            <TextField size="small" type="number" label="From °" value={fromAz} sx={{ width: 90 }}
                                       inputProps={{ min: -180, max: 180, step: 5 }} onChange={(e) => setFromAz(e.target.value)} />
                        </Tooltip>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'flex-start' }}>
                        {/* blend */}
                        <Box>
                            <Typography variant="subtitle2" sx={{ mb: 0.75 }}>Blend</Typography>
                            <Tooltip title={TIPS.engine.blend}>
                                <Box>
                                    <BlendPad anchors={anchors} blend={blend} onChange={setBlendWeights} disabled={!connected} />
                                </Box>
                            </Tooltip>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                                <Tooltip title={TIPS.engine.wander}>
                                    <FormControlLabel control={
                                        <Switch size="small" checked={!!params['/scene/wander/on']?.value}
                                                onChange={(e) => setParam('/scene/wander/on', e.target.checked)} disabled={!connected} />
                                    } label={<Typography variant="caption">Wander</Typography>} />
                                </Tooltip>
                                <Tooltip title={TIPS.engine.oneshot}>
                                    <span>
                                        <Button size="small" variant="text" startIcon={<RenderIcon size={14} />} onClick={renderOneshot}
                                                disabled={!connected || modelState !== 'ready' || (oneshot && oneshot.state !== 'done' && oneshot.state !== 'error')}>
                                            Render 10 s
                                        </Button>
                                    </span>
                                </Tooltip>
                            </Box>
                            {oneshot && (
                                <Box sx={{ mt: 0.5 }}>
                                    {oneshot.state === 'done' ? (
                                        <audio controls src={oneshot.url} style={{ width: 260, height: 32 }} />
                                    ) : oneshot.state === 'error' ? (
                                        <Typography variant="caption" color="error">{oneshot.error}</Typography>
                                    ) : (
                                        <LinearProgress variant={oneshot.progress ? 'determinate' : 'indeterminate'} value={(oneshot.progress || 0) * 100} sx={{ width: 260 }} />
                                    )}
                                </Box>
                            )}
                        </Box>

                        {/* knobs */}
                        <Box sx={{ flex: 1, minWidth: 260 }}>
                            <Typography variant="subtitle2" sx={{ mb: 0.75 }}>Conductor</Typography>
                            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(84px, 1fr))', gap: 1.5 }}>
                                {KNOBS.map((k) => {
                                    const [lo, hi] = knobRange(k);
                                    const knob = (
                                        <RotaryKnob value={knobVal(k)} min={lo} max={hi} step={k.step} center={lo} detent={0} size={44}
                                                    format={k.fmt} label={k.label} ariaLabel={k.label}
                                                    onChange={(v) => setParam(k.path, v)} busy={!connected} />
                                    );
                                    return (
                                        <Tooltip key={k.path} title={TIPS.engine.knobs[k.path] || k.label}>
                                            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                                                {k.midi ? (
                                                    <MidiMappable id={k.midi} label={`Engine ${k.label}`} kind="continuous" min={lo} max={hi}
                                                                  value={knobVal(k)} onChange={(v) => setParam(k.path, v)}>
                                                        {knob}
                                                    </MidiMappable>
                                                ) : knob}
                                            </Box>
                                        </Tooltip>
                                    );
                                })}
                            </Box>
                            <Box sx={{ mt: 1.5, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                                <Typography variant="caption" color="textSecondary">Trajectories</Typography>
                                <ToggleButtonGroup size="small" value={params['/spatial/events/trajectories']?.value || []}
                                                   onChange={(_, v) => v.length && setParam('/spatial/events/trajectories', v)}
                                                   sx={{ '& .MuiToggleButton-root': { px: 1.2, py: 0.2, fontSize: 11 } }} disabled={!connected}>
                                    {['static', 'passby', 'orbit', 'drift'].map((t) => <ToggleButton key={t} value={t}>{t}</ToggleButton>)}
                                </ToggleButtonGroup>
                            </Box>
                        </Box>
                    </Box>
                </Paper>
            </Box>

            {/* the rail: space, buffers, machine */}
            <Box sx={{ width: { xs: '100%', lg: 300 }, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <RailSection title="Space">
                    <SpatialView beds={beds} events={liveEvents} size={268} />
                    {status?.conductor && (
                        <Typography variant="caption" color="textSecondary">
                            {Object.keys(status.conductor.blend || {}).length
                                ? Object.entries(status.conductor.blend).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(' · ')
                                : 'no blend'}
                            {status.conductor.transition < 1 ? ` · transition ${Math.round(status.conductor.transition * 100)}%` : ''}
                        </Typography>
                    )}
                </RailSection>
                <RailSection title="Buffers">
                    {!running && <Typography variant="caption" color="textSecondary">Off.</Typography>}
                    {running && (
                        <>
                            {Object.entries(beds).map(([k, b]) => (
                                <Box key={k}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="caption">bed {k.split(':')[1]} · {b.state}</Typography>
                                        <Typography variant="caption" color="textSecondary">{Math.round(b.ahead_s)} s ahead{b.starvations ? ` · ${b.starvations} loops` : ''}</Typography>
                                    </Box>
                                    <LinearProgress variant="determinate" value={Math.min(100, (b.ahead_s / Math.max(1, 2 * Number(params['/bed/clip_length']?.value || 60))) * 100)} />
                                </Box>
                            ))}
                            {Object.keys(beds).length === 0 && <Typography variant="caption" color="textSecondary">No bed playing.</Typography>}
                            <Divider />
                            {Object.entries(pools).map(([t, p]) => (
                                <Box key={t} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption">{t}</Typography>
                                    <Typography variant="caption" color="textSecondary">{p.ready} ready{p.pending ? ` · ${p.pending} rendering` : ''}</Typography>
                                </Box>
                            ))}
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">events fired</Typography>
                                <Typography variant="caption" color="textSecondary">{status?.conductor?.events_fired ?? 0} · next in {status?.conductor?.next_onset_in ?? '–'} s</Typography>
                            </Box>
                        </>
                    )}
                </RailSection>
                <RailSection title="Machine">
                    {!running && <Typography variant="caption" color="textSecondary">Off.</Typography>}
                    {running && (
                        <>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">queue</Typography>
                                <Typography variant="caption" color="textSecondary">{rend.queued ?? 0} · {rend.renders ?? 0} done{rend.errors ? ` · ${rend.errors} failed` : ''}</Typography>
                            </Box>
                            {rend.current && (
                                <Box>
                                    <Typography variant="caption">rendering {rend.current.role}{rend.current.event_type ? ` · ${rend.current.event_type}` : ''} · {Math.round(rend.current.duration_s)} s</Typography>
                                    <LinearProgress variant="determinate" value={(rend.progress || 0) * 100} />
                                </Box>
                            )}
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">render time</Typography>
                                <Typography variant="caption" color="textSecondary">{rend.avg_render_s ? `${rend.avg_render_s} s avg` : '–'}</Typography>
                            </Box>
                            {rend.gpu_total_gb && (
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption">GPU</Typography>
                                    <Typography variant="caption" color="textSecondary">{rend.gpu_used_gb} / {rend.gpu_total_gb} GB</Typography>
                                </Box>
                            )}
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">output</Typography>
                                <Typography variant="caption" color="textSecondary">{status?.audio?.sink}{meters ? ` · ${meters.peak_db?.[0]} dB` : ''}{meters?.limiter_db < -0.5 ? ` · lim ${meters.limiter_db}` : ''}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">OSC</Typography>
                                <Typography variant="caption" color="textSecondary">in :{status?.osc?.in_port ?? 'off'} · out {status?.osc?.out || 'off'}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">API</Typography>
                                <Typography variant="caption" color="textSecondary">{base}</Typography>
                            </Box>
                            {status?.model?.error && <Typography variant="caption" color="error">{status.model.error}</Typography>}
                            {log.length > 0 && (
                                <Box sx={{ maxHeight: 120, overflow: 'auto' }}>
                                    {log.slice(-8).reverse().map((l) => (
                                        <Typography key={l.t} variant="caption" color="textSecondary" sx={{ display: 'block', fontSize: 10.5 }}>{l.text}</Typography>
                                    ))}
                                </Box>
                            )}
                        </>
                    )}
                </RailSection>
            </Box>
        </Box>
    );
}
