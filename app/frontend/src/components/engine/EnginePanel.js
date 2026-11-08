import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Box, Typography, Paper, Button, Select, MenuItem, TextField, FormControl, InputLabel,
    Chip, Switch, FormControlLabel, LinearProgress, Alert, Divider, ToggleButton, ToggleButtonGroup, Slider, Checkbox,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
    Infinity as EngineIcon, Play as PlayIcon, Square as StopIcon, Circle as RecordIcon,
    Headphones as ListenIcon, Radio as PowerIcon, RefreshCw as RefreshIcon, Waves as RenderIcon,
    ChevronUp, ChevronDown, X as RemoveIcon, Plus as AddIcon, SkipForward as NextIcon, Save as SaveIcon, ListMusic as ProgramIcon, Pencil as EditIcon,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import RotaryKnob from '../RotaryKnob';
import RailSection from '../RailSection';
import BetaBadge from '../bend/BetaBadge';
import { MidiMappable } from '../MidiContext';
import { TIPS } from '../../tooltips';
import { appStyles } from '../../theme';
import useEngine, { v1 } from './useEngine';
import api from '../../api';
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
    { path: '/render/steps', label: 'Steps', fmt: (v) => `${Math.round(v)}`, step: 5 },
    { path: '/render/cfg', label: 'CFG', fmt: (v) => Number(v).toFixed(1), step: 0.5 },
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
    const [loraRuns, setLoraRuns] = useState([]);      // /api/loras?base_model=<modelId>: only adapters that fit the base
    const [loraRun, setLoraRun] = useState(stored.loraRun || '');
    const [layout, setLayout] = useState(stored.layout || 'stereo');
    const [radioUrl, setRadioUrl] = useState(stored.radioUrl || '');
    const [busy, setBusy] = useState('');
    const [error, setError] = useState('');

    // the engine's parameters, anchors, scenes
    const [params, setParams] = useState({});
    const [anchors, setAnchors] = useState([]);
    const [blend, setBlend] = useState({});
    const [scenes, setScenes] = useState([]);
    // the program: the piece as timed steps (decision 11)
    const [programs, setPrograms] = useState([]);
    const [programId, setProgramId] = useState(stored.programId || 'tehran_hour');
    const [progDraft, setProgDraft] = useState(null);
    const [progProblems, setProgProblems] = useState([]);
    const [progSaved, setProgSaved] = useState('');      // JSON of the last loaded/saved list; Save lights up when the draft differs
    const [showMeta, setShowMeta] = useState(false);
    const progDirty = !!progDraft && JSON.stringify(progDraft) !== progSaved;
    // places (scenes), anchors and words are data the user edits here (E8.B / E8.D)
    const [vocab, setVocab] = useState({ fields: {}, order: [] });
    const [wordsDraft, setWordsDraft] = useState(null);
    const [wordInput, setWordInput] = useState({});
    const [placeDraft, setPlaceDraft] = useState(null);
    const [anchorDraft, setAnchorDraft] = useState(null);
    const [listening, setListening] = useState(false);
    // live output: device, channel count, one route per layer (/v1/output)
    const [devices, setDevices] = useState([]);
    const [outDraft, setOutDraft] = useState(null);
    const [outLayers, setOutLayers] = useState(['bed:0', 'bed:1', 'events', 'soundmarks']);
    const [oneshot, setOneshot] = useState(null);     // {id, state, url}
    const audioRef = useRef(null);
    const throttles = useRef({});

    useEffect(() => {
        try {
            window.localStorage.setItem(STORE_KEY, JSON.stringify({ modelId, loraPath, loraRun, layout, radioUrl, programId }));
        } catch { /* noop */ }
    }, [modelId, loraPath, loraRun, layout, radioUrl, programId]);

    // /api/loras gives all_checkpoints as path strings (older shapes: objects with .path)
    const ckptsOf = (run) => ((run?.all_checkpoints) || []).map((c) => (typeof c === 'string' ? { path: c } : c)).filter((c) => c.path);
    // LoRAs that fit the chosen base (the backend knows each run's base model); a run has many checkpoints
    useEffect(() => {
        let gone = false;
        api.get(`/api/loras?base_model=${encodeURIComponent(modelId)}`).then((r) => {
            if (gone) return;
            const runs = r.data.loras || [];
            setLoraRuns(runs);
            const run = runs.find((x) => x.id === loraRun) || runs.find((x) => ckptsOf(x).some((c) => c.path === loraPath));
            if (!run) { setLoraRun(''); setLoraPath(''); }
            else if (run.id !== loraRun) setLoraRun(run.id);
        }).catch(() => setLoraRuns([]));
        return () => { gone = true; };
    }, [modelId]);   // eslint-disable-line react-hooks/exhaustive-deps
    const runCkpts = useMemo(() => ckptsOf(loraRuns.find((x) => x.id === loraRun)), [loraRuns, loraRun]);
    const stepOf = (c) => { const m = /step=(\d+)/.exec(c.path || ''); return m ? `step ${m[1]}` : (c.path || '').split('/').pop(); };
    const pickRun = (id) => {
        setLoraRun(id);
        const ck = ckptsOf(loraRuns.find((x) => x.id === id));
        setLoraPath(ck.length ? ck[ck.length - 1].path : '');
    };

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
            try { const vw = await v1(base, 'GET', '/v1/vocabulary'); setVocab(vw); } catch { /* older engine */ }
            try {
                const pr = await v1(base, 'GET', '/v1/programs');
                setPrograms(pr.programs || []);
                const cur = pr.current?.id || programId;
                if (cur && !progDraft) {
                    const full = await v1(base, 'GET', `/v1/programs/${encodeURIComponent(cur)}`);
                    setProgDraft(full); setProgSaved(JSON.stringify(full));
                    if (pr.current?.id) setProgramId(pr.current.id);
                }
            } catch { /* an engine without programs */ }
            const [o, d] = await Promise.all([v1(base, 'GET', '/v1/output'), v1(base, 'GET', '/v1/output/devices')]);
            setOutLayers(o.layers || outLayers);
            setDevices(d.devices || []);
            setOutDraft((prev) => prev || { device: o.device ?? '', channels: o.channels, routing: o.routing || {}, sink: o.sink });
        } catch (e) {
            setError(errorText(e, 'Could not read the engine'));
        }
    }, [base]);   // eslint-disable-line react-hooks/exhaustive-deps

    const routeValue = (layer) => {
        const r = outDraft?.routing?.[layer] || (layer.startsWith('bed:') ? outDraft?.routing?.bed : null);
        return !r || r.mode !== 'direct' ? 'spatial' : `direct:${(r.channels || []).join('-')}`;
    };
    const setRoute = (layer, value) => {
        const routing = { ...(outDraft.routing || {}) };
        if (value === 'spatial') routing[layer] = { mode: 'spatial' };
        else routing[layer] = { mode: 'direct', channels: value.slice(7).split('-').map(Number) };
        setOutDraft({ ...outDraft, routing });
    };
    const applyOutput = async () => {
        if (!outDraft) return;
        setBusy('output');
        try {
            const hasDevice = outDraft.device !== '' && outDraft.device !== null && outDraft.device !== undefined;
            const r = await call('PUT', '/v1/output', {
                device: hasDevice ? outDraft.device : null, channels: Number(outDraft.channels) || 2,
                routing: outDraft.routing, sink: hasDevice ? 'sounddevice' : 'null',
            });
            if (r.note) setError(r.note);
            setOutDraft({ device: r.device ?? '', channels: r.channels, routing: r.routing || {}, sink: r.sink });
            eng.refreshStatus();
        } catch (e) { setError(errorText(e, 'Output change refused')); } finally { setBusy(''); }
    };
    const channelPairs = useMemo(() => {
        const n = Math.min(64, Math.max(1, Number(outDraft?.channels) || 2));
        const opts = [];
        for (let i = 1; i < n; i += 1) opts.push(`${i}-${i + 1}`);
        for (let i = 1; i <= n; i += 1) opts.push(`${i}`);
        return opts;
    }, [outDraft?.channels]);

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
                model_id: modelId, loras: loraList, layout: radio ? 'stereo' : layout, scene: 'places',
                program: programId || undefined,
                sink: radio ? 'radio' : 'auto', radio: radio && radioUrl ? [radioUrl] : [],
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
    // --- program ---------------------------------------------------------------
    const pickProgram = async (id) => {
        setProgramId(id);
        if (!connected) { setProgDraft(null); return; }
        try { const full = await v1(base, 'GET', `/v1/programs/${encodeURIComponent(id)}`); setProgDraft(full); setProgSaved(JSON.stringify(full)); setProgProblems([]); setShowMeta(false); } catch (e) { setError(errorText(e, 'Could not read the program')); }
    };
    const loadProgram = async () => {
        if (!programId) return;
        try {
            const r = await call('POST', `/v1/programs/${encodeURIComponent(programId)}/load`, { step: 0 });
            setProgDraft(r.program); setProgSaved(JSON.stringify(r.program)); setProgProblems(r.problems || []); eng.refreshStatus();
        } catch (e) { setError(errorText(e, 'Could not load the program')); }
    };
    const saveProgram = async () => {
        if (!progDraft) return;
        const id = (progDraft.id || 'program').toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
        try {
            const r = await call('PUT', `/v1/programs/${encodeURIComponent(id)}`, { ...progDraft, id });
            setProgDraft(r.program); setProgSaved(JSON.stringify(r.program)); setProgProblems(r.problems || []); setProgramId(id); setShowMeta(false);
            const pr = await v1(base, 'GET', '/v1/programs'); setPrograms(pr.programs || []);
        } catch (e) { setError(errorText(e, 'Could not save the program')); }
    };
    const nextStep = async () => { try { await call('POST', '/v1/program/next'); eng.refreshStatus(); } catch (e) { setError(errorText(e, 'Next step failed')); } };
    const newProgram = () => { setProgDraft({ id: 'my_piece', label: 'My piece', loop: true, scenes: '', steps: [{ scene: scenes[0]?.id || '', minutes: 10, transition_s: 25, from_azimuth: 0, beds: 'generated', label: '' }] }); setProgSaved(''); setShowMeta(true); };
    const editStep = (i, patch) => setProgDraft((d) => ({ ...d, steps: d.steps.map((s, k) => (k === i ? { ...s, ...patch } : s)) }));
    const moveStep = (i, dir) => setProgDraft((d) => {
        const j = i + dir; if (j < 0 || j >= d.steps.length) return d;
        const steps = [...d.steps]; [steps[i], steps[j]] = [steps[j], steps[i]]; return { ...d, steps };
    });
    const removeStep = (i) => setProgDraft((d) => ({ ...d, steps: d.steps.filter((_, k) => k !== i) }));
    const addStep = () => setProgDraft((d) => ({ ...d, steps: [...d.steps, { scene: scenes[0]?.id || '', minutes: 10, transition_s: 25, from_azimuth: 0, beds: 'generated', label: '' }] }));
    const mmss = (s) => `${Math.floor((s || 0) / 60)}:${String(Math.floor((s || 0) % 60)).padStart(2, '0')}`;
    // --- places / anchors / words ------------------------------------------------------
    const realAnchors = useMemo(() => anchors.filter((a) => !String(a.id).includes('|')), [anchors]);
    const slug = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    const newPlace = () => setPlaceDraft({ id: '', label: '', blend: {}, params: { '/events/density': 0.4, '/events/sparseness': 0.3, '/events/max_concurrent': 3 }, events: {}, modulators: {}, soundmarks: [], _new: true });
    const editPlace = (s) => setPlaceDraft({ ...s, blend: { ...(s.blend || {}) }, params: { ...(s.params || {}) }, events: { ...(s.events || {}) } });
    const savePlace = async () => {
        if (!placeDraft) return;
        const id = placeDraft._new ? slug(placeDraft.label || placeDraft.id) : placeDraft.id;
        if (!id) { setError('A place needs a name'); return; }
        const blend = Object.fromEntries(Object.entries(placeDraft.blend || {}).filter(([, w]) => Number(w) > 0));
        if (!Object.keys(blend).length) { setError('Give the place at least one anchor weight'); return; }
        const { _new, ...rest } = placeDraft;
        try {
            await call('POST', '/v1/scenes', { ...rest, id, label: placeDraft.label || id, blend });
            setPlaceDraft(null); await loadAll();
        } catch (e) { setError(errorText(e, 'Could not save the place')); }
    };
    const deletePlace = async () => {
        if (!placeDraft || placeDraft._new) { setPlaceDraft(null); return; }
        try { await call('DELETE', `/v1/scenes/${encodeURIComponent(placeDraft.id)}`); setPlaceDraft(null); await loadAll(); } catch (e) { setError(errorText(e, 'Could not delete the place')); }
    };
    const setPlaceEvent = (word, on, prob) => setPlaceDraft((d) => {
        const ev = { ...(d.events || {}) };
        if (!on) delete ev[word]; else ev[word] = { ...(ev[word] || {}), probability: prob ?? ev[word]?.probability ?? 0.3 };
        return { ...d, events: ev };
    });
    const newAnchor = () => setAnchorDraft({ id: '', label: '', setting: 'exterior', perspective: 'mid', density: 'moderate', sources: [] });
    const saveAnchor = async () => {
        const id = slug(anchorDraft.id || anchorDraft.label);
        if (!id) { setError('An anchor needs a place word'); return; }
        const fields = { type: 'ambience', place: id, perspective: anchorDraft.perspective, density: anchorDraft.density, setting: anchorDraft.setting, sources: anchorDraft.sources };
        try {
            await call('POST', '/v1/anchors', { id, label: anchorDraft.label || id, fields, variants: [{ density: anchorDraft.density === 'dense' ? 'moderate' : 'dense' }, { perspective: 'far' }] });
            setAnchorDraft(null); await loadAll();
        } catch (e) { setError(errorText(e, 'Could not save the anchor')); }
    };
    const deleteAnchor = async (id) => { try { await call('DELETE', `/v1/anchors/${encodeURIComponent(id)}`); await loadAll(); } catch (e) { setError(errorText(e, 'Could not delete the anchor')); } };
    const words = wordsDraft || vocab.fields || {};
    const wordsDirty = !!wordsDraft && JSON.stringify(wordsDraft) !== JSON.stringify(vocab.fields || {});
    const addWord = (field) => {
        const w = slug(wordInput[field] || '').replace(/_/g, ' ');
        if (!w) return;
        setWordsDraft({ ...words, [field]: [...(words[field] || []).filter((x) => x !== w), w] });
        setWordInput({ ...wordInput, [field]: '' });
    };
    const removeWord = (field, w) => setWordsDraft({ ...words, [field]: (words[field] || []).filter((x) => x !== w) });
    const saveWords = async () => { try { const r = await call('PUT', '/v1/vocabulary', { fields: wordsDraft }); setVocab((v) => ({ ...v, fields: r.fields })); setWordsDraft(null); } catch (e) { setError(errorText(e, 'Could not save the words')); } };
    const prog = status?.conductor?.program;
    const padAnchors = useMemo(() => {
        const steps = progDraft?.steps || [];
        if (!prog || !steps.length) return anchors;
        const used = new Set();
        steps.forEach((s) => { const sc = scenes.find((x) => x.id === s.scene); Object.keys(sc?.blend || {}).forEach((k) => used.add(k)); });
        const sub = anchors.filter((a) => used.has(a.id));
        return sub.length ? sub : anchors;
    }, [anchors, scenes, progDraft, prog]);

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
                        <Typography variant="h6" sx={appStyles.sectionCardTitle}>Perpetual Engine</Typography>
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
                                <Select value={loraRuns.some((x) => x.id === loraRun) ? loraRun : ''} label="LoRA" onChange={(e) => pickRun(e.target.value)}>
                                    <MenuItem value="">None (plain base)</MenuItem>
                                    {loraRuns.map((l) => <MenuItem key={l.id} value={l.id}>{l.name}</MenuItem>)}
                                </Select>
                            </Tooltip>
                        </FormControl>
                        {runCkpts.length > 1 && (
                            <FormControl size="small" sx={{ minWidth: 140 }}>
                                <InputLabel>Checkpoint</InputLabel>
                                <Tooltip title={TIPS.engine.checkpoint}>
                                    <Select value={runCkpts.some((c) => c.path === loraPath) ? loraPath : ''} label="Checkpoint" onChange={(e) => setLoraPath(e.target.value)}>
                                        {runCkpts.map((c) => <MenuItem key={c.path} value={c.path}>{stepOf(c)}</MenuItem>)}
                                    </Select>
                                </Tooltip>
                            </FormControl>
                        )}
                        <FormControl size="small" sx={{ minWidth: 130 }}>
                            <InputLabel>Layout</InputLabel>
                            <Tooltip title={TIPS.engine.layout}>
                                <Select value={layout} label="Layout" onChange={(e) => setLayout(e.target.value)}>
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

                    {/* scenes: the piece as timed steps (a program file underneath) */}
                    <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center', flexWrap: 'wrap', mb: 0.75 }}>
                        <ProgramIcon size={14} />
                        <Typography variant="subtitle2" sx={{ mr: 0.5 }}>Scenes</Typography>
                        <Tooltip title={TIPS.engine.program}>
                            <Select size="small" value={programs.some((x) => x.id === programId) ? programId : ''} displayEmpty onChange={(e) => pickProgram(e.target.value)}
                                    disabled={!connected && programs.length === 0} sx={{ minWidth: 210, '& .MuiSelect-select': { py: 0.6, fontSize: 13 } }}
                                    renderValue={(v) => { const x = programs.find((p) => p.id === v); return x ? `${x.label}${x.total_s ? ` · ${Math.round(x.total_s / 60)} min` : ''}` : 'Choose a list…'; }}>
                                {programs.map((x) => <MenuItem key={x.id} value={x.id}>{x.label}{x.total_s ? ` · ${Math.round(x.total_s / 60)} min` : ''}</MenuItem>)}
                            </Select>
                        </Tooltip>
                        <Tooltip title={TIPS.engine.programLoad}><span>
                            <Button size="small" variant="outlined" onClick={loadProgram} disabled={!connected || !programId}>Load</Button>
                        </span></Tooltip>
                        <Tooltip title={TIPS.engine.programNext}><span>
                            <Button size="small" variant="text" startIcon={<NextIcon size={14} />} onClick={nextStep} disabled={!connected || !prog || !playing}>Next</Button>
                        </span></Tooltip>
                        <Button size="small" variant="text" startIcon={<AddIcon size={14} />} onClick={newProgram} disabled={!connected}>New</Button>
                        {prog && playing && (
                            <Typography variant="caption" color="textSecondary" sx={{ ml: 'auto' }}>
                                {prog.step + 1}/{prog.steps}{prog.step_label ? ` · ${prog.step_label}` : ''} · {mmss(prog.step_elapsed_s)} / {mmss(prog.step_total_s)}{prog.ended ? ' · ended' : ''}
                            </Typography>
                        )}
                    </Box>
                    {prog && playing && prog.step_total_s > 0 && (
                        <LinearProgress variant="determinate" value={Math.min(100, (100 * prog.step_elapsed_s) / prog.step_total_s)} sx={{ mb: 1, height: 3 }} />
                    )}
                    {progDraft && (() => {
                        const cols = '22px minmax(150px, 1.4fr) 66px 78px 72px 104px minmax(110px, 1fr) 82px';
                        const cell = { '& .MuiInputBase-input': { py: 0.45, px: 1, fontSize: 12 }, '& .MuiSelect-select': { py: 0.45, fontSize: 12 } };
                        const head = (txt) => <Typography variant="caption" color="textSecondary" sx={{ fontSize: 11 }}>{txt}</Typography>;
                        const total = Math.round((progDraft.steps || []).reduce((a, s) => a + Number(s.minutes || 0), 0));
                        return (
                            <Box sx={{ mb: 1.5, p: 1, pt: 0.75, border: `1px solid ${theme.palette.divider}`, borderRadius: 1 }}>
                                {showMeta && (
                                    <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center', mb: 1 }}>
                                        <TextField size="small" label="Id" value={progDraft.id || ''} sx={{ width: 160, ...cell }} onChange={(e) => setProgDraft({ ...progDraft, id: e.target.value })} />
                                        <TextField size="small" label="Name" value={progDraft.label || ''} sx={{ width: 220, ...cell }} onChange={(e) => setProgDraft({ ...progDraft, label: e.target.value })} />
                                    </Box>
                                )}
                                <Box sx={{ display: 'grid', gridTemplateColumns: cols, gap: 0.75, alignItems: 'center', px: 0.5, mb: 0.25 }}>
                                    {head('')}{head('Place')}{head('Minutes')}{head('Arrives s')}{head('From °')}{head('Beds')}{head('Label')}{head('')}
                                </Box>
                                {(progDraft.steps || []).map((s, i) => {
                                    const live = prog && prog.id === progDraft.id && prog.step === i && playing;
                                    return (
                                        <Box key={i} sx={{ display: 'grid', gridTemplateColumns: cols, gap: 0.75, alignItems: 'center', px: 0.5, py: 0.25, borderRadius: 1,
                                                           bgcolor: live ? theme.palette.action.selected : 'transparent' }}>
                                            <Typography variant="caption" color={live ? 'primary' : 'textSecondary'} sx={{ textAlign: 'right', fontSize: 11 }}>{i + 1}</Typography>
                                            <Select size="small" value={scenes.some((x) => x.id === s.scene) ? s.scene : ''} displayEmpty onChange={(e) => editStep(i, { scene: e.target.value })} sx={cell}>
                                                <MenuItem value="" disabled>place…</MenuItem>
                                                {scenes.map((x) => <MenuItem key={x.id} value={x.id}>{x.label}</MenuItem>)}
                                            </Select>
                                            <TextField size="small" type="number" value={s.minutes} inputProps={{ min: 0.1, step: 0.5 }} sx={cell} onChange={(e) => editStep(i, { minutes: Number(e.target.value) })} />
                                            <TextField size="small" type="number" value={s.transition_s ?? ''} inputProps={{ min: 0, max: 120, step: 1 }} sx={cell} onChange={(e) => editStep(i, { transition_s: e.target.value === '' ? null : Number(e.target.value) })} />
                                            <TextField size="small" type="number" value={s.from_azimuth ?? ''} inputProps={{ min: -180, max: 180, step: 5 }} sx={cell} onChange={(e) => editStep(i, { from_azimuth: e.target.value === '' ? null : Number(e.target.value) })} />
                                            <Tooltip title={TIPS.engine.programBeds}>
                                                <Select size="small" value={s.beds || 'generated'} onChange={(e) => editStep(i, { beds: e.target.value })} sx={cell}>
                                                    <MenuItem value="generated">Generated</MenuItem>
                                                    <MenuItem value="pool">Pool</MenuItem>
                                                    <MenuItem value="both">Both</MenuItem>
                                                </Select>
                                            </Tooltip>
                                            <TextField size="small" value={s.label || ''} placeholder={scenes.find((x) => x.id === s.scene)?.label || ''} sx={cell} onChange={(e) => editStep(i, { label: e.target.value })} />
                                            <Box sx={{ display: 'flex', justifyContent: 'flex-end', color: 'text.secondary' }}>
                                                <Button size="small" variant="text" color="inherit" sx={{ minWidth: 0, p: 0.4 }} onClick={() => moveStep(i, -1)} disabled={i === 0}><ChevronUp size={13} /></Button>
                                                <Button size="small" variant="text" color="inherit" sx={{ minWidth: 0, p: 0.4 }} onClick={() => moveStep(i, 1)} disabled={i === progDraft.steps.length - 1}><ChevronDown size={13} /></Button>
                                                <Button size="small" variant="text" color="inherit" sx={{ minWidth: 0, p: 0.4 }} onClick={() => removeStep(i)}><RemoveIcon size={13} /></Button>
                                            </Box>
                                        </Box>
                                    );
                                })}
                                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.75 }}>
                                    <Button size="small" variant="text" startIcon={<AddIcon size={14} />} onClick={addStep}>Add scene</Button>
                                    <Tooltip title={TIPS.engine.programLoop}>
                                        <FormControlLabel sx={{ ml: 0.5 }} control={<Switch size="small" checked={!!progDraft.loop} onChange={(e) => setProgDraft({ ...progDraft, loop: e.target.checked })} />}
                                                          label={<Typography variant="caption">Loop</Typography>} />
                                    </Tooltip>
                                    <Typography variant="caption" color="textSecondary">{total} min</Typography>
                                    <Button size="small" variant="text" color="inherit" startIcon={<EditIcon size={12} />} onClick={() => setShowMeta((v) => !v)} sx={{ color: 'text.secondary', textTransform: 'none' }}>
                                        {progDraft.label || progDraft.id}
                                    </Button>
                                    <Tooltip title={TIPS.engine.programSave}><span>
                                        <Button size="small" variant={progDirty ? 'contained' : 'outlined'} startIcon={<SaveIcon size={14} />} onClick={saveProgram} disabled={!connected || !progDirty} sx={{ ml: 'auto' }}>Save</Button>
                                    </span></Tooltip>
                                </Box>
                                {progProblems.length > 0 && <Alert severity="warning" sx={{ mt: 0.75 }}>{progProblems.join(' · ')}</Alert>}
                            </Box>
                        );
                    })()}

                    <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'flex-start' }}>
                        {/* blend */}
                        <Box>
                            <Typography variant="subtitle2" sx={{ mb: 0.75 }}>Blend</Typography>
                            <Tooltip title={TIPS.engine.blend}>
                                <Box>
                                    <BlendPad anchors={padAnchors} blend={blend} onChange={setBlendWeights} disabled={!connected} />
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
                <RailSection title="Places" defaultOpen={false}>
                    {!connected && <Typography variant="caption" color="textSecondary">Start the engine to edit places.</Typography>}
                    {connected && (
                        <>
                            <Tooltip title={TIPS.engine.places}>
                                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
                                    {scenes.map((s) => (
                                        <Chip key={s.id} label={s.label} size="small" clickable onClick={() => editPlace(s)}
                                              color={placeDraft && placeDraft.id === s.id ? 'primary' : 'default'} variant={placeDraft && placeDraft.id === s.id ? 'filled' : 'outlined'} />
                                    ))}
                                    <Button size="small" variant="text" startIcon={<AddIcon size={14} />} onClick={newPlace}>New place</Button>
                                </Box>
                            </Tooltip>
                            {placeDraft && (
                                <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
                                    <TextField size="small" label="Name" value={placeDraft.label || ''} onChange={(e) => setPlaceDraft({ ...placeDraft, label: e.target.value })} fullWidth />
                                    <Typography variant="caption" color="textSecondary">Blend of anchors</Typography>
                                    {realAnchors.map((a) => (
                                        <Box key={a.id} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="caption" sx={{ width: 76, flexShrink: 0 }}>{a.label || a.id}</Typography>
                                            <Slider size="small" min={0} max={1} step={0.05} value={Number(placeDraft.blend?.[a.id] || 0)} sx={{ flex: 1 }}
                                                    onChange={(_, v) => setPlaceDraft({ ...placeDraft, blend: { ...placeDraft.blend, [a.id]: v } })} />
                                            <Typography variant="caption" color="textSecondary" sx={{ width: 32, textAlign: 'right' }}>{Math.round((placeDraft.blend?.[a.id] || 0) * 100)}%</Typography>
                                        </Box>
                                    ))}
                                    {[['/events/density', 'Density'], ['/events/sparseness', 'Silence']].map(([k, lbl]) => (
                                        <Box key={k} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="caption" sx={{ width: 76, flexShrink: 0 }}>{lbl}</Typography>
                                            <Slider size="small" min={0} max={1} step={0.05} value={Number(placeDraft.params?.[k] ?? 0.4)} sx={{ flex: 1 }}
                                                    onChange={(_, v) => setPlaceDraft({ ...placeDraft, params: { ...placeDraft.params, [k]: v } })} />
                                            <Typography variant="caption" color="textSecondary" sx={{ width: 32, textAlign: 'right' }}>{Math.round((placeDraft.params?.[k] ?? 0.4) * 100)}%</Typography>
                                        </Box>
                                    ))}
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography variant="caption" sx={{ width: 76 }}>Max events</Typography>
                                        <TextField size="small" type="number" value={placeDraft.params?.['/events/max_concurrent'] ?? 3} inputProps={{ min: 1, max: 8 }} sx={{ width: 70 }}
                                                   onChange={(e) => setPlaceDraft({ ...placeDraft, params: { ...placeDraft.params, '/events/max_concurrent': Number(e.target.value) } })} />
                                    </Box>
                                    <Typography variant="caption" color="textSecondary">Events: which sources, how likely</Typography>
                                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 1 }}>
                                        {(vocab.fields?.sources || []).map((w) => {
                                            const on = !!placeDraft.events?.[w];
                                            return (
                                                <Box key={w} sx={{ display: 'flex', alignItems: 'center', gap: 0.25, minHeight: 28 }}>
                                                    <Checkbox size="small" checked={on} onChange={(e) => setPlaceEvent(w, e.target.checked)} sx={{ p: 0.25 }} />
                                                    <Typography variant="caption" sx={{ flex: 1 }}>{w}</Typography>
                                                    {on && <TextField size="small" type="number" value={placeDraft.events[w].probability ?? 0.3} inputProps={{ min: 0, max: 1, step: 0.05, style: { padding: '2px 4px', fontSize: 11, width: 40 } }}
                                                                       onChange={(e) => setPlaceEvent(w, true, Number(e.target.value))} />}
                                                </Box>
                                            );
                                        })}
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                        <Button size="small" variant="contained" startIcon={<SaveIcon size={14} />} onClick={savePlace}>Save place</Button>
                                        <Button size="small" variant="text" onClick={() => setPlaceDraft(null)}>Cancel</Button>
                                        {!placeDraft._new && <Button size="small" variant="text" color="error" onClick={deletePlace} sx={{ ml: 'auto' }}>Delete</Button>}
                                    </Box>
                                </Box>
                            )}
                            <Divider sx={{ my: 1 }} />
                            <Tooltip title={TIPS.engine.anchorsList}>
                                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
                                    <Typography variant="caption" color="textSecondary" sx={{ mr: 0.5 }}>Anchors</Typography>
                                    {realAnchors.map((a) => <Chip key={a.id} label={a.label || a.id} size="small" variant="outlined" onDelete={() => deleteAnchor(a.id)} />)}
                                    <Button size="small" variant="text" startIcon={<AddIcon size={14} />} onClick={newAnchor}>Anchor</Button>
                                </Box>
                            </Tooltip>
                            {anchorDraft && (
                                <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                        <FormControl size="small" sx={{ flex: 1 }}>
                                            <InputLabel shrink>Place word</InputLabel>
                                            <Select value={anchorDraft.id} label="Place word" displayEmpty notched onChange={(e) => setAnchorDraft({ ...anchorDraft, id: e.target.value, label: anchorDraft.label || e.target.value })}>
                                                <MenuItem value="" disabled>from Words › place</MenuItem>
                                                {(vocab.fields?.place || []).filter((w) => !realAnchors.some((a) => a.id === w)).map((w) => <MenuItem key={w} value={w}>{w}</MenuItem>)}
                                            </Select>
                                        </FormControl>
                                        <TextField size="small" label="Label" value={anchorDraft.label} onChange={(e) => setAnchorDraft({ ...anchorDraft, label: e.target.value })} sx={{ flex: 1 }} />
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                        {[['setting', 'Setting'], ['perspective', 'Perspective'], ['density', 'Density']].map(([k, lbl]) => (
                                            <FormControl key={k} size="small" sx={{ flex: 1 }}>
                                                <InputLabel>{lbl}</InputLabel>
                                                <Select value={anchorDraft[k]} label={lbl} onChange={(e) => setAnchorDraft({ ...anchorDraft, [k]: e.target.value })}>
                                                    {(vocab.fields?.[k] || []).map((w) => <MenuItem key={w} value={w}>{w}</MenuItem>)}
                                                </Select>
                                            </FormControl>
                                        ))}
                                    </Box>
                                    <FormControl size="small" fullWidth>
                                        <InputLabel>Sources</InputLabel>
                                        <Select multiple value={anchorDraft.sources} label="Sources" onChange={(e) => setAnchorDraft({ ...anchorDraft, sources: e.target.value })} renderValue={(v) => v.join(', ')}>
                                            {(vocab.fields?.sources || []).map((w) => <MenuItem key={w} value={w}>{w}</MenuItem>)}
                                        </Select>
                                    </FormControl>
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                        <Button size="small" variant="contained" onClick={saveAnchor}>Save anchor</Button>
                                        <Button size="small" variant="text" onClick={() => setAnchorDraft(null)}>Cancel</Button>
                                    </Box>
                                </Box>
                            )}
                        </>
                    )}
                </RailSection>
                <RailSection title="Words" defaultOpen={false}>
                    {!connected && <Typography variant="caption" color="textSecondary">Start the engine to edit the vocabulary.</Typography>}
                    {connected && (
                        <Tooltip title={TIPS.engine.words}>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                {(vocab.order || Object.keys(words)).filter((f) => f !== 'type' && f !== 'mood').map((f) => (
                                    <Box key={f}>
                                        <Typography variant="caption" color="textSecondary" sx={{ textTransform: 'capitalize' }}>{f}</Typography>
                                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
                                            {(words[f] || []).map((w) => <Chip key={w} label={w} size="small" variant="outlined" onDelete={() => removeWord(f, w)} />)}
                                            <TextField size="small" placeholder="add…" value={wordInput[f] || ''} sx={{ width: 96, '& .MuiInputBase-input': { py: 0.4, px: 1, fontSize: 12 } }}
                                                       onChange={(e) => setWordInput({ ...wordInput, [f]: e.target.value })}
                                                       onKeyDown={(e) => { if (e.key === 'Enter') addWord(f); }} />
                                        </Box>
                                    </Box>
                                ))}
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                    <Button size="small" variant={wordsDirty ? 'contained' : 'outlined'} startIcon={<SaveIcon size={14} />} onClick={saveWords} disabled={!wordsDirty}>Save words</Button>
                                    {wordsDirty && <Button size="small" variant="text" onClick={() => setWordsDraft(null)}>Cancel</Button>}
                                </Box>
                            </Box>
                        </Tooltip>
                    )}
                </RailSection>
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
                <RailSection title="Output" defaultOpen={false}>
                    {!running && <Typography variant="caption" color="textSecondary">Off.</Typography>}
                    {running && !outDraft && <Typography variant="caption" color="textSecondary">…</Typography>}
                    {running && outDraft && (
                        <>
                            <FormControl size="small" fullWidth>
                                <InputLabel shrink>Device</InputLabel>
                                <Tooltip title={TIPS.engine.device}>
                                    <Select value={outDraft.device ?? ''} label="Device" displayEmpty notched
                                            renderValue={(v) => (v === '' || v === null || v === undefined)
                                                ? 'None (monitor only)'
                                                : (devices.find((d) => d.index === v)?.name || String(v))}
                                            onChange={(e) => setOutDraft({ ...outDraft, device: e.target.value })}>
                                        <MenuItem value="">None (monitor only)</MenuItem>
                                        {devices.map((d) => (
                                            <MenuItem key={d.index} value={d.index}>{d.name} · {d.channels} ch{d.default ? ' · default' : ''}</MenuItem>
                                        ))}
                                    </Select>
                                </Tooltip>
                            </FormControl>
                            <Tooltip title={TIPS.engine.channels}>
                                <TextField size="small" type="number" label="Channels" value={outDraft.channels} inputProps={{ min: 1, max: 64, step: 1 }}
                                           onChange={(e) => setOutDraft({ ...outDraft, channels: e.target.value })} sx={{ width: 110 }} />
                            </Tooltip>
                            {outLayers.map((layer) => (
                                <Box key={layer} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="caption" sx={{ flex: 1 }}>{layer.replace(':', ' ')}</Typography>
                                    <Tooltip title={TIPS.engine.route}>
                                        <Select size="small" value={routeValue(layer)} onChange={(e) => setRoute(layer, e.target.value)}
                                                sx={{ minWidth: 118, '& .MuiSelect-select': { py: 0.4, fontSize: 12 } }}>
                                            <MenuItem value="spatial">Spatial</MenuItem>
                                            {channelPairs.map((p) => <MenuItem key={p} value={`direct:${p}`}>→ {p}</MenuItem>)}
                                        </Select>
                                    </Tooltip>
                                </Box>
                            ))}
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Button size="small" variant="outlined" onClick={applyOutput} disabled={!connected || busy === 'output'}>Apply</Button>
                                <Typography variant="caption" color="textSecondary">
                                    {status?.output?.sink === 'sounddevice' ? `${status.output.device} · ${status.output.channels} ch` : 'monitor only'}
                                </Typography>
                            </Box>
                        </>
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
