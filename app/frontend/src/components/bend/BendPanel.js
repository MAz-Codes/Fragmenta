import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Box, Typography, Paper, Button, IconButton, Select, MenuItem, TextField,
    Slider, FormControl, InputLabel, ToggleButton, ToggleButtonGroup, Menu,
    LinearProgress, Accordion, AccordionSummary, AccordionDetails, Chip,
    FormControlLabel, Checkbox, Alert, Switch, Snackbar, Portal, useTheme,
} from '@mui/material';
import {
    CircuitBoard as CircuitBoardIcon, Dices as DicesIcon,
    Save as SaveIcon, ChevronDown as ExpandMoreIcon, Square as StopIcon,
    RotateCcw as UnbendIcon, NotebookPen as LogIcon, Footprints as NearbyIcon,
    Shuffle as NewInputIcon, Check as KeepIcon, Wrench as RackIcon,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import api from '../../api';
import { TIPS } from '../../tooltips';
import { appStyles } from '../../theme';
import { BEND_MODEL } from '../../utils/macModels';
import SignalPath from './SignalPath';
import LoraStack from '../LoraStack';
import BendModuleCard from './BendModuleCard';
import BendingLogPanel from './BendingLogPanel';
import BreakPanel from './BreakPanel';
import BendTakes from './BendTakes';
import ProbeBoard from './ProbeBoard';
import RailSection from '../RailSection';
import {
    buildPatch, chancePatch, newModule, nextId, mutatePatch,
    boardColumns, probeModules, jumperPadId, parseJumperId, isNoopWire,
    isWireId, isChordId, chordId, circuitFingerprint,
} from './bendUtils';

/**
 * Bend — network bending & model bending as an instrument.
 *
 * Two views of one engine. The Probe board is the default and the naive
 * companion: the model as an unlabelled board of contacts — touch, hear,
 * name what you found (ProbeBoard). The Rack is the case opened: signal
 * path + module rack with every target, operator and parameter, presets,
 * Chance. Break (training-time model bending — BreakPanel) stays out of
 * sight unless `localStorage['fragmenta.bend.advanced'] = '1'` (reload
 * after setting it). Performance channels generate through saved presets
 * either way. Grounded in Ghazala's anti-theory (via
 * Kotowski & Font, NIME 2026), Broad et al. (2021/2022) and Dzwonczyk et
 * al. (DAFx 2024); the write-up is phd/network-bending.md.
 *
 * Everything here is declarative and per-request: the rack describes the
 * patch, the patch rides each /api/generate call, and the backend
 * guarantees the model comes back pristine afterwards. UNBEND is
 * UI-state reset + reassurance, not cleanup.
 */

const STORE_KEY = 'fragmenta.bend.v1';
// The models the board can bend (see the comment at downloadedModels).
const BEND_MODELS = [BEND_MODEL];
// Break is hidden unless this flag is set; see the header comment.
const ADVANCED_KEY = 'fragmenta.bend.advanced';
const readAdvanced = () => {
    try { return window.localStorage.getItem(ADVANCED_KEY) === '1'; } catch { return false; }
};
// The board needs something to pass through it; an empty prompt shouldn't
// stop anyone from touching it.
const DEFAULT_PROBE_PROMPT = 'warm synth melody over a simple drum groove';
const randomSeed32 = () => Math.floor(Math.random() * 0xffffffff);

// /api/generate errors arrive as APIResponse.error — {error: {message, …}} —
// while the bend routes return {error: "…"}. Always hand React a string.
const errorText = (err, fallback) => {
    const e = err?.response?.data?.error;
    if (typeof e === 'string') return e;
    if (e && typeof e.message === 'string') return e.message;
    return fallback;
};

const readStore = () => {
    try {
        const raw = window.localStorage.getItem(STORE_KEY);
        return raw ? JSON.parse(raw) : {};
    } catch { return {}; }
};

export default function BendPanel({ models, active = true, isDocker = false }) {
    const theme = useTheme();
    const accent = theme.palette.bend?.main || '#AEB9C4';
    const stored = useMemo(readStore, []);
    const advanced = useMemo(readAdvanced, []);

    // Without the advanced flag a remembered Break state is ignored.
    const [mode, setMode] = useState(advanced ? (stored.mode || 'bend') : 'bend');
    const [modelId, setModelId] = useState(stored.modelId || '');
    const [registry, setRegistry] = useState(null);
    const [modules, setModules] = useState(stored.modules || []);
    const [prompt, setPrompt] = useState(stored.prompt || '');
    const [duration, setDuration] = useState(stored.duration || 8);
    const [steps, setSteps] = useState(stored.steps || null);
    const [seed, setSeed] = useState(stored.seed ?? 0);
    const [randomSeed, setRandomSeed] = useState(stored.randomSeed !== false);

    const [busy, setBusy] = useState(false);          // 'bent' | 'clean' | false
    const [progress, setProgress] = useState(0);
    const [statusMsg, setStatusMsg] = useState('');
    // The bottom-corner pop-up: what a bend did (warning) or why a request
    // failed (error). {severity, lines} | null
    const [notice, setNotice] = useState(null);
    const notify = (severity, lines) => setNotice({ severity, lines: [].concat(lines) });
    const [bentAudio, setBentAudio] = useState(null);  // {url, seed, filename}
    const [cleanAudio, setCleanAudio] = useState(null);
    // The last bent request minus its patch: Clean A/B replays exactly this
    // (model, prompt, duration, steps, seed), so editing the prompt after a
    // bent take can't make the comparison apples-to-oranges.
    const [lastRunBody, setLastRunBody] = useState(null);

    const [presets, setPresets] = useState([]);
    const [presetAnchor, setPresetAnchor] = useState(null);
    const [chanceAnchor, setChanceAnchor] = useState(null);
    const [logEntries, setLogEntries] = useState([]);
    const [logOpen, setLogOpen] = useState(false);
    // The log is per model by default; "all models" shows every row (the
    // board used to offer more models, and those rows are still finds).
    const [logAll, setLogAll] = useState(!!stored.logAll);

    // Probe board, wired like a real one: a soldered wire is in the circuit
    // for every render until it is disconnected; contacts are momentary
    // (press) or latched (shift-press → `held`). Every contact always does
    // the same thing.
    const [view, setView] = useState(stored.view || 'probe');
    // Wires that join points already joined (block n → block n+1) change
    // nothing; older sessions could hold them, so they are dropped on load.
    const [wires, setWires] = useState(() => Array.from(new Set([
        ...(stored.wires || []), ...(stored.held || []).filter(id => id.startsWith('j.')),
    ])).filter((id) => {
        const j = parseJumperId(id);
        return j && !isNoopWire(j.from, j.fromPart, j.to, j.toPart);
    }));
    // Only contacts latch; a wire is either connected (live) or not.
    const [held, setHeld] = useState(() => (stored.held || []).filter(id => !id.startsWith('j.')));
    const [wet, setWet] = useState(stored.wet ?? 0.8);
    const [probeSeed, setProbeSeed] = useState(stored.probeSeed ?? randomSeed32());
    // Off: every touch passes the same input (comparable, one clean render).
    // On: each touch draws fresh noise — the same contact never sounds twice.
    const [freshInput, setFreshInput] = useState(!!stored.freshInput);
    const [probeDuration, setProbeDuration] = useState(stored.probeDuration || 5);
    const [showWiring, setShowWiring] = useState(!!stored.showWiring);
    // Adapters loaded under the bend (both views): [{path, strength, bypassed}].
    const [loraStack, setLoraStack] = useState(stored.loraStack || []);
    const [boardNames, setBoardNames] = useState({});
    const [lastTouched, setLastTouched] = useState(null);
    // The last bent take, for Nearby / Keep / Open in rack.
    const [lastTake, setLastTake] = useState(null);
    const [findName, setFindName] = useState('');
    const [autoPlay, setAutoPlay] = useState(null);
    const cleanKeyRef = useRef(null);

    const abortRef = useRef(null);
    const tickerRef = useRef(null);
    // The backend reports phase "loading" until sampling starts. That is a
    // blink when the model is warm, but ~20 s when its weights load (first
    // render, model switch, new LoRAs) — only that long wait is shown.
    const [loadingWeights, setLoadingWeights] = useState(false);
    const loadingSinceRef = useRef(null);

    // Bending hooks PyTorch modules, so MLX models can't be bent; on a Mac
    // the Checkpoint Manager offers this PyTorch model next to its MLX twin,
    // for the Bend tab only, and it runs on the MPS path. Until more models
    // are verified, the board offers this one model on every platform
    // (verified contact by contact 4 Nov 2026 on CUDA: tests/verify_bend_board.py).
    const downloadedModels = useMemo(
        () => (models || []).filter(m => m.downloaded && BEND_MODELS.includes(m.name)), [models]);

    // Also replaces a remembered model that is no longer offered.
    useEffect(() => {
        if (!downloadedModels.length) { if (modelId) setModelId(''); return; }
        if (modelId && downloadedModels.some(m => m.name === modelId)) return;
        setModelId(downloadedModels[0].name);
    }, [downloadedModels, modelId]);

    // Registry per model. A slow response for a model the user has already
    // switched away from must not overwrite the current one.
    useEffect(() => {
        let stale = false;
        const id = modelId || BEND_MODELS[0];
        api.get(`/api/bend/targets?model_id=${encodeURIComponent(id)}`)
            .then(({ data }) => { if (!stale) setRegistry(data); })
            .catch(() => {});
        return () => { stale = true; };
    }, [modelId]);

    const refreshLog = useCallback(() => {
        const q = modelId && !logAll ? `?model_id=${encodeURIComponent(modelId)}` : '';
        api.get(`/api/bend/log${q}`)
            .then(({ data }) => setLogEntries(data.entries || []))
            .catch(() => {});
    }, [modelId, logAll]);

    const refreshPresets = useCallback(() => {
        api.get('/api/bend/presets')
            .then(({ data }) => setPresets(data.presets || []))
            .catch(() => {});
    }, []);
    // The panel stays mounted across tab switches; re-read on return, since
    // Performance channels generating through a preset append to the log.
    useEffect(() => {
        if (!active) return;
        refreshLog();
        refreshPresets();
    }, [active, refreshLog, refreshPresets]);

    // Persist the session (per-viewer convenience only).
    useEffect(() => {
        try {
            window.localStorage.setItem(STORE_KEY, JSON.stringify({
                mode, modelId, modules, prompt, duration, steps, seed, randomSeed,
                view, held, wires, wet, probeSeed, freshInput, probeDuration, showWiring, loraStack,
                logAll,
            }));
        } catch { /* ignore */ }
    }, [mode, modelId, modules, prompt, duration, steps, seed, randomSeed,
        view, held, wires, wet, probeSeed, freshInput, probeDuration, showWiring, loraStack, logAll]);

    // The user's names for this model's contacts.
    useEffect(() => {
        if (!modelId) return undefined;
        let stale = false;
        api.get(`/api/bend/boards/${encodeURIComponent(modelId)}`)
            .then(({ data }) => { if (!stale) setBoardNames(data.pads || {}); })
            .catch(() => { if (!stale) setBoardNames({}); });
        return () => { stale = true; };
    }, [modelId]);

    const saveBoardNames = (next) => {
        setBoardNames(next);
        api.put(`/api/bend/boards/${encodeURIComponent(modelId)}`, { pads: next })
            .catch(() => notify('error', 'Could not save the board — the name is kept for this session only.'));
    };

    // --- rack operations ---------------------------------------------------
    const addModule = (stage) => {
        setModules(m => [...m, newModule(stage, registry)]);
    };
    const updateModule = (id, next) => {
        setModules(m => m.map(x => (x.id === id ? next : x)));
    };
    const removeModule = (id) => setModules(m => m.filter(x => x.id !== id));
    const moveModule = (id, dir) => {
        setModules(m => {
            const i = m.findIndex(x => x.id === id);
            const j = i + dir;
            if (i < 0 || j < 0 || j >= m.length) return m;
            const next = [...m];
            [next[i], next[j]] = [next[j], next[i]];
            return next;
        });
    };
    const randomizeModule = (id) => {
        setModules(m => m.map(x => {
            if (x.id !== id) return x;
            const [replacement] = chancePatch(registry, 'bend', { stage: x.target?.stage });
            return replacement ? { ...replacement, id: x.id, enabled: x.enabled } : x;
        }));
    };
    const doChance = (level) => {
        setChanceAnchor(null);
        if (!registry) return;
        setModules(chancePatch(registry, level).map(m => ({ ...m, id: nextId() })));
    };
    const unbend = () => {
        setModules([]);
        setStatusMsg('Rack cleared — the model is never left bent between generations anyway.');
    };

    // --- generation ---------------------------------------------------------
    // The LoRA stack rides every bent and clean request alike, so A/B
    // compares the bend and nothing else. Bypassed slots keep their place
    // in the load order but send strength 0 (as on the Generation tab).
    const withLoras = (body) => {
        const active = (loraStack || []).filter(s => s.path);
        return active.length ? {
            ...body,
            loras: active.map(s => ({ path: s.path, strength: s.bypassed ? 0 : s.strength })),
        } : body;
    };

    const stopTicker = () => {
        if (tickerRef.current) { clearInterval(tickerRef.current); tickerRef.current = null; }
        loadingSinceRef.current = null;
        setLoadingWeights(false);
    };
    useEffect(() => () => { stopTicker(); abortRef.current?.abort?.(); }, []);

    /**
     * One generation. Without `override` it renders the rack (bent) or
     * replays the last bent request clean. `override` lets the Probe board
     * and Nearby render their own modules: {modules, body, probe, padId,
     * autoPlay} for a bent take, {body} for a clean one. Resolves to the
     * take, or null.
     */
    const generate = async (withBend, override = null) => {
        if (!withBend && !(override?.body || lastRunBody)) return null;
        if (withBend && !modelId) { setStatusMsg('Pick a model first.'); return null; }
        if (withBend && !override && !prompt.trim()) { setStatusMsg('Write a prompt.'); return null; }
        const activeModules = override?.modules ?? modules.filter(m => m.enabled !== false);
        if (withBend && !activeModules.length) {
            setStatusMsg('The rack is empty — attach a module on the signal path, or hit Chance.');
            return null;
        }
        // A/B contract: "clean" replays the last bent request verbatim
        // (same seed, prompt, model, duration, steps) with the patch
        // removed; a fresh bent run resolves its seed here.
        let body;
        let patch = null;
        if (withBend && override?.body) {
            body = { ...override.body };
            patch = buildPatch(activeModules, modelId, body.seed);
            if (override.probe) patch.probe = override.probe;
        } else if (withBend) {
            const freshSeed = randomSeed
                ? Math.floor(Math.random() * 0xffffffff) : (parseInt(seed, 10) || 0);
            body = withLoras({
                model_id: modelId,
                prompt: prompt.trim(),
                duration: Number(duration),
                seed: freshSeed,
                batch_size: 1,
            });
            // Steps is only shown (and meaningful) for base models; a value
            // left over from one must not push a distilled model to 50 steps.
            if (steps && modelId.endsWith('-base')) body.steps = Number(steps);
            patch = buildPatch(activeModules, modelId, freshSeed);
        } else {
            body = { ...(override?.body || lastRunBody) };
        }
        const runSeed = body.seed;

        const controller = new AbortController();
        abortRef.current = controller;
        setBusy(withBend ? 'bent' : 'clean');
        setNotice(null);
        setProgress(0);
        setStatusMsg(withBend ? 'Generating bent…'
            : override ? 'Rendering the unbent model for comparison…'
                : 'Generating clean (same seed)…');
        tickerRef.current = setInterval(async () => {
            try {
                const r = await api.get('/api/generation-progress');
                const pct = Number(r.data?.progress) || 0;
                setProgress(prev => Math.max(prev, Math.min(95, pct)));
                if (r.data?.phase === 'loading') {
                    loadingSinceRef.current ??= Date.now();
                    if (Date.now() - loadingSinceRef.current > 700) setLoadingWeights(true);
                } else {
                    loadingSinceRef.current = null;
                    setLoadingWeights(false);
                }
            } catch { /* non-fatal */ }
        }, 250);

        try {
            const response = await api.post('/api/generate',
                patch ? { ...body, bend_patch: patch } : body, {
                    responseType: 'blob', signal: controller.signal,
                });
            stopTicker();
            setProgress(100);
            const url = URL.createObjectURL(response.data);
            const filename = response.headers?.['x-fragment-filename'] || '';
            try {
                const w = JSON.parse(response.headers?.['x-bend-warnings'] || '[]');
                if (Array.isArray(w) && w.length) notify('warning', w);
            } catch { /* no report */ }
            const result = {
                url, blob: response.data, seed: runSeed, filename, body, key: JSON.stringify(body),
            };
            const logId = response.headers?.['x-bend-log-id'] || null;
            if (withBend) {
                if (bentAudio?.url?.startsWith('blob:')) URL.revokeObjectURL(bentAudio.url);
                setBentAudio(result);
                setCleanAudio(prev => {
                    // A clean take of a different request (seed, prompt,
                    // model, duration, steps) is no longer comparable.
                    if (prev && prev.key !== result.key) {
                        if (prev.url?.startsWith('blob:')) URL.revokeObjectURL(prev.url);
                        return null;
                    }
                    return prev;
                });
                setLastRunBody(body);
                setLastTake({
                    modules: activeModules, body, logId, modelId,
                    padId: override?.padId ?? null,
                    // The whole circuit that was heard: what Keep names.
                    pads: override?.probe?.pads || null,
                    source: override?.source || (override ? 'probe' : 'rack'),
                });
                setFindName('');
                if (override?.autoPlay) setAutoPlay({ key: 'bent', token: Date.now() });
                refreshLog();
                // On the board the take plays by itself — nothing to say.
                setStatusMsg(override
                    ? ''
                    : `Bent fragment ready (seed ${runSeed}). Note what it did in the log below.`);
            } else {
                if (cleanAudio?.url?.startsWith('blob:')) URL.revokeObjectURL(cleanAudio.url);
                setCleanAudio(result);
                cleanKeyRef.current = result.key;
                setStatusMsg(override
                    ? ''
                    : `Clean reference ready (seed ${runSeed}) — A/B against the bent take.`);
            }
            setTimeout(() => setProgress(0), 1500);
            return result;
        } catch (err) {
            stopTicker();
            setProgress(0);
            if (err?.name === 'AbortError') {
                setStatusMsg('Stopped.');
            } else {
                setStatusMsg('');
                notify('error', errorText(err, `Generation failed: ${err.message}`));
            }
            return null;
        } finally {
            stopTicker();
            setBusy(false);
            abortRef.current = null;
        }
    };

    const stopGeneration = () => {
        api.post('/api/stop-generation').catch(() => {});
        abortRef.current?.abort?.();
    };

    // --- Probe board -----------------------------------------------------------
    const columns = useMemo(() => boardColumns(registry), [registry]);
    const allWires = wires;
    // What a render passes through: every connected wire, every latched
    // contact, and whatever is being pressed right now.
    const circuit = (extra = [], { wiresNow = wires, heldNow = held } = {}) =>
        Array.from(new Set([...wiresNow, ...heldNow, ...extra]));

    const probeBody = (seed = probeSeed) => {
        const body = {
            model_id: modelId,
            prompt: prompt.trim() || DEFAULT_PROBE_PROMPT,
            duration: Number(probeDuration),
            seed,
            batch_size: 1,
        };
        if (steps && modelId.endsWith('-base')) body.steps = Number(steps);
        return withLoras(body);
    };

    // After a bent take, render the unbent model once per (model,
    // prompt, seed, duration), so A/B is always there without asking.
    const ensureClean = async (take) => {
        if (take && cleanKeyRef.current !== take.key) await generate(false, { body: take.body });
    };

    // `fresh`: a touch (not a dry/wet re-listen) with "new input each touch" on.
    const probe = async (padIds, touchedId, at = wet, { fresh = freshInput } = {}) => {
        if (busy || !registry || !modelId) return;
        const mods = probeModules(padIds, columns, registry, at);
        if (!mods.length) return;
        setLastTouched(touchedId);
        let seed = probeSeed;
        if (fresh) {
            seed = randomSeed32();
            setProbeSeed(seed);
        }
        const take = await generate(true, {
            modules: mods, body: probeBody(seed), padId: touchedId, autoPlay: true,
            probe: { pads: padIds, wet: at, ...(touchedId ? { touched: touchedId } : {}) },
        });
        await ensureClean(take);
    };

    const touchPad = (id, { hold = false } = {}) => {
        if (busy) return;
        if (isChordId(id)) {
            // A chord find: its wires are soldered back on and its contacts
            // held (added to what is held when the hold key is down, else
            // in place of it), then the circuit plays.
            const pads = boardNames[id]?.pads || [];
            if (!pads.length) return;
            const wiresNow = Array.from(new Set([...wires, ...pads.filter(isWireId)]));
            const chordHeld = pads.filter(p => !isWireId(p));
            const heldNow = hold ? Array.from(new Set([...held, ...chordHeld])) : chordHeld;
            setWires(wiresNow);
            setHeld(heldNow);
            probe(circuit([], { wiresNow, heldNow }), id);
            return;
        }
        if (isWireId(id)) {
            // A wire is already live: clicking it plays the board again (and
            // makes it the find Keep will name). A named wire picked from the
            // finds is soldered back on first.
            const wiresNow = wires.includes(id) ? wires : [...wires, id];
            setWires(wiresNow);
            probe(circuit([], { wiresNow }), id);
            return;
        }
        if (hold) {
            const heldNow = held.includes(id) ? held.filter(x => x !== id) : [...held, id];
            setHeld(heldNow);
            const pads = circuit([], { heldNow });
            if (pads.length) probe(pads, heldNow.includes(id) ? id : null);
            else setLastTouched(null);
            return;
        }
        probe(circuit([id]), id);
    };

    const solderWire = (from, fromPart, to, toPart) => {
        if (busy) return;
        const id = jumperPadId(from, to, fromPart, toPart);
        const wiresNow = wires.includes(id) ? wires : [...wires, id];
        setWires(wiresNow);
        probe(circuit([], { wiresNow }), id);
    };

    const disconnectWire = (id) => {
        if (busy) return;
        setWires(w => w.filter(x => x !== id));
        setLastTouched(t => (t === id ? null : t));
    };

    // Release every contact and remove every wire. Names are kept: they are
    // what you found, and touching one brings it back.
    const clearBoard = () => {
        setHeld([]);
        setWires([]);
        setLastTouched(null);
    };

    const forgetFind = (id) => {
        const next = { ...boardNames };
        delete next[id];
        saveBoardNames(next);
    };

    // Nearby: move the wire a little — same operators, small steps.
    const nearby = async () => {
        if (busy || !lastTake) return;
        const base = view === 'rack' ? modules : lastTake.modules;
        const moved = mutatePatch(base, registry);
        if (view === 'rack') setModules(moved);
        const take = await generate(true, {
            modules: moved.filter(m => m.enabled !== false),
            body: lastTake.body,
            autoPlay: view === 'probe',
            source: view,
        });
        if (view === 'probe') await ensureClean(take);
    };

    // Keep names what was heard: the one contact or wire, or — when more
    // than one was in the circuit — the chord of all of them, with a
    // fingerprint of what the circuit did so a later change shows.
    const keepFind = () => {
        const name = findName.trim();
        if (!name || !lastTake) return;
        if (lastTake.logId) {
            api.patch(`/api/bend/log/${encodeURIComponent(lastTake.logId)}`, { note: name })
                .then(refreshLog).catch(() => {});
        }
        const pads = lastTake.pads?.length ? lastTake.pads : (lastTake.padId ? [lastTake.padId] : []);
        if (pads.length && lastTake.modelId === modelId) {
            const id = pads.length === 1 ? pads[0] : chordId(pads);
            const find = { name, does: circuitFingerprint(pads, columns, registry) };
            if (pads.length > 1) find.pads = pads;
            saveBoardNames({ ...boardNames, [id]: find });
        }
        setFindName('');
    };

    const openInRack = () => {
        if (!lastTake) return;
        setModules(lastTake.modules.map(m => ({ ...m, id: nextId() })));
        setPrompt(lastTake.body.prompt || prompt);
        setDuration(lastTake.body.duration);
        setRandomSeed(false);
        setSeed(lastTake.body.seed);
        // The rack now holds exactly this take, so Nearby can walk from it.
        setLastTake(t => (t ? { ...t, source: 'rack' } : t));
        setView('rack');
    };

    // --- presets -------------------------------------------------------------
    const savePreset = async () => {
        setPresetAnchor(null);
        if (!modules.length) {
            setStatusMsg('The rack is empty — nothing to save yet.');
            return;
        }
        const name = window.prompt('Preset name:')?.trim();
        if (!name) return;
        if (presets.some(p => p.name === name)
            && !window.confirm(`A preset named “${name}” exists. Replace it?`)) return;
        // Record the seed of the take the user just heard, not a stale
        // manual-seed field that random-seed mode never used.
        const presetSeed = bentAudio?.seed ?? (parseInt(seed, 10) || 0);
        try {
            await api.post('/api/bend/presets', {
                name,
                patch: buildPatch(modules, modelId, presetSeed, name,
                                  { keepDisabled: true }),
            });
            refreshPresets();
            setStatusMsg(`Preset “${name}” saved.`);
        } catch (err) {
            notify('error', errorText(err, 'Preset save failed.'));
        }
    };
    const loadPreset = (p) => {
        setPresetAnchor(null);
        setModules((p.patch?.modules || []).map(m => ({ ...m, id: nextId() })));
        setStatusMsg(`Preset “${p.name}” loaded${
            p.model_id && p.model_id !== modelId ? ` (saved against ${p.model_id} — unresolvable targets are skipped)` : ''}.`);
    };
    const deletePreset = async (p, e) => {
        e.stopPropagation();
        try {
            await api.delete(`/api/bend/presets/${encodeURIComponent(p.name)}`);
            refreshPresets();
        } catch { /* ignore */ }
    };

    const recallLogEntry = (entry) => {
        if (busy) return;
        const loras = (entry.loras || []).filter(l => l.path);
        if (Array.isArray(entry.loras)) {
            setLoraStack(loras.map(l => ({ path: l.path, strength: l.strength, bypassed: false })));
        }
        const otherModel = entry.model_id && entry.model_id !== modelId ? entry.model_id : null;
        const offered = otherModel && downloadedModels.some(m => m.name === otherModel);
        const origin = entry.patch?.probe;
        if (origin && Array.isArray(origin.pads) && origin.pads.length) {
            // Found on the board: the same contacts go back under the same
            // finger — held ones held, the pressed one pressed — with the
            // same input, and it plays again.
            const touched = origin.touched && origin.pads.includes(origin.touched) ? origin.touched : null;
            const wiresNow = origin.pads.filter(isWireId);
            const heldNow = origin.pads.filter(id => !isWireId(id) && id !== touched);
            const wetNow = typeof origin.wet === 'number' ? origin.wet : wet;
            setView('probe');
            setWires(wiresNow);
            setHeld(heldNow);
            setWet(wetNow);
            if (entry.prompt) setPrompt(entry.prompt);
            if (entry.duration) setProbeDuration(entry.duration);
            setSteps(entry.steps || null);
            setProbeSeed(entry.seed);
            // Hearing it again needs that exact input.
            setFreshInput(false);
            setLastTouched(touched);
            setMode('bend');
            if (otherModel) {
                // Another model: its contacts are the same places on the
                // board, but the sound is that model's. Switch when it is
                // offered; either way wait for a touch.
                if (offered) setModelId(otherModel);
                notify('warning', offered
                    ? `This was found on ${otherModel} — switched to it. Touch the board to hear it.`
                    : `This was found on ${otherModel}, which the board no longer offers. `
                      + `Its contacts are back on the board; touch them to hear what they do on ${modelId}.`);
                return;
            }
            const mods = probeModules(origin.pads, columns, registry, wetNow);
            if (!mods.length) return;
            const body = {
                model_id: modelId,
                prompt: entry.prompt || prompt.trim() || DEFAULT_PROBE_PROMPT,
                duration: Number(entry.duration || probeDuration),
                seed: entry.seed,
                batch_size: 1,
            };
            if (entry.steps && modelId.endsWith('-base')) body.steps = Number(entry.steps);
            if (loras.length) body.loras = loras.map(l => ({ path: l.path, strength: l.strength }));
            generate(true, {
                modules: mods, body, padId: touched, autoPlay: true,
                probe: { pads: origin.pads, wet: wetNow, ...(touched ? { touched } : {}) },
            }).then(ensureClean);
            return;
        }
        setModules((entry.patch?.modules || []).map(m => ({ ...m, id: nextId() })));
        if (entry.prompt) setPrompt(entry.prompt);
        // Duration and steps shape the noise and the schedule: without them
        // the same patch + seed is a different sound.
        // Kept exact even past the slider's range (Performance entries).
        if (entry.duration) setDuration(entry.duration);
        setSteps(entry.steps || null);
        setRandomSeed(false);
        setSeed(entry.seed);
        setMode('bend');
        setView('rack');
        if (otherModel && offered) setModelId(otherModel);
        setStatusMsg(`Recalled bend from the log (seed ${entry.seed})${
            otherModel ? (offered ? `, switched to ${otherModel}` : ` — logged on ${otherModel}, playing on ${modelId}`) : ''}.`);
    };

    const activeCount = modules.filter(m => m.enabled !== false).length;

    // Steps only exist on base models; the field lives with the prompt.
    const stepsField = modelId.endsWith('-base') && (
        <TextField size="small" label="Steps" type="number"
                   value={steps ?? ''} placeholder="50"
                   onChange={(e) => setSteps(e.target.value ? parseInt(e.target.value, 10) : null)}
                   sx={{ width: 110 }} />
    );

    // The rail: what goes in (instrument, prompt) and what came out (takes).
    // The board and the log have the page.
    const rail = mode !== 'break' && (
        <Box sx={{
            width: { xs: '100%', lg: 340 }, flex: 'none',
            display: 'flex', flexDirection: 'column', gap: 2,
            position: { lg: 'sticky' }, top: { lg: 16 },
        }}>
            <RailSection title="Instrument">
                <Tooltip title={TIPS.bend.model}>
                    <FormControl size="small" fullWidth>
                        <InputLabel>Model</InputLabel>
                        <Select value={modelId} label="Model"
                                onChange={(e) => setModelId(e.target.value)}>
                            {downloadedModels.map(m => (
                                <MenuItem key={m.name} value={m.name}>{m.displayName || m.name}</MenuItem>
                            ))}
                            {!downloadedModels.length && (
                                <MenuItem value="" disabled>
                                    Bending needs Small - Music (the PyTorch build) — get it in the Checkpoint Manager
                                </MenuItem>
                            )}
                        </Select>
                    </FormControl>
                </Tooltip>
                <LoraStack selectedModel={modelId} value={loraStack} onChange={setLoraStack} bare />
            </RailSection>

            <RailSection title="Prompt">
                <TextField
                    fullWidth multiline minRows={2} maxRows={6} size="small" value={prompt}
                    placeholder={view === 'probe' ? DEFAULT_PROBE_PROMPT
                        : 'Describe the sound — then bend what the model does with it…'}
                    onChange={(e) => setPrompt(e.target.value)}
                />
                {stepsField}
            </RailSection>

            {(bentAudio || cleanAudio) && (
                <RailSection title="Takes">
                    <BendTakes
                        compact
                        isDocker={isDocker}
                        onMessage={setStatusMsg}
                        autoPlay={autoPlay}
                        takes={[
                            bentAudio && {
                                ...bentAudio, key: 'bent', title: 'Bent',
                                color: accent, titleColor: accent,
                            },
                            cleanAudio && { ...cleanAudio, key: 'clean', title: 'Clean' },
                        ].filter(Boolean)}
                    />
                    {/* the find: step nearby, name it, or open the case */}
                    {view === 'probe' && lastTake?.source === 'probe' && bentAudio && (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                <TextField
                                    size="small" value={findName} sx={{ flex: 1, minWidth: 0 }}
                                    placeholder="Name what you heard…"
                                    inputProps={{ maxLength: 60 }}
                                    onChange={(e) => setFindName(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === 'Enter') keepFind(); }}
                                />
                                <Tooltip title={TIPS.bend.keep}>
                                    <span>
                                        <Button size="small" variant="contained" color="bend"
                                                startIcon={<KeepIcon size={14} />}
                                                disabled={!findName.trim()} onClick={keepFind}>
                                            Keep
                                        </Button>
                                    </span>
                                </Tooltip>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                <Tooltip title={TIPS.bend.nearby}>
                                    <span>
                                        <Button size="small" variant="outlined" disabled={!!busy}
                                                startIcon={<NearbyIcon size={14} />} onClick={nearby}>
                                            Nearby
                                        </Button>
                                    </span>
                                </Tooltip>
                                <Tooltip title={TIPS.bend.openInRack}>
                                    <Button size="small" variant="text" startIcon={<RackIcon size={14} />}
                                            disabled={!!busy} onClick={openInRack}>
                                        Open in rack
                                    </Button>
                                </Tooltip>
                            </Box>
                        </Box>
                    )}
                </RailSection>
            )}
        </Box>
    );

    return (
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', lg: 'row' }, alignItems: 'flex-start', gap: 2 }}>
        <Box sx={{ flex: 1, minWidth: 0, width: '100%' }}>
        <Paper sx={appStyles.elevatedInfoCard}>
            <Box sx={appStyles.sectionCardHeader}>
                <Box component="span" sx={appStyles.sectionCardIcon}>
                    <CircuitBoardIcon size={20} />
                </Box>
                <Typography variant="h6" sx={appStyles.sectionCardTitle}>Bend</Typography>
                <Box sx={{ flex: 1 }} />
                {advanced && (
                    <ToggleButtonGroup
                        size="small" exclusive value={mode}
                        onChange={(_, v) => v && setMode(v)}
                        sx={{ '& .MuiToggleButton-root': { px: 2, py: 0.4 } }}
                    >
                        <Tooltip title={TIPS.bend.modeBend}>
                            <ToggleButton value="bend">Bend</ToggleButton>
                        </Tooltip>
                        <Tooltip title={TIPS.bend.modeBreak}>
                            <ToggleButton value="break">Break</ToggleButton>
                        </Tooltip>
                    </ToggleButtonGroup>
                )}
            </Box>

            {mode === 'break' ? (
                <BreakPanel defaultBase={
                    modelId && modelId.endsWith('-base') ? modelId : 'sa3-small-music-base'} />
            ) : (
            <>
                {/* toolbar: the view, and the Rack's tools */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    <ToggleButtonGroup
                        size="small" exclusive value={view}
                        onChange={(_, v) => v && setView(v)}
                        sx={{ '& .MuiToggleButton-root': { px: 1.5, py: 0.4 } }}
                    >
                        <Tooltip title={TIPS.bend.viewProbe}>
                            <ToggleButton value="probe">Board</ToggleButton>
                        </Tooltip>
                        <Tooltip title={TIPS.bend.viewRack}>
                            <ToggleButton value="rack">Rack</ToggleButton>
                        </Tooltip>
                    </ToggleButtonGroup>
                    <Box sx={{ flex: 1 }} />
                    {view === 'rack' && (<>
                    <Tooltip title={TIPS.bend.presets}>
                        <Button size="small" variant="outlined" startIcon={<SaveIcon size={14} />}
                                onClick={(e) => setPresetAnchor(e.currentTarget)}>
                            Presets
                        </Button>
                    </Tooltip>
                    <Menu anchorEl={presetAnchor} open={!!presetAnchor}
                          onClose={() => setPresetAnchor(null)}>
                        <MenuItem onClick={savePreset}>Save current rack…</MenuItem>
                        {presets.length > 0 && <MenuItem disabled>— load —</MenuItem>}
                        {presets.map(p => (
                            <MenuItem key={p.name} onClick={() => loadPreset(p)}
                                      sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
                                <span>{p.name} <Typography component="span" variant="caption" color="textSecondary">
                                    ({p.modules})</Typography></span>
                                <Typography variant="caption" color="error"
                                            onClick={(e) => deletePreset(p, e)}
                                            sx={{ cursor: 'pointer' }}>delete</Typography>
                            </MenuItem>
                        ))}
                    </Menu>
                    <Tooltip title={TIPS.bend.chance}>
                        <span>
                            <Button size="small" variant="contained" color="bend"
                                    startIcon={<DicesIcon size={14} />}
                                    disabled={!registry}
                                    onClick={(e) => setChanceAnchor(e.currentTarget)}>
                                Chance
                            </Button>
                        </span>
                    </Tooltip>
                    <Menu anchorEl={chanceAnchor} open={!!chanceAnchor}
                          onClose={() => setChanceAnchor(null)}>
                        <MenuItem onClick={() => doChance('nudge')}>Nudge — one gentle bend</MenuItem>
                        <MenuItem onClick={() => doChance('bend')}>Bend — a couple, committed</MenuItem>
                        <MenuItem onClick={() => doChance('snap')}>Snap — no hypothesis, all in</MenuItem>
                    </Menu>
                    <Tooltip title={TIPS.bend.unbend}>
                        <Button size="small" variant="outlined" startIcon={<UnbendIcon size={14} />}
                                onClick={unbend} disabled={!modules.length}>
                            Unbend
                        </Button>
                    </Tooltip>
                    </>)}
                </Box>

                {view === 'probe' ? (
                <>
                    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center', mb: 1 }}>
                        <Tooltip title={TIPS.bend.dryWet}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 190 }}>
                                <Typography variant="caption" color="textSecondary">dry/wet</Typography>
                                <Slider size="small" value={wet} min={0.05} max={1} step={0.05}
                                        color="bend" sx={{ width: 110, color: 'bend.main' }}
                                        onChange={(_, v) => setWet(v)}
                                        onChangeCommitted={(_, v) => {
                                            const pads = circuit();
                                            // A re-listen at the new mix: same input.
                                            if (pads.length && !busy) probe(pads, null, v, { fresh: false });
                                        }} />
                            </Box>
                        </Tooltip>
                        <Tooltip title={TIPS.bend.duration}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 150 }}>
                                <Typography variant="caption" color="textSecondary">dur</Typography>
                                <Slider size="small" value={probeDuration} min={2} max={15} step={1}
                                        onChange={(_, v) => setProbeDuration(v)} valueLabelDisplay="auto"
                                        sx={{ width: 80 }} />
                                <Typography variant="caption">{probeDuration}s</Typography>
                            </Box>
                        </Tooltip>
                        <Box sx={{ flex: 1 }} />
                        <Tooltip title={TIPS.bend.newInput}>
                            <span>
                                <Button size="small" variant="text" startIcon={<NewInputIcon size={14} />}
                                        disabled={!!busy || freshInput} onClick={() => setProbeSeed(randomSeed32())}>
                                    New seed
                                </Button>
                            </span>
                        </Tooltip>
                        {/* Same control and wording as the Rack's seed checkbox. */}
                        <Tooltip title={TIPS.bend.freshInput}>
                            <FormControlLabel
                                sx={{ ml: -0.5, mr: 1 }}
                                control={<Checkbox size="small" checked={freshInput}
                                                   onChange={(e) => setFreshInput(e.target.checked)} />}
                                label={<Typography variant="caption">Random seed</Typography>}
                            />
                        </Tooltip>
                        <Tooltip title={TIPS.bend.clearBoard}>
                            <span>
                                <Button size="small" variant="text" startIcon={<UnbendIcon size={14} />}
                                        disabled={!!busy || (!held.length && !allWires.length)}
                                        onClick={clearBoard}>
                                    Clear board
                                </Button>
                            </span>
                        </Tooltip>
                        {busy && (
                            <Button size="small" variant="outlined" color="error"
                                    startIcon={<StopIcon size={14} />} onClick={stopGeneration}>
                                Stop
                            </Button>
                        )}
                        <Tooltip title={TIPS.bend.showWiring}>
                            <FormControlLabel
                                sx={{ ml: 0.5, mr: 0 }}
                                control={<Switch size="small" checked={showWiring}
                                                 onChange={(e) => setShowWiring(e.target.checked)} />}
                                label={<Typography variant="caption">Show wiring</Typography>}
                            />
                        </Tooltip>
                    </Box>
                    <ProbeBoard
                        registry={registry}
                        loading={!!busy && loadingWeights}
                        names={boardNames}
                        held={held}
                        lastTouched={lastTouched}
                        wires={allWires}
                        showWiring={showWiring}
                        disabled={!!busy || !modelId}
                        onTouch={touchPad}
                        onWire={solderWire}
                        onForget={forgetFind}
                        onDisconnect={disconnectWire}
                    />
                </>
                ) : (
                <>
                {/* signal path */}
                <SignalPath registry={registry} modules={modules} onAddModule={addModule} />

                {/* rack */}
                {modules.length === 0 ? (
                    <Box sx={{
                        textAlign: 'center', py: 3, px: 2, mb: 1.5,
                        border: '1px dashed', borderColor: 'divider', borderRadius: 2.5,
                    }}>
                        <Typography variant="body2" color="textSecondary">
                            The rack is empty. Click a stage on the signal path to
                            attach a bend module — or press <b>Chance</b> and just listen.
                            No theory required; that's the point.
                        </Typography>
                    </Box>
                ) : (
                    <Box sx={{ mb: 1.5 }}>
                        {modules.map((m, i) => (
                            <BendModuleCard
                                key={m.id}
                                module={m}
                                registry={registry}
                                onChange={(next) => updateModule(m.id, next)}
                                onRemove={() => removeModule(m.id)}
                                onRandomize={() => randomizeModule(m.id)}
                                onMoveUp={() => moveModule(m.id, -1)}
                                onMoveDown={() => moveModule(m.id, +1)}
                                isFirst={i === 0}
                                isLast={i === modules.length - 1}
                            />
                        ))}
                    </Box>
                )}

                {/* generate strip — the prompt lives in the rail */}
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center', mb: 1.5 }}>
                    <Tooltip title={TIPS.bend.duration}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 170 }}>
                            <Typography variant="caption" color="textSecondary">dur</Typography>
                            <Slider size="small" value={duration} min={1} max={30} step={1}
                                    onChange={(_, v) => setDuration(v)} valueLabelDisplay="auto"
                                    sx={{ width: 100 }} />
                            <Typography variant="caption">{duration}s</Typography>
                        </Box>
                    </Tooltip>
                    <FormControlLabel
                        control={<Checkbox size="small" checked={randomSeed}
                                           onChange={(e) => setRandomSeed(e.target.checked)} />}
                        label={<Typography variant="caption">Random seed</Typography>}
                    />
                    {!randomSeed && (
                        <TextField size="small" label="Seed" type="number" value={seed}
                                   onChange={(e) => setSeed(e.target.value)} sx={{ width: 120 }} />
                    )}
                    <Box sx={{ flex: 1 }} />
                    {busy ? (
                        <Button variant="outlined" color="error" startIcon={<StopIcon size={14} />}
                                onClick={stopGeneration}>
                            Stop
                        </Button>
                    ) : (
                        <>
                            <Tooltip title={TIPS.bend.nearby}>
                                <span>
                                    <Button variant="outlined" disabled={lastTake?.source !== 'rack' || !activeCount}
                                            startIcon={<NearbyIcon size={14} />}
                                            onClick={nearby}>
                                        Nearby
                                    </Button>
                                </span>
                            </Tooltip>
                            <Tooltip title={TIPS.bend.abCompare}>
                                <span>
                                    <Button variant="outlined" disabled={!bentAudio}
                                            onClick={() => generate(false)}>
                                        Clean A/B
                                    </Button>
                                </span>
                            </Tooltip>
                            <Tooltip title={TIPS.bend.generate}>
                                <span>
                                    <Button variant="contained" color="bend"
                                            disabled={!activeCount || !modelId}
                                            onClick={() => generate(true)}>
                                        Generate bent
                                    </Button>
                                </span>
                            </Tooltip>
                        </>
                    )}
                </Box>
                </>
                )}

                {busy && (
                    <LinearProgress variant={loadingWeights ? 'indeterminate' : 'determinate'}
                                    value={progress} sx={{ mb: loadingWeights && view === 'rack' ? 0.5 : 1.5 }} />
                )}
                {/* The board carries its own loading overlay. */}
                {busy && loadingWeights && view === 'rack' && (
                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 1.5 }}>
                        Loading model weights…
                    </Typography>
                )}
                {/* What the bend did that the user should know (level
                    reduced, silenced, a module that never acted), or why a
                    request failed: the app's bottom-corner notice, gone after
                    a few seconds. */}
                <Portal>
                    <Snackbar
                        open={!!notice}
                        autoHideDuration={6000}
                        onClose={(_e, reason) => { if (reason !== 'clickaway') setNotice(null); }}
                        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                    >
                        {notice ? (
                            <Alert severity={notice.severity} variant="filled" onClose={() => setNotice(null)}
                                   sx={{ minWidth: 280, maxWidth: 480, boxShadow: 6 }}>
                                {notice.lines.map((w, i) => <Box key={i}>{w}</Box>)}
                            </Alert>
                        ) : undefined}
                    </Snackbar>
                </Portal>
                {/* The Rack talks back; the board doesn't (progress is on the bar). */}
                {statusMsg && view === 'rack' && (
                    <Typography variant="caption" color="textSecondary"
                                sx={{ display: 'block', mb: 1.5 }}>
                        {statusMsg}
                    </Typography>
                )}

                {/* Bending Log */}
                <Accordion disableGutters expanded={logOpen}
                           onChange={(_, v) => { setLogOpen(v); if (v) refreshLog(); }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon size={18} />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LogIcon size={16} />
                            <Typography variant="subtitle1">Bending Log</Typography>
                            {logEntries.length > 0 && (
                                <Chip size="small" label={logEntries.length}
                                      sx={{ height: 18, fontSize: '0.65rem' }} />
                            )}
                        </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 0.5 }}>
                            <Tooltip title={TIPS.bend.logAll}>
                                <FormControlLabel
                                    sx={{ mr: 0 }}
                                    control={<Switch size="small" checked={logAll}
                                                     onChange={(e) => setLogAll(e.target.checked)} />}
                                    label={<Typography variant="caption">All models</Typography>}
                                />
                            </Tooltip>
                        </Box>
                        <BendingLogPanel
                            entries={logEntries}
                            registry={registry}
                            modelId={modelId}
                            showWiring={view === 'rack' || showWiring}
                            onRecall={recallLogEntry}
                            onChanged={refreshLog}
                        />
                    </AccordionDetails>
                </Accordion>
            </>
            )}
        </Paper>
        </Box>
        {rail}
        </Box>
    );
}
